import os
import requests
import time
from datetime import datetime
import pandas as pd
import numpy as np
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import sklearn
import queue  # For thread-safe command communication

# Import SQLAlchemy components
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

# Import for Telegram bot functionality
import asyncio
from telegram import Bot
from telegram.error import TelegramError

# Import PyTorch for the centralized critic
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------- Global Settings -------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
engine = create_engine("postgresql://postgres:Rw.1010419@localhost:5432/watchlist")
VERBOSE = False  # Set to True for detailed per-token logs

# Global initialization variables for the progress bar
init_complete = False
init_message = ""
init_progress = 0.0  # Value between 0 and 1

# Global dictionaries
persistent_agents = {}       # Persist agents per token
global_trade_stats = {}      # { token: { 'sum': total_outcome, 'count': total_trades } }
global_trade_outcomes = {}   # Latest trade profit/loss outcomes
global_held_trades = {}      # For currently active trades
held_trades_lock = threading.Lock()  # Lock for active trade dictionary

# ------------------------- Helper Functions -------------------------
def is_complete_token(entry):
    """Return True if the token entry has all required fields."""
    required_fields = [
        "chainId", "dexId", "url", "pairAddress", "baseToken", "quoteToken",
        "priceNative", "priceUsd", "liquidity", "fdv", "marketCap", "pairCreatedAt",
        "info", "boosts"
    ]
    for field in required_fields:
        if field not in entry or not entry[field]:
            return False
    for token_field in ["baseToken", "quoteToken"]:
        for subfield in ["address", "name", "symbol"]:
            if subfield not in entry[token_field] or not entry[token_field][subfield]:
                return False
    liquidity = entry["liquidity"]
    for subfield in ["usd", "base", "quote"]:
        if subfield not in liquidity or not liquidity[subfield]:
            return False
    if "imageUrl" not in entry["info"] or not entry["info"]["imageUrl"]:
        return False
    if "active" not in entry["boosts"]:
        return False
    return True

def clear_console():
    """Clears the console."""
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

# ------------------------- Aggregation (Seeder) Code -------------------------
class DataMining:
    """
    Retrieves token data and seeds the database.
    Only tokens with complete data are used.
    """
    def __init__(self, source):
        self.source = source
        self.data = []
        self.latest_boosted_tokens_data = []
        self.most_active_boosts_data = []
        self.latest_boosted_token_addresses = []
        self.most_active_boosted_token_addresses = []
        self.blacklisted_tokens = []
        self.token_details = []
        self.config = {}
        self.init_db()
        global init_message
        init_message = "Database initialized successfully."
    
    def init_db(self):
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS watchlist (
                chainid TEXT,
                dexid TEXT,
                url TEXT,
                pairaddress TEXT,
                labels TEXT[],
                basetoken_address TEXT,
                basetoken_name TEXT,
                basetoken_symbol TEXT,
                quotetoken_address TEXT,
                quotetoken_name TEXT,
                quotetoken_symbol TEXT,
                pricenative NUMERIC,
                priceusd NUMERIC,
                liquidity_usd NUMERIC,
                liquidity_base NUMERIC,
                liquidity_quote NUMERIC,
                fdv NUMERIC,
                marketcap NUMERIC,
                paircreatedat BIGINT,
                imageurl TEXT,
                websites TEXT[],
                socials TEXT[],
                boosts_active INT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            with engine.connect() as conn:
                conn.execute(create_table_query)
                conn.commit()
        except SQLAlchemyError as e:
            pass
    
    def fetch_latest_boosted_tokens(self):
        global init_message, init_progress
        init_message = "Fetching latest boosted tokens..."
        response = requests.get("https://api.dexscreener.com/token-boosts/latest/v1", headers={})
        if response.status_code != 200:
            init_message = f"Warning: Failed to fetch latest tokens (HTTP {response.status_code})"
            return []
        self.latest_boosted_tokens_data = response.json()
        self.latest_boosted_token_addresses = [
            {"tokenAddress": entry["tokenAddress"], "chainId": entry.get("chainId", "default_chain"), "blacklistStatus": None}
            for entry in self.latest_boosted_tokens_data[:20]
        ]
        init_progress = 0.25
        init_message = "Latest boosted tokens fetched."
        return self.latest_boosted_tokens_data

    def fetch_top_boosted_tokens(self):
        global init_message, init_progress
        init_message = "Fetching top boosted tokens..."
        response = requests.get("https://api.dexscreener.com/token-boosts/top/v1", headers={})
        if response.status_code != 200:
            init_message = f"Warning: Failed to fetch top tokens (HTTP {response.status_code})"
            return []
        self.most_active_boosts_data = response.json()
        self.most_active_boosted_token_addresses = [
            {"tokenAddress": entry["tokenAddress"], "chainId": entry.get("chainId", "default_chain"), "blacklistStatus": None}
            for entry in self.most_active_boosts_data[:20]
        ]
        init_progress = 0.5
        init_message = "Top boosted tokens fetched."
        return self.most_active_boosts_data

    def fetch_token_details(self):
        global init_message, init_progress
        desired_count = 20
        init_message = "Fetching token details..."
        while True:
            token_list = self.latest_boosted_token_addresses + self.most_active_boosted_token_addresses
            tokens_by_chain = {}
            for token in token_list:
                chain = token.get("chainId", "default_chain")
                tokens_by_chain.setdefault(chain, []).append(token["tokenAddress"])
            init_message = "Grouped tokens by chain."
            details = []
            for chain, addresses in tokens_by_chain.items():
                url = f"https://api.dexscreener.com/tokens/v1/{chain}/{','.join(addresses)}"
                init_message = f"Requesting token details for chain {chain}..."
                response = requests.get(url, headers={})
                if response.status_code != 200:
                    init_message = f"Warning: Failed to fetch details for chain {chain} (HTTP {response.status_code})"
                    continue
                token_data = response.json()
                for entry in token_data:
                    if is_complete_token(entry):
                        details.append({"symbol": entry["baseToken"]["symbol"], "data": entry})
            if len(details) >= desired_count:
                break
            else:
                init_message = f"Only {len(details)} complete tokens found; retrying in 10 seconds..."
                self.fetch_latest_boosted_tokens()
                self.fetch_top_boosted_tokens()
                time.sleep(10)
        self.token_details = details
        init_progress = 0.75
        init_message = "Complete token details fetched."
        return self.token_details

    def seed_watchlist(self, token_details):
        global init_message, init_progress, init_complete
        if not token_details:
            init_message = "No token details available to seed the watchlist."
            return
        init_message = "Seeding watchlist..."
        insert_query = text("""
            INSERT INTO watchlist (
                chainid, dexid, url, pairaddress, labels, 
                basetoken_address, basetoken_name, basetoken_symbol,
                quotetoken_address, quotetoken_name, quotetoken_symbol,
                pricenative, priceusd, liquidity_usd, liquidity_base, 
                liquidity_quote, fdv, marketcap, paircreatedat,
                imageurl, websites, socials, boosts_active, timestamp
            ) VALUES (
                :chainid, :dexid, :url, :pairaddress, :labels, 
                :basetoken_address, :basetoken_name, :basetoken_symbol,
                :quotetoken_address, :quotetoken_name, :quotetoken_symbol,
                :pricenative, :priceusd, :liquidity_usd, :liquidity_base, 
                :liquidity_quote, :fdv, :marketcap, :paircreatedat,
                :imageurl, :websites, :socials, :boosts_active, CURRENT_TIMESTAMP
            )
        """)
        try:
            with engine.begin() as conn:
                for token in token_details:
                    entry = token["data"]
                    required_fields = [
                        "chainId", "dexId", "url", "pairAddress", "baseToken", "quoteToken",
                        "priceNative", "priceUsd", "liquidity", "fdv", "marketCap",
                        "pairCreatedAt", "info", "boosts"
                    ]
                    missing_fields = [field for field in required_fields if field not in entry or not entry[field]]
                    if missing_fields:
                        continue
                    params = {
                        "chainid": entry["chainId"],
                        "dexid": entry["dexId"],
                        "url": entry["url"],
                        "pairaddress": entry["pairAddress"],
                        "labels": entry.get("labels", []),
                        "basetoken_address": entry["baseToken"]["address"],
                        "basetoken_name": entry["baseToken"]["name"],
                        "basetoken_symbol": entry["baseToken"]["symbol"],
                        "quotetoken_address": entry["quoteToken"]["address"],
                        "quotetoken_name": entry["quoteToken"]["name"],
                        "quotetoken_symbol": entry["quoteToken"]["symbol"],
                        "pricenative": entry["priceNative"],
                        "priceusd": entry["priceUsd"],
                        "liquidity_usd": entry["liquidity"]["usd"],
                        "liquidity_base": entry["liquidity"]["base"],
                        "liquidity_quote": entry["liquidity"]["quote"],
                        "fdv": entry["fdv"],
                        "marketcap": entry["marketCap"],
                        "paircreatedat": entry["pairCreatedAt"],
                        "imageurl": entry["info"]["imageUrl"],
                        "websites": [site["url"] for site in entry["info"].get("websites", [])],
                        "socials": [social.get("platform") for social in entry["info"].get("socials", [])],
                        "boosts_active": entry["boosts"]["active"]
                    }
                    conn.execute(insert_query, params)
            init_progress = 0.90
            init_message = "Seeding complete. Waiting 3 minutes for data stabilization..."
        except SQLAlchemyError as e:
            init_message = f"Error during seeding: {e}"

    def post_seed_wait(self):
        global init_message, init_progress, init_complete
        wait_time = 180  # 3 minutes
        start = time.time()
        while time.time() - start < wait_time:
            elapsed = time.time() - start
            init_progress = 0.90 + 0.10 * (elapsed / wait_time)
            time.sleep(1)
        init_progress = 1.0
        init_message = "Initialization complete."
        init_complete = True

    def clock_function(self):
        while True:
            self.fetch_token_details()
            self.seed_watchlist(self.token_details)
            time.sleep(60)

    def run(self):
        self.fetch_latest_boosted_tokens()
        self.fetch_top_boosted_tokens()
        token_details = self.fetch_token_details()
        if token_details:
            self.token_details = token_details
        else:
            init_message = "Warning: No token details were fetched; seeding will be skipped."
        self.seed_watchlist(self.token_details)
        self.post_seed_wait()  # 3-minute wait post-seeding
        self.clock_function()

# ------------------------- Global Shared Objects -------------------------
class GlobalKnowledge:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()
    def update(self, token_address, agent_results):
        with self.lock:
            self.data[token_address] = agent_results
    def get_all(self):
        with self.lock:
            return self.data.copy()
    def update_field(self, token, field, value):
        with self.lock:
            if token in self.data:
                self.data[token][field] = value

global_knowledge = GlobalKnowledge()

# ------------------------- Trade Statistics -------------------------
def update_trade_stats(token, outcome):
    if token not in global_trade_stats:
        global_trade_stats[token] = {'sum': 0.0, 'count': 0}
    global_trade_stats[token]['sum'] += outcome
    global_trade_stats[token]['count'] += 1

def get_trade_ratio(token):
    if token in global_trade_stats and global_trade_stats[token]['count'] > 0:
        return global_trade_stats[token]['sum'] / global_trade_stats[token]['count']
    else:
        return 0

# ------------------------- Configuration and Agent Classes -------------------------
class TradingConfig:
    def __init__(self,
                 profit_threshold=0.01,
                 reward_value=0.05,
                 punishment_value=0.05,
                 idle_duration=5,           # Duration to remain idle
                 bidding_duration=5,
                 holding_duration=10,
                 positive_trade_multiplier_rate=0.15,
                 negative_trade_multiplier_rate=0.1,
                 signal_refresh_interval=120,
                 dashboard_refresh_interval=1):  # Dashboard refresh interval in seconds
        self.profit_threshold = profit_threshold
        self.reward_value = reward_value
        self.punishment_value = punishment_value
        self.idle_duration = idle_duration
        self.bidding_duration = bidding_duration
        self.holding_duration = holding_duration
        self.positive_trade_multiplier_rate = positive_trade_multiplier_rate
        self.negative_trade_multiplier_rate = negative_trade_multiplier_rate
        self.signal_refresh_interval = signal_refresh_interval
        self.dashboard_refresh_interval = dashboard_refresh_interval

class TradingAgent:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.weight_score = 0.80
    def update_reward(self, predicted_action: str, actual_profit: float, confidence: float,
                      reward_multiplier=1.0, punishment_multiplier=1.0):
        if predicted_action == 'buy':
            if actual_profit > self.config.profit_threshold:
                self.weight_score += self.config.reward_value * confidence * reward_multiplier
            else:
                self.weight_score -= self.config.punishment_value * (1 - confidence) * punishment_multiplier
        elif predicted_action == 'sell':
            if actual_profit < -self.config.profit_threshold:
                self.weight_score += self.config.reward_value * confidence * reward_multiplier
            else:
                self.weight_score -= self.config.punishment_value * (1 - confidence) * punishment_multiplier
        elif predicted_action == 'hold':
            if abs(actual_profit) < self.config.profit_threshold:
                self.weight_score += self.config.reward_value * confidence * reward_multiplier
            else:
                self.weight_score -= self.config.punishment_value * (1 - confidence) * punishment_multiplier
        self.weight_score = max(0.0, min(self.weight_score, 1.0))

# --- All Agents include a global_bias parameter in predict and force string output ---
class ReverseForestAgent(TradingAgent):
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
    def predict(self, sample: pd.DataFrame, global_bias=0):
        pred_probs = self.model.predict_proba(sample)[0]
        classes = list(self.model.classes_)
        try:
            buy_idx = classes.index('buy')
            sell_idx = classes.index('sell')
        except ValueError:
            buy_idx = None
            sell_idx = None
        if global_bias != 0 and buy_idx is not None and sell_idx is not None:
            bias_factor = 0.5
            bias = global_bias * bias_factor
            if bias > 0:
                pred_probs[buy_idx] += bias
                pred_probs[sell_idx] -= bias
            else:
                pred_probs[buy_idx] -= abs(bias)
                pred_probs[sell_idx] += abs(bias)
        pred_idx = np.argmax(pred_probs)
        action = str(classes[pred_idx])
        confidence = pred_probs[pred_idx] / (np.sum(pred_probs) + 1e-6)
        return action, confidence

class RLAgent(TradingAgent):
    def __init__(self, config: TradingConfig, num_bins=10, actions=['idle', 'buy', 'sell', 'hold']):
        super().__init__(config)
        self.num_bins = num_bins
        self.actions = actions  # Includes "idle"
        self.q_table = np.random.rand(num_bins, len(actions))
        self.alpha = 0.1
        self.gamma = 0.95
        self.bins = None
    def train(self, price_series: pd.Series, price_change_series: pd.Series):
        self.bins = np.linspace(price_series.min(), price_series.max(), self.num_bins + 1)
        price_bins = np.digitize(price_series, self.bins) - 1
        for i in range(1, len(price_series)):
            current_state = min(price_bins[i - 1], self.num_bins - 1)
            next_state = min(price_bins[i], self.num_bins - 1)
            if price_change_series.iloc[i] > self.config.profit_threshold:
                reward_vector = [0, 1, -1, 0]
            elif price_change_series.iloc[i] < -self.config.profit_threshold:
                reward_vector = [0, -1, 1, 0]
            else:
                reward_vector = [0, 0, 0, 1]
            action_idx = np.argmax(self.q_table[current_state])
            best_next_q = np.max(self.q_table[next_state])
            self.q_table[current_state, action_idx] += self.alpha * (
                reward_vector[action_idx] + self.gamma * best_next_q - self.q_table[current_state, action_idx]
            )
    def predict(self, current_price: float, global_bias=0):
        if self.bins is None:
            raise ValueError("The agent must be trained before prediction.")
        state = np.digitize([current_price], self.bins) - 1
        state = min(state[0], self.num_bins - 1)
        q_values = self.q_table[state].copy()
        if global_bias != 0:
            bias_factor = 0.5
            bias = global_bias * bias_factor
            if bias > 0:
                q_values[1] += bias
                q_values[2] -= bias
            else:
                q_values[1] -= abs(bias)
                q_values[2] += abs(bias)
        action_idx = np.argmax(q_values)
        action = str(self.actions[action_idx])
        confidence = q_values[action_idx] / (np.sum(q_values) + 1e-6)
        return action, confidence

class XGBoostAgent(TradingAgent):
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.model = XGBClassifier(verbosity=0, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        self.le = LabelEncoder()
    def train(self, X: pd.DataFrame, y: pd.Series):
        y_encoded = self.le.fit_transform(y)
        self.model.fit(X, y_encoded)
    def predict(self, sample: pd.DataFrame, global_bias=0):
        pred_probs = self.model.predict_proba(sample)[0]
        classes = list(self.model.classes_)
        try:
            buy_idx = classes.index('buy')
            sell_idx = classes.index('sell')
        except ValueError:
            buy_idx = None
            sell_idx = None
        if global_bias != 0 and buy_idx is not None and sell_idx is not None:
            bias_factor = 0.5
            bias = global_bias * bias_factor
            if bias > 0:
                pred_probs[buy_idx] += bias
                pred_probs[sell_idx] -= bias
            else:
                pred_probs[buy_idx] -= abs(bias)
                pred_probs[sell_idx] += abs(bias)
        pred_idx = np.argmax(pred_probs)
        action = str(classes[pred_idx])
        confidence = pred_probs[pred_idx] / (np.sum(pred_probs) + 1e-6)
        return action, confidence

# ------------------------- Token Processing -------------------------
def process_token(token_address: str, config: TradingConfig):
    """
    Continuously processes a token.
    Uses persistent agent instances so that each token's agents learn over time.
    Enforces an idle phase before bidding.
    Updates trade statistics to compute a trade ratio, which scales a global bias.
    All agents apply the same global bias method.
    """
    global persistent_agents
    if token_address not in persistent_agents:
        persistent_agents[token_address] = {
            'rf': ReverseForestAgent(config),
            'rl': RLAgent(config, num_bins=10),
            'xgb': XGBoostAgent(config),
            'global_value': 0
        }
    agents = persistent_agents[token_address]

    while True:
        try:
            query = f"""
            SELECT chainid, dexid, url, pairaddress, labels, 
                   basetoken_address, basetoken_name, basetoken_symbol,
                   quotetoken_address, quotetoken_name, quotetoken_symbol,
                   pricenative, priceusd, liquidity_usd, liquidity_base, 
                   liquidity_quote, fdv, marketcap, paircreatedat,
                   imageurl, websites, socials, boosts_active, timestamp
            FROM watchlist
            WHERE basetoken_address = '{token_address}' OR quotetoken_address = '{token_address}'
            ORDER BY timestamp ASC;
            """
            df = pd.read_sql(query, engine)
            df.columns = df.columns.str.lower()
        except Exception as e:
            time.sleep(5)
            continue

        if df.empty:
            time.sleep(5)
            continue

        required_columns = ['priceusd', 'marketcap', 'liquidity_usd']
        if not all(col in df.columns for col in required_columns):
            time.sleep(5)
            continue

        features = ['priceusd', 'liquidity_usd', 'marketcap']
        X = df[features].copy()
        for col in features:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        try:
            price_change = X['priceusd'].pct_change().fillna(0)
        except Exception as e:
            time.sleep(5)
            continue
        y = np.where(price_change > 0.01, 'buy', np.where(price_change < -0.01, 'sell', 'hold'))
        y = pd.Series(y, index=X.index)
        latest_sample = X.iloc[[-1]]
        latest_price = latest_sample['priceusd'].values[0]

        # --- Idle Phase ---
        idle_action = "idle"
        instance_results = {
            'rf': {'action': idle_action, 'confidence': 1.0, 'weight': agents['rf'].weight_score},
            'rl': {'action': idle_action, 'confidence': 1.0, 'weight': agents['rl'].weight_score},
            'xgb': {'action': idle_action, 'confidence': 1.0, 'weight': agents['xgb'].weight_score},
            'token_symbol': df['basetoken_symbol'].iloc[0] if 'basetoken_symbol' in df.columns else token_address,
            'global_value': agents.get('global_value', 0)
        }
        global_knowledge.update(token_address, instance_results)
        time.sleep(config.idle_duration)

        # --- Bidding Phase ---
        agents['rf'].train(X, y)
        agents['rl'].train(X['priceusd'], price_change)
        agents['xgb'].train(X, y)

        trade_ratio = get_trade_ratio(token_address)
        global_value = agents.get('global_value', 0)
        bias_factor = 0.5
        global_bias = global_value * trade_ratio * bias_factor

        rf_action, rf_confidence = agents['rf'].predict(latest_sample, global_bias=global_bias)
        rl_action, rl_confidence = agents['rl'].predict(latest_price, global_bias=global_bias)
        xgb_action, xgb_confidence = agents['xgb'].predict(latest_sample, global_bias=global_bias)
        token_symbol = df['basetoken_symbol'].iloc[0] if 'basetoken_symbol' in df.columns else token_address

        instance_results = {
            'rf': {'action': rf_action, 'confidence': rf_confidence, 'weight': agents['rf'].weight_score},
            'rl': {'action': rl_action, 'confidence': rl_confidence, 'weight': agents['rl'].weight_score},
            'xgb': {'action': xgb_action, 'confidence': xgb_confidence, 'weight': agents['xgb'].weight_score},
            'token_symbol': token_symbol,
            'global_value': agents.get('global_value', 0)
        }
        global_knowledge.update(token_address, instance_results)

        if VERBOSE:
            print(f"[{token_symbol}] Recs: RF={rf_action}({rf_confidence:.2f}), RL={rl_action}({rl_confidence:.2f}), XGB={xgb_action}({xgb_confidence:.2f})")
            consensus = max(set([rf_action, rl_action, xgb_action]), key=lambda x: str(x))
            print(f"[{token_symbol}] Consensus: {str(consensus).upper()}")
            print(f"[{token_symbol}] Weights: RF={agents['rf'].weight_score:.2f}, RL={agents['rl'].weight_score:.2f}, XGB={agents['xgb'].weight_score:.2f}")
            print(f"[{token_symbol}] Global Bias: {global_bias:.4f}")
            print(f"[{token_symbol}] Entering bidding phase for {config.bidding_duration} seconds.")

        time.sleep(config.bidding_duration)
        trade_start_time = time.time()
        with held_trades_lock:
            global_held_trades[token_address] = {
                "token_symbol": token_symbol,
                "start_time": trade_start_time,
                "holding_duration": config.holding_duration
            }
        if VERBOSE:
            print(f"[{token_symbol}] Trade executed. Holding for {config.holding_duration} seconds.")
        time.sleep(config.holding_duration)
        with held_trades_lock:
            if token_address in global_held_trades:
                del global_held_trades[token_address]
        simulated_profit = random.uniform(-0.05, 0.05)
        if VERBOSE:
            print(f"[{token_symbol}] Trade closed. Profit: {simulated_profit*100:.2f}%")
        global_trade_outcomes[token_address] = simulated_profit
        update_trade_stats(token_address, simulated_profit)
        if simulated_profit > config.profit_threshold:
            reward_multiplier = 1 + (config.holding_duration * config.positive_trade_multiplier_rate)
            punishment_multiplier = 1.0
        else:
            reward_multiplier = 1.0
            punishment_multiplier = 1 + (config.holding_duration * config.negative_trade_multiplier_rate)
        
        agents['rf'].update_reward(rf_action, simulated_profit, rf_confidence, reward_multiplier, punishment_multiplier)
        agents['rl'].update_reward(rl_action, simulated_profit, rl_confidence, reward_multiplier, punishment_multiplier)
        agents['xgb'].update_reward(xgb_action, simulated_profit, xgb_confidence, reward_multiplier, punishment_multiplier)
        
        instance_results = {
            'rf': {'action': rf_action, 'confidence': rf_confidence, 'weight': agents['rf'].weight_score},
            'rl': {'action': rl_action, 'confidence': rl_confidence, 'weight': agents['rl'].weight_score},
            'xgb': {'action': xgb_action, 'confidence': xgb_confidence, 'weight': agents['xgb'].weight_score},
            'token_symbol': token_symbol,
            'global_value': agents.get('global_value', 0)
        }
        global_knowledge.update(token_address, instance_results)
        
        time.sleep(2)

# ------------------------- Signal Interface -------------------------
class Signal:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.command_queue = queue.Queue()
    def display_agent_weights(self):
        output = "\n------- Agent Weights -------\n"
        for token, results in global_knowledge.get_all().items():
            token_symbol = results.get('token_symbol', token)
            rf_w = results['rf']['weight']
            rl_w = results['rl']['weight']
            xgb_w = results['xgb']['weight']
            global_val = results.get('global_value', 0)
            output += f"Token: {token_symbol} | RF: {rf_w:.2f}, RL: {rl_w:.2f}, XGB: {xgb_w:.2f} | Global Value: {global_val:.4f}\n"
        output += "-----------------------------\n"
        return output
    def display_latest_signals(self):
        output = "---- Latest Signals ----\n"
        for token, results in global_knowledge.get_all().items():
            token_symbol = results.get('token_symbol', token)
            votes = [results['rf']['action'], results['rl']['action'], results['xgb']['action']]
            consensus = max(set(votes), key=lambda x: str(x))
            output += f"Token: {token_symbol} - Signal: {str(consensus).upper()}\n"
        output += "------------------------\n"
        return output
    def display_trade_outcomes(self):
        output = "\n---- Latest Trade Outcomes ----\n"
        all_data = global_knowledge.get_all()
        for token, info in all_data.items():
            token_symbol = info.get('token_symbol', token)
            outcome = global_trade_outcomes.get(token)
            if outcome is None:
                outcome_str = "N/A"
            else:
                outcome_str = f"{outcome*100:.2f}%"
            output += f"Token: {token_symbol} - Outcome: {outcome_str}\n"
        output += "--------------------------------\n"
        return output
    def display_active_trades(self):
        output = "\n---- Current Active Trades ----\n"
        with held_trades_lock:
            if not global_held_trades:
                output += "No active trades.\n"
            else:
                current_time = time.time()
                for token, trade_info in global_held_trades.items():
                    token_symbol = trade_info.get("token_symbol", token)
                    elapsed = current_time - trade_info["start_time"]
                    remaining = max(0, trade_info["holding_duration"] - elapsed)
                    output += f"Token: {token_symbol} - Time Remaining: {remaining:.1f} seconds\n"
        output += "-------------------------------\n"
        return output
    def listen_for_input(self):
        while True:
            cmd = input("Enter command (P for outcomes, C for active, Q to quit): ")
            self.command_queue.put(cmd.strip().lower())
    def run(self):
        input_thread = threading.Thread(target=self.listen_for_input, daemon=True)
        input_thread.start()
        while True:
            clear_console()  # Wipe the console completely before printing new output
            dashboard = ""
            dashboard += self.display_agent_weights()
            dashboard += self.display_latest_signals()
            dashboard += "\n---- Options ----\n"
            dashboard += "P - Display latest trade outcomes\n"
            dashboard += "C - Display current active trades\n"
            dashboard += "Q - Quit\n"
            dashboard += f"Dashboard auto-refreshes every {self.config.dashboard_refresh_interval} second(s).\n"
            print(dashboard)  # Using print ensures a newline and clears the previous content.
            try:
                cmd = self.command_queue.get_nowait()
                if cmd == 'p':
                    clear_console()
                    print(self.display_trade_outcomes())
                    input("Press ENTER to return to dashboard...")
                elif cmd == 'c':
                    clear_console()
                    print(self.display_active_trades())
                    input("Press ENTER to return to dashboard...")
                elif cmd == 'q':
                    print("Exiting Signal interface.")
                    break
            except queue.Empty:
                pass
            time.sleep(self.config.dashboard_refresh_interval)

# ------------------------- TokenLists Class -------------------------
class TokenLists:
    def __init__(self, latest_boosted_token_addresses, most_active_boosted_token_addresses):
        self.latest_boosted_token_addresses = latest_boosted_token_addresses
        self.most_active_boosted_token_addresses = most_active_boosted_token_addresses

# ------------------------- Final Database Check -------------------------
def final_database_check(required_rows=1, poll_interval=10):
    try:
        with engine.connect() as conn:
            while True:
                result = conn.execute(text("SELECT COUNT(*) FROM watchlist"))
                count = result.scalar()
                print(f"Final Check: {count} rows in watchlist.")
                if count >= required_rows:
                    break
                time.sleep(poll_interval)
    except Exception as e:
        print("Error during final database check:", e)
    print("Database seeding verified. Proceeding to start agent threads.")

# ------------------------- Initialization Display -------------------------
def display_initialization():
    bar_length = 30
    while not init_complete:
        filled_length = int(round(bar_length * init_progress))
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f"\rInitializing... [{bar}] {int(init_progress*100)}% | {init_message}")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\nInitialization complete. Launching dashboard...\n")
    sys.stdout.flush()

# ------------------------- Global Master (Centralized Critic) -------------------------
class MasterCritic(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1):
        super(MasterCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GlobalMaster:
    """
    Aggregates global state from persistent agents,
    trains a centralized critic to predict profit outcomes,
    and broadcasts a global value estimate back to local agents.
    """
    def __init__(self, input_dim, hidden_dim=32, lr=0.001, update_interval=30):
        self.critic = MasterCritic(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.update_interval = update_interval
        self.running = False
    def aggregate_global_state(self):
        global_states = []
        outcomes = []
        tokens_list = []
        data = global_knowledge.get_all()
        for token, info in data.items():
            tokens_list.append(token)
            rf_weight = info['rf']['weight']
            rl_weight = info['rl']['weight']
            xgb_weight = info['xgb']['weight']
            rf_conf = info['rf']['confidence']
            rl_conf = info['rl']['confidence']
            xgb_conf = info['xgb']['confidence']
            avg_conf = (rf_conf + rl_conf + xgb_conf) / 3
            state_vector = [rf_weight, rl_weight, xgb_weight, avg_conf]
            global_states.append(state_vector)
            outcome = global_trade_outcomes.get(token, 0)
            outcomes.append([outcome])
        if global_states:
            state_tensor = torch.tensor(global_states, dtype=torch.float)
            target_tensor = torch.tensor(outcomes, dtype=torch.float)
            return state_tensor, target_tensor, tokens_list
        else:
            return None, None, None
    def train_step(self):
        state_tensor, target_tensor, tokens_list = self.aggregate_global_state()
        if state_tensor is not None:
            self.optimizer.zero_grad()
            predictions = self.critic(state_tensor)
            loss = nn.MSELoss()(predictions, target_tensor)
            loss.backward()
            self.optimizer.step()
            print(f"Global Master Training Loss: {loss.item():.4f}")
            for token, pred in zip(tokens_list, predictions):
                global_knowledge.update_field(token, 'global_value', pred.item())
                if token in persistent_agents:
                    persistent_agents[token]['global_value'] = pred.item()
        else:
            print("Global Master: No global states to train on.")
    def run(self):
        self.running = True
        while self.running:
            self.train_step()
            time.sleep(self.update_interval)
    def stop(self):
        self.running = False

# ------------------------- Telegram Bot Class -------------------------
class TelegramBot:
    def __init__(self, token, chat_id, global_knowledge, poll_interval=30):
        self.token = token
        self.chat_id = chat_id
        self.global_knowledge = global_knowledge
        self.poll_interval = poll_interval
        self.bot = Bot(token=self.token)
        self.running = False
    def construct_message(self):
        data = self.global_knowledge.get_all()
        if not data:
            return "No agent signals available."
        message_lines = ["*Agent Signals Update:*"]
        for token, info in data.items():
            token_symbol = info.get("token_symbol", token)
            votes = [
                info["rf"]["action"],
                info["rl"]["action"],
                info["xgb"]["action"]
            ]
            consensus = max(set(votes), key=lambda x: str(x))
            global_val = info.get("global_value", 0)
            message_lines.append(f"Token: {token_symbol} -> Consensus: {str(consensus).upper()} | Global Value: {global_val:.4f}")
        return "\n".join(message_lines)
    def relay_signals(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while self.running:
            message = self.construct_message()
            try:
                loop.run_until_complete(self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="Markdown"))
            except TelegramError as e:
                print(f"Error sending message: {e}")
            time.sleep(self.poll_interval)
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.relay_signals, daemon=True)
        self.thread.start()
    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()

# ------------------------- Main Execution -------------------------
def main():
    global init_complete
    aggregator = DataMining(source="API")
    aggregator_thread = threading.Thread(target=aggregator.run, daemon=True)
    aggregator_thread.start()
    print("Aggregator thread started. Waiting 3 minutes for seeding to complete...")
    display_initialization()
    final_database_check(required_rows=1, poll_interval=10)
    
    config = TradingConfig(
        profit_threshold=0.01,
        reward_value=0.05,
        punishment_value=0.05,
        idle_duration=5,
        bidding_duration=5,
        holding_duration=60,
        positive_trade_multiplier_rate=0.1,
        negative_trade_multiplier_rate=0.1,
        signal_refresh_interval=30,
        dashboard_refresh_interval=1
    )
    
    token_lists = TokenLists(
        latest_boosted_token_addresses=aggregator.latest_boosted_token_addresses,
        most_active_boosted_token_addresses=aggregator.most_active_boosted_token_addresses
    )
    latest = [entry["tokenAddress"] for entry in token_lists.latest_boosted_token_addresses]
    most_active = [entry["tokenAddress"] for entry in token_lists.most_active_boosted_token_addresses]
    all_tokens = list(set(latest + most_active))
    all_tokens = all_tokens[:20]
    
    print("Starting processing for tokens")
    
    executor = ThreadPoolExecutor(max_workers=min(len(all_tokens), 20))
    for token in all_tokens:
        executor.submit(process_token, token, config)
    
    TELEGRAM_BOT_TOKEN = "7937809727:AAF84eC3iKCwhYvbFeaU-TJTp3H6RQqr45Y"
    TELEGRAM_CHAT_ID = "-1002338904119"
    telegram_bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, global_knowledge, poll_interval=30)
    telegram_bot.start()
    
    global_master = GlobalMaster(input_dim=4, hidden_dim=32, lr=0.001, update_interval=30)
    master_thread = threading.Thread(target=global_master.run, daemon=True)
    master_thread.start()
    
    signal = Signal(config)
    signal.run()

if __name__ == "__main__":
    main()
