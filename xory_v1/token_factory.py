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

# Import SQLAlchemy components
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
# ------------------------- Global Settings -------------------------



# Suppress XGBoost warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# ------------------------- Global Settings -------------------------
engine = create_engine("postgresql://postgres:Rw.1010419@localhost:5432/watchlist")
VERBOSE = False  # Set to True for detailed per-token logs

# Global initialization variables for the progress bar
init_complete = False
init_message = ""
init_progress = 0.0  # Value between 0 and 1

# ------------------------- Helper Function -------------------------
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

# ------------------------- End Aggregation Code -------------------------

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

global_knowledge = GlobalKnowledge()
global_trade_outcomes = {}  # Latest trade profit/loss outcomes

def get_global_reinforcement_factor(predicted_action):
    all_data = global_knowledge.get_all()
    total_conf = 0.0
    count = 0
    for token, results in all_data.items():
        votes = [results['rf']['action'], results['rl']['action'], results['xgb']['action']]
        consensus = max(set(votes), key=votes.count)
        if consensus == predicted_action:
            conf_sum = 0.0
            vote_count = 0
            for agent in results.values():
                if isinstance(agent, dict) and 'action' in agent and agent['action'] == predicted_action:
                    conf_sum += agent['confidence']
                    vote_count += 1
            if vote_count > 0:
                total_conf += (conf_sum / vote_count)
                count += 1
    return (total_conf / count) if count > 0 else 0

global_held_trades = {}
held_trades_lock = threading.Lock()

# ------------------------- Configuration and Agent Classes -------------------------
class TradingConfig:
    def __init__(self,
                 profit_threshold=0.01,
                 reward_value=0.05,
                 punishment_value=0.05,
                 bidding_duration=5,
                 holding_duration=10,
                 positive_trade_multiplier_rate=0.1,
                 negative_trade_multiplier_rate=0.1,
                 signal_refresh_interval=30):
        self.profit_threshold = profit_threshold
        self.reward_value = reward_value
        self.punishment_value = punishment_value
        self.bidding_duration = bidding_duration
        self.holding_duration = holding_duration
        self.positive_trade_multiplier_rate = positive_trade_multiplier_rate
        self.negative_trade_multiplier_rate = negative_trade_multiplier_rate
        self.signal_refresh_interval = signal_refresh_interval

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

class ReverseForestAgent(TradingAgent):
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
    def predict(self, sample: pd.DataFrame):
        pred_probs = self.model.predict_proba(sample)[0]
        pred_idx = np.argmax(pred_probs)
        action = self.model.classes_[pred_idx]
        confidence = pred_probs[pred_idx]
        return action, confidence

class RLAgent(TradingAgent):
    def __init__(self, config: TradingConfig, num_bins=10, actions=['buy', 'sell', 'hold']):
        super().__init__(config)
        self.num_bins = num_bins
        self.actions = actions
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
                reward_vector = [1, -1, 0]
            elif price_change_series.iloc[i] < -self.config.profit_threshold:
                reward_vector = [-1, 1, 0]
            else:
                reward_vector = [0, 0, 1]
            action_idx = np.argmax(self.q_table[current_state])
            best_next_q = np.max(self.q_table[next_state])
            self.q_table[current_state, action_idx] += self.alpha * (
                reward_vector[action_idx] + self.gamma * best_next_q - self.q_table[current_state, action_idx]
            )
    def predict(self, current_price: float):
        if self.bins is None:
            raise ValueError("The agent must be trained before prediction.")
        state = np.digitize([current_price], self.bins) - 1
        state = min(state[0], self.num_bins - 1)
        action_idx = np.argmax(self.q_table[state])
        action = self.actions[action_idx]
        confidence = self.q_table[state, action_idx] / (np.sum(self.q_table[state]) + 1e-6)
        return action, confidence

class XGBoostAgent(TradingAgent):
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.model = XGBClassifier(verbosity=0, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        self.le = LabelEncoder()
    def train(self, X: pd.DataFrame, y: pd.Series):
        y_encoded = self.le.fit_transform(y)
        self.model.fit(X, y_encoded)
    def predict(self, sample: pd.DataFrame):
        y_pred = self.model.predict(sample)
        action = self.le.inverse_transform(y_pred)[0]
        pred_probs = self.model.predict_proba(sample)[0]
        pred_idx = self.le.transform([action])[0]
        confidence = pred_probs[pred_idx]
        return action, confidence

# ------------------------- Token Processing -------------------------
def process_token(token_address: str, config: TradingConfig):
    """
    Continuously processes a token (rebidding after each trading round).
    Displays minimal output (using token symbols) while updating global state.
    """
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
            WHERE basetoken_address = '{token_address}' OR quotetoken_address = '{token_address}';
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

        rf_agent = ReverseForestAgent(config)
        rf_agent.train(X, y)
        rl_agent = RLAgent(config, num_bins=10)
        rl_agent.train(X['priceusd'], price_change)
        xgb_agent = XGBoostAgent(config)
        xgb_agent.train(X, y)

        rf_action, rf_confidence = rf_agent.predict(latest_sample)
        rl_action, rl_confidence = rl_agent.predict(latest_price)
        xgb_action, xgb_confidence = xgb_agent.predict(latest_sample)
        token_symbol = df['basetoken_symbol'].iloc[0] if 'basetoken_symbol' in df.columns else token_address

        # Update global knowledge with minimal info for dashboard aggregation
        instance_results = {
            'rf': {'action': rf_action, 'confidence': rf_confidence, 'weight': rf_agent.weight_score},
            'rl': {'action': rl_action, 'confidence': rl_confidence, 'weight': rl_agent.weight_score},
            'xgb': {'action': xgb_action, 'confidence': xgb_confidence, 'weight': xgb_agent.weight_score},
            'token_symbol': token_symbol
        }
        global_knowledge.update(token_address, instance_results)

        if VERBOSE:
            print(f"[{token_symbol}] Recs: RF={rf_action}({rf_confidence:.2f}), RL={rl_action}({rl_confidence:.2f}), XGB={xgb_action}({xgb_confidence:.2f})")
            print(f"[{token_symbol}] Consensus: {max(set([rf_action, rl_action, xgb_action]), key=[rf_action, rl_action, xgb_action].count)}")
            print(f"[{token_symbol}] Weights: RF={rf_agent.weight_score:.2f}, RL={rl_agent.weight_score:.2f}, XGB={xgb_agent.weight_score:.2f}")
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
        if simulated_profit > config.profit_threshold:
            reward_multiplier = 1 + (config.holding_duration * config.positive_trade_multiplier_rate)
            punishment_multiplier = 1.0
        else:
            reward_multiplier = 1.0
            punishment_multiplier = 1 + (config.holding_duration * config.negative_trade_multiplier_rate)
        rf_agent.update_reward(rf_action, simulated_profit, rf_confidence, reward_multiplier, punishment_multiplier)
        rl_agent.update_reward(rl_action, simulated_profit, rl_confidence, reward_multiplier, punishment_multiplier)
        xgb_agent.update_reward(xgb_action, simulated_profit, xgb_confidence, reward_multiplier, punishment_multiplier)
        if VERBOSE:
            print(f"[{token_symbol}] Final Weights: RF={rf_agent.weight_score:.2f}, RL={rl_agent.weight_score:.2f}, XGB={xgb_agent.weight_score:.2f}")
            print(f"[{token_symbol}] Round complete.\n")
        time.sleep(2)

# ------------------------- Signal Interface -------------------------
class Signal:
    def __init__(self, config: TradingConfig):
        self.config = config
    def display_agent_weights(self):
        output = "\n------- Agent Weights -------\n"
        for token, results in global_knowledge.get_all().items():
            token_symbol = results.get('token_symbol', token)
            rf_w = results['rf']['weight']
            rl_w = results['rl']['weight']
            xgb_w = results['xgb']['weight']
            output += f"Token: {token_symbol} | RF: {rf_w:.2f}, RL: {rl_w:.2f}, XGB: {xgb_w:.2f}\n"
        output += "-----------------------------\n"
        return output
    def display_latest_signals(self):
        output = "---- Latest Signals ----\n"
        for token, results in global_knowledge.get_all().items():
            token_symbol = results.get('token_symbol', token)
            votes = [results['rf']['action'], results['rl']['action'], results['xgb']['action']]
            consensus = max(set(votes), key=votes.count)
            output += f"Token: {token_symbol} - Signal: {consensus.upper()}\n"
        output += "------------------------\n"
        return output
    def display_trade_outcomes(self):
        output = "\n---- Latest Trade Outcomes ----\n"
        if not global_trade_outcomes:
            output += "No trade outcomes available.\n"
        else:
            for token, outcome in global_trade_outcomes.items():
                token_symbol = global_knowledge.get_all().get(token, {}).get('token_symbol', token)
                output += f"Token: {token_symbol} - Outcome: {outcome*100:.2f}%\n"
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
    def run(self):
        while True:
            # Clear screen and move cursor to top-left
            sys.stdout.write("\033[2J\033[H")
            dashboard = ""
            dashboard += self.display_agent_weights()
            dashboard += self.display_latest_signals()
            dashboard += "\n---- Options ----\n"
            dashboard += "P - Latest trade profit/loss outcomes\n"
            dashboard += "C - Current active trades\n"
            dashboard += "Q - Quit\n"
            sys.stdout.write(dashboard)
            sys.stdout.flush()
            user_input = input("Enter option (or press ENTER to refresh): ").strip().lower()
            if user_input == 'p':
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write(self.display_trade_outcomes())
                input("Press ENTER to continue...")
            elif user_input == 'c':
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write(self.display_active_trades())
                input("Press ENTER to continue...")
            elif user_input == 'q':
                print("Exiting Signal interface.")
                break
            else:
                time.sleep(self.config.signal_refresh_interval)

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
    """Display a persistent progress bar until initialization is complete."""
    bar_length = 30
    while not init_complete:
        filled_length = int(round(bar_length * init_progress))
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f"\rInitializing... [{bar}] {int(init_progress*100)}% | {init_message}")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\nInitialization complete. Launching dashboard...\n")
    sys.stdout.flush()

# ------------------------- Main Execution -------------------------
def main():
    global init_complete
    aggregator = DataMining(source="API")
    aggregator_thread = threading.Thread(target=aggregator.run, daemon=True)
    aggregator_thread.start()
    print("Aggregator thread started. Waiting 10 minutes for seeding to complete...")
    display_initialization()
    final_database_check(required_rows=1, poll_interval=10)
    
    config = TradingConfig(
        profit_threshold=0.01,
        reward_value=0.05,
        punishment_value=0.05,
        bidding_duration=5,
        holding_duration=10,
        positive_trade_multiplier_rate=0.1,
        negative_trade_multiplier_rate=0.1,
        signal_refresh_interval=30
    )
    
    token_lists = TokenLists(
        latest_boosted_token_addresses=aggregator.latest_boosted_token_addresses,
        most_active_boosted_token_addresses=aggregator.most_active_boosted_token_addresses
    )
    latest = [entry["tokenAddress"] for entry in token_lists.latest_boosted_token_addresses]
    most_active = [entry["tokenAddress"] for entry in token_lists.most_active_boosted_token_addresses]
    all_tokens = list(set(latest + most_active))
    all_tokens = all_tokens[:20]  # Limit to 20 tokens
    
    print("Starting processing for tokens:")
    for t in all_tokens:
        print("  ", t)
    
    executor = ThreadPoolExecutor(max_workers=min(len(all_tokens), 20))
    for token in all_tokens:
        executor.submit(process_token, token, config)
    
    # Immediately launch the Signal dashboard (main thread)
    signal = Signal(config)
    signal.run()

if __name__ == "__main__":
    main()
