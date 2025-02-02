import psycopg2
import pandas as pd
import numpy as np
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Machine Learning libraries
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =============================================================================
# GLOBAL KNOWLEDGE OBJECT (shared among token instances)
# =============================================================================
class GlobalKnowledge:
    def __init__(self):
        self.data = {}  # Maps token_address -> instance results (a dict)
        self.lock = threading.Lock()

    def update(self, token_address, agent_results):
        with self.lock:
            self.data[token_address] = agent_results

    def get_all(self):
        with self.lock:
            return self.data.copy()

# Global instance of the knowledge base.
global_knowledge = GlobalKnowledge()

def get_global_reinforcement_factor(predicted_action):
    """
    Calculate an average reinforcement factor (based on confidence) for the given
    predicted_action across all token instances.
    """
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
                # Skip the token_symbol field if present.
                if isinstance(agent, dict) and 'action' in agent and agent['action'] == predicted_action:
                    conf_sum += agent['confidence']
                    vote_count += 1
            if vote_count > 0:
                total_conf += (conf_sum / vote_count)
                count += 1
    return (total_conf / count) if count > 0 else 0

# =============================================================================
# GLOBAL HELD TRADES (tracks tokens that are currently "in_trade")
# =============================================================================
global_held_trades = {}
held_trades_lock = threading.Lock()

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================
class TradingConfig:
    def __init__(self,
                 profit_threshold=0.01,         # Minimum profit % to consider a winning trade
                 reward_value=0.05,             # Base reward increment factor
                 punishment_value=0.05,         # Base punishment decrement factor
                 bidding_duration=5,            # Seconds in bidding phase (no new vote)
                 holding_duration=10,           # Seconds to hold the trade before evaluation
                 positive_trade_multiplier_rate=0.1,  # Rate at which reward multiplier increases per second of holding
                 negative_trade_multiplier_rate=0.1   # Rate at which punishment multiplier increases per second of holding
                ):
        self.profit_threshold = profit_threshold
        self.reward_value = reward_value
        self.punishment_value = punishment_value
        self.bidding_duration = bidding_duration
        self.holding_duration = holding_duration
        self.positive_trade_multiplier_rate = positive_trade_multiplier_rate
        self.negative_trade_multiplier_rate = negative_trade_multiplier_rate

# =============================================================================
# BASE TRADING AGENT (handles reward/punishment updates)
# =============================================================================
class TradingAgent:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.weight_score = 0.80  # starting weight

    def update_reward(self, predicted_action: str, actual_profit: float, confidence: float,
                      reward_multiplier=1.0, punishment_multiplier=1.0):
        """
        Update the agent's weight score based on the outcome and multipliers.
        For:
          - 'buy': if actual_profit > profit_threshold → reward (using reward_multiplier),
                   otherwise, punish (using punishment_multiplier).
          - 'sell': similar logic (inversely for negative profit).
          - 'hold': reward if price change is small.
        """
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

# =============================================================================
# AGENT 1 – Reverse Forest Agent (using RandomForestClassifier)
# =============================================================================
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

# =============================================================================
# AGENT 2 – Reinforcement Learning Agent (simplified Q-Learning)
# =============================================================================
class RLAgent(TradingAgent):
    def __init__(self, config: TradingConfig, num_bins=10, actions=['buy', 'sell', 'hold']):
        super().__init__(config)
        self.num_bins = num_bins
        self.actions = actions
        self.q_table = np.random.rand(num_bins, len(actions))
        self.alpha = 0.1   # learning rate
        self.gamma = 0.95  # discount factor
        self.bins = None

    def train(self, price_series: pd.Series, price_change_series: pd.Series):
        self.bins = np.linspace(price_series.min(), price_series.max(), self.num_bins + 1)
        price_bins = np.digitize(price_series, self.bins) - 1

        for i in range(1, len(price_series)):
            current_state = price_bins[i - 1]
            next_state = price_bins[i]
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
        state = state[0]
        action_idx = np.argmax(self.q_table[state])
        action = self.actions[action_idx]
        confidence = self.q_table[state, action_idx] / (np.sum(self.q_table[state]) + 1e-6)
        return action, confidence

# =============================================================================
# AGENT 3 – XGBoost Agent (Hybrid/Ensemble Approach)
# =============================================================================
class XGBoostAgent(TradingAgent):
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, sample: pd.DataFrame):
        pred_probs = self.model.predict_proba(sample)[0]
        pred_idx = np.argmax(pred_probs)
        action = self.model.classes_[pred_idx]
        confidence = pred_probs[pred_idx]
        return action, confidence

# =============================================================================
# TOKEN PROCESSOR FUNCTION (runs one token instance)
# =============================================================================
def process_token(token_address: str, config: TradingConfig):
    thread_name = threading.current_thread().name
    print(f"\n[Token: {token_address} | Thread: {thread_name}] Processing started.")

    # --- Database Connection & Data Fetching ---
    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="xory_db",
            user="your_username",
            password="your_password"
        )
    except Exception as e:
        print(f"[Token: {token_address}] Error connecting to database: {e}")
        return

    query = f"""
    SELECT chainId, dexId, url, pairAddress, labels, 
           baseToken_address, baseToken_name, baseToken_symbol,
           quoteToken_address, quoteToken_name, quoteToken_symbol,
           priceNative, priceUsd, liquidity_usd, liquidity_base, 
           liquidity_quote, fdv, marketCap, pairCreatedAt,
           imageUrl, websites, socials, boosts_active, timestamp
    FROM watchlist
    WHERE baseToken_address = '{token_address}' OR quoteToken_address = '{token_address}';
    """
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        print(f"[Token: {token_address}] Error executing SQL: {e}")
        conn.close()
        return
    conn.close()

    if df.empty:
        print(f"[Token: {token_address}] No data found.")
        return

    # --- Data Preprocessing & Target Simulation ---
    features = ['priceUsd', 'liquidity_usd', 'marketCap']
    X = df[features].fillna(0)
    price_change = X['priceUsd'].pct_change().fillna(0)
    y = np.where(price_change > 0.01, 'buy', np.where(price_change < -0.01, 'sell', 'hold'))
    y = pd.Series(y, index=X.index)
    latest_sample = X.iloc[[-1]]
    latest_price = latest_sample['priceUsd'].values[0]

    # --- Instantiate & Train Agents for this token ---
    rf_agent = ReverseForestAgent(config)
    rf_agent.train(X, y)

    rl_agent = RLAgent(config, num_bins=10)
    rl_agent.train(X['priceUsd'], price_change)

    xgb_agent = XGBoostAgent(config)
    xgb_agent.train(X, y)

    # --- Agents Make Their Predictions ---
    rf_action, rf_confidence = rf_agent.predict(latest_sample)
    rl_action, rl_confidence = rl_agent.predict(latest_price)
    xgb_action, xgb_confidence = xgb_agent.predict(latest_sample)

    # --- Retrieve token symbol from the database row ---
    token_symbol = df['baseToken_symbol'].iloc[0] if 'baseToken_symbol' in df.columns else token_address

    print(f"[Token: {token_address}] Initial Recommendations:")
    print(f"  Reverse Forest -> Action: {rf_action}, Confidence: {rf_confidence:.2f}, Weight: {rf_agent.weight_score:.2f}")
    print(f"  RL Agent       -> Action: {rl_action}, Confidence: {rl_confidence:.2f}, Weight: {rl_agent.weight_score:.2f}")
    print(f"  XGBoost Agent  -> Action: {xgb_action}, Confidence: {xgb_confidence:.2f}, Weight: {xgb_agent.weight_score:.2f}")

    # --- Intra-Instance (Local) Consensus ---
    votes = [rf_action, rl_action, xgb_action]
    consensus = max(set(votes), key=votes.count)
    print(f"[Token: {token_address}] Local Consensus: {consensus}")
    bonus = config.reward_value * 0.1
    penalty = config.punishment_value * 0.1
    if rf_action == consensus:
        rf_agent.weight_score += bonus
    else:
        rf_agent.weight_score -= penalty
    if rl_action == consensus:
        rl_agent.weight_score += bonus
    else:
        rl_agent.weight_score -= penalty
    if xgb_action == consensus:
        xgb_agent.weight_score += bonus
    else:
        xgb_agent.weight_score -= penalty

    # --- Global Sharing: Update Global Knowledge (including token_symbol) ---
    instance_results = {
        'rf': {'action': rf_action, 'confidence': rf_confidence, 'weight': rf_agent.weight_score},
        'rl': {'action': rl_action, 'confidence': rl_confidence, 'weight': rl_agent.weight_score},
        'xgb': {'action': xgb_action, 'confidence': xgb_confidence, 'weight': xgb_agent.weight_score},
        'token_symbol': token_symbol
    }
    global_knowledge.update(token_address, instance_results)

    # --- Inter-Instance Reinforcement ---
    global_rf_factor = get_global_reinforcement_factor(rf_action)
    global_rl_factor = get_global_reinforcement_factor(rl_action)
    global_xgb_factor = get_global_reinforcement_factor(xgb_action)
    rf_agent.weight_score += config.reward_value * global_rf_factor
    rl_agent.weight_score += config.reward_value * global_rl_factor
    xgb_agent.weight_score += config.reward_value * global_xgb_factor

    print(f"[Token: {token_address}] Post Global Reinforcement:")
    print(f"  Reverse Forest New Weight: {rf_agent.weight_score:.2f}")
    print(f"  RL Agent New Weight:       {rl_agent.weight_score:.2f}")
    print(f"  XGBoost New Weight:        {xgb_agent.weight_score:.2f}")

    # --- STATE MANAGEMENT: Trading Phases ---
    # Enter bidding phase
    trade_state = "bidding"
    print(f"[Token: {token_address}] Entering bidding phase for {config.bidding_duration} seconds.")
    time.sleep(config.bidding_duration)

    # Enter in_trade phase and record the trade
    trade_state = "in_trade"
    trade_start_time = time.time()
    # Record the trade in the global held trades dictionary.
    with held_trades_lock:
        global_held_trades[token_address] = {
            "token_symbol": token_symbol,
            "start_time": trade_start_time,
            "holding_duration": config.holding_duration
        }
    print(f"[Token: {token_address}] Bidding complete. Trade executed. Holding trade for {config.holding_duration} seconds.")
    time.sleep(config.holding_duration)

    # Remove from held trades once the trade is closed.
    with held_trades_lock:
        if token_address in global_held_trades:
            del global_held_trades[token_address]

    # --- Trade Outcome Simulation & Dynamic Multipliers ---
    simulated_profit = random.uniform(-0.05, 0.05)
    print(f"[Token: {token_address}] Trade closed. Simulated Profit Outcome: {simulated_profit:.2%}")

    if simulated_profit > config.profit_threshold:
        reward_multiplier = 1 + (config.holding_duration * config.positive_trade_multiplier_rate)
        punishment_multiplier = 1.0
    else:
        reward_multiplier = 1.0
        punishment_multiplier = 1 + (config.holding_duration * config.negative_trade_multiplier_rate)

    trade_state = "closed"

    # --- Final Reward Update Using Dynamic Multipliers ---
    rf_agent.update_reward(rf_action, simulated_profit, rf_confidence, reward_multiplier, punishment_multiplier)
    rl_agent.update_reward(rl_action, simulated_profit, rl_confidence, reward_multiplier, punishment_multiplier)
    xgb_agent.update_reward(xgb_action, simulated_profit, xgb_confidence, reward_multiplier, punishment_multiplier)

    print(f"[Token: {token_address}] Final Agent Status After Outcome Update:")
    print(f"  Reverse Forest -> Final Weight: {rf_agent.weight_score:.2f}")
    print(f"  RL Agent       -> Final Weight: {rl_agent.weight_score:.2f}")
    print(f"  XGBoost Agent  -> Final Weight: {xgb_agent.weight_score:.2f}")
    print(f"[Token: {token_address}] Processing complete.\n")

# =============================================================================
# SIGNAL CLASS (Human Interface)
# =============================================================================
class Signal:
    def __init__(self, config: TradingConfig):
        self.config = config

    def display_signals(self):
        """
        Print the majority vote (BUY or SELL) along with the token symbol for each token instance.
        If the consensus is HOLD, nothing is printed for that token.
        """
        global_data = global_knowledge.get_all()
        print("\n--- SIGNALS ---")
        for token, results in global_data.items():
            token_symbol = results.get('token_symbol', token)
            votes = [results['rf']['action'], results['rl']['action'], results['xgb']['action']]
            consensus = max(set(votes), key=votes.count)
            if consensus in ['buy', 'sell']:
                print(f"Token: {token_symbol} - Signal: {consensus.upper()}")
        print("----------------\n")

    def display_held_trades(self):
        """
        Display all currently held trades with the time remaining on each hold.
        """
        with held_trades_lock:
            if not global_held_trades:
                print("No held trades currently.")
            else:
                print("\n--- Held Trades ---")
                current_time = time.time()
                for token, trade_info in global_held_trades.items():
                    token_symbol = trade_info.get("token_symbol", token)
                    trade_start = trade_info["start_time"]
                    holding_duration = trade_info["holding_duration"]
                    elapsed = current_time - trade_start
                    remaining = max(0, holding_duration - elapsed)
                    print(f"Token: {token_symbol} - Time Remaining: {remaining:.1f} seconds")
                print("-------------------\n")

    def run(self):
        """
        Main interface: display signals and offer the user an option to view held trades.
        """
        self.display_signals()
        print("Press 'H' to display all held trades, or any other key to exit.")
        user_input = input().strip().lower()
        if user_input == 'h':
            self.display_held_trades()
        else:
            print("Exiting Signal interface.")

# =============================================================================
# SIMULATED CLASS PROVIDING TOKEN ADDRESS LISTS
# =============================================================================
class TokenLists:
    def __init__(self):
        self.latest_boosted_token_addresses = [
            '0xTokenAddress1', '0xTokenAddress2'
        ]
        self.most_active_boosted_token_addresses = [
            '0xTokenAddress3', '0xTokenAddress4'
        ]

# =============================================================================
# MAIN EXECUTION WITH MULTITHREADING AND SIGNAL INTERFACE
# =============================================================================
def main():
    config = TradingConfig(
        profit_threshold=0.01,
        reward_value=0.05,
        punishment_value=0.05,
        bidding_duration=5,
        holding_duration=10,
        positive_trade_multiplier_rate=0.1,
        negative_trade_multiplier_rate=0.1
    )
    token_lists = TokenLists()
    all_tokens = list(set(token_lists.latest_boosted_token_addresses + token_lists.most_active_boosted_token_addresses))
    
    print("Starting processing for tokens:")
    for t in all_tokens:
        print("  ", t)
    
    # Process each token concurrently.
    with ThreadPoolExecutor(max_workers=len(all_tokens)) as executor:
        futures = [executor.submit(process_token, token, config) for token in all_tokens]
        for future in futures:
            future.result()

    # Once all token processing is complete (or while some trades may still be held),
    # present the Signal interface.
    signal = Signal(config)
    signal.run()

if __name__ == "__main__":
    main()
