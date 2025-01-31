import requests
import psycopg2
import time
from datetime import datetime

class DataMining:
    """
    This class handles data retrieval and extraction from various sources.
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
        self.config = {}  # Config structure for future additions
        self.init_db()
        print("Database initialized successfully.")

    def init_db(self):
        """
        Initializes the PostgreSQL database and creates the watchlist table if it does not exist.
        """
        self.conn = psycopg2.connect(
            dbname="xory_db", user="postgres", password="yourpassword", host="localhost", port="5432"
        )
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                chainId TEXT,
                dexId TEXT,
                url TEXT,
                pairAddress TEXT,
                labels TEXT[],
                baseToken_address TEXT,
                baseToken_name TEXT,
                baseToken_symbol TEXT,
                quoteToken_address TEXT,
                quoteToken_name TEXT,
                quoteToken_symbol TEXT,
                priceNative TEXT,
                priceUsd TEXT,
                liquidity_usd FLOAT,
                liquidity_base FLOAT,
                liquidity_quote FLOAT,
                fdv FLOAT,
                marketCap FLOAT,
                pairCreatedAt BIGINT,
                imageUrl TEXT,
                websites TEXT[],
                socials TEXT[],
                boosts_active INT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        print("Watchlist table verified/created successfully.")

    def fetch_latest_boosted_tokens(self):
        response = requests.get("https://api.dexscreener.com/token-boosts/latest/v1", headers={})
        if response.status_code != 200:
            print(f"Warning: Failed to fetch latest boosted tokens. HTTP Code: {response.status_code}")
            return []
        self.latest_boosted_tokens_data = response.json()
        self.latest_boosted_token_addresses = [{"tokenAddress": entry["tokenAddress"], "blacklistStatus": None} for entry in self.latest_boosted_tokens_data[:20]]
        print("Fetched latest boosted tokens successfully.")
        return self.latest_boosted_tokens_data

    def fetch_top_boosted_tokens(self):
        response = requests.get("https://api.dexscreener.com/token-boosts/top/v1", headers={})
        if response.status_code != 200:
            print(f"Warning: Failed to fetch top boosted tokens. HTTP Code: {response.status_code}")
            return []
        self.most_active_boosts_data = response.json()
        self.most_active_boosted_token_addresses = [{"tokenAddress": entry["tokenAddress"], "blacklistStatus": None} for entry in self.most_active_boosts_data[:20]]
        print("Fetched top boosted tokens successfully.")
        return self.most_active_boosts_data
    
    def fetch_token_details(self):
        """
        Fetch detailed information for all tokens and store in a structured format.
        """
        token_addresses = [entry["tokenAddress"] for entry in self.latest_boosted_token_addresses + self.most_active_boosted_token_addresses]
        chain_id = "default_chain"  # This should be dynamically determined
        url = f"https://api.dexscreener.com/tokens/v1/{chain_id}/{','.join(token_addresses)}"
        response = requests.get(url, headers={})
        if response.status_code != 200:
            print(f"Warning: Failed to fetch token details. HTTP Code: {response.status_code}")
            return []
        token_data = response.json()
        self.token_details = [{"symbol": entry["baseToken"]["symbol"], "data": entry} for entry in token_data]
        print("Fetched token details successfully.")
        return self.token_details

    def seed_watchlist(self):
        """
        Seeds the watchlist table with token data.
        """
        for token in self.token_details:
            entry = token["data"]
            required_fields = ["chainId", "dexId", "url", "pairAddress", "baseToken", "quoteToken", "priceNative", "priceUsd", "liquidity", "fdv", "marketCap", "pairCreatedAt", "info", "boosts"]
            missing_fields = [field for field in required_fields if field not in entry or not entry[field]]
            if missing_fields:
                print(f"Warning: Token {token['symbol']} is missing fields: {missing_fields}")
                continue
            self.cursor.execute("""
                INSERT INTO watchlist (
                    chainId, dexId, url, pairAddress, labels, 
                    baseToken_address, baseToken_name, baseToken_symbol,
                    quoteToken_address, quoteToken_name, quoteToken_symbol,
                    priceNative, priceUsd, liquidity_usd, liquidity_base, 
                    liquidity_quote, fdv, marketCap, pairCreatedAt,
                    imageUrl, websites, socials, boosts_active, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (
                entry["chainId"], entry["dexId"], entry["url"], entry["pairAddress"], entry.get("labels", []),
                entry["baseToken"]["address"], entry["baseToken"]["name"], entry["baseToken"]["symbol"],
                entry["quoteToken"]["address"], entry["quoteToken"]["name"], entry["quoteToken"]["symbol"],
                entry["priceNative"], entry["priceUsd"], entry["liquidity"]["usd"], entry["liquidity"]["base"],
                entry["liquidity"]["quote"], entry["fdv"], entry["marketCap"], entry["pairCreatedAt"],
                entry["info"]["imageUrl"], [site["url"] for site in entry["info"].get("websites", [])],
                [social["platform"] for social in entry["info"].get("socials", [])], entry["boosts"]["active"]
            ))
            print(f"Updated {token['symbol']} at {datetime.now()}")
        self.conn.commit()

    def clock_function(self):
        while True:
            self.fetch_token_details()
            self.seed_watchlist()
            time.sleep(60)

    def run(self):
        self.fetch_latest_boosted_tokens()
        self.fetch_top_boosted_tokens()
        self.fetch_token_details()
        self.seed_watchlist()
        self.clock_function()
