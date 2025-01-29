import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData
from typing import Dict, List, Optional
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Enhanced Configuration
CONFIG = {
    "DB": {
        "dbname": "dexscreener",
        "user": "admin",
        "password": "your_password",
        "host": "localhost",
        "port": "5432"
    },
    "FILTERS": {
        "min_liquidity": 5000,  # USD
        "min_age_days": 3,
        "coin_blacklist": [
            "0x123...def",  # Known scam token address
            "SUSPECTCOIN"   # Blacklisted symbol
        ],
        "dev_blacklist": [
            "0x456...abc",  # Known rug developer address
            "0x789...fed"   # Another scam developer
        ],
        "chain_whitelist": ["ethereum", "binance-smart-chain"]
    }
}

class EnhancedDexScreenerBot:
    def __init__(self):
        self.engine = create_engine(
            f'postgresql+psycopg2://{CONFIG["DB"]["user"]}:{CONFIG["DB"]["password"]}'
            f'@{CONFIG["DB"]["host"]}/{CONFIG["DB"]["dbname"]}'
        )
        self.metadata = MetaData()
        self.tokens_table = Table(
            'tokens', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String),
            Column('symbol', String),
            Column('price', Float),
            Column('volume', Float),
            Column('liquidity', Float),
            Column('chain', String),
            Column('launch_date', String)
        )
        self.metadata.create_all(self.engine)
        self._init_db()
    
    def fetch_dexscreener_data(self):
        """Fetches real-time data from DexScreener API."""
        url = "https://api.dexscreener.com/latest/dex/tokens"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Failed to fetch data from DexScreener")
            return None
    
    def check_rug_status(self, contract_address):
        """Check token contract status on RugCheck.xyz"""
        url = f"http://rugcheck.xyz/api/check/{contract_address}"
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            return result.get("status") == "Good"
        return False

    def _init_db(self):
        """Initialize database with blacklist table"""
        with self.engine.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blacklist (
                    address VARCHAR(42) PRIMARY KEY,
                    type VARCHAR(20) CHECK (type IN ('coin', 'dev')),
                    reason TEXT,
                    listed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
            )

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply security and quality filters to the dataset"""
        df = df[df['chain'].isin(CONFIG["FILTERS"]["chain_whitelist"])]
        df = df[df['liquidity'] >= CONFIG["FILTERS"]["min_liquidity"]]
        min_age = datetime.utcnow() - timedelta(days=CONFIG["FILTERS"]["min_age_days"])
        df = df[pd.to_datetime(df['created_at']) < min_age]
        return df

    def process_data(self, raw_data: Dict) -> pd.DataFrame:
        """Processes raw DexScreener data and applies triple check."""
        df = pd.DataFrame(raw_data['pairs'])
        df = df.rename(columns={
            'pairAddress': 'pair_address',
            'baseToken': 'base_token',
            'quoteToken': 'quote_token',
            'priceUsd': 'price',
            'liquidity': 'liquidity',
            'volume': 'volume_24h',
            'chainId': 'chain',
            'dexId': 'exchange',
            'createdAt': 'created_at'
        })
        df = self.apply_filters(df)
        df = df[df['pair_address'].apply(self.check_rug_status)]
        return df

    def save_to_db(self, df: pd.DataFrame):
        """Saves processed data to the database."""
        df.to_sql('tokens', self.engine, if_exists='append', index=False)
    
    def check_for_alerts(self, df: pd.DataFrame):
        """Checks for specific trading patterns and generates alerts."""
        for index, row in df.iterrows():
            if row['volume_24h'] > 1000000 and row['price'] < 0.0001:
                print(f"Alert: {row['symbol']} has high volume and low price!")
    
    def run(self):
        """Main loop that fetches, processes, stores, and analyzes token data"""
        while True:
            try:
                raw_data = self.fetch_dexscreener_data()
                if raw_data:
                    processed_data = self.process_data(raw_data)
                    self.save_to_db(processed_data)
                    self.check_for_alerts(processed_data)
                    print(processed_data.head())  # Debugging output
            except Exception as e:
                print(f"Runtime error: {e}")

# Run bot
if __name__ == "__main__":
    bot = EnhancedDexScreenerBot()
    bot.run()
