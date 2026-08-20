"""Loads the generated CSVs into a local SQLite database (data/fraud.db)."""
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/fraud.db")
DB_PATH.unlink(missing_ok=True)

conn = sqlite3.connect(DB_PATH)
conn.executescript(Path("sql/schema.sql").read_text())

pd.read_csv("data/customers.csv").to_sql("customers", conn, if_exists="append", index=False)
pd.read_csv("data/merchants.csv").to_sql("merchants", conn, if_exists="append", index=False)
pd.read_csv("data/transactions.csv").to_sql("transactions", conn, if_exists="append", index=False)

counts = conn.execute("SELECT (SELECT COUNT(*) FROM customers), (SELECT COUNT(*) FROM merchants), (SELECT COUNT(*) FROM transactions)").fetchone()
print(f"Loaded into {DB_PATH} -> customers={counts[0]}, merchants={counts[1]}, transactions={counts[2]}")
conn.close()
