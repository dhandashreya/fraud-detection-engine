CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    home_city TEXT NOT NULL,
    avg_monthly_spend REAL NOT NULL,
    account_age_days INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id INTEGER PRIMARY KEY,
    category TEXT NOT NULL,
    merchant_risk_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    merchant_id INTEGER NOT NULL REFERENCES merchants(merchant_id),
    category TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    amount REAL NOT NULL,
    seconds_since_prev_txn INTEGER NOT NULL,
    merchant_risk_score REAL NOT NULL,
    is_fraud INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_txn_merchant ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_txn_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_txn_is_fraud ON transactions(is_fraud);
