-- ============================================================
-- Fraud Detection Engine — analyst SQL queries (SQLite dialect)
-- Run after: python src/load_to_sqlite.py
-- ============================================================

-- 1. Headline fraud rate
SELECT
    COUNT(*)                                   AS total_transactions,
    SUM(is_fraud)                              AS fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount END), 2) AS fraud_dollar_loss
FROM transactions;

-- 2. Fraud rate by merchant category, ranked
SELECT
    category,
    COUNT(*)                                    AS txns,
    SUM(is_fraud)                               AS fraud_txns,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)  AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount END), 2) AS fraud_dollar_loss
FROM transactions
GROUP BY category
ORDER BY fraud_rate_pct DESC;

-- 3. Fraud rate by hour of day (odd-hours pattern)
SELECT
    CAST(strftime('%H', timestamp) AS INTEGER)  AS hour_of_day,
    COUNT(*)                                    AS txns,
    SUM(is_fraud)                               AS fraud_txns,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)  AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- 4. Top 10 customers by confirmed fraud dollar loss
SELECT
    c.customer_id,
    c.home_city,
    COUNT(*)                                    AS fraud_txns,
    ROUND(SUM(t.amount), 2)                     AS total_fraud_amount
FROM transactions t
JOIN customers c ON c.customer_id = t.customer_id
WHERE t.is_fraud = 1
GROUP BY c.customer_id, c.home_city
ORDER BY total_fraud_amount DESC
LIMIT 10;

-- 5. Weekly fraud-rate trend (window function: 3-week rolling average)
WITH weekly AS (
    SELECT
        strftime('%Y-%W', timestamp)                AS iso_week,
        COUNT(*)                                     AS txns,
        SUM(is_fraud)                                AS fraud_txns
    FROM transactions
    GROUP BY iso_week
)
SELECT
    iso_week,
    txns,
    fraud_txns,
    ROUND(100.0 * fraud_txns / txns, 3)                                             AS fraud_rate_pct,
    ROUND(AVG(100.0 * fraud_txns / txns) OVER (
        ORDER BY iso_week ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 3)                                                                            AS rolling_3wk_avg_fraud_rate
FROM weekly
ORDER BY iso_week;

-- 6. Card-testing pattern: rapid repeat transactions per customer
--    (real gap-to-previous-transaction computed via LAG, independent of the
--     pre-baked seconds_since_prev_txn feature used for modeling)
WITH ordered AS (
    SELECT
        customer_id,
        transaction_id,
        timestamp,
        amount,
        is_fraud,
        LAG(timestamp) OVER (PARTITION BY customer_id ORDER BY timestamp) AS prev_ts
    FROM transactions
),
gaps AS (
    SELECT
        *,
        (julianday(timestamp) - julianday(prev_ts)) * 86400 AS gap_seconds
    FROM ordered
    WHERE prev_ts IS NOT NULL
)
SELECT
    customer_id,
    COUNT(*)                                    AS rapid_repeat_txns,
    SUM(is_fraud)                               AS of_which_fraud,
    ROUND(AVG(amount), 2)                       AS avg_amount
FROM gaps
WHERE gap_seconds < 120
GROUP BY customer_id
HAVING COUNT(*) >= 2
ORDER BY rapid_repeat_txns DESC
LIMIT 15;

-- 7. Does merchant_risk_score actually predict fraud? (validation query)
SELECT
    CASE
        WHEN merchant_risk_score < 0.10 THEN '0.00-0.10'
        WHEN merchant_risk_score < 0.20 THEN '0.10-0.20'
        WHEN merchant_risk_score < 0.35 THEN '0.20-0.35'
        ELSE '0.35+'
    END                                          AS risk_bucket,
    COUNT(*)                                     AS txns,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)   AS actual_fraud_rate_pct
FROM transactions
GROUP BY risk_bucket
ORDER BY risk_bucket;

-- 8. Customer spend-anomaly flag: single transaction > 3x their implied daily spend
SELECT
    t.transaction_id,
    t.customer_id,
    t.amount,
    ROUND(c.avg_monthly_spend / 30.0, 2)         AS implied_daily_spend,
    ROUND(t.amount / (c.avg_monthly_spend / 30.0), 1) AS x_over_daily_norm,
    t.is_fraud
FROM transactions t
JOIN customers c ON c.customer_id = t.customer_id
WHERE t.amount > 3 * (c.avg_monthly_spend / 30.0)
ORDER BY x_over_daily_norm DESC
LIMIT 20;
