-- ATM-to-ECOM debit-card adoption dataset.
-- Select the project schema in MySQL Workbench before running this query.
-- Source transactions begin on 2026-01-01, so the first snapshot has a full
-- 90-day feature window. The final 30-day outcome window ends 2026-10-30.
--
-- Cohort: customer has >=1 successful ATM transaction on a DEBIT card in the
-- previous 90 days, and no successful ECOM transaction on a DEBIT card in
-- that same lookback. Customers may have used other channels or ECOM with a
-- non-debit card; the modeled outcome is specifically ECOM debit-card usage.
-- Label: >=1 successful ECOM transaction linked to a DEBIT card during the
-- next 30 days. This predicts usage, not campaign-caused incremental lift.

WITH snapshots AS (
    SELECT DATE('2026-04-01') AS snapshot_date
    UNION ALL SELECT DATE('2026-05-01')
    UNION ALL SELECT DATE('2026-06-01')
    UNION ALL SELECT DATE('2026-07-01')
    UNION ALL SELECT DATE('2026-08-01')
    UNION ALL SELECT DATE('2026-09-01')
    UNION ALL SELECT DATE('2026-10-01')
),
customer_features AS (
    SELECT
        s.snapshot_date,
        t.customer_id,
        COUNT(t.transaction_id) AS successful_txn_count_90d,
        SUM(CASE WHEN cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS debit_txn_count_90d,
        SUM(CASE WHEN cd.card_type = 'DEBIT' THEN t.transaction_amount ELSE 0 END) AS debit_txn_amount_90d,
        SUM(CASE WHEN t.channel = 'ATM' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS atm_debit_txn_count_90d,
        SUM(CASE WHEN t.channel = 'ATM' AND cd.card_type = 'DEBIT' THEN t.transaction_amount ELSE 0 END) AS atm_debit_amount_90d,
        MAX(CASE WHEN t.channel = 'ATM' AND cd.card_type = 'DEBIT' THEN t.transaction_date END) AS last_atm_debit_datetime_90d,
        SUM(CASE WHEN t.channel = 'UPI' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS upi_debit_txn_count_90d,
        SUM(CASE WHEN t.channel = 'POS' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS pos_debit_txn_count_90d,
        SUM(CASE WHEN t.channel = 'MOBILE_APP' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS mobile_app_debit_txn_count_90d,
        SUM(CASE WHEN t.channel = 'NETBANKING' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS netbanking_debit_txn_count_90d,
        SUM(CASE WHEN t.channel = 'ECOM' AND cd.card_type <> 'DEBIT' THEN 1 ELSE 0 END) AS ecom_non_debit_txn_count_90d,
        SUM(CASE WHEN t.transaction_type = 'PURCHASE' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS debit_purchase_txn_count_90d,
        SUM(CASE WHEN t.transaction_type = 'BILL_PAYMENT' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS debit_bill_payment_txn_count_90d,
        SUM(CASE WHEN t.transaction_type = 'CASH_WITHDRAWAL' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS debit_cash_withdrawal_txn_count_90d,
        SUM(CASE WHEN t.transaction_type = 'TRANSFER' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS debit_transfer_txn_count_90d,
        SUM(CASE WHEN t.transaction_type = 'REFUND' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) AS debit_refund_txn_count_90d
    FROM snapshots AS s
    JOIN transactions AS t
      ON t.transaction_date >= DATE_SUB(s.snapshot_date, INTERVAL 90 DAY)
     AND t.transaction_date < s.snapshot_date
     AND t.transaction_status = 'SUCCESS'
    LEFT JOIN card AS cd
      ON cd.card_id = t.card_id
    GROUP BY s.snapshot_date, t.customer_id
    HAVING SUM(CASE WHEN t.channel = 'ATM' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) > 0
       AND SUM(CASE WHEN t.channel = 'ECOM' AND cd.card_type = 'DEBIT' THEN 1 ELSE 0 END) = 0
)
SELECT
    f.snapshot_date,
    f.customer_id,
    f.successful_txn_count_90d,
    f.debit_txn_count_90d,
    f.debit_txn_amount_90d,
    f.atm_debit_txn_count_90d,
    f.atm_debit_amount_90d,
    TIMESTAMPDIFF(DAY, f.last_atm_debit_datetime_90d, f.snapshot_date) AS days_since_atm_debit,
    f.upi_debit_txn_count_90d,
    f.pos_debit_txn_count_90d,
    f.mobile_app_debit_txn_count_90d,
    f.netbanking_debit_txn_count_90d,
    f.ecom_non_debit_txn_count_90d,
    f.debit_purchase_txn_count_90d,
    f.debit_bill_payment_txn_count_90d,
    f.debit_cash_withdrawal_txn_count_90d,
    f.debit_transfer_txn_count_90d,
    f.debit_refund_txn_count_90d,
    CASE WHEN EXISTS (
        SELECT 1
        FROM transactions AS future_txn
        JOIN card AS future_card
          ON future_card.card_id = future_txn.card_id
         AND future_card.card_type = 'DEBIT'
        WHERE future_txn.customer_id = f.customer_id
          AND future_txn.transaction_date >= f.snapshot_date
          AND future_txn.transaction_date < DATE_ADD(f.snapshot_date, INTERVAL 30 DAY)
          AND future_txn.transaction_status = 'SUCCESS'
          AND future_txn.channel = 'ECOM'
    ) THEN 1 ELSE 0 END AS ecom_debit_next_30d
FROM customer_features AS f
ORDER BY f.snapshot_date, f.customer_id;
