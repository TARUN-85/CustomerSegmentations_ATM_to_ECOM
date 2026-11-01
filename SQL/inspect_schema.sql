-- Select your project database/schema in MySQL Workbench before running this script.
-- This lists the actual columns and types for the tables used by the project.

SELECT
    TABLE_NAME,
    ORDINAL_POSITION,
    COLUMN_NAME,
    COLUMN_TYPE,
    IS_NULLABLE,
    COLUMN_KEY,
    EXTRA
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME IN (
      'customer',
      'account',
      'card',
      'transactions',
      'offer',
      'offer_rules',
      'points',
      'redemption',
      'burn',
      'refund'
  )
ORDER BY TABLE_NAME, ORDINAL_POSITION;

-- Optional: row estimates for the same tables.
-- InnoDB TABLE_ROWS is an estimate, not an exact count.
SELECT
    TABLE_NAME,
    TABLE_ROWS AS estimated_rows
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME IN (
      'customer',
      'account',
      'card',
      'transactions',
      'offer',
      'offer_rules',
      'points',
      'redemption',
      'burn',
      'refund'
  )
ORDER BY TABLE_NAME;
