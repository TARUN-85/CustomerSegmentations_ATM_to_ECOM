"""Export the ATM-to-ECOM debit-card training dataset to CSV."""

from __future__ import annotations

import logging
from pathlib import Path
import csv

from sqlalchemy import text

from db_connection import create_mysql_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = PROJECT_ROOT / "SQL" / "build_ecom_debit_training_dataset.sql"
OUTPUT_PATH = PROJECT_ROOT / "Data" / "ecom_debit_training_dataset.csv"
CHUNK_SIZE = 5_000
LOGGER = logging.getLogger(__name__)


def export_dataset() -> int:
    query = SQL_PATH.read_text(encoding="utf-8-sig").strip().removesuffix(";")
    engine = create_mysql_engine()
    temporary_path = OUTPUT_PATH.with_suffix(".csv.tmp")
    row_count = 0
    try:
        with engine.connect().execution_options(stream_results=True) as connection:
            result = connection.execute(text(query))
            fieldnames = list(result.keys())
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with temporary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                while rows := result.fetchmany(CHUNK_SIZE):
                    writer.writerows(dict(row._mapping) for row in rows)
                    row_count += len(rows)
        temporary_path.replace(OUTPUT_PATH)
        return row_count
    except Exception:
        temporary_path.unlink(missing_ok=True)
        LOGGER.error("Export failed. Check database access and SQL configuration.")
        raise
    finally:
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        exported_rows = export_dataset()
    except Exception:
        raise SystemExit("Export did not complete. Error details were withheld to protect connection information.")
    print(f"Export complete: {exported_rows:,} rows written to {OUTPUT_PATH}")
