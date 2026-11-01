"""MySQL connection helpers for the ATM-to-digital adoption project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "Config" / "config.yaml"
LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration and return its database section."""
    try:
        with config_path.open("r", encoding="utf-8-sig") as config_file:
            config = yaml.safe_load(config_file) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc

    database = config.get("database")
    if not isinstance(database, dict):
        raise ValueError("The config file must contain a 'database' section.")

    required = ("host", "port", "name", "username", "password")
    missing = [key for key in required if database.get(key) in (None, "", f"YOUR_{key.upper()}")]
    if missing:
        raise ValueError(
            "Fill these database settings in Config/config.yaml before connecting: "
            + ", ".join(missing)
        )
    return database


def create_mysql_engine(config_path: Path = DEFAULT_CONFIG_PATH) -> Engine:
    """Create a SQLAlchemy engine using MySQL Connector/Python."""
    database = load_config(config_path)
    url = URL.create(
        drivername="mysql+mysqlconnector",
        username=str(database["username"]),
        password=str(database["password"]),
        host=str(database["host"]),
        port=int(database["port"]),
        database=str(database["name"]),
        query={"connection_timeout": "10"},
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=1800)


def check_connection(config_path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Run a lightweight query and return whether the database is reachable."""
    engine = create_mysql_engine(config_path)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        # Avoid logging connection details or credentials.
        LOGGER.exception("MySQL connection check failed.")
        raise
    finally:
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        check_connection()
    except Exception:
        raise SystemExit("MySQL connection failed. Check the settings and database availability.")
    print("MySQL connection successful (SELECT 1).")
