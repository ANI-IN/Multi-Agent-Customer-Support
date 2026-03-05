"""
Database module for the Chinook SQLite database.
Handles downloading, loading, and providing access to the sample database.

FIX #14: Bundle SQL locally with GitHub fallback.
"""

import os
import re
import sqlite3
import requests
import logging
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

_engine = None
_db = None

CHINOOK_SQL_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/"
    "master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)

LOCAL_SQL_PATH = os.path.join(os.path.dirname(__file__), "Chinook_Sqlite.sql")


def _load_sql_script() -> str:
    """Load Chinook SQL script from local file first, then fallback to GitHub."""
    # Try local file first (FIX #14)
    if os.path.isfile(LOCAL_SQL_PATH):
        logger.info(f"Loading Chinook SQL from local file: {LOCAL_SQL_PATH}")
        with open(LOCAL_SQL_PATH, "r", encoding="utf-8") as f:
            return f.read()

    # Fallback to GitHub download
    logger.info("Local SQL file not found. Downloading from GitHub...")
    response = requests.get(CHINOOK_SQL_URL, timeout=60)
    response.raise_for_status()
    sql_script = response.text

    # Cache locally for future startups
    try:
        with open(LOCAL_SQL_PATH, "w", encoding="utf-8") as f:
            f.write(sql_script)
        logger.info(f"Cached SQL script to {LOCAL_SQL_PATH}")
    except Exception as e:
        logger.warning(f"Could not cache SQL script locally: {e}")

    return sql_script


def _create_engine():
    """Load Chinook SQL script and create an in-memory SQLite engine."""
    sql_script = _load_sql_script()

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    logger.info("Chinook database loaded successfully into memory.")

    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def get_engine():
    """Get or create the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        _engine = _create_engine()
    return _engine


def get_db() -> SQLDatabase:
    """Get or create the LangChain SQLDatabase instance (singleton)."""
    global _db
    if _db is None:
        _db = SQLDatabase(get_engine())
    return _db


def run_query_safe(query: str, params: dict = None) -> str:
    """
    Run a parameterized SQL query safely using SQLAlchemy text().
    Returns the result as a string.

    FIX #1: Parameterized queries to prevent SQL injection.
    """
    engine = get_engine()
    try:
        with engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
            if not rows:
                return "[]"
            # Return as list of dicts for clarity (FIX #10)
            results_list = [dict(zip(columns, row)) for row in rows]
            return str(results_list)
    except Exception as e:
        logger.error(f"Query error: {e} | query={query} | params={params}")
        raise


def normalize_phone(phone: str) -> str:
    """
    Normalize a phone number by stripping everything except digits and leading +.
    FIX #4: Phone number normalization.
    """
    if not phone:
        return ""
    phone = phone.strip()
    if phone.startswith("+"):
        return "+" + re.sub(r"[^\d]", "", phone[1:])
    return re.sub(r"[^\d]", "", phone)


def verify_database() -> bool:
    """Verify the database is loaded and accessible."""
    try:
        db = get_db()
        result = db.run("SELECT COUNT(*) FROM Customer;")
        logger.info(f"Database verification OK. Customer count query returned: {result}")
        return True
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False
