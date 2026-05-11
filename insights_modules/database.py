"""
modules/database.py
-------------------
Data layer for the Smart Budget Reallocation System.
Handles all SQLite interactions: schema creation, expense insertion, and aggregations.

The database file is stored in the project's /data/ directory.
Path is resolved relative to this file's location so the project is portable.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────
# Resolve: <project_root>/data/budget_tracker.db
_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
_DATA_DIR    = os.path.join(_PROJECT_ROOT, "data")
os.makedirs(_DATA_DIR, exist_ok=True)          # create data/ if missing
DB_PATH = os.path.join(_DATA_DIR, "budget_tracker.db")


# ── Connection Helper ───────────────────────────────────────────────────────────
def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set to dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema Bootstrap ────────────────────────────────────────────────────────────
def init_db() -> None:
    """
    Create the expenses table if it does not already exist.
    Safe to call multiple times (idempotent).
    """
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT    NOT NULL,
                category    TEXT    NOT NULL,
                amount      REAL    NOT NULL CHECK(amount > 0),
                date        TEXT    NOT NULL   -- ISO-8601 string: YYYY-MM-DD
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_category
            ON expenses (user_id, category)
        """)
        
        # Monthly summaries table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS monthly_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                month INTEGER NOT NULL,
                year INTEGER NOT NULL,
                total_spent REAL NOT NULL,
                spending_data TEXT,
                created_at TEXT,
                saving REAL DEFAULT 0,
                saving_ratio REAL DEFAULT 0,
                total_cumulative REAL DEFAULT 0,
                UNIQUE(user_id, month, year)
            )
        """)
        conn.commit()


# ── Write Operations ──────────────────────────────────────────────────────────
def add_expense(
    user_id:  str,
    category: str,
    amount:   float,
    date:     Optional[str] = None,
) -> int:
    """
    Insert a new expense record.

    Returns
    -------
    Auto-incremented row id of the inserted record.

    Raises
    ------
    ValueError if amount ≤ 0 or required fields are empty.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty.")
    if not category or not category.strip():
        raise ValueError("category must not be empty.")
    if amount <= 0:
        raise ValueError(f"amount must be positive; got {amount}.")

    date = date or datetime.today().strftime("%Y-%m-%d")

    with _get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO expenses (user_id, category, amount, date) VALUES (?, ?, ?, ?)",
            (user_id.strip(), category.strip().lower(), amount, date),
        )
        conn.commit()
        return cursor.lastrowid


# ── Read Operations ───────────────────────────────────────────────────────────
def get_total_spent(user_id: str, category: str) -> float:
    """Total amount spent by a user in a given category (all time). Returns 0.0 if none."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0.0) AS total FROM expenses "
            "WHERE user_id = ? AND category = ?",
            (user_id.strip(), category.strip().lower()),
        ).fetchone()
    return float(row["total"])


def get_all_categories_spent(user_id: str) -> dict[str, float]:
    """Return {category: total_spent} for all categories with at least one expense."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT category, COALESCE(SUM(amount), 0.0) AS total "
            "FROM expenses WHERE user_id = ? GROUP BY category",
            (user_id.strip(),),
        ).fetchall()
    return {row["category"]: float(row["total"]) for row in rows}


def get_monthly_spending(user_id: str, month: int, year: int) -> dict[str, float]:
    """Return {category: total_spent} for a specific month/year."""
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-31"
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT category, COALESCE(SUM(amount), 0.0) AS total "
            "FROM expenses WHERE user_id = ? AND date BETWEEN ? AND ? GROUP BY category",
            (user_id.strip(), start_date, end_date),
        ).fetchall()
    return {row["category"]: float(row["total"]) for row in rows}


def get_monthly_totals(user_id: str, year: int) -> dict[int, float]:
    """Return {month: total_spent} for all months in a year."""
    result = {m: 0.0 for m in range(1, 13)}
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT strftime('%m', date) AS month, COALESCE(SUM(amount), 0.0) AS total "
            "FROM expenses WHERE user_id = ? AND strftime('%Y', date) = ? "
            "GROUP BY month",
            (user_id.strip(), str(year)),
        ).fetchall()
    for row in rows:
        month_num = int(row["month"])
        result[month_num] = float(row["total"])
    return result


def get_expenses_by_date_range(
    user_id:    str,
    start_date: str,
    end_date:   str,
) -> list[dict]:
    """Fetch individual expense rows within an inclusive date range (YYYY-MM-DD)."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT id, user_id, category, amount, date FROM expenses "
            "WHERE user_id = ? AND date BETWEEN ? AND ? ORDER BY date ASC",
            (user_id.strip(), start_date, end_date),
        ).fetchall()
    return [dict(row) for row in rows]


def get_weekly_summary(user_id: str) -> dict[str, float]:
    """Aggregate spending for the last 7 calendar days per category."""
    today          = datetime.today()
    seven_days_ago = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    today_str      = today.strftime("%Y-%m-%d")

    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT category, COALESCE(SUM(amount), 0.0) AS total "
            "FROM expenses WHERE user_id = ? AND date BETWEEN ? AND ? "
            "GROUP BY category",
            (user_id.strip(), seven_days_ago, today_str),
        ).fetchall()
    return {row["category"]: float(row["total"]) for row in rows}


def get_transaction_amounts(user_id: str, category: str) -> list[float]:
    """
    Return chronologically ordered individual transaction amounts for a user+category.
    Used by the z-score anomaly detector.  Returns [] if no transactions exist.
    """
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT amount FROM expenses WHERE user_id = ? AND category = ? "
            "ORDER BY date ASC, id ASC",
            (user_id.strip(), category.strip().lower()),
        ).fetchall()
    return [float(row["amount"]) for row in rows]


def save_monthly_summary(user_id: str, month: int, year: int, end_date: str, saving: float = 0, saving_ratio: float = 0) -> None:
    """
    Archive current month's spending to monthly_summary table.
    Includes saving data for cumulative tracking.
    """
    spending = get_monthly_spending(user_id, month, year)
    total = sum(spending.values())
    
    # Get total saved so far from previous months
    prev_savings = get_total_savings(user_id, year, upto_month=month-1)
    total_savings = prev_savings + saving
    
    with _get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO monthly_summary 
            (user_id, month, year, total_spent, spending_data, created_at, saving, saving_ratio, total_cumulative)
            VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?)
        """, (user_id, month, year, total, str(spending), saving, saving_ratio, total_savings))
        conn.commit()


def get_total_savings(user_id: str, year: int, upto_month: int = 12) -> float:
    """Get cumulative savings from start of year up to (not including) a month."""
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT COALESCE(SUM(saving), 0) as total FROM monthly_summary
            WHERE user_id = ? AND year = ? AND month < ?
        """, (user_id, year, upto_month)).fetchone()
        return float(rows["total"]) if rows else 0.0


def get_monthly_summaries(user_id: str, year: int) -> list[dict]:
    """Get all monthly summaries for a user in a year."""
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT month, year, total_spent, spending_data, created_at
            FROM monthly_summary 
            WHERE user_id = ? AND year = ?
            ORDER BY month DESC
        """, (user_id.strip(), year)).fetchall()
    return [
        {
            "month": row["month"],
            "year": row["year"],
            "total_spent": row["total_spent"],
            "spending_data": eval(row["spending_data"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def clear_month_expenses(user_id: str, month: int, year: int) -> int:
    """
    Delete all expenses for a specific month/year to start fresh.
    Returns number of deleted records.
    """
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-31"
    with _get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM expenses WHERE user_id = ? AND date BETWEEN ? AND ?",
            (user_id.strip(), start_date, end_date)
        )
        conn.commit()
        return cursor.rowcount


def clear_user_expenses(user_id: str) -> int:
    """Delete all expense records for a user. Returns count of deleted rows."""
    with _get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM expenses WHERE user_id = ?", (user_id.strip(),)
        )
        conn.commit()
        return cursor.rowcount


# ── Module Init ───────────────────────────────────────────────────────────────
init_db()
