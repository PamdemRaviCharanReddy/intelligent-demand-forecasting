"""SQLite persistent run storage for historical forecasts, simulations, and planner reviews (P6-T06)."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/runs.db")


class RunStorage:
    """SQLite run store managing simulation history and planner review records."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    policy_name TEXT NOT NULL,
                    lead_time INTEGER NOT NULL,
                    holding_cost REAL NOT NULL,
                    stockout_cost REAL NOT NULL,
                    ordering_cost REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    fill_rate REAL NOT NULL,
                    stockout_periods INTEGER NOT NULL,
                    n_orders INTEGER NOT NULL,
                    planner_decision TEXT DEFAULT 'PENDING',
                    planner_notes TEXT DEFAULT ''
                )
                """
            )
            conn.commit()

    def save_run(
        self,
        model_name: str,
        policy_name: str,
        lead_time: int,
        holding_cost: float,
        stockout_cost: float,
        ordering_cost: float,
        total_cost: float,
        fill_rate: float,
        stockout_periods: int,
        n_orders: int,
        planner_decision: str = "PENDING",
        planner_notes: str = "",
        run_id: str | None = None,
    ) -> str:
        """Save a simulation or forecast run into SQLite database."""
        rid = run_id or str(uuid.uuid4())[:8]
        ts = pd.Timestamp.now().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO simulation_runs (
                    run_id, timestamp, model_name, policy_name, lead_time,
                    holding_cost, stockout_cost, ordering_cost, total_cost,
                    fill_rate, stockout_periods, n_orders, planner_decision, planner_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    ts,
                    model_name,
                    policy_name,
                    lead_time,
                    holding_cost,
                    stockout_cost,
                    ordering_cost,
                    total_cost,
                    fill_rate,
                    stockout_periods,
                    n_orders,
                    planner_decision,
                    planner_notes,
                ),
            )
            conn.commit()
        return rid

    def get_recent_runs(self, limit: int = 25) -> pd.DataFrame:
        """Retrieve recent simulation runs sorted descending by timestamp."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM simulation_runs ORDER BY timestamp DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        return df

    def get_all_runs(self) -> pd.DataFrame:
        """Retrieve all simulation runs sorted descending by timestamp."""
        return self.get_recent_runs(limit=10000)

    def get_run_by_id(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a specific run by its run_id."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM simulation_runs WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return dict(row)

    def update_planner_decision(
        self,
        run_id: str,
        decision: str,  # 'APPROVED', 'OVERRIDDEN', 'REJECTED'
        notes: str = "",
    ) -> bool:
        """Update planner approval status and qualitative review notes."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE simulation_runs
                SET planner_decision = ?, planner_notes = ?
                WHERE run_id = ?
                """,
                (decision, notes, run_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def update_planner_review(
        self,
        run_id: str,
        decision: str,
        notes: str = "",
    ) -> bool:
        """Convenience alias for update_planner_decision."""
        return self.update_planner_decision(run_id=run_id, decision=decision, notes=notes)
