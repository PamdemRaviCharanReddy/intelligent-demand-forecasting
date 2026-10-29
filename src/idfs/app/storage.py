"""Persistent run storage for historical forecasts, simulations, and planner reviews (P6-T06).

Supports SQLite with automatic JSON fallback if SQLite DLL is blocked by OS policies.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Check if sqlite3 DLL is permitted by host OS
_SQLITE_AVAILABLE = False
try:
    import sqlite3

    # Test dummy connection to verify DLL loading
    _test_conn = sqlite3.connect(":memory:")
    _test_conn.close()
    _SQLITE_AVAILABLE = True
except Exception as e:
    logger.warning("SQLite unavailable (%s). Falling back to JSON persistent storage.", e)
    _SQLITE_AVAILABLE = False

DEFAULT_DB_PATH = Path("data/runs.db")
DEFAULT_JSON_PATH = Path("data/simulation_runs.json")


class RunStorage:
    """Run store managing simulation history and planner review records (SQLite / JSON fallback)."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.json_path = self.db_path.parent / "simulation_runs.json"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_sqlite = _SQLITE_AVAILABLE

        if self.use_sqlite:
            self._init_sqlite_db()
        else:
            self._init_json_store()

    def _get_connection(self) -> Any:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite_db(self) -> None:
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

    def _init_json_store(self) -> None:
        if not self.json_path.exists():
            self.json_path.write_text("[]", encoding="utf-8")

    def _read_json_runs(self) -> list[dict[str, Any]]:
        self._init_json_store()
        try:
            return json.loads(self.json_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write_json_runs(self, runs: list[dict[str, Any]]) -> None:
        self.json_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")

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
        """Save a simulation or forecast run into persistent storage."""
        rid = run_id or str(uuid.uuid4())[:8]
        ts = pd.Timestamp.now().isoformat()

        if self.use_sqlite:
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
        else:
            runs = self._read_json_runs()
            runs.insert(
                0,
                {
                    "run_id": rid,
                    "timestamp": ts,
                    "model_name": model_name,
                    "policy_name": policy_name,
                    "lead_time": lead_time,
                    "holding_cost": holding_cost,
                    "stockout_cost": stockout_cost,
                    "ordering_cost": ordering_cost,
                    "total_cost": total_cost,
                    "fill_rate": fill_rate,
                    "stockout_periods": stockout_periods,
                    "n_orders": n_orders,
                    "planner_decision": planner_decision,
                    "planner_notes": planner_notes,
                },
            )
            self._write_json_runs(runs)

        return rid

    def get_recent_runs(self, limit: int = 25) -> pd.DataFrame:
        """Retrieve recent simulation runs sorted descending by timestamp."""
        if self.use_sqlite:
            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM simulation_runs ORDER BY timestamp DESC LIMIT ?",
                    conn,
                    params=(limit,),
                )
            return df
        else:
            runs = self._read_json_runs()
            if not runs:
                return pd.DataFrame(
                    columns=[
                        "run_id",
                        "timestamp",
                        "model_name",
                        "policy_name",
                        "lead_time",
                        "holding_cost",
                        "stockout_cost",
                        "ordering_cost",
                        "total_cost",
                        "fill_rate",
                        "stockout_periods",
                        "n_orders",
                        "planner_decision",
                        "planner_notes",
                    ]
                )
            return pd.DataFrame(runs[:limit])

    def get_all_runs(self) -> pd.DataFrame:
        """Retrieve all simulation runs sorted descending by timestamp."""
        return self.get_recent_runs(limit=10000)

    def get_run_by_id(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a specific run by its run_id."""
        if self.use_sqlite:
            with self._get_connection() as conn:
                cur = conn.execute("SELECT * FROM simulation_runs WHERE run_id = ?", (run_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return dict(row)
        else:
            runs = self._read_json_runs()
            for r in runs:
                if r.get("run_id") == run_id:
                    return r
            return None

    def update_planner_decision(
        self,
        run_id: str,
        decision: str,
        notes: str = "",
    ) -> bool:
        """Update planner approval status and qualitative review notes."""
        if self.use_sqlite:
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
        else:
            runs = self._read_json_runs()
            updated = False
            for r in runs:
                if r.get("run_id") == run_id:
                    r["planner_decision"] = decision
                    r["planner_notes"] = notes
                    updated = True
                    break
            if updated:
                self._write_json_runs(runs)
            return updated

    def update_planner_review(
        self,
        run_id: str,
        decision: str,
        notes: str = "",
    ) -> bool:
        """Convenience alias for update_planner_decision."""
        return self.update_planner_decision(run_id=run_id, decision=decision, notes=notes)
