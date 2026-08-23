import sqlite3
import logging
from typing import Optional

class DatabaseManager:
    def __init__(self, db_path: str = "safedrive.db") -> None:
        self.db_path = db_path
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize_schema(self) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS session_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            vision_status TEXT NOT NULL,
            confidence REAL NOT NULL
        )
        """
        with self._get_connection() as conn:
            conn.execute(query)

    def log_prediction(self, status: str, confidence: float) -> None:
        query = "INSERT INTO session_logs (vision_status, confidence) VALUES (?, ?)"
        try:
            with self._get_connection() as conn:
                conn.execute(query, (status, confidence))
        except sqlite3.Error as e:
            logging.error(f"Database transaction failed: {e}")