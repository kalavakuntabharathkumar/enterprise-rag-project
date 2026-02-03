"""Append-only query/retrieval log backed by a CSV file, read back as a
pandas DataFrame for analysis.
"""
import os

import pandas as pd

from backend.config import Config

LOG_COLUMNS = [
    "timestamp",
    "question",
    "intent",
    "query_type",
    "num_docs_retrieved",
    "top_similarity",
    "avg_similarity",
    "confidence",
    "latency_ms",
    "answered",
    "llm_skipped",
    "estimated_tokens",
    "estimated_cost_usd",
    "llm_backend",
]


class QueryAnalytics:
    def __init__(self, path: str = None):
        self.path = path or os.path.join(Config.LOGS_PATH, "query_analytics.csv")

    def log(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        row = {col: record.get(col) for col in LOG_COLUMNS}
        df = pd.DataFrame([row], columns=LOG_COLUMNS)
        write_header = not os.path.exists(self.path)
        df.to_csv(self.path, mode="a", header=write_header, index=False)

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=LOG_COLUMNS)
        return pd.read_csv(self.path)
