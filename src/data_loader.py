import pandas as pd
from pathlib import Path

REQUIRED_COLS = ["Customer_ID", "Gender", "Age", "Annual_Income", "Spending_Score"]

def load_data(csv_path: Path, logger) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded shape: {df.shape}")

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df
    except Exception:
        logger.exception("Failed to load dataset.")
        raise