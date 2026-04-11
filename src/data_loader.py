# =========================
# Data Loading Module
# =========================
# This file handles loading and validating the dataset
# used for clustering analysis.

import pandas as pd
from pathlib import Path

# Required columns for clustering
# Ensures dataset has all necessary features
REQUIRED_COLS = ["Gender", "Age", "Annual_Income", "Spending_Score"]


def load_data(csv_path: Path, logger) -> pd.DataFrame:
    """
    Load dataset from CSV and validate required columns.

    Parameters:
    csv_path (Path): Path to the dataset file
    logger: Logger object for tracking execution

    Returns:
    pd.DataFrame: Loaded and validated dataset

    Purpose:
    - Load dataset into memory
    - Validate required columns for clustering
    - Prevent downstream errors due to missing data
    """
    try:
        # =========================
        # Load Dataset
        # =========================
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Log dataset shape (rows, columns)
        logger.info(f"Loaded shape: {df.shape}")

        # =========================
        # Validate Required Columns
        # =========================
        # Check if all required columns exist in dataset
        missing = [c for c in REQUIRED_COLS if c not in df.columns]

        if missing:
            # Raise error if required columns are missing
            raise ValueError(f"Missing required columns: {missing}")

        return df

    except Exception:
        # Log full error details for debugging
        logger.exception("Failed to load dataset.")
        raise