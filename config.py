from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "mall_customers.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "app.log"

# Notebook uses these
FEATURE_SET_2D = ["Annual_Income", "Spending_Score"]
FEATURE_SET_3D = ["Age", "Annual_Income", "Spending_Score"]

K_MIN = 3
K_MAX = 8
DEFAULT_K = 5
RANDOM_STATE = 42