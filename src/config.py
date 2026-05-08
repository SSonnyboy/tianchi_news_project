"""Global configuration for the unified Tianchi News Recommendation codebase."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "outputs", "cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data files
TRAIN_CLICK_PATH = os.path.join(DATA_DIR, "train_click_log.csv")
ARTICLES_PATH = os.path.join(DATA_DIR, "articles.csv")
ARTICLES_EMB_PATH = os.path.join(DATA_DIR, "articles_emb.csv")

RANDOM_SEED = 42

# Column name constants
COL_USER = "user_id"
COL_ITEM = "click_article_id"
COL_TIME = "click_timestamp"
COL_CATEGORY = "category_id"
COL_WORDS = "words_count"
COL_CREATED = "created_at_ts"

# Context feature columns
CONTEXT_COLS = [
    "click_environment", "click_deviceGroup", "click_os",
    "click_country", "click_region", "click_referrer_type",
]
