from os import getenv
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

TELEGRAM_BOT_TOKEN = getenv("TELEGRAM_BOT_TOKEN", "")
DATA_PATH = Path(getenv("MIRAGE_DATA_PATH", user_data_dir("mirage", "mirage")))
CACHE_PATH = Path(getenv("MIRAGE_CACHE_PATH", user_cache_dir("mirage", "mirage")))
