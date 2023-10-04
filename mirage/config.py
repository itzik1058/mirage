from os import getenv

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

TELEGRAM_BOT_TOKEN = getenv("TELEGRAM_BOT_TOKEN", "")
