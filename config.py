import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model")
INDEX_PATH = os.getenv("INDEX_PATH", "data/retrieval_data")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
