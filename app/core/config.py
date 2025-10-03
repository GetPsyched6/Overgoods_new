import os
from dotenv import load_dotenv

load_dotenv()

# Watsonx Configuration
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "your_api_key_here")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "your_project_id_here")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# Application Configuration
UPLOAD_DIR = "uploads"
ASSETS_DIR = "data/assets"  # Directory for the 10 real overgood images
CHROMA_DB_PATH = "./chroma_db"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
