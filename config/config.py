import os
from dotenv import load_dotenv

# טעינת משתנים מקובץ .env
load_dotenv()

# משתני הסביבה
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

DALLE_API_KEY = os.getenv("DALLE_API_KEY")
DALLE_ENDPOINT = os.getenv("DALLE_ENDPOINT")
DALLE_API_VERSION = os.getenv("DALLE_API_VERSION")

# Microsoft authentication settings
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
MS_TENANT_ID = os.getenv("MS_TENANT_ID")
MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI")

# JWT secret key
SECRET_KEY = os.getenv("SECRET_KEY")

# Google Serper API key for web search
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# Rate Limiting Configuration
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "3"))
MAX_CONCURRENT_EMBEDDING_CALLS = int(os.getenv("MAX_CONCURRENT_EMBEDDING_CALLS", "2"))

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"