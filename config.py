import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Holds all configuration for the application."""
    TAVILY_API_KEY = os.getenv("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")
    ALPHA_VANTAGE_KEY = os.getenv("AZ6V1R6SBW2Q3TA7")
    GEMINI_API_KEY = os.getenv("AIzaSyDflgWPLuoRH5mczK7gu0i_iOlg332Oliw")

    MODEL_PATH = "ai_plus_model.pth"
    PREPROCESSOR_PATH = "preprocessor.pkl"

    # Caching configuration
    CACHE_TTL_SECONDS = 3600  # Cache API calls for 1 hour

    # Monte Carlo Simulation parameters
    SIMULATION_RUNS = 2000
    SIMULATION_MONTHS = 36

# Instantiate the config
config = Config()