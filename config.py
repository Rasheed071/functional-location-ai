# config.py
# This file contains constants and default values for the OpenAI API
# including the API key, model names, and other parameters.

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API key    
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI API settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
