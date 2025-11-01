# config.py
import os
from dotenv import load_dotenv

# Load .env if present (local dev)
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_VOICE      = os.getenv("DEFAULT_VOICE", "fa-IR-DilaraNeural")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
MODEL_PATH         = os.getenv("MODEL_PATH", "model-weights/homo-t5")
PROMPT_FILE        = os.getenv("PROMPT_FILE", "prompt_base.txt")
