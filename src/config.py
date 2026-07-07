"""Load environment variables and provide API keys."""

import os
from dotenv import load_dotenv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


def get_api_key(provider: str) -> str:
    if provider == "google":
        key = os.getenv("GOOGLE_API_KEY")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if not key:
        raise ValueError(
            f"API key for '{provider}' not found. "
            f"Make sure your .env file contains the right key."
        )

    return key