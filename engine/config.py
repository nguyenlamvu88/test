import os
from dataclasses import dataclass


DEFAULT_SUBREDDITS = (
    "pennystocks",
    "10xPennyStocks",
    "100xPennyStocks",
    "Shortsqueeze",
    "RobinHoodPennyStocks",
    "wallstreetbets",
    "biotech_stocks",
    "SPACs",
)


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "")
    arctic_base_url: str = os.getenv(
        "ARCTIC_BASE_URL", "https://arctic-shift.photon-reddit.com"
    )
    untouched_holdout_year: int = int(os.getenv("UNTOUCHED_HOLDOUT_YEAR", "2026"))
    target_return: float = float(os.getenv("TARGET_RETURN", "0.50"))
    max_adverse_excursion: float = float(os.getenv("MAX_ADVERSE_EXCURSION", "0.20"))
    outcome_horizon_sessions: int = int(os.getenv("OUTCOME_HORIZON_SESSIONS", "5"))


settings = Settings()
