"""
analyzer/cleaner.py
───────────────────
Stage 2: Data Cleaning & Preprocessing

Purpose: Cleans raw Play Store reviews, handles missing text, normalizes dates,
performs language detection, and outputs analysis-ready datasets.

Usage:
    python analyzer/cleaner.py
    OR imported in run_pipeline.py via run_cleaning()

Output:
    data/processed/reviews_clean.csv
    data/processed/reviews_clean.json
    data/processed/clean_metadata.json
"""

import os
import sys
import re
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_CSV_PATH,
    RAW_JSON_PATH,
    PROCESSED_DATA_DIR,
    PROCESSED_CSV_PATH,
    PROCESSED_JSON_PATH,
    CLEAN_METADATA_PATH,
)

# Optional dependencies with safe fallbacks
try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataCleaner")


# 1. Text Cleaning Utilities

def clean_text_for_nlp(text: str) -> str:
    """
    Cleans raw review text for NLP tasks (TF-IDF, topic modeling, word clouds):
    - Lowercases text
    - Removes URLs and HTML artifacts
    - Converts/strips emojis
    - Normalizes punctuation and repeated characters
    - Strips excess whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    cleaned = text.lower()

    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Remove HTML tags & entities
    cleaned = re.sub(r"<.*?>", " ", cleaned)
    cleaned = re.sub(r"&[a-z]+;", " ", cleaned)

    # Convert or strip emojis
    if HAS_EMOJI:
        # demojize converts 😡 to :enraged_face:
        cleaned = emoji.demojize(cleaned, delimiters=(" ", " "))
        # Replace colons and underscores from demojize tags
        cleaned = re.sub(r":([a-zA-Z0-9_]+):", r" \1 ", cleaned)
    else:
        # Fallback: remove non-ascii characters
        cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)

    # Normalize repetitive characters (e.g. "sooooo baaaad" -> "soo baad")
    cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)

    # Keep letters, numbers, and basic punctuation for phrasing
    cleaned = re.sub(r"[^a-zA-Z0-9\s.,!?'\-_]", " ", cleaned)

    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def detect_language_safe(text: str) -> str:
    """
    Detects the language of a review safely.
    Handles short texts, Hinglish, and numeric content gracefully.
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return "unknown"

    cleaned = text.strip()

    # Fast-path for simple short English words
    common_short_words = {"good", "bad", "worst", "nice", "best", "great", "ok", "okay", "fine", "slow", "poor"}
    words_in_text = set(re.findall(r"\b[a-zA-Z]+\b", cleaned.lower()))
    if words_in_text and words_in_text.issubset(common_short_words):
        return "en"

    if HAS_LANGDETECT:
        try:
            return detect(cleaned)
        except Exception:
            pass

    # Heuristic fallback: check if Latin / ASCII characters dominate
    ascii_chars = sum(1 for c in cleaned if ord(c) < 128)
    if ascii_chars / max(len(cleaned), 1) > 0.85:
        return "en"

    return "unknown"


def extract_rating_class(star: int) -> str:
    """Classifies 1-5 star ratings into standard sentiment buckets."""
    if star in [1, 2]:
        return "Negative"
    elif star == 3:
        return "Neutral"
    elif star in [4, 5]:
        return "Positive"
    return "Unknown"


# 2. Main Cleaning Pipeline

def load_raw_data() -> pd.DataFrame:
    """Loads raw reviews from CSV (or fallback to JSON)."""
    if os.path.exists(RAW_CSV_PATH):
        logger.info(f"Loading raw data from CSV: {RAW_CSV_PATH}")
        df = pd.read_csv(RAW_CSV_PATH)
    elif os.path.exists(RAW_JSON_PATH):
        logger.info(f"CSV not found. Loading from JSON: {RAW_JSON_PATH}")
        with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise FileNotFoundError(
            f"No raw data found at '{RAW_CSV_PATH}' or '{RAW_JSON_PATH}'. "
            "Please run Stage 1 (scraper) first."
        )

    logger.info(f"Raw reviews loaded: {len(df):,} records")
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Executes full transformation & cleaning pipeline on reviews dataframe.
    Returns cleaned dataframe and a metadata audit dictionary.
    """
    initial_count = len(df)
    metrics = {
        "initial_count": initial_count,
        "duplicates_removed": 0,
        "empty_text_reviews": 0,
        "language_distribution": {},
        "rating_distribution": {},
        "date_range": {},
        "processed_timestamp": datetime.now().isoformat()
    }

    # Standardize column names if different
    col_mapping = {
        "reviewId": "review_id",
        "userName": "user_name",
        "content": "review_text",
        "score": "star_rating",
        "thumbsUpCount": "thumbs_up",
        "at": "review_date",
        "replyContent": "dev_reply",
        "repliedAt": "dev_reply_date",
        "appVersion": "app_version",
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    # Ensure required columns exist
    for col in ["review_id", "user_name", "review_text", "star_rating", "review_date"]:
        if col not in df.columns:
            df[col] = None

    # Step 1: Deduplication
    # Deduplicate primarily on review_id
    if "review_id" in df.columns and df["review_id"].notna().any():
        df = df.drop_duplicates(subset=["review_id"])
    
    # Secondary deduplicate on identical user, text, and rating
    df = df.drop_duplicates(subset=["user_name", "review_text", "star_rating"])
    metrics["duplicates_removed"] = initial_count - len(df)
    logger.info(f"Deduplication complete: {metrics['duplicates_removed']} duplicate records dropped.")

    # Step 2: Handle missing text & rating-only reviews
    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    df["has_text_content"] = df["review_text"].apply(lambda t: len(t) > 0)
    metrics["empty_text_reviews"] = int((~df["has_text_content"]).sum())
    logger.info(f"Reviews with text: {int(df['has_text_content'].sum())} | Rating-only: {metrics['empty_text_reviews']}")

    # Step 3: Date normalization & temporal features
    df["review_date_parsed"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True)
    # Remove timezone info for clean universal timestamps
    df["review_date_dt"] = df["review_date_parsed"].dt.tz_localize(None)

    df["date"] = df["review_date_dt"].dt.strftime("%Y-%m-%d")
    df["review_date"] = df["review_date_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["year"] = df["review_date_dt"].dt.year
    df["month"] = df["review_date_dt"].dt.month
    df["year_month"] = df["review_date_dt"].dt.strftime("%Y-%m")
    df["day_of_week"] = df["review_date_dt"].dt.day_name()
    df["hour"] = df["review_date_dt"].dt.hour

    min_date = df["date"].min()
    max_date = df["date"].max()
    metrics["date_range"] = {"oldest": min_date, "newest": max_date}
    logger.info(f"Date range: {min_date} to {max_date}")

    # Drop temporary datetime column
    df = df.drop(columns=["review_date_parsed", "review_date_dt"], errors="ignore")

    # Step 4: Text Cleaning for NLP & metrics
    logger.info("Cleaning review text and generating NLP tokens...")
    df["review_text_clean"] = df["review_text"].apply(clean_text_for_nlp)
    df["char_count"] = df["review_text"].str.len()
    df["word_count"] = df["review_text"].apply(lambda t: len(t.split()) if t else 0)

    # Step 5: Language Detection
    logger.info("Detecting languages...")
    df["detected_language"] = df["review_text"].apply(detect_language_safe)
    # English flag: "en" or unknown ASCII text
    df["is_english"] = df["detected_language"].isin(["en", "unknown"])
    lang_counts = df["detected_language"].value_counts().to_dict()
    metrics["language_distribution"] = {str(k): int(v) for k, v in lang_counts.items()}
    logger.info(f"Language distribution: {metrics['language_distribution']}")

    # Step 6: Star Rating & Sentiment Class
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce").fillna(0).astype(int)
    df["rating_class"] = df["star_rating"].apply(extract_rating_class)
    rating_counts = df["star_rating"].value_counts().sort_index().to_dict()
    metrics["rating_distribution"] = {str(k): int(v) for k, v in rating_counts.items()}

    # Step 7: App Version normalization
    if "app_version" in df.columns:
        df["app_version"] = df["app_version"].fillna("Unknown").astype(str).str.strip()
        df["app_version"] = df["app_version"].replace("", "Unknown")
        # Extract major.minor version (e.g., 12.19 from 12.19.0)
        df["app_version_short"] = df["app_version"].apply(
            lambda v: re.match(r"^(\d+\.\d+)", v).group(1) if re.match(r"^(\d+\.\d+)", v) else v
        )
    else:
        df["app_version"] = "Unknown"
        df["app_version_short"] = "Unknown"

    # Step 8: Developer response features
    if "dev_reply" in df.columns:
        df["dev_reply"] = df["dev_reply"].fillna("").astype(str).str.strip()
        df["has_dev_reply"] = df["dev_reply"].apply(lambda r: len(r) > 0)
        
        if "dev_reply_date" in df.columns:
            dev_dt = pd.to_datetime(df["dev_reply_date"], errors="coerce", utc=True).dt.tz_localize(None)
            df["dev_reply_date"] = dev_dt.dt.strftime("%Y-%m-%d %H:%M:%S")
            # Calculate response time delay in days
            rev_dt = pd.to_datetime(df["review_date"], errors="coerce")
            df["dev_reply_delay_days"] = (dev_dt - rev_dt).dt.total_seconds() / (24 * 3600)
            df["dev_reply_delay_days"] = df["dev_reply_delay_days"].apply(
                lambda d: round(d, 1) if pd.notnull(d) and d >= 0 else None
            )
        else:
            df["dev_reply_date"] = ""
            df["dev_reply_delay_days"] = None
    else:
        df["dev_reply"] = ""
        df["has_dev_reply"] = False
        df["dev_reply_date"] = ""
        df["dev_reply_delay_days"] = None

    # Step 9: Reorder columns logically
    ordered_cols = [
        "review_id",
        "user_name",
        "star_rating",
        "rating_class",
        "review_date",
        "date",
        "year",
        "month",
        "year_month",
        "day_of_week",
        "review_text",
        "review_text_clean",
        "has_text_content",
        "char_count",
        "word_count",
        "detected_language",
        "is_english",
        "app_version",
        "app_version_short",
        "thumbs_up",
        "has_dev_reply",
        "dev_reply",
        "dev_reply_date",
        "dev_reply_delay_days",
    ]
    # Keep any additional columns at the end
    existing_cols = [c for c in ordered_cols if c in df.columns]
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[existing_cols + extra_cols]

    metrics["final_clean_count"] = len(df)
    return df, metrics


# ─────────────────────────────────────────────────────────────────
# 3. Execution & Export
# ─────────────────────────────────────────────────────────────────

def save_clean_data(df: pd.DataFrame, metrics: dict):
    """Saves processed data and metadata to data/processed/ directory."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Save CSV
    df.to_csv(PROCESSED_CSV_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved clean CSV: {PROCESSED_CSV_PATH} ({len(df):,} rows)")

    # Save JSON
    df.to_json(PROCESSED_JSON_PATH, orient="records", date_format="iso", indent=2)
    logger.info(f"Saved clean JSON: {PROCESSED_JSON_PATH}")

    # Save Metadata audit
    with open(CLEAN_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved clean metadata: {CLEAN_METADATA_PATH}")


def print_clean_summary(metrics: dict):
    """Prints a professional validation summary to the terminal."""
    print("\n" + "=" * 54)
    print("  Stage 2 Preprocessing & Data Cleaning Summary")
    print("=" * 54)
    print(f"  Initial Reviews Loaded  : {metrics.get('initial_count', 0):,}")
    print(f"  Duplicates Removed      : {metrics.get('duplicates_removed', 0):,}")
    print(f"  Final Cleaned Reviews   : {metrics.get('final_clean_count', 0):,}")
    print(f"  Rating-Only (No text)   : {metrics.get('empty_text_reviews', 0):,}")
    date_range = metrics.get('date_range', {})
    print(f"  Date Range Covered      : {date_range.get('oldest', 'N/A')} to {date_range.get('newest', 'N/A')}")
    print("\n  Star Rating Breakdown:")
    for star, count in metrics.get('rating_distribution', {}).items():
        print(f"    ⭐ {star}-Star : {count:,} reviews")
    print("\n  Language Breakdown:")
    for lang, count in sorted(metrics.get('language_distribution', {}).items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    - {lang:<8} : {count:,} reviews")
    print("=" * 54)
    print("  [SUCCESS] Stage 2 Complete! Output ready in data/processed/")
    print("=" * 54 + "\n")


def run_cleaning() -> pd.DataFrame:
    """Entry point callable from run_pipeline.py or directly."""
    df_raw = load_raw_data()
    df_clean, metrics = clean_dataframe(df_raw)
    save_clean_data(df_clean, metrics)
    print_clean_summary(metrics)
    return df_clean


if __name__ == "__main__":
    run_cleaning()
