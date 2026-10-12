"""
scraper/playstore_scraper.py
─────────────────────────────
Stage 1: Data Collection
Fetches reviews from the Google Play Store for the MoveInSync employee app.

Usage:
    python scraper/playstore_scraper.py

Output:
    data/raw/reviews_raw.json
    data/raw/reviews_raw.csv
    data/raw/scrape_metadata.json
"""

import os
import json
import time
import sys
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    APP_ID,
    APP_NAME,
    SCRAPE_LANGUAGE,
    SCRAPE_COUNTRY,
    REVIEWS_PER_BATCH,
    MAX_REVIEWS_TOTAL,
    SCRAPE_RATINGS,
    RAW_DATA_DIR,
    RAW_JSON_PATH,
    RAW_CSV_PATH,
    SCRAPE_METADATA_PATH,
)

try:
    from google_play_scraper import reviews, Sort, app as get_app_info
except ImportError:
    print("ERROR: google-play-scraper not installed.")
    print("Run: pip install google-play-scraper")
    sys.exit(1)

# Helpers

def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)


def serialize_review(review: dict) -> dict:
    """Convert datetime objects to ISO strings for JSON serialization."""
    serialized = {}
    for k, v in review.items():
        if isinstance(v, datetime):
            serialized[k] = v.isoformat()
        else:
            serialized[k] = v
    return serialized


# App Info

def fetch_app_info() -> dict:
    """Fetch current app metadata (rating, installs, version, etc.)."""
    print(f"\n📱 Fetching app metadata for: {APP_ID}")
    try:
        info = get_app_info(APP_ID, lang=SCRAPE_LANGUAGE, country=SCRAPE_COUNTRY)
        print(f"   App Name    : {info.get('title')}")
        print(f"   Current Rating : {info.get('score', 'N/A')}")
        print(f"   Total Ratings  : {info.get('ratings', 'N/A'):,}")
        print(f"   Installs    : {info.get('installs', 'N/A')}")
        print(f"   Version     : {info.get('version', 'N/A')}")
        return info
    except Exception as e:
        print(f"   WARNING: Could not fetch app info — {e}")
        return {}


# Core Scraper

def scrape_reviews_by_rating(star_rating: int, max_count: int) -> list:
    """
    Scrape reviews filtered by a specific star rating.
    Uses pagination (continuation_token) to fetch large volumes.
    """
    all_reviews = []
    continuation_token = None
    batch_num = 0

    print(f"\n  ⭐ Scraping {star_rating}-star reviews (target: {max_count})...")

    with tqdm(total=max_count, unit="reviews", leave=False) as pbar:
        while len(all_reviews) < max_count:
            try:
                batch_size = min(REVIEWS_PER_BATCH, max_count - len(all_reviews))

                result, continuation_token = reviews(
                    APP_ID,
                    lang=SCRAPE_LANGUAGE,
                    country=SCRAPE_COUNTRY,
                    sort=Sort.NEWEST,
                    count=batch_size,
                    filter_score_with=star_rating,
                    continuation_token=continuation_token,
                )

                if not result:
                    break  # No more reviews available

                all_reviews.extend(result)
                pbar.update(len(result))
                batch_num += 1

                if continuation_token is None:
                    break  # Reached the end

                time.sleep(0.5)  # Polite delay between batches

            except Exception as e:
                print(f"\n  WARNING: Error fetching {star_rating}-star reviews (batch {batch_num}): {e}")
                time.sleep(2)
                break

    print(f"  ✓ Collected {len(all_reviews)} reviews for {star_rating}-star")
    return all_reviews


def scrape_all_reviews() -> list:
    """
    Scrape reviews for all star ratings and combine them.
    Deduplicates based on reviewId.
    """
    # Distribute the total review budget across star ratings
    # Weight toward lower ratings (more important for diagnosis)
    rating_budgets = {
        1: int(MAX_REVIEWS_TOTAL * 0.35),  # 35% from 1-star
        2: int(MAX_REVIEWS_TOTAL * 0.20),  # 20% from 2-star
        3: int(MAX_REVIEWS_TOTAL * 0.15),  # 15% from 3-star
        4: int(MAX_REVIEWS_TOTAL * 0.15),  # 15% from 4-star
        5: int(MAX_REVIEWS_TOTAL * 0.15),  # 15% from 5-star
    }

    all_reviews = []
    seen_ids = set()

    for rating in SCRAPE_RATINGS:
        budget = rating_budgets.get(rating, REVIEWS_PER_BATCH)
        rating_reviews = scrape_reviews_by_rating(rating, budget)

        # Deduplicate
        for r in rating_reviews:
            rid = r.get("reviewId")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                all_reviews.append(r)

    print(f"\n✅ Total unique reviews collected: {len(all_reviews)}")
    return all_reviews


# Save Output

def save_reviews(all_reviews: list, app_info: dict):
    """Save reviews to JSON and CSV, and write scrape metadata."""

    # ── Serialize for JSON (convert datetime objects)
    serialized = [serialize_review(r) for r in all_reviews]

    # ── Save JSON
    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved JSON  → {RAW_JSON_PATH}")

    # ── Save CSV
    df = pd.DataFrame(serialized)

    # Rename columns for readability
    df = df.rename(columns={
        "reviewId": "review_id",
        "userName": "user_name",
        "userImage": "user_image",
        "content": "review_text",
        "score": "star_rating",
        "thumbsUpCount": "thumbs_up",
        "reviewCreatedVersion": "app_version",
        "at": "review_date",
        "replyContent": "dev_reply",
        "repliedAt": "dev_reply_date",
        "appVersion": "app_version_code",
    })

    # Keep only relevant columns
    keep_cols = [
        "review_id", "user_name", "review_text", "star_rating",
        "thumbs_up", "app_version", "review_date", "dev_reply", "dev_reply_date"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8")
    print(f"💾 Saved CSV   → {RAW_CSV_PATH}")

    # ── Save Metadata
    dates = [r.get("at") for r in all_reviews if r.get("at")]
    if dates:
        dates_parsed = []
        for d in dates:
            if isinstance(d, str):
                try:
                    dates_parsed.append(datetime.fromisoformat(d))
                except Exception:
                    pass
            elif isinstance(d, datetime):
                dates_parsed.append(d)

        oldest = min(dates_parsed).isoformat() if dates_parsed else "unknown"
        newest = max(dates_parsed).isoformat() if dates_parsed else "unknown"
    else:
        oldest = newest = "unknown"

    rating_dist = df["star_rating"].value_counts().sort_index().to_dict() if "star_rating" in df.columns else {}

    metadata = {
        "app_id": APP_ID,
        "app_name": APP_NAME,
        "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_reviews": len(all_reviews),
        "date_range": {"oldest": oldest, "newest": newest},
        "rating_distribution": {str(k): int(v) for k, v in rating_dist.items()},
        "current_app_rating": app_info.get("score"),
        "total_app_ratings": app_info.get("ratings"),
        "app_installs": app_info.get("installs"),
        "app_version": app_info.get("version"),
        "scrape_settings": {
            "language": SCRAPE_LANGUAGE,
            "country": SCRAPE_COUNTRY,
            "max_reviews": MAX_REVIEWS_TOTAL,
        },
    }

    with open(SCRAPE_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved Meta  → {SCRAPE_METADATA_PATH}")

    return df


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame):
    """Run basic data quality checks and print a summary report."""
    print("\n" + "=" * 50)
    print("📊 SCRAPE VALIDATION REPORT")
    print("=" * 50)

    print(f"  Total reviews        : {len(df)}")

    if "star_rating" in df.columns:
        print(f"\n  Star Rating Distribution:")
        for star in [1, 2, 3, 4, 5]:
            count = len(df[df["star_rating"] == star])
            bar = "█" * (count // 5)
            print(f"    {star}★  {count:>5}  {bar}")

    if "review_text" in df.columns:
        empty = df["review_text"].isna().sum() + (df["review_text"] == "").sum()
        print(f"\n  Empty review texts   : {empty}")

    if "review_date" in df.columns:
        dates = pd.to_datetime(df["review_date"], errors="coerce")
        print(f"  Oldest review        : {dates.min()}")
        print(f"  Newest review        : {dates.max()}")

    if "review_id" in df.columns:
        dupes = df["review_id"].duplicated().sum()
        print(f"  Duplicate review IDs : {dupes}")

    print("\n✅ Validation complete.")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 MoveInSync Play Store Scraper — Stage 1")
    print("=" * 50)

    ensure_dirs()

    # Step 1: Get app metadata
    app_info = fetch_app_info()

    # Step 2: Scrape reviews
    print(f"\n📥 Starting review scrape for: {APP_ID}")
    print(f"   Max reviews target: {MAX_REVIEWS_TOTAL}")
    all_reviews = scrape_all_reviews()

    if not all_reviews:
        print("\n❌ No reviews collected. Check your internet connection or App ID.")
        sys.exit(1)

    # Step 3: Save to disk
    df = save_reviews(all_reviews, app_info)

    # Step 4: Validate
    validate_data(df)

    print(f"\n🎉 Stage 1 Complete! Data saved to: data/raw/")
