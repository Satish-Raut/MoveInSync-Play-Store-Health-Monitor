"""
analyzer/categorizer.py
───────────────────────
Stage 3: AI Analysis & Categorization
Performs dual-layer sentiment scoring (VADER), detects rating divergence (sarcasm/inversion),
categorizes reviews into business complaint themes, and computes an engineering Priority Matrix.

Usage:
    python analyzer/categorizer.py
    OR imported in run_pipeline.py via run_categorization()

Output:
    data/processed/reviews_analyzed.csv
    data/processed/reviews_analyzed.json
    data/processed/analysis_summary.json
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
    PROCESSED_CSV_PATH,
    ANALYZED_CSV_PATH,
    ANALYZED_JSON_PATH,
    ANALYSIS_SUMMARY_PATH,
    COMPLAINT_CATEGORIES,
)

# Optional NLP imports with graceful fallbacks
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    vader_analyzer = None
    HAS_VADER = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Categorizer")


# ─────────────────────────────────────────────────────────────────
# 1. Sentiment Scoring & Divergence Detection
# ─────────────────────────────────────────────────────────────────

def score_sentiment(text: str) -> dict:
    """
    Computes sentiment compound and component scores using VADER (or TextBlob fallback).
    Returns dict with compound, pos, neu, neg, and text_sentiment class.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "sentiment_score": 0.0,
            "sentiment_pos": 0.0,
            "sentiment_neu": 1.0,
            "sentiment_neg": 0.0,
            "text_sentiment": "Neutral",
        }

    if HAS_VADER:
        scores = vader_analyzer.polarity_scores(text)
        compound = round(scores["compound"], 4)
        pos = round(scores["pos"], 4)
        neu = round(scores["neu"], 4)
        neg = round(scores["neg"], 4)
    elif HAS_TEXTBLOB:
        polarity = TextBlob(text).sentiment.polarity
        compound = round(polarity, 4)
        pos = round(max(0.0, polarity), 4)
        neu = round(1.0 - abs(polarity), 4)
        neg = round(max(0.0, -polarity), 4)
    else:
        # Simple heuristic fallback
        pos_words = {"good", "great", "excellent", "awesome", "best", "love", "smooth", "helpful", "nice"}
        neg_words = {"bad", "worst", "terrible", "horrible", "poor", "hate", "pathetic", "useless", "crash", "bug"}
        words = set(re.findall(r"\b\w+\b", text.lower()))
        pos_count = len(words.intersection(pos_words))
        neg_count = len(words.intersection(neg_words))
        if pos_count > neg_count:
            compound, pos, neu, neg = 0.5, 0.6, 0.4, 0.0
        elif neg_count > pos_count:
            compound, pos, neu, neg = -0.5, 0.0, 0.4, 0.6
        else:
            compound, pos, neu, neg = 0.0, 0.0, 1.0, 0.0

    # Categorize compound score
    if compound >= 0.05:
        text_sentiment = "Positive"
    elif compound <= -0.05:
        text_sentiment = "Negative"
    else:
        text_sentiment = "Neutral"

    return {
        "sentiment_score": compound,
        "sentiment_pos": pos,
        "sentiment_neu": neu,
        "sentiment_neg": neg,
        "text_sentiment": text_sentiment,
    }


def classify_divergence(star_rating: int, text_sentiment: str) -> tuple[str, bool]:
    """
    Compares explicit star rating against NLP text sentiment.
    Identifies Sarcasm (high stars + negative text) and Inverted ratings (low stars + positive text).
    """
    if star_rating >= 4 and text_sentiment == "Negative":
        return "Sarcasm / Sarcastic High Rating", True
    elif star_rating <= 2 and text_sentiment == "Positive":
        return "Accidental / Inverted Low Rating", True
    elif star_rating >= 4 and text_sentiment == "Positive":
        return "Aligned Positive", False
    elif star_rating <= 2 and text_sentiment == "Negative":
        return "Aligned Negative", False
    else:
        return "Mixed / Neutral", False


# ─────────────────────────────────────────────────────────────────
# 2. Domain-Specific Complaint Taxonomy Classifier
# ─────────────────────────────────────────────────────────────────

CATEGORY_PATTERNS = {
    "Notifications & OTP": [
        r"\botp\b", r"\bcode\b", r"\bverification\b", r"\bverify\b",
        r"\bnotification\b", r"\bnotifications\b", r"\balert\b", r"\balerts\b",
        r"\bsms\b", r"\bno otp\b", r"\botp not (received|coming|sent)\b"
    ],
    "Live Tracking": [
        r"\btracking\b", r"\blive track(ing)?\b", r"\bgps\b", r"\blocation\b",
        r"\bmap\b", r"\bmaps\b", r"\bwhere is (the )?cab\b", r"\bcar location\b",
        r"\bvan location\b", r"\bwrong location\b", r"\bnot moving\b", r"\bfreeze.*map\b",
        r"\bgoogle map\b"
    ],
    "Login & Authentication": [
        r"\blogin\b", r"\blog in\b", r"\blogged out\b", r"\blogs out\b",
        r"\bsign in\b", r"\bsigning in\b", r"\bpassword\b", r"\bauth(entication)?\b",
        r"\bemployee id\b", r"\bcredentials\b", r"\baccount\b", r"\bdomain\b",
        r"\bunable to log(in)?\b"
    ],
    "App Performance & Crashes": [
        r"\bcrash(es|ed|ing)?\b", r"\bfreeze(s|d)?\b", r"\bfreezing\b", r"\bforce close\b",
        r"\bslow\b", r"\blag(gy|ging)?\b", r"\bbug(gy|s)?\b", r"\bblank screen\b",
        r"\bwhite screen\b", r"\bblack screen\b", r"\bloading forever\b", r"\bstuck\b",
        r"\bopen(ing)? again\b", r"\breinstall\b", r"\buninstall\b", r"\bheavy app\b",
        r"\bhands\b", r"\bhang(s|ing)?\b"
    ],
    "Cab Booking Issues": [
        r"\bbook(ing)?\b", r"\bschedule\b", r"\breschedule\b", r"\bcancel(lation|led|ing)?\b",
        r"\ballocat(ion|ed)?\b", r"\broster\b", r"\bshift\b", r"\bseat\b",
        r"\broute\b", r"\brouting\b", r"\bclub(bing|bed)?\b", r"\bdetour\b",
        r"\beta\b", r"\bdelay(ed)?\b", r"\bwaitlist\b", r"\bno cab\b", r"\bcab not (arrived|coming|booked)\b",
        r"\btrip\b", r"\brange\b", r"\bpickup\b", r"\bdrop\b"
    ],
    "Driver Issues": [
        r"\bdriver(s)?\b", r"\brude\b", r"\bbehav(ior|iour)\b", r"\brash\b",
        r"\bdriving\b", r"\brefuse(d)?\b", r"\bcall(ed|ing)?\b", r"\bunreachable\b",
        r"\bdriver cancelled\b", r"\bunprofessional\b", r"\bswitched off\b"
    ],
    "Safety Concerns": [
        r"\bsafety\b", r"\bsafe\b", r"\bunsafe\b", r"\bwomen\b", r"\bwoman\b",
        r"\bnight\b", r"\bpanic\b", r"\bsos\b", r"\bemergency\b", r"\bsecurity\b",
        r"\bescort\b", r"\bguard\b", r"\bthreat\b"
    ],
    "Billing & Payroll": [
        r"\bbill(ing)?\b", r"\bpayroll\b", r"\bsalary\b", r"\bdeduct(ion|ed)?\b",
        r"\bcharge(s|d)?\b", r"\bpayment\b", r"\bmoney\b", r"\ballowance\b",
        r"\breimburse(ment)?\b"
    ],
    "UI & Navigation": [
        r"\bui\b", r"\bux\b", r"\binterface\b", r"\bdark mode\b", r"\bfont\b",
        r"\bbutton\b", r"\blayout\b", r"\bupdate\b", r"\bnew update\b", r"\bnew version\b",
        r"\bcluttered\b", r"\bconfusing\b", r"\bcalendar\b", r"\bmonthly view\b",
        r"\bday view\b", r"\bnavigation\b", r"\bfeature(s)?\b", r"\bdesign\b"
    ],
    "Positive Feedback": [
        r"\bgreat\b", r"\bgood\b", r"\bawesome\b", r"\bexcellent\b", r"\blove\b",
        r"\bsmooth\b", r"\bhelpful\b", r"\bnice\b", r"\bsuper(b)?\b", r"\bfantastic\b",
        r"\bbest\b", r"\bconvenient\b", r"\beasy to use\b", r"\bhappy\b", r"\bthank(s| you)?\b"
    ],
}


def categorize_review(text: str, star_rating: int, text_sentiment: str) -> tuple[str, list]:
    """
    Classifies a review into primary and secondary complaint categories.
    Considers star rating, domain keywords, and sentiment context.
    """
    if not isinstance(text, str) or not text.strip():
        if star_rating >= 4:
            return "Positive Feedback", []
        return "Other", []

    text_lower = text.lower()
    matched_scores = {}

    for category, patterns in CATEGORY_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower))
            score += matches
        if score > 0:
            matched_scores[category] = score

    # Positive reviews with 4 or 5 stars and no strong negative complaint
    if star_rating >= 4 and text_sentiment == "Positive":
        if "Positive Feedback" in matched_scores or not matched_scores:
            return "Positive Feedback", [c for c in matched_scores if c != "Positive Feedback"]

    # If it's a negative or low star rating, deprioritize "Positive Feedback"
    if star_rating <= 2 or text_sentiment == "Negative":
        matched_scores.pop("Positive Feedback", None)

    if not matched_scores:
        if star_rating >= 4:
            return "Positive Feedback", []
        return "Other", []

    # Sort categories by match frequency
    sorted_matches = sorted(matched_scores.items(), key=lambda x: x[1], reverse=True)
    primary_category = sorted_matches[0][0]
    secondary_categories = [cat for cat, _ in sorted_matches[1:]]

    return primary_category, secondary_categories


# ─────────────────────────────────────────────────────────────────
# 3. Engineering Priority Matrix Calculation
# ─────────────────────────────────────────────────────────────────

def calculate_priority_matrix(df: pd.DataFrame) -> list[dict]:
    """
    Computes business and engineering priority scores for each complaint category:
    Priority Score = Frequency Weight * Severity Weight * Community Impact
    """
    categories = [c for c in COMPLAINT_CATEGORIES if c not in ["Positive Feedback", "Other"]]
    total_reviews = len(df)
    total_negative = len(df[df["star_rating"] <= 2]) or 1

    matrix = []

    for cat in categories:
        cat_df = df[df["category"] == cat]
        count = len(cat_df)
        if count == 0:
            continue

        neg_df = cat_df[cat_df["star_rating"] <= 2]
        neg_count = len(neg_df)
        avg_stars = round(cat_df["star_rating"].mean(), 2)
        avg_sentiment = round(cat_df["sentiment_score"].mean(), 3)
        thumbs_up_total = int(cat_df["thumbs_up"].sum())

        # Frequency: share of negative reviews (0 to 100)
        freq_pct = round((neg_count / total_negative) * 100, 1)

        # Severity: lower rating + negative sentiment compound = higher severity (0 to 10 scale)
        # avg_stars is 1-5, so (5 - avg_stars) gives 0-4
        # avg_sentiment is -1 to 1, so (1 - avg_sentiment) gives 0-2
        severity_score = round(((5.0 - avg_stars) * 1.5) + ((1.0 - avg_sentiment) * 2.0), 2)

        # Community impact factor (thumbs up indicates widespread agreement)
        thumbs_factor = 1.0 + min(thumbs_up_total / 50.0, 1.5)

        # Composite Priority Score (0 to 100 scale)
        composite_score = round(min(100.0, (freq_pct * 0.5) + (severity_score * 4.5) * thumbs_factor), 1)

        # Priority Rank
        if composite_score >= 65:
            rank = "P0 - Critical"
        elif composite_score >= 45:
            rank = "P1 - High"
        elif composite_score >= 25:
            rank = "P2 - Medium"
        else:
            rank = "P3 - Low"

        matrix.append({
            "category": cat,
            "total_reviews": count,
            "negative_reviews": neg_count,
            "negative_share_pct": freq_pct,
            "avg_star_rating": avg_stars,
            "avg_sentiment": avg_sentiment,
            "thumbs_up_total": thumbs_up_total,
            "severity_score": severity_score,
            "priority_score": composite_score,
            "priority_rank": rank,
        })

    # Sort descending by priority score
    matrix = sorted(matrix, key=lambda x: x["priority_score"], reverse=True)
    return matrix


# ─────────────────────────────────────────────────────────────────
# 4. Main Stage 3 Execution
# ─────────────────────────────────────────────────────────────────

def run_categorization() -> pd.DataFrame:
    """Executes Stage 3 AI Sentiment Scoring & Complaint Categorization."""
    logger.info("Starting Stage 3: AI Analysis & Categorization...")

    if not os.path.exists(PROCESSED_CSV_PATH):
        raise FileNotFoundError(
            f"Cleaned reviews not found at '{PROCESSED_CSV_PATH}'. "
            "Please run Stage 2 first."
        )

    df = pd.read_csv(PROCESSED_CSV_PATH)
    logger.info(f"Loaded {len(df):,} cleaned reviews for analysis.")

    # 1. Sentiment Scoring
    logger.info("Computing sentiment scores (VADER)...")
    sentiment_results = df["review_text"].apply(score_sentiment)
    sentiment_df = pd.DataFrame(sentiment_results.tolist())
    for col in sentiment_df.columns:
        df[col] = sentiment_df[col]

    # 2. Rating Divergence & Sarcasm Detection
    logger.info("Detecting rating discrepancies & sarcasm...")
    divergence_data = [
        classify_divergence(star, sent)
        for star, sent in zip(df["star_rating"], df["text_sentiment"])
    ]
    df["divergence_type"] = [d[0] for d in divergence_data]
    df["is_divergent"] = [d[1] for d in divergence_data]

    # 3. Categorization
    logger.info("Classifying complaint categories...")
    cat_results = [
        categorize_review(text, star, sent)
        for text, star, sent in zip(df["review_text"], df["star_rating"], df["text_sentiment"])
    ]
    df["category"] = [c[0] for c in cat_results]
    df["secondary_categories"] = [json.dumps(c[1]) for c in cat_results]

    # 4. Priority Matrix
    logger.info("Computing Engineering Priority Matrix...")
    priority_matrix = calculate_priority_matrix(df)

    # 5. Save Analyzed Datasets
    os.makedirs(os.path.dirname(ANALYZED_CSV_PATH), exist_ok=True)
    df.to_csv(ANALYZED_CSV_PATH, index=False, encoding="utf-8")
    df.to_json(ANALYZED_JSON_PATH, orient="records", date_format="iso", indent=2)
    logger.info(f"Saved analyzed CSV: {ANALYZED_CSV_PATH}")
    logger.info(f"Saved analyzed JSON: {ANALYZED_JSON_PATH}")

    # 6. Generate Summary Audit
    sentiment_dist = df["text_sentiment"].value_counts().to_dict()
    category_dist = df["category"].value_counts().to_dict()
    divergent_count = int(df["is_divergent"].sum())
    sarcasm_count = int((df["divergence_type"] == "Sarcasm / Sarcastic High Rating").sum())
    inverted_count = int((df["divergence_type"] == "Accidental / Inverted Low Rating").sum())

    summary = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_analyzed": len(df),
        "sentiment_distribution": {str(k): int(v) for k, v in sentiment_dist.items()},
        "category_distribution": {str(k): int(v) for k, v in category_dist.items()},
        "divergence_metrics": {
            "total_divergent": divergent_count,
            "sarcasm_high_rating_negative_text": sarcasm_count,
            "accidental_low_rating_positive_text": inverted_count,
        },
        "priority_matrix": priority_matrix,
    }

    with open(ANALYSIS_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved analysis summary: {ANALYSIS_SUMMARY_PATH}")

    # Print Summary Report
    print_summary_report(summary, priority_matrix)

    return df


def print_summary_report(summary: dict, priority_matrix: list[dict]):
    """Prints a structured summary of Stage 3 analysis."""
    print("\n" + "=" * 60)
    print("  Stage 3: AI Sentiment Analysis & Categorization Summary")
    print("=" * 60)
    print(f"  Total Reviews Analyzed  : {summary['total_analyzed']:,}")
    
    print("\n  Text Sentiment Distribution (NLP):")
    for sent, count in summary["sentiment_distribution"].items():
        pct = (count / summary['total_analyzed']) * 100
        print(f"    - {sent:<10} : {count:,} ({pct:.1f}%)")

    div = summary["divergence_metrics"]
    print(f"\n  Rating Divergence & Sarcasm:")
    print(f"    - Sarcastic High Ratings (4-5★ but Negative Text) : {div['sarcasm_high_rating_negative_text']:,}")
    print(f"    - Accidental Low Ratings (1-2★ but Positive Text) : {div['accidental_low_rating_positive_text']:,}")

    print("\n  Engineering Priority Matrix (Ranked by Impact):")
    print("  " + "-" * 56)
    print(f"  {'Rank':<14} {'Category':<24} {'Neg %':<8} {'Score':<6}")
    print("  " + "-" * 56)
    for row in priority_matrix[:6]:
        print(f"  {row['priority_rank']:<14} {row['category']:<24} {row['negative_share_pct']:>5.1f}%  {row['priority_score']:>5.1f}")
    print("  " + "-" * 56)
    print("=" * 60)
    print("  [SUCCESS] Stage 3 Complete! Output ready in data/processed/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_categorization()
