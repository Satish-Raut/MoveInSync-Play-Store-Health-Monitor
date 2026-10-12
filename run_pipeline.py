"""
run_pipeline.py
───────────────
One-command runner for the full pipeline:
  Stage 1: Scrape Play Store reviews
  Stage 2: Clean & preprocess data
  Stage 3: Analyze & categorize (AI)

Usage:
    python run_pipeline.py              # Run all stages
    python run_pipeline.py --stage 1   # Run only Stage 1 (scrape)
    python run_pipeline.py --stage 2   # Run only Stage 2 (clean)
    python run_pipeline.py --stage 3   # Run only Stage 3 (analyze)
"""

import sys
import time
import argparse
from datetime import datetime


def print_banner(text: str):
    width = 54
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def run_stage_1():
    print_banner("Stage 1: Scraping Play Store Reviews")
    from scraper.playstore_scraper import (
        ensure_dirs, fetch_app_info, scrape_all_reviews, save_reviews, validate_data
    )
    ensure_dirs()
    app_info = fetch_app_info()
    all_reviews = scrape_all_reviews()
    if not all_reviews:
        print("ERROR: No reviews collected. Aborting.")
        sys.exit(1)
    df = save_reviews(all_reviews, app_info)
    validate_data(df)
    return df


def run_stage_2():
    print_banner("Stage 2: Cleaning & Preprocessing")
    try:
        from analyzer.cleaner import run_cleaning
        run_cleaning()
    except ImportError:
        print("  [INFO] cleaner.py not yet implemented — skipping Stage 2.")


def run_stage_3():
    print_banner("Stage 3: AI Analysis & Categorization")
    try:
        from analyzer.categorizer import run_categorization
        run_categorization()
    except ImportError:
        print("  [INFO] categorizer.py not yet implemented — skipping Stage 3.")


def main():
    parser = argparse.ArgumentParser(description="MoveInSync Play Store Health Monitor Pipeline")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        help="Run a specific stage only (1=scrape, 2=clean, 3=analyze). Default: all stages."
    )
    args = parser.parse_args()

    start = time.time()
    print_banner(f"MoveInSync Play Store Health Monitor")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.stage == 1:
        run_stage_1()
    elif args.stage == 2:
        run_stage_2()
    elif args.stage == 3:
        run_stage_3()
    else:
        # Run all stages in sequence
        run_stage_1()
        run_stage_2()
        run_stage_3()

    elapsed = time.time() - start
    print_banner(f"Pipeline Complete in {elapsed:.1f}s")
    print("  Next step: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
