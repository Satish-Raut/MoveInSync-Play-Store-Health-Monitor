# MoveInSync Play Store Health Monitor — Configuration

# Target App Details
# Actual URL: https://play.google.com/store/apps/details?id=com.moveinsync.ets&hl=en_IN

APP_ID = "com.moveinsync.ets"
APP_NAME = "MoveInSync For Employees"
PLAY_STORE_URL = f"https://play.google.com/store/apps/details?id={APP_ID}"

# Scraper Settings
REVIEWS_PER_BATCH = 200          # Reviews fetched per API call
MAX_REVIEWS_TOTAL = 2000         # Hard cap on total reviews to scrape
SCRAPE_LANGUAGE = "en"           # Language filter for reviews
SCRAPE_COUNTRY = "in"            # Country: India (primary market)
SCRAPE_SORT = "newest"           # Sort: "newest" | "most_relevant" | "rating"

# Star ratings to scrape (scrape each separately to ensure coverage of 1-star reviews)
SCRAPE_RATINGS = [1, 2, 3, 4, 5]

# ENV variables
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Complaint Categories
COMPLAINT_CATEGORIES = [
    "Cab Booking Issues",
    "Live Tracking",
    "Notifications & OTP",
    "App Performance & Crashes",
    "Driver Issues",
    "Safety Concerns",
    "Billing & Payroll",
    "Login & Authentication",
    "UI & Navigation",
    "Positive Feedback",
    "Other",
]
