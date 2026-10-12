# MoveInSync — Play Store Health Monitor
> **A repeatable tool to track, analyze, and improve MoveInSync's Play Store ratings.**

---

## What This Does

This tool automatically:
1. **Scrapes** Play Store reviews for the MoveInSync employee app
2. **Analyzes** complaints using AI (sentiment + categorization)
3. **Visualizes** insights in an interactive Streamlit dashboard
4. **Exports** a PDF report with prioritized recommendations
5. **Runs automatically** every week via GitHub Actions

---

## Quick Start (Local Setup)

### 1. Clone the repo
```bash
https://github.com/Satish-Raut/MoveInSync-Play-Store-Health-Monitor.git
cd playstore-health-monitor
```

### 2. Create & activate virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
# Copy the template
cp .env.example .env

# Edit .env and add your API keys
# GEMINI_API_KEY=your_key_here
```

### 5. Run the full pipeline (scrape + analyze)
```bash
python run_pipeline.py
```

### 6. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

Open your browser at: **http://localhost:8501**

---

## Project Structure

```
playstore-health-monitor/
├── data/
│   ├── raw/                  ← Scraped reviews (JSON + CSV)
│   └── processed/            ← Cleaned + analyzed data
├── scraper/
│   └── playstore_scraper.py  ← Fetches reviews from Play Store
├── analyzer/
│   ├── cleaner.py            ← Text cleaning & preprocessing
│   ├── categorizer.py        ← AI complaint categorization
│   └── sentiment.py          ← Sentiment scoring
├── report/
│   └── report_generator.py   ← PDF report export
├── app/
│   └── streamlit_app.py      ← Interactive dashboard
├── .github/workflows/
│   └── weekly_report.yml     ← Auto-runs every Monday
├── config.py                 ← All settings in one place
├── run_pipeline.py           ← One-command pipeline runner
├── requirements.txt
└── .env.example              ← API key template
```

---

## Running Individual Stages

```bash
# Stage 1: Scrape reviews only
python scraper/playstore_scraper.py

# Stage 2: Clean data only
python analyzer/cleaner.py

# Stage 3: Analyze + categorize
python analyzer/categorizer.py

# Stage 4: Launch dashboard
streamlit run app/streamlit_app.py

# All stages at once
python run_pipeline.py
```

---

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path**: `app/streamlit_app.py`
5. Add secrets in the Streamlit dashboard:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
6. Click **Deploy** — your app is live at:
   `https://your-app-name.streamlit.app`

---

## Automation (GitHub Actions)

The pipeline runs **every Monday at 9:00 AM IST** automatically.

To set up:
1. Go to your GitHub repo → **Settings → Secrets**
2. Add `GEMINI_API_KEY` as a repository secret
3. The workflow in `.github/workflows/weekly_report.yml` handles the rest

---

## Tools Used

| Tool | Purpose |
|------|---------|
| `google-play-scraper` | Fetching Play Store reviews |
| `pandas` | Data manipulation |
| `VADER / TextBlob` | Sentiment analysis |
| `Gemini / OpenAI API` | AI complaint categorization |
| `Streamlit` | Interactive dashboard |
| `Plotly` | Charts and visualizations |
| `fpdf2` | PDF report generation |
| `GitHub Actions` | Weekly automation |

---

## Assumptions

- Reviews scraped are public and available via the Play Store API
- English reviews only (primary language filter: `en`, country: `in`)
- AI categorization uses zero-shot classification — no training data required
- Priority scoring formula: `Priority = (Frequency × Rating Weight) + (Sentiment Severity × 2)`

---

## Stage Progress

| Stage | Status |
|-------|--------|
| Stage 1: Data Collection | ✅ Done |
| Stage 2: Cleaning | ⏳ Pending |
| Stage 3: AI Analysis | ⏳ Pending |
| Stage 4: Dashboard | ⏳ Pending |
| Stage 5: PDF Export | ⏳ Pending |
| Stage 6: Automation | ⏳ Pending |
| Stage 7: Submission | ⏳ Pending |
