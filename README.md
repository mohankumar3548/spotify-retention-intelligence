# Spotify Churn Radar

AI solution 02 of 14 · Marketing · built for **Spotify**

Prepared by **K Mohan Kumar**.

**Live website:** https://mohankumar3548.github.io/spotify-retention-intelligence/

## Problem statement

A subscription business is experiencing increasing customer churn despite continued marketing expenditure. Develop an AI system that identifies behavioral patterns associated with churn, segments at-risk customers, determines potential causes, and generates personalized retention strategies.

## Why Spotify

Spotify runs on monthly subscriptions and competes with Apple Music and YouTube Music for the same listeners. This tool scores every subscriber for churn risk, explains why, and assigns a retention play.

## Project structure

```
spotify-retention-intelligence/
├── index.html                 Website entry point (served by GitHub Pages)
├── frontend/
│   ├── css/style.css          Layout, typography, light and dark themes
│   └── js/
│       ├── engine.js          Maths, models, charts, page layout and AI analyst panel
│       └── model.js           Sample data and the algorithm for this problem (browser version)
├── app.py                     Streamlit dashboard (Python version of the website)
├── src/
│   ├── data_generator.py      Builds the sample dataset and saves it to data/
│   ├── model.py               The algorithm in Python (pandas, NumPy, scikit-learn)
│   ├── report.py              Writes results to reports/
│   ├── utils.py               Formatting and forecasting helpers
│   └── meta.json              Problem statement, company and algorithm steps
├── data/                      Sample data as CSV
├── reports/                   Results: results.json, summary.md and CSV tables
├── tests/test_model.py        pytest tests for the data and the model
├── requirements.txt
└── pytest.ini
```

## Algorithm

1. **Collect subscriber data**: Listening, skips, logins, billing, plan, support tickets, reviews
2. **Engineer behaviour features**: Change in weekly listening, days since login, failed payments
3. **Train churn classifier**: Logistic regression, 70/30 split, class-weighted
4. **Explain each user**: Per-user feature contributions (SHAP-style) and global drivers
5. **Time to churn**: Monthly hazard turned into expected months until cancel
6. **Segment at-risk users**: Group by each user's top churn driver
7. **Text signals**: Keyword sentiment on reviews and tickets
8. **Retention plays**: Rules + AI analyst pick an action per segment

The same algorithm is implemented twice: in JavaScript (`frontend/js/model.js`) so the website runs in any browser with no server, and in Python (`src/model.py`) for analysis, the Streamlit dashboard and the tests.

## How to run

**Website:** open the live link above, or open `index.html` in a browser.

**Python:**

```bash
pip install -r requirements.txt
python -m src.data_generator     # create the sample data in data/
python -m src.model              # run the algorithm and write reports/
streamlit run app.py             # open the interactive dashboard
pytest                           # run the tests
```

## Data

3,000 Premium subscribers with 90 days of listening, billing and support history, scaled 2,000x to a 6M-subscriber market (sample data). Files: `data/subscribers.csv`. All figures are illustrative and are not Spotify's actual numbers.

## Results (default settings, Python run)

| Measure | Value | Note |
|---|---|---|
| 90-day churn | 14.0% | 420 of 3,000 subscribers |
| Active users at risk | 334,000 | Score >= 35%, scaled to 6M base |
| Revenue at risk | $52.7M | Annual subscription value |
| Revenue saved | $13.2M | At 25% play success |
| Model AUC | 0.86 | 1.0 = perfect, 0.5 = coin flip |

### Recommended actions

1. **Disengaged listeners: Re-engagement.** 224,000 users at 50% average risk, about 2.1 months from cancelling. a fresh Daily Mix and a push for their most played artist.
2. **Payment friction: Payment rescue.** 80,000 users at 50% average risk, about 2.1 months from cancelling. retry on payday, card update reminder, 7-day grace period.
3. **Price-sensitive: Plan-switch offer.** 24,000 users at 40% average risk, about 2.6 months from cancelling. Duo or annual plan at a lower monthly cost.
4. **Service issues: Support recovery.** 6,000 users at 43% average risk, about 2.4 months from cancelling. priority callback and a free month for open tickets.
5. **Stop discounting users the model calls safe.** 4,128,000 users score under 15%. Take them out of win-back offers and keep the budget for the segments above.

Full output: [`reports/summary.md`](reports/summary.md) and [`reports/results.json`](reports/results.json).

## AI analyst panel

The website includes an "Ask the AI analyst" panel that answers questions about the results through Claude. It works when the page is opened as a Claude artifact on claude.ai. On GitHub Pages the models, charts and controls run in full and the panel shows that it is unavailable.
