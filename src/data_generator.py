"""Sample data for Spotify: 3,000 Premium subscribers with 90 days of behaviour."""
from pathlib import Path

import numpy as np
import pandas as pd

PLANS = ["Individual", "Duo", "Family", "Student"]
PRICE = {"Individual": 11.99, "Duo": 16.99, "Family": 19.99, "Student": 5.99}
NEG = ["too expensive", "price went up", "ads came back", "app keeps crashing", "cannot find new music",
       "same songs every day", "payment failed again", "support never replied"]
POS = ["love my daily mix", "great podcasts", "worth the price", "discover weekly is perfect"]


def generate(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = 3000
    plan = rng.choice(PLANS, n, p=np.array([5, 1.5, 2.5, 1.5]) / 10.5)
    tenure = rng.integers(1, 61, n)
    hrs = np.clip(rng.normal(14, 6, n), 0.3, None)
    chg = np.clip(rng.normal(-0.05, 0.3, n), -0.9, 1.2)
    skip = np.clip(rng.normal(0.25, 0.1, n), 0.02, 0.8)
    idle = np.round(np.abs(rng.normal(0, 6, n)) + np.where(chg < -0.4, rng.uniform(4, 20, n), 0)).astype(int)
    fail = np.where(rng.random(n) < 0.08, rng.integers(1, 4, n), 0)
    tickets = np.where(rng.random(n) < 0.15, rng.integers(1, 5, n), 0)
    price_hit = ((plan != "Student") & (rng.random(n) < 0.35)).astype(int)
    new_music = np.clip(rng.normal(0.3, 0.12, n), 0, 0.9)
    z = (-2.6 - 2.2 * chg + 3.2 * (skip - 0.25) + 0.11 * idle + 0.9 * fail + 0.38 * tickets + 1.1 * price_hit
         - 0.018 * tenure - 0.045 * hrs - 2.2 * (new_music - 0.3) - 0.5 * (plan == "Family"))
    p = 1 / (1 + np.exp(-z))
    churn = (rng.random(n) < p).astype(int)
    neg = rng.random(n) < 1 / (1 + np.exp(-(z + 1)))
    review = np.where(neg, rng.choice(NEG, n), rng.choice(POS, n))
    df = pd.DataFrame({
        "user_id": [f"U{10000 + i}" for i in range(n)], "plan": plan, "monthly_price": [PRICE[x] for x in plan],
        "tenure_months": tenure, "weekly_hours": hrs.round(2), "listening_change_90d": chg.round(3), "skip_rate": skip.round(3),
        "days_since_login": idle, "failed_payments": fail, "support_tickets": tickets, "hit_by_price_increase": price_hit,
        "new_music_share": new_music.round(3), "last_review": review, "churned": churn,
    })
    return {"subscribers": df}


# ---------------------------------------------------------------------------
# Save and load helpers
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def save(data: dict, folder: Path = DATA_DIR) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(folder / f"{name}.csv", index=False)


def load_or_generate(folder: Path = DATA_DIR, seed: int = 42) -> dict:
    """Read the CSVs in data/ if they exist, otherwise generate fresh sample data."""
    folder = Path(folder)
    names = list(generate(seed).keys())
    if all((folder / f"{n}.csv").exists() for n in names):
        return {n: pd.read_csv(folder / f"{n}.csv") for n in names}
    return generate(seed)


if __name__ == "__main__":
    d = generate(42)
    save(d)
    for n, df in d.items():
        print(f"data/{n}.csv  {len(df):,} rows x {df.shape[1]} columns")
