"""Churn scoring, driver explanation, risk segments and retention plays for Spotify."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import bar, kpi, money, num, pct, rec, table

META = json.loads(Path(__file__).with_name("meta.json").read_text())
PARAMS = [
    {"name": "threshold", "label": "At-risk threshold", "min": 0.30, "max": 0.80, "step": 0.05, "default": 0.35},
    {"name": "save_rate", "label": "Expected save rate of plays (%)", "min": 10, "max": 50, "step": 5, "default": 25},
]
SCALE = 2000  # 3,000-user sample scaled to a 6M-subscriber market
FEATURES = {"weekly_hours": "Weekly listening (hrs)", "listening_change_90d": "Listening change (90d)", "skip_rate": "Skip rate",
            "days_since_login": "Days since last login", "failed_payments": "Failed payments", "support_tickets": "Support tickets",
            "hit_by_price_increase": "Hit by price increase", "tenure_months": "Tenure (months)", "new_music_share": "New-music share", "is_family": "Family plan"}
SEGMENT = {"weekly_hours": "Disengaged listeners", "listening_change_90d": "Disengaged listeners", "days_since_login": "Disengaged listeners",
           "skip_rate": "Content mismatch", "new_music_share": "Content mismatch", "failed_payments": "Payment friction",
           "support_tickets": "Service issues", "hit_by_price_increase": "Price-sensitive", "is_family": "Price-sensitive", "tenure_months": "New and unsettled"}
PLAY = {"Disengaged listeners": "Re-engagement: a fresh Daily Mix and a push for their most played artist",
        "Content mismatch": "Discovery boost: retuned Discover Weekly and a 'new for you' playlist",
        "Payment friction": "Payment rescue: retry on payday, card update reminder, 7-day grace period",
        "Service issues": "Support recovery: priority callback and a free month for open tickets",
        "Price-sensitive": "Plan-switch offer: Duo or annual plan at a lower monthly cost",
        "New and unsettled": "Onboarding nudge: build their first playlists in the first 30 days"}
THEMES = {"too expensive": "Price", "price went up": "Price", "ads came back": "Price", "app keeps crashing": "App quality",
          "cannot find new music": "Content", "same songs every day": "Content", "payment failed again": "Billing", "support never replied": "Support"}


def run(data, threshold=0.35, save_rate=25):
    df = data["subscribers"].copy()
    df["is_family"] = (df["plan"] == "Family").astype(int)
    cols = list(FEATURES)
    X, y = df[cols].to_numpy(float), df["churned"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    auc = roc_auc_score(yte, m.predict_proba(sc.transform(Xte))[:, 1])
    Z = sc.transform(X)
    contrib = Z * m.coef_[0]                       # per-user feature contributions (log-odds)
    df["risk"] = m.predict_proba(Z)[:, 1]
    imp = pd.DataFrame({"driver": [FEATURES[c] for c in cols], "importance": np.abs(contrib).mean(0).round(3),
                        "direction": np.where(m.coef_[0] > 0, "raises churn", "lowers churn")}).sort_values("importance", ascending=False)

    at = df[(df.risk >= threshold) & (df.churned == 0)].copy()
    top_idx = contrib[at.index].argmax(1)
    at["top_driver"] = [FEATURES[cols[j]] for j in top_idx]
    at["segment"] = [SEGMENT[cols[j]] for j in top_idx]
    hazard = np.minimum(0.95, at["risk"] * 0.6)
    at["months_to_cancel"] = np.maximum(0.5, np.log(0.5) / np.log(1 - hazard))
    seg = at.groupby("segment").agg(users=("user_id", "count"), avg_risk=("risk", "mean"), months_to_cancel=("months_to_cancel", "mean"), mrr=("monthly_price", "sum")).reset_index()
    seg["users"] *= SCALE
    seg["annual_revenue_at_risk"] = (seg["mrr"] * 12 * SCALE).round(0)
    seg["play"] = seg["segment"].map(PLAY)
    seg = seg.sort_values("annual_revenue_at_risk", ascending=False).drop(columns="mrr")
    rev_risk = seg["annual_revenue_at_risk"].sum()
    saved = rev_risk * save_rate / 100

    ch = df[df.churned == 1]
    themes = ch["last_review"].map(THEMES).value_counts(normalize=True).rename_axis("theme").reset_index(name="share_of_churners")
    hist = pd.DataFrame({"risk_band": [f"{b * 10}-{b * 10 + 10}%" for b in range(10)],
                         "subscribers": [int(((df.risk >= b / 10) & (df.risk < (b + 1) / 10)).sum()) for b in range(10)]})
    top = at.assign(value=at.risk * at.monthly_price).sort_values("value", ascending=False).head(10)
    top_tbl = top[["user_id", "plan", "risk", "top_driver", "segment"]].assign(risk=lambda d: d.risk.round(2))
    top_tbl["action"] = top_tbl["segment"].map(lambda s: PLAY[s].split(":")[0])

    recs = [rec(f"{r.segment}: {r.play.split(':')[0]}", f"{num(r.users)} users at {pct(r.avg_risk, 0)} average risk, about {r.months_to_cancel:.1f} months from cancelling. {r.play.split(': ')[1]}.", money(r.annual_revenue_at_risk * save_rate / 100) + "/yr saved") for r in seg.head(4).itertuples()]
    safe = int(((df.risk < 0.15) & (df.churned == 0)).sum())
    recs.append(rec("Stop discounting users the model calls safe", f"{num(safe * SCALE)} users score under 15%. Take them out of win-back offers and keep the budget for the segments above.", "Lower CAC"))
    return {
        "steps": [f"{len(df):,} subscribers, {pct(y.mean())} churned in 90 days", f"{len(cols)} behaviour features per subscriber",
                  f"Test AUC {auc:.3f} on {len(yte):,} held-out users", f"Top driver: {imp.iloc[0].driver}",
                  f"Median time to cancel {at.months_to_cancel.median():.1f} months", f"{len(seg)} segments, {len(at)} of the sample at risk",
                  f"Top complaint theme among churners: {themes.iloc[0].theme}", f"{len(seg)} plays assigned"],
        "kpis": [kpi("90-day churn", pct(y.mean()), f"{int(y.sum())} of {len(df):,} subscribers"),
                 kpi("Active users at risk", num(len(at) * SCALE), f"Score >= {pct(threshold, 0)}, scaled to 6M base"),
                 kpi("Revenue at risk", money(rev_risk), "Annual subscription value"),
                 kpi("Revenue saved", money(saved), f"At {save_rate}% play success"),
                 kpi("Model AUC", f"{auc:.2f}", "1.0 = perfect, 0.5 = coin flip")],
        "charts": [bar("What drives churn", imp, "driver", "importance", "Average size of each feature's push on a user's churn score."),
                   bar("Risk score distribution", hist, "risk_band", "subscribers", horizontal=False),
                   bar("At-risk segments (users)", seg, "segment", "users"),
                   bar("What churners said", themes, "theme", "share_of_churners")],
        "tables": [table("Retention plan by segment", seg.round(2)), table("Highest-value users to save first", top_tbl, "Ranked by churn probability x monthly price.")],
        "recs": recs,
    }


if __name__ == "__main__":
    from src.data_generator import load_or_generate
    from src.report import write_reports

    out = run(load_or_generate(), **{p["name"]: p["default"] for p in PARAMS})
    for k in out["kpis"]:
        print(f"{k['label']}: {k['value']}  {k['note']}")
    print("Reports written to", write_reports(out, META, Path(__file__).resolve().parent.parent / "reports"))
