"""NIL vs. production evaluation.

Two complementary methods, both anchored to the *position market*:

1. Market percentile method (transparent)
   - Each position has a pay curve (low / typical / elite) from market data.
   - A player's production percentile maps to a "market-fair NIL" on that curve.
   - His actual NIL maps back to a "pay percentile" on the same curve.
   - Value gap = production percentile - pay percentile; ratio = actual / fair.

2. Expected-NIL model (statistical)
   - Ridge regression of log(NIL) on production, the position's market level,
     Power-4 membership and roster talent (a market/brand proxy), fit on all
     players with known NIL.
   - Residual = how much more/less a player earns than similar players do.
     A positive residual is "brand premium" not explained by production.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
G5_MARKET_FACTOR = 0.5  # market ranges are Power-4 figures; G5 programs pay roughly half
PCT_ANCHORS = np.array([0, 50, 80, 95, 100])

VERDICTS = [  # (min ratio, label)
    (1.75, "Overpaid"),
    (1.25, "Above market"),
    (0.80, "Fair value"),
    (0.55, "Below market"),
    (0.0, "Bargain"),
]


def verdict(ratio: float) -> str:
    if ratio is None or np.isnan(ratio):
        return "n/a"
    for threshold, label in VERDICTS:
        if ratio >= threshold:
            return label
    return "Bargain"


# ---------- Position market ----------

def load_market(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "position_market.csv")


def _anchors(row: pd.Series) -> np.ndarray:
    vals = np.array([row.low_usd, row.typical_low_usd, row.typical_high_usd,
                     max(row.elite_low_usd, row.typical_high_usd), row.elite_high_usd], dtype=float)
    return np.maximum.accumulate(vals)  # enforce a monotonic curve


def market_fair_value(pct: float, market_position: str, power4: bool, market: pd.DataFrame) -> float:
    """Production percentile -> market-fair NIL (log-linear interpolation on the pay curve)."""
    row = market[market["market_position"] == market_position]
    if row.empty or pd.isna(pct):
        return np.nan
    a = _anchors(row.iloc[0])
    val = float(np.exp(np.interp(pct, PCT_ANCHORS, np.log(a))))
    return val if power4 else val * G5_MARKET_FACTOR


def pay_percentile(nil: float, market_position: str, power4: bool, market: pd.DataFrame) -> float:
    """Actual NIL -> where it sits on the position's pay curve (0-100, >100 = above the elite ceiling)."""
    row = market[market["market_position"] == market_position]
    if row.empty or pd.isna(nil) or nil <= 0:
        return np.nan
    a = np.log(_anchors(row.iloc[0]) * (1 if power4 else G5_MARKET_FACTOR))
    x = np.log(nil)
    if x > a[-1]:  # extrapolate above the ceiling using the top segment's slope
        slope = (PCT_ANCHORS[-1] - PCT_ANCHORS[-2]) / max(a[-1] - a[-2], 1e-9)
        return float(min(100 + (x - a[-1]) * slope, 150))
    a_unique, idx = np.unique(a, return_index=True)
    return float(np.interp(x, a_unique, PCT_ANCHORS[idx]))


def market_midpoint(market_position: str, market: pd.DataFrame) -> float:
    row = market[market["market_position"] == market_position]
    if row.empty:
        return np.nan
    r = row.iloc[0]
    return float(np.sqrt(r.typical_low_usd * r.typical_high_usd))


# ---------- NIL data + matching ----------

def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_nil(path: Path | None = None) -> pd.DataFrame:
    nil = pd.read_csv(path or DATA_DIR / "nil_values.csv")
    nil["team"] = nil["team"].fillna("")
    return nil


def match_nil(players: pd.DataFrame, nil: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach NIL values to scored players. Returns (players_with_nil, unmatched_nil_rows)."""
    p = players.copy()
    p["_nkey"] = p["player"].map(norm_name)
    p["_rkey"] = p["roster_name"].map(norm_name) if "roster_name" in p else p["_nkey"]
    p["nil_value_usd"] = np.nan
    p["nil_source"] = None
    p["nil_as_of"] = None
    p["nil_listed_position"] = None
    unmatched = []
    for _, row in nil.iterrows():
        key = norm_name(row["player"])
        cands = p[(p["_nkey"] == key) | (p["_rkey"] == key)]
        if len(cands) > 1 and row["team"]:
            same_team = cands[cands["team"].str.lower() == row["team"].lower()]
            cands = same_team if not same_team.empty else cands
        if len(cands) > 1:  # still ambiguous: prefer the most productive/visible record
            cands = cands.sort_values("production_score", ascending=False).head(1)
        if cands.empty:
            unmatched.append(row)
            continue
        i = cands.index[0]
        p.loc[i, ["nil_value_usd", "nil_source", "nil_as_of", "nil_listed_position"]] = [
            row["nil_value_usd"], row["source"], row["as_of"], row["position"]]
    p = p.drop(columns=["_nkey", "_rkey"])
    return p, pd.DataFrame(unmatched)


# ---------- Evaluation ----------

def fit_expected_nil(df: pd.DataFrame, market: pd.DataFrame, ridge: float = 1.0):
    """Fit log(NIL) ~ production + position market + power4 + talent. Returns (predict_fn, coefs)."""
    known = df[df["nil_value_usd"].notna() & (df["nil_value_usd"] > 0)].copy()

    def features(x: pd.DataFrame) -> np.ndarray:
        talent = x["talent"].fillna(x["talent"].median() if x["talent"].notna().any() else 500)
        return np.column_stack([
            np.ones(len(x)),
            x["production_score"].fillna(0).values / 100,
            np.log(x["market_position"].map(lambda m: market_midpoint(m, market)).astype(float).values),
            x["power4"].astype(float).values,
            (talent.values - 650) / 200,
        ])

    names = ["intercept", "production (0-1)", "log position market", "power 4", "roster talent (z-ish)"]
    if len(known) < 8:
        return None, None
    X, y = features(known), np.log(known["nil_value_usd"].values)
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0)
    sd[sd == 0] = 1
    Xs = X.copy()
    Xs[:, 1:] = (X[:, 1:] - mu) / sd
    pen = ridge * np.eye(X.shape[1])
    pen[0, 0] = 0
    beta_s = np.linalg.solve(Xs.T @ Xs + pen, Xs.T @ y)
    beta = beta_s.copy()  # back to original feature units
    beta[1:] = beta_s[1:] / sd
    beta[0] = beta_s[0] - np.sum(beta_s[1:] * mu / sd)
    resid = y - Xs @ beta_s
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan

    def predict(x: pd.DataFrame) -> np.ndarray:
        return np.exp(features(x) @ beta)

    coefs = pd.DataFrame({"feature": names, "coefficient": beta})
    coefs.attrs["r2"] = r2
    coefs.attrs["n"] = len(known)
    return predict, coefs


def evaluate(players: pd.DataFrame, nil: pd.DataFrame, market: pd.DataFrame):
    """Full pipeline on scored players. Returns (evaluated_df, unmatched_nil, model_coefs)."""
    df, unmatched = match_nil(players, nil)
    # The NIL listing's position (e.g. EDGE vs CFBD's generic DL) is the better market bucket.
    listed = df["nil_listed_position"].where(df["nil_listed_position"].isin(market["market_position"]))
    df["market_position"] = listed.fillna(df["market_position"])

    df["market_fair_nil"] = [market_fair_value(s, m, p4, market) for s, m, p4 in
                             zip(df["production_score"], df["market_position"], df["power4"])]
    df["pay_percentile"] = [pay_percentile(n, m, p4, market) for n, m, p4 in
                            zip(df["nil_value_usd"], df["market_position"], df["power4"])]
    df["value_gap_pts"] = df["production_score"] - df["pay_percentile"]
    df["market_ratio"] = df["nil_value_usd"] / df["market_fair_nil"]
    df["market_verdict"] = df["market_ratio"].map(verdict)

    predict, coefs = fit_expected_nil(df, market)
    if predict is not None:
        df["expected_nil"] = predict(df)
        df["model_ratio"] = df["nil_value_usd"] / df["expected_nil"]
        df["model_verdict"] = df["model_ratio"].map(verdict)
        df["brand_premium_usd"] = df["nil_value_usd"] - df["expected_nil"]
    else:
        df["expected_nil"] = np.nan
        df["model_ratio"] = np.nan
        df["model_verdict"] = "n/a"
        df["brand_premium_usd"] = np.nan
    return df, unmatched, coefs
