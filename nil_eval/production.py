"""Position-specific production scoring.

For each scoring group we compute a handful of metrics, z-score them among
"qualified" players (enough volume to judge), take a weighted sum, add a
strength-of-schedule adjustment, and convert to a 0-100 percentile within the
position group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# metric name -> (weight, higher_is_better)
WEIGHTS: dict[str, dict[str, tuple[float, bool]]] = {
    "QB": {
        "ppa_total": (0.30, True),      # total expected points added (volume x efficiency)
        "ppa_pass_avg": (0.25, True),   # EPA per dropback
        "pass_ypa": (0.10, True),
        "td_rate": (0.10, True),
        "int_rate": (0.10, False),
        "rush_ypg": (0.10, True),
        "pass_ypg": (0.05, True),
    },
    "RB": {
        "ppa_total": (0.25, True),
        "ppa_rush_avg": (0.20, True),
        "rush_ypg": (0.20, True),
        "rush_ypc": (0.10, True),
        "scrim_td_pg": (0.10, True),
        "rec_ypg": (0.10, True),
        "usage_rush": (0.05, True),
    },
    "WR": {
        "ppa_pass_total": (0.25, True),
        "rec_ypg": (0.25, True),
        "usage_pass": (0.15, True),     # target-share proxy
        "ppa_pass_avg": (0.10, True),
        "rec_ypr": (0.10, True),
        "rec_td_pg": (0.15, True),
    },
    "TE": {
        "ppa_pass_total": (0.25, True),
        "rec_ypg": (0.25, True),
        "usage_pass": (0.15, True),
        "ppa_pass_avg": (0.10, True),
        "rec_ypr": (0.10, True),
        "rec_td_pg": (0.15, True),
    },
    "EDGE": {
        "sacks_pg": (0.35, True),
        "tfl_pg": (0.25, True),
        "hurries_pg": (0.25, True),
        "tackles_pg": (0.10, True),
        "pd_pg": (0.05, True),
    },
    "DL": {
        "tfl_pg": (0.30, True),
        "sacks_pg": (0.25, True),
        "hurries_pg": (0.20, True),
        "tackles_pg": (0.20, True),
        "pd_pg": (0.05, True),
    },
    "LB": {
        "tackles_pg": (0.30, True),
        "tfl_pg": (0.25, True),
        "sacks_pg": (0.15, True),
        "pd_pg": (0.10, True),
        "ints_pg": (0.10, True),
        "hurries_pg": (0.10, True),
    },
    "DB": {
        "pd_pg": (0.35, True),
        "ints_pg": (0.30, True),
        "tackles_pg": (0.20, True),
        "tfl_pg": (0.10, True),
        "def_td_pg": (0.05, True),
    },
    # No individual OL stats exist publicly; we use team run-blocking/pass-pro proxies.
    "OL": {
        "ol_line_yards": (0.35, True),
        "ol_stuff_rate": (0.25, False),
        "ol_power_success": (0.15, True),
        "ol_pd_success": (0.15, True),
        "ol_rush_ppa": (0.10, True),
    },
}

METRIC_LABELS = {
    "ppa_total": "Total EPA", "ppa_pass_avg": "EPA / pass play", "ppa_rush_avg": "EPA / rush",
    "ppa_pass_total": "Receiving EPA", "pass_ypa": "Yards / attempt", "td_rate": "TD rate",
    "int_rate": "INT rate", "rush_ypg": "Rush yds / game", "pass_ypg": "Pass yds / game",
    "rush_ypc": "Yards / carry", "scrim_td_pg": "Scrimmage TD / game", "rec_ypg": "Rec yds / game",
    "usage_rush": "Rush usage share", "usage_pass": "Target share (usage)", "rec_ypr": "Yards / reception",
    "rec_td_pg": "Rec TD / game", "sacks_pg": "Sacks / game", "tfl_pg": "TFL / game",
    "hurries_pg": "QB hurries / game", "tackles_pg": "Tackles / game", "pd_pg": "Passes defended / game",
    "ints_pg": "INT / game", "def_td_pg": "Defensive TD / game", "ol_line_yards": "Team line yards",
    "ol_stuff_rate": "Team stuff rate allowed", "ol_power_success": "Team power success",
    "ol_pd_success": "Team passing-down success", "ol_rush_ppa": "Team EPA / rush",
}

# Volume needed to be "qualified" (per game played) — keeps 2-game samples from topping lists.
QUALIFY = {
    "QB": ("passing_ATT", 10), "RB": ("rushing_CAR", 5), "WR": ("receiving_REC", 1.5),
    "TE": ("receiving_REC", 1.0), "EDGE": ("defensive_TOT", 1.5), "DL": ("defensive_TOT", 1.5),
    "LB": ("defensive_TOT", 2.5), "DB": ("defensive_TOT", 2.0), "OL": (None, 0),
}

SOS_WEIGHT = 0.25  # z-units added per 1 SD of conference strength


def _derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Rates are per game PLAYED, so a starter who missed games isn't penalized on rate stats.
    # Season-total metrics (total EPA) still reflect missed time. Falls back to team games
    # where games played is unknown (OL).
    gp = d["games_played"] if "games_played" in d else pd.Series(np.nan, index=d.index)
    g = gp.where(gp > 0, d["team_games"]).fillna(d["team_games"]).clip(lower=1)
    d["games_for_rates"] = g
    f = lambda c: d[c].fillna(0) if c in d else pd.Series(0.0, index=d.index)  # noqa: E731
    att = f("passing_ATT").replace(0, np.nan)
    car = f("rushing_CAR").replace(0, np.nan)
    rec = f("receiving_REC").replace(0, np.nan)
    d["ppa_total"] = f("ppa_tot_all")
    d["ppa_pass_total"] = f("ppa_tot_pass")
    d["ppa_pass_avg"] = d["ppa_avg_pass"]
    d["ppa_rush_avg"] = d["ppa_avg_rush"]
    d["pass_ypa"] = f("passing_YDS") / att
    d["td_rate"] = f("passing_TD") / att
    d["int_rate"] = f("passing_INT") / att
    d["pass_ypg"] = f("passing_YDS") / g
    d["rush_ypg"] = f("rushing_YDS") / g
    d["rush_ypc"] = f("rushing_YDS") / car
    d["rec_ypg"] = f("receiving_YDS") / g
    d["rec_ypr"] = f("receiving_YDS") / rec
    d["rec_td_pg"] = f("receiving_TD") / g
    d["scrim_td_pg"] = (f("rushing_TD") + f("receiving_TD")) / g
    d["sacks_pg"] = f("defensive_SACKS") / g
    d["tfl_pg"] = f("defensive_TFL") / g
    d["hurries_pg"] = f("defensive_QB_HUR") / g
    d["tackles_pg"] = f("defensive_TOT") / g
    d["pd_pg"] = f("defensive_PD") / g
    d["ints_pg"] = f("interceptions_INT") / g
    d["def_td_pg"] = (f("defensive_TD") + f("interceptions_TD")) / g
    return d


def score_production(players: pd.DataFrame) -> pd.DataFrame:
    """Return players with production_score (0-100), qualified flag and per-metric percentiles."""
    d = _derive_metrics(players)
    cs = d["conf_strength"]
    d["sos_z"] = ((cs - cs.mean()) / cs.std()).fillna(0)
    out = []
    for group, weights in WEIGHTS.items():
        sub = d[d["group"] == group].copy()
        if sub.empty:
            continue
        col, per_game = QUALIFY[group]
        if col:
            enough_volume = sub[col].fillna(0) >= per_game * sub["games_for_rates"]
            # Must have played at least 40% of team games (min 3) to set the baseline.
            enough_games = sub["games_played"].fillna(0) >= np.maximum(3, 0.4 * sub["team_games"])
            sub["qualified"] = enough_volume & enough_games
        else:
            sub["qualified"] = True
        has_snaps = sub["snap_share"].notna() if "snap_share" in sub else pd.Series(False, index=sub.index)
        if has_snaps.any():
            # With snap data, part-time players can't qualify (and OL backups stop inheriting starter value).
            min_share = 0.4 if group == "OL" else 0.25
            sub.loc[has_snaps, "qualified"] &= sub.loc[has_snaps, "snap_share"] >= min_share
        q = sub[sub["qualified"]]
        composite = pd.Series(0.0, index=sub.index)
        for metric, (w, higher) in weights.items():
            vals = sub[metric].astype(float)
            ref = q[metric].astype(float)
            mu, sd = ref.mean(), ref.std() or 1.0
            z = ((vals - mu) / sd).clip(-3, 3).fillna(-1.0)  # missing => below average
            if not higher:
                z = -z
            composite += w * z
            # Per-metric percentile vs qualified peers (for the player card).
            sorted_ref = np.sort(ref.dropna().values if higher else -ref.dropna().values)
            v = vals.values if higher else -vals.values
            pct = np.searchsorted(sorted_ref, v, side="right") / max(len(sorted_ref), 1) * 100
            sub[f"pct_{metric}"] = np.where(np.isnan(v), np.nan, pct)
        composite += SOS_WEIGHT * sub["sos_z"]
        sub["composite"] = composite
        ref_comp = np.sort(sub.loc[sub["qualified"], "composite"].values)
        sub["production_score"] = (np.searchsorted(ref_comp, sub["composite"].values, side="right")
                                   / max(len(ref_comp), 1) * 100).round(1)
        # Rank among qualified players only (low-volume players get no rank).
        sub["position_rank"] = sub["composite"].where(sub["qualified"]).rank(ascending=False, method="min").astype("Int64")
        if group == "OL":
            sub["data_confidence"] = np.where(has_snaps, "Team proxy, snap-weighted",
                                              "Team proxy (no individual OL stats)")
        else:
            missed = (sub["team_games"] - sub["games_played"].fillna(0)).clip(lower=0).astype(int)
            sub["data_confidence"] = np.select(
                [~sub["qualified"], missed >= 2],
                ["Low volume", "Individual stats · missed " + missed.astype(str) + " games"],
                "Individual stats")
        out.append(sub)
    return pd.concat(out, ignore_index=True)
