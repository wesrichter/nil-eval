"""Transfer portal board.

The portal cycle that follows season N is cycle N+1 (e.g. the January 2027 window
follows the 2026 season). Entries come from CFBD's /player/portal feed; before the
window opens the feed is empty, so the board falls back to a watchlist of every
qualified player.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cfbd
from .evaluate import market_fair_value, norm_name


def load_portal(cycle_year: int) -> pd.DataFrame:
    rows = cfbd.get("player/portal", ttl_hours=3, year=cycle_year)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["player"] = (df["firstName"].fillna("") + " " + df["lastName"].fillna("")).str.strip()
    df["_key"] = df["player"].map(norm_name)
    return df


def build_board(evaluated: pd.DataFrame, cycle_year: int, market: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return (board, n_portal_entries). Board has one row per player with prices at P4 and G5."""
    board = evaluated.copy()
    board["_key"] = board["player"].map(norm_name)
    portal = load_portal(cycle_year)
    board["in_portal"] = False
    for col in ["portal_destination", "portal_eligibility", "portal_stars", "portal_date"]:
        board[col] = None
    if not portal.empty:
        # Match on name + origin school (the season's team), falling back to unique name.
        p = portal.rename(columns={"origin": "team"})
        merged = board.reset_index().merge(
            p[["_key", "team", "destination", "eligibility", "stars", "transferDate"]],
            on=["_key", "team"], how="inner").set_index("index")
        board.loc[merged.index, "in_portal"] = True
        board.loc[merged.index, "portal_destination"] = merged["destination"]
        board.loc[merged.index, "portal_eligibility"] = merged["eligibility"]
        board.loc[merged.index, "portal_stars"] = merged["stars"]
        board.loc[merged.index, "portal_date"] = merged["transferDate"]

    board["price_p4"] = [market_fair_value(s, m, True, market)
                         for s, m in zip(board["production_score"], board["market_position"])]
    board["price_g5"] = [market_fair_value(s, m, False, market)
                         for s, m in zip(board["production_score"], board["market_position"])]
    # OL scores are team-level proxies, so pricing an individual lineman from them is misleading.
    ol = board["group"] == "OL"
    board.loc[ol, ["price_p4", "price_g5"]] = np.nan
    # Production points per $100K of Power-4 price: a simple value-for-money index.
    board["value_index"] = board["production_score"] / (board["price_p4"] / 1e5)
    board = board.drop(columns="_key")
    return board, len(portal)


def shortlist(board: pd.DataFrame, needs: dict[str, int], budget: float, price_col: str = "price_p4",
              flex: float = 1.5) -> pd.DataFrame:
    """Greedy roster-building: fill each need with the most productive player whose price fits
    the remaining budget's fair share (remaining budget / slots left x flex)."""
    picks, remaining = [], budget
    slots_left = sum(needs.values())
    taken: set = set()
    for group, count in needs.items():
        for _ in range(count):
            cap = remaining / max(slots_left, 1) * flex
            pool = board[(board["group"] == group) & board[price_col].notna() & (~board["player_id"].isin(taken))
                         & (board[price_col] <= min(cap, remaining))]
            if pool.empty:
                slots_left -= 1
                continue
            best = pool.sort_values("composite", ascending=False).iloc[0]
            picks.append(best)
            taken.add(best["player_id"])
            remaining -= best[price_col]
            slots_left -= 1
    out = pd.DataFrame(picks)
    out.attrs["spent"] = budget - remaining
    return out
