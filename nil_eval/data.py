"""Build a one-row-per-player production table for a season from CFBD."""
from __future__ import annotations

import pandas as pd

from . import cfbd

POWER_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC"}
POWER_INDEPENDENTS = {"Notre Dame"}

# CFBD position -> scoring group
SCORING_GROUP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "OL": "OL", "OT": "OL", "T": "OL", "G": "OL", "OG": "OL", "C": "OL", "IOL": "OL",
    "DE": "EDGE", "EDGE": "EDGE", "OLB": "EDGE",
    "DL": "DL", "DT": "DL", "NT": "DL",
    "LB": "LB", "ILB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "SAF": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
}

# CFBD position -> position market (pay scale) bucket
MARKET_POSITION = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "OL": "OT", "OT": "OT", "T": "OT", "G": "IOL", "OG": "IOL", "C": "IOL", "IOL": "IOL",
    "DE": "EDGE", "EDGE": "EDGE", "OLB": "EDGE",
    "DL": "DL", "DT": "DL", "NT": "DL",
    "LB": "LB", "ILB": "LB", "MLB": "LB",
    "CB": "CB", "DB": "CB", "S": "S", "SAF": "S", "FS": "S", "SS": "S",
}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def team_context(year: int) -> pd.DataFrame:
    """Per-team: conference, power flag, games played, SP+, talent, OL metrics."""
    teams = pd.DataFrame(cfbd.get("teams/fbs", year=year))[["school", "conference"]]
    teams = teams.rename(columns={"school": "team"})
    teams["power4"] = teams["conference"].isin(POWER_CONFERENCES) | teams["team"].isin(POWER_INDEPENDENTS)

    games = pd.DataFrame(cfbd.get("games", year=year, seasonType="both"))
    games = games[games["completed"] == True]  # noqa: E712
    home = games.get("homeTeam", games.get("home_team"))
    away = games.get("awayTeam", games.get("away_team"))
    gp = pd.concat([home, away]).value_counts().rename("team_games")
    teams = teams.merge(gp, left_on="team", right_index=True, how="left")

    sp = pd.DataFrame(cfbd.get("ratings/sp", year=year))
    if not sp.empty:
        teams = teams.merge(sp[["team", "rating"]].rename(columns={"rating": "sp_rating"}),
                            on="team", how="left")
    talent = pd.DataFrame(cfbd.get("talent", year=year))
    if not talent.empty:
        teams = teams.merge(talent[["team", "talent"]], on="team", how="left")

    adv = cfbd.get("stats/season/advanced", year=year)
    ol_rows = []
    for t in adv:
        o = t.get("offense") or {}
        ol_rows.append({
            "team": t["team"],
            "ol_line_yards": o.get("lineYards"),
            "ol_stuff_rate": o.get("stuffRate"),
            "ol_power_success": o.get("powerSuccess"),
            "ol_pd_success": (o.get("passingDowns") or {}).get("successRate"),
            "ol_rush_ppa": (o.get("rushingPlays") or {}).get("ppa"),
        })
    teams = teams.merge(pd.DataFrame(ol_rows), on="team", how="left")

    # Strength of schedule proxy: average SP+ of the team's conference
    # (independents use their own rating).
    teams["conf_strength"] = teams.groupby("conference")["sp_rating"].transform("mean")
    indep = teams["conference"].str.contains("Independent", na=False)
    teams.loc[indep, "conf_strength"] = teams.loc[indep, "sp_rating"]
    teams["team_games"] = teams["team_games"].fillna(teams["team_games"].median())
    return teams


def player_stats(year: int) -> pd.DataFrame:
    """Season box-score stats, PPA and usage pivoted to one row per player."""
    raw = pd.DataFrame(cfbd.get("stats/player/season", year=year))
    raw["key"] = raw["category"] + "_" + raw["statType"].str.replace(" ", "_")
    raw["stat"] = raw["stat"].map(_num)
    wide = raw.pivot_table(index=["playerId", "player", "position", "team"],
                           columns="key", values="stat", aggfunc="sum").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"playerId": "player_id"})
    # A player can appear twice if listed at two positions; keep the row with more stats.
    wide["_n"] = wide.notna().sum(axis=1)
    wide = wide.sort_values("_n", ascending=False).drop_duplicates("player_id").drop(columns="_n")

    ppa = cfbd.get("ppa/players/season", year=year)
    ppa_rows = [{
        "player_id": str(p["id"]),
        "ppa_avg_all": (p.get("averagePPA") or {}).get("all"),
        "ppa_avg_pass": (p.get("averagePPA") or {}).get("pass"),
        "ppa_avg_rush": (p.get("averagePPA") or {}).get("rush"),
        "ppa_tot_all": (p.get("totalPPA") or {}).get("all"),
        "ppa_tot_pass": (p.get("totalPPA") or {}).get("pass"),
        "ppa_tot_rush": (p.get("totalPPA") or {}).get("rush"),
    } for p in ppa]
    usage = cfbd.get("player/usage", year=year)
    usage_rows = [{
        "player_id": str(u["id"]),
        "usage_overall": (u.get("usage") or {}).get("overall"),
        "usage_pass": (u.get("usage") or {}).get("pass"),
        "usage_rush": (u.get("usage") or {}).get("rush"),
    } for u in usage]

    wide["player_id"] = wide["player_id"].astype(str)
    wide = wide.merge(pd.DataFrame(ppa_rows).drop_duplicates("player_id"), on="player_id", how="left")
    wide = wide.merge(pd.DataFrame(usage_rows).drop_duplicates("player_id"), on="player_id", how="left")
    return wide


def roster(year: int) -> pd.DataFrame:
    """Full FBS rosters (needed for offensive linemen, who have no box-score stats)."""
    r = pd.DataFrame(cfbd.get("roster", year=year))
    r["player"] = (r["firstName"].fillna("") + " " + r["lastName"].fillna("")).str.strip()
    r = r.rename(columns={"id": "player_id", "year": "class_year"})
    r["player_id"] = r["player_id"].astype(str)
    return r[["player_id", "player", "team", "position", "class_year", "height", "weight"]]


def build_players(year: int) -> pd.DataFrame:
    teams = team_context(year)
    stats = player_stats(year)
    ros = roster(year)

    # Add offensive linemen (and anyone else on a roster with no stats) from rosters.
    missing = ros[~ros["player_id"].isin(stats["player_id"])]
    missing = missing[missing["position"].map(SCORING_GROUP) == "OL"]
    players = pd.concat([stats, missing[["player_id", "player", "team", "position"]]], ignore_index=True)
    # Roster spelling can differ from the stats feed (e.g. "A.J." vs "Anthony"); keep both for matching.
    players = players.merge(ros[["player_id", "class_year", "player"]].rename(columns={"player": "roster_name"})
                            .drop_duplicates("player_id"), on="player_id", how="left")

    players = players.merge(teams, on="team", how="inner")  # FBS only
    players["group"] = players["position"].map(SCORING_GROUP)
    players["market_position"] = players["position"].map(MARKET_POSITION)
    players = players[players["group"].notna()].reset_index(drop=True)
    players["season"] = year
    return players
