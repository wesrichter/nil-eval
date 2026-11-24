"""NIL Evaluator — Streamlit dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nil_eval import build_players, evaluate, load_market, load_nil, score_production
from nil_eval.evaluate import VERDICTS, _anchors, market_fair_value
from nil_eval.portal import build_board, shortlist
from nil_eval.production import METRIC_LABELS, WEIGHTS

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="NIL Evaluator", layout="wide")

# Diverging verdict scale: blue (bargain) -> gray (fair) -> red (overpaid).
VERDICT_COLORS = {
    "Bargain": "#1c5cab", "Below market": "#6da7ec", "Fair value": "#9a9992",
    "Above market": "#e66767", "Overpaid": "#b22c2c", "n/a": "#c3c2b7",
}
VERDICT_ICON = {"Bargain": "▼▼", "Below market": "▼", "Fair value": "●",
                "Above market": "▲", "Overpaid": "▲▲", "n/a": ""}
VERDICT_ORDER = ["Bargain", "Below market", "Fair value", "Above market", "Overpaid"]
BLUE = "#2a78d6"
GP = lambda d: [("—" if pd.isna(g) else f"{int(g)}/{int(t)}") for g, t in zip(d["games_played"], d["team_games"])]  # noqa: E731
SNAP = lambda d: d["snap_share"].map(lambda v: "—" if pd.isna(v) else f"{v:.0%}")  # noqa: E731
MONEY = lambda v: "—" if pd.isna(v) else (f"${v/1e6:.2f}M" if v >= 995e3 else f"${v/1e3:.0f}K")  # noqa: E731
LOG_TICKS = dict(tickvals=[1e5, 2.5e5, 5e5, 1e6, 1.5e6, 2e6, 3e6, 4e6, 6e6, 8e6],
                 ticktext=["$100K", "$250K", "$500K", "$1M", "$1.5M", "$2M", "$3M", "$4M", "$6M", "$8M"])


@st.cache_data(show_spinner="Pulling season data from CollegeFootballData…", ttl=6 * 3600)
def scored_players(year: int) -> pd.DataFrame:
    return score_production(build_players(year))


# ---------------- Sidebar ----------------
st.sidebar.title("NIL Evaluator")
st.sidebar.caption("College football · production vs. NIL")
year = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=0)
method = st.sidebar.radio("Verdict method", ["Expected-NIL model", "Position market"],
                          help="Model: compares NIL to what similar players (production, position, "
                               "program) earn. Market: compares NIL to the position's pay curve "
                               "at the player's production percentile.")
upload = st.sidebar.file_uploader("Override NIL values (CSV)", type="csv",
                                  help="Columns: player,team,position,nil_value_usd,source,as_of")

nil_df = pd.read_csv(upload) if upload else load_nil()
nil_df["team"] = nil_df["team"].fillna("")
market = load_market()

try:
    players = scored_players(year)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load CFBD data for {year}: {e}")
    st.stop()

df, unmatched, coefs = evaluate(players, nil_df, market)
verdict_col = "model_verdict" if method == "Expected-NIL model" else "market_verdict"
ratio_col = "model_ratio" if method == "Expected-NIL model" else "market_ratio"
bench_col = "expected_nil" if method == "Expected-NIL model" else "market_fair_nil"
bench_label = "Expected NIL" if method == "Expected-NIL model" else "Market-fair NIL"

groups = sorted(df["group"].unique())
sel_groups = st.sidebar.multiselect("Position groups", groups, default=groups)
p4_only = st.sidebar.checkbox("Power 4 only", value=False)

view = df[df["group"].isin(sel_groups)]
if p4_only:
    view = view[view["power4"]]
nil_view = view[view["nil_value_usd"].notna()].copy()
nil_view["verdict"] = nil_view[verdict_col]

tab_eval, tab_card, tab_market, tab_portal, tab_finder, tab_data = st.tabs(
    ["Evaluation", "Player card", "Position market", "Transfer portal", "Bargain finder", "Data & method"])

# ---------------- Evaluation ----------------
with tab_eval:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players with NIL data", len(nil_view))
    c2.metric("Total NIL tracked", MONEY(nil_view["nil_value_usd"].sum()))
    c3.metric("Overpaid / above market",
              int(nil_view["verdict"].isin(["Overpaid", "Above market"]).sum()))
    c4.metric("Bargains / below market",
              int(nil_view["verdict"].isin(["Bargain", "Below market"]).sum()))

    st.subheader("Production vs. NIL")
    st.caption(f"Each dot is a player with a known NIL value, colored by **{method}** verdict. "
               "x = production percentile within position group; y = NIL (log scale).")
    # Label selectively: biggest earners + the most extreme over/under-valuations.
    lr = np.log(nil_view[ratio_col].astype(float))
    label_set = set(nil_view.nlargest(3, "nil_value_usd")["player"]) | \
        set(nil_view.loc[lr.nlargest(3).index, "player"]) | set(nil_view.loc[lr.nsmallest(3).index, "player"])
    fig = go.Figure()
    for v in VERDICT_ORDER:
        s = nil_view[nil_view["verdict"] == v]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["production_score"], y=s["nil_value_usd"], mode="markers+text",
            name=f"{VERDICT_ICON[v]} {v}",
            text=np.where(s["player"].isin(label_set), s["player"], ""),
            textposition="top center", textfont=dict(size=11),
            marker=dict(size=11, color=VERDICT_COLORS[v], line=dict(width=2, color="rgba(255,255,255,0.9)")),
            customdata=np.stack([s["player"], s["team"], s["market_position"],
                                 s[bench_col].map(MONEY), s[ratio_col].round(2)], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]} (%{customdata[2]})<br>"
                          "Production: %{x:.0f}th pct<br>NIL: $%{y:,.0f}<br>"
                          f"{bench_label}: " + "%{customdata[3]}<br>Ratio: %{customdata[4]}x<extra></extra>",
        ))
    fig.update_layout(height=520, yaxis_type="log", xaxis_title="Production percentile (within position)",
                      yaxis_title="NIL value (USD)", xaxis_range=[0, 105], legend_title_text="Verdict",
                      margin=dict(t=10, l=10, r=10, b=10), yaxis=LOG_TICKS)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Evaluation table")
    tbl = nil_view.sort_values("nil_value_usd", ascending=False)
    st.dataframe(pd.DataFrame({
        "Player": tbl["player"], "Team": tbl["team"], "Pos": tbl["market_position"],
        "Production pct": tbl["production_score"].round(0),
        "NIL": tbl["nil_value_usd"].map(MONEY),
        "Market-fair NIL": tbl["market_fair_nil"].map(MONEY),
        "Pay pct (position)": tbl["pay_percentile"].round(0),
        "Market verdict": tbl["market_verdict"].map(lambda v: f"{VERDICT_ICON[v]} {v}"),
        "Expected NIL (model)": tbl["expected_nil"].map(MONEY),
        "Model verdict": tbl["model_verdict"].map(lambda v: f"{VERDICT_ICON[v]} {v}"),
        "Brand premium": tbl["brand_premium_usd"].map(lambda v: ("+" if v > 0 else "−") + MONEY(abs(v)) if pd.notna(v) else "—"),
        "GP": GP(tbl), "Snap %": SNAP(tbl),
        "Data": tbl["data_confidence"],
    }), hide_index=True, width="stretch")
    st.caption("**Pay pct** = where the NIL sits on the position's pay curve (>100 = above the elite range). "
               "**Brand premium** = NIL minus model-expected NIL — money not explained by production, "
               "position or program.")

# ---------------- Player card ----------------
with tab_card:
    options = view.sort_values(["nil_value_usd", "composite"], ascending=False, na_position="last")
    options = options[options["nil_value_usd"].notna() | options["qualified"]]
    labels = options["player"] + " — " + options["team"] + " (" + options["position"] + ")"
    pick = st.selectbox("Player", labels.tolist())
    p = options.loc[labels[labels == pick].index[0]]

    a, b, c, d = st.columns(4)
    a.metric("Production percentile", f"{p['production_score']:.0f}",
             help=(f"Rank #{p['position_rank']} among qualified {p['group']}s" if pd.notna(p['position_rank'])
                   else "Below the volume threshold — not ranked"))
    b.metric("NIL value", MONEY(p["nil_value_usd"]))
    c.metric("Market-fair NIL", MONEY(p["market_fair_nil"]),
             delta=None if pd.isna(p["market_ratio"]) else f"{VERDICT_ICON[p['market_verdict']]} {p['market_verdict']}",
             delta_color="off")
    d.metric("Model-expected NIL", MONEY(p["expected_nil"]),
             delta=None if pd.isna(p["model_ratio"]) else f"{VERDICT_ICON[p['model_verdict']]} {p['model_verdict']}",
             delta_color="off")
    st.caption(f"{p['team']} · {p['conference']} · {'Power 4' if p['power4'] else 'Group of 5'} · "
               f"{p['data_confidence']}" + (f" · NIL source: {p['nil_source']} ({p['nil_as_of']})"
                                             if pd.notna(p["nil_value_usd"]) else ""))

    left, right = st.columns(2)
    with left:
        st.markdown("**Production breakdown** — percentile vs. qualified peers")
        mets = list(WEIGHTS[p["group"]].keys())
        pcts = [p.get(f"pct_{m}", np.nan) for m in mets]
        bar = go.Figure(go.Bar(
            x=pcts, y=[f"{METRIC_LABELS[m]} ({WEIGHTS[p['group']][m][0]:.0%})" for m in mets],
            orientation="h", marker_color=BLUE, text=[f"{v:.0f}" if pd.notna(v) else "–" for v in pcts],
            textposition="outside", hovertemplate="%{y}: %{x:.0f}th pct<extra></extra>"))
        bar.update_layout(height=60 + 42 * len(mets), xaxis_range=[0, 110], margin=dict(t=5, l=5, r=5, b=5),
                          yaxis_autorange="reversed", xaxis_title="Percentile (weight in label)")
        bar.update_xaxes(showgrid=False)
        st.plotly_chart(bar, width="stretch")
    with right:
        st.markdown(f"**{p['market_position']} pay curve** — where this player sits")
        mrow = market[market["market_position"] == p["market_position"]]
        if not mrow.empty:
            xs = np.linspace(0, 100, 101)
            curve = [market_fair_value(x, p["market_position"], p["power4"], market) for x in xs]
            mf = go.Figure(go.Scatter(x=xs, y=curve, mode="lines", line=dict(color=BLUE, width=2),
                                      name="Market-fair NIL", hovertemplate="%{x:.0f}th pct → $%{y:,.0f}<extra></extra>"))
            mf.add_trace(go.Scatter(x=[p["production_score"]], y=[p["market_fair_nil"]], mode="markers",
                                    name="Fair at his production", marker=dict(size=12, color=BLUE,
                                    line=dict(width=2, color="white"))))
            if pd.notna(p["nil_value_usd"]):
                mf.add_trace(go.Scatter(x=[p["production_score"]], y=[p["nil_value_usd"]], mode="markers",
                                        name="Actual NIL", marker=dict(size=13, symbol="diamond",
                                        color=VERDICT_COLORS[p["market_verdict"]], line=dict(width=2, color="white"))))
            if pd.notna(p["expected_nil"]):
                mf.add_trace(go.Scatter(x=[p["production_score"]], y=[p["expected_nil"]], mode="markers",
                                        name="Model-expected", marker=dict(size=12, symbol="square-open",
                                        color="#6250d6", line=dict(width=2))))
            mf.update_layout(height=360, yaxis_type="log", yaxis=LOG_TICKS, margin=dict(t=5, l=5, r=5, b=5),
                             xaxis_title="Production percentile", legend=dict(orientation="h", y=-0.25))
            mf.update_xaxes(showgrid=False)
            st.plotly_chart(mf, width="stretch")

    raw_cols = {"QB": ["passing_ATT", "passing_COMPLETIONS", "passing_YDS", "passing_TD", "passing_INT", "rushing_YDS", "ppa_tot_all"],
                "RB": ["rushing_CAR", "rushing_YDS", "rushing_TD", "receiving_REC", "receiving_YDS", "ppa_tot_all"],
                "WR": ["receiving_REC", "receiving_YDS", "receiving_TD", "receiving_LONG", "ppa_tot_pass"],
                "TE": ["receiving_REC", "receiving_YDS", "receiving_TD", "ppa_tot_pass"],
                "OL": ["ol_line_yards", "ol_stuff_rate", "ol_power_success"]}
    cols = raw_cols.get(p["group"], ["defensive_TOT", "defensive_SOLO", "defensive_TFL", "defensive_SACKS",
                                     "defensive_QB_HUR", "defensive_PD", "interceptions_INT"])
    gp_txt = "games played unknown (OL)" if pd.isna(p["games_played"]) else \
        f"played {int(p['games_played'])} of {int(p['team_games'])} team games"
    snap_txt = "" if pd.isna(p.get("snap_share")) else f" · {int(p['snaps'])} snaps ({p['snap_share']:.0%})"
    st.markdown(f"**Season totals** ({gp_txt}{snap_txt}; rate stats are per game played)")
    nice = lambda c: c.replace("defensive_", "").replace("interceptions_INT", "INT").replace("ppa_tot_all", "Total EPA") \
        .replace("ppa_tot_pass", "Rec EPA").replace("_", " ").title()  # noqa: E731
    st.dataframe(pd.DataFrame([{nice(c): (round(p.get(c), 2) if pd.notna(p.get(c)) else None) for c in cols}]),
                 hide_index=True)

# ---------------- Position market ----------------
with tab_market:
    st.subheader("What each position costs")
    st.caption("Power-4 pay curves by position, built from **College Sports Commission** data (median and "
               "90th percentile of revenue share + cleared third-party NIL, Jan–Sep 2026) with the top observed "
               "On3 valuation as the ceiling. Tracked players overlaid. Group of 5 pays ~11–12% of Power 4 "
               "(from CSC's non-Power-4 QB figures).")
    mk = market.copy()
    mk["order"] = mk["ceiling_usd"]
    mk = mk.sort_values("order")
    fig = go.Figure()
    for _, r in mk.iterrows():
        a = _anchors(r)
        fig.add_trace(go.Scatter(x=[a[0], a[-1]], y=[r.market_position] * 2, mode="lines",
                                 line=dict(color="#b7d3f6", width=6), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r.p50_usd, r.p90_usd], y=[r.market_position] * 2,
                                 mode="lines", line=dict(color=BLUE, width=14), showlegend=False,
                                 hovertemplate=f"<b>{r.market_position}</b><br>CSC median: {MONEY(r.p50_usd)}<br>"
                                               f"CSC 90th pct: {MONEY(r.p90_usd)}<br>"
                                               f"Top of market: {MONEY(r.ceiling_usd)}<extra></extra>"))
    for v in VERDICT_ORDER:
        s = nil_view[nil_view["market_verdict"] == v]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["nil_value_usd"], y=s["market_position"], mode="markers", name=f"{VERDICT_ICON[v]} {v}",
            marker=dict(size=10, symbol="diamond", color=VERDICT_COLORS[v], line=dict(width=2, color="white")),
            customdata=np.stack([s["player"], s["team"], s["production_score"].round(0)], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>NIL: $%{x:,.0f}<br>"
                          "Production pct: %{customdata[2]}<extra></extra>"))
    fig.update_layout(height=520, xaxis_type="log", xaxis=LOG_TICKS, legend_title_text="Market verdict",
                      margin=dict(t=10, l=10, r=10, b=10), xaxis_title="Annual NIL / rev-share value (log)")
    fig.update_yaxes(categoryorder="array", categoryarray=mk["market_position"].tolist())
    st.plotly_chart(fig, width="stretch")
    st.caption("Thick bar = CSC median → 90th percentile; thin bar extends to the top of the market. Diamonds = players.")

    st.markdown("**Market-fair NIL by production percentile (Power 4)**")
    grid = pd.DataFrame({"Position": market["market_position"]})
    for pct in [25, 50, 75, 90, 95, 100]:
        grid[f"{pct}th pct"] = [MONEY(market_fair_value(pct, m, True, market)) for m in market["market_position"]]
    st.dataframe(grid, hide_index=True, width="stretch")
    st.dataframe(market, hide_index=True, width="stretch")

# ---------------- Transfer portal ----------------
with tab_portal:
    cycle = year + 1
    board, n_portal = build_board(df, cycle, market)
    st.subheader(f"{cycle} transfer portal board")
    if n_portal == 0:
        st.info(f"The {cycle} portal feed is empty — the FBS window opens in early January {cycle} "
                "(Jan 2–16 under current rules; a 10-day window has been proposed). Until then this is a "
                "**watchlist** of every qualified player with projected prices. Entrants are flagged automatically "
                "once CollegeFootballData publishes them (refreshes every 3 hours).")
        only_portal = False
    else:
        st.caption(f"{n_portal:,} portal entries in the feed; {int(board['in_portal'].sum()):,} matched to FBS "
                   f"{year} production (the rest are FCS/lower-division or had no recorded stats).")
        only_portal = st.toggle("Portal entrants only", value=True)

    f1, f2, f3, f4, f5 = st.columns(5)
    has_ol_snaps = board.loc[board["group"] == "OL", "snap_share"].notna().any()
    pg = f1.multiselect("Position group", [g for g in groups if g != "OL" or has_ol_snaps], default=["QB"],
                        help="OL is excluded unless snap counts are loaded (Data & method tab): team line grades "
                             "can't price an individual lineman.")
    origin = f2.selectbox("Coming from", ["Anywhere", "Group of 5 only", "Power 4 only"])
    min_prod = f3.slider("Min production pct", 0, 100, 70)
    min_avail = f5.slider("Min games played (%)", 0, 100, 60,
                          help="Share of team games with a recorded stat. Screens out injury/part-time risk.")
    max_price = f4.number_input("Max P4 price ($)", min_value=0, value=3_000_000, step=100_000)

    bv = board[board["qualified"] & board["group"].isin(pg) & (board["production_score"] >= min_prod)
               & (board["availability"].fillna(0) * 100 >= min_avail)
               & (board["price_p4"] <= max_price)]
    if only_portal:
        bv = bv[bv["in_portal"]]
    if origin == "Group of 5 only":
        bv = bv[~bv["power4"]]
    elif origin == "Power 4 only":
        bv = bv[bv["power4"]]
    bv = bv.sort_values("composite", ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Players on board", len(bv))
    k2.metric("Median Power-4 price", MONEY(bv["price_p4"].median()) if len(bv) else "—")
    k3.metric("From Group of 5", int((~bv["power4"]).sum()))

    st.dataframe(pd.DataFrame({
        "Player": bv["player"], "From": bv["team"], "Conf": bv["conference"], "Pos": bv["position"],
        "Production pct": bv["production_score"].round(1), "Rank": bv["position_rank"],
        "GP": GP(bv), "Snap %": SNAP(bv),
        "Power-4 price": bv["price_p4"].map(MONEY), "G5 price": bv["price_g5"].map(MONEY),
        "Known NIL": bv["nil_value_usd"].map(MONEY), "Value index": bv["value_index"].round(1),
        "In portal": bv["in_portal"].map({True: "Yes", False: ""}),
        "Stars": bv["portal_stars"], "Eligibility": bv["portal_eligibility"],
        "Destination": bv["portal_destination"],
    }), hide_index=True, width="stretch", height=420)
    st.caption("**Power-4 price** = position market curve at his production percentile (what a P4 program "
               "should expect to pay). **Value index** = production points per $100K of that price. "
               "G5 production is already schedule-adjusted, but still carries more projection risk.")

    st.markdown("#### Shortlist builder")
    st.caption("Fill positions of need within a budget. Greedy: each slot takes the most productive player "
               "whose price fits the remaining budget's fair share (×1.5 flex).")
    b1, b2 = st.columns([1, 2])
    budget = b1.number_input("Portal budget ($)", min_value=0, value=5_000_000, step=250_000)
    need_cols = b2.columns(4)
    needs = {}
    for i, g in enumerate([g for g in groups if g != "OL"]):
        n = need_cols[i % 4].number_input(g, min_value=0, max_value=5, value=1 if g in ("QB", "EDGE", "DB") else 0,
                                          key=f"need_{g}")
        if n:
            needs[g] = n
    pool = board[board["qualified"]]
    if only_portal:
        pool = pool[pool["in_portal"]]
    sl = shortlist(pool, needs, budget)
    if len(sl):
        st.dataframe(pd.DataFrame({
            "Need": sl["group"], "Player": sl["player"], "From": sl["team"],
            "Production pct": sl["production_score"].round(1), "Power-4 price": sl["price_p4"].map(MONEY),
        }), hide_index=True, width="stretch")
        st.write(f"**Spent:** {MONEY(sl.attrs['spent'])} of {MONEY(budget)} · "
                 f"{len(sl)} of {sum(needs.values())} needs filled")
    else:
        st.write("No players fit those needs within the budget.")

# ---------------- Bargain finder ----------------
with tab_finder:
    st.subheader("Top producers and what they'd be worth")
    st.caption("All qualified FBS players, ranked by production. *Market-fair NIL* is what the position "
               "market pays for that production level — use it to spot under-paid producers and portal targets.")
    g = st.selectbox("Position group", groups, index=groups.index("QB") if "QB" in groups else 0)
    fv = df[(df["group"] == g) & df["qualified"]]
    if p4_only:
        fv = fv[fv["power4"]]
    fv = fv.sort_values("composite", ascending=False).head(50)
    if g == "OL":
        st.warning("No public individual OL stats exist — OL scores are team-level line metrics shared by every "
                   "lineman on the roster. Treat as program context, not individual grades.")
    st.dataframe(pd.DataFrame({
        "Rank": fv["position_rank"], "Player": fv["player"], "Team": fv["team"], "Pos": fv["position"],
        "GP": GP(fv), "Snap %": SNAP(fv),
        "Conf": fv["conference"], "Production pct": fv["production_score"].round(1),
        "Market-fair NIL": fv["market_fair_nil"].map(MONEY), "Known NIL": fv["nil_value_usd"].map(MONEY),
    }), hide_index=True, width="stretch", height=600)

# ---------------- Data & method ----------------
with tab_data:
    st.subheader("NIL values")
    st.caption("Edit values here and save to update `data/nil_values.csv`. Figures are public estimates "
               "(mostly On3's deal-based valuations), not confirmed contracts.")
    edited = st.data_editor(nil_df, num_rows="dynamic", width="stretch", key="nil_editor")
    if st.button("Save NIL values"):
        edited.to_csv(DATA / "nil_values.csv", index=False)
        st.success("Saved. Reload to re-run the evaluation.")
    if len(unmatched):
        st.warning("These NIL rows didn't match a player with stats this season (name/team mismatch, injury, "
                   "or not on an FBS roster): " + ", ".join(unmatched["player"].astype(str)))

    st.subheader("Snap counts (optional)")
    st.caption("No free source publishes college snap counts (CFBD doesn't carry them; PFF and Sports Info "
               "Solutions sell them). Upload a CSV with columns `player, team, snaps` (optional `team_snaps`) "
               f"and it's saved as `data/snaps_{year}.csv`. With snaps loaded, players under 25% of team snaps "
               "(40% for OL) stop qualifying, so backup linemen no longer inherit their starters' team grade.")
    snap_up = st.file_uploader("Snap counts CSV", type="csv", key="snaps")
    if snap_up is not None and st.button("Save snap counts"):
        sdf = pd.read_csv(snap_up)
        missing = {"player", "team", "snaps"} - set(sdf.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
        else:
            sdf.to_csv(DATA / f"snaps_{year}.csv", index=False)
            scored_players.clear()
            st.success(f"Saved {len(sdf)} rows. Re-running with snap counts.")
            st.rerun()

    st.subheader("Expected-NIL model")
    if coefs is not None:
        st.write(f"Ridge regression on log(NIL), n = {coefs.attrs['n']}, in-sample R² = {coefs.attrs['r2']:.2f}")
        st.dataframe(coefs.round(3), hide_index=True)
    else:
        st.info("Need at least 8 matched NIL values to fit the model.")

    st.subheader("Methodology")
    st.markdown(f"""
**Production score (0–100).** Each position group has its own weighted metrics (see the player card
for weights): EPA/PPA, efficiency, per-game volume, and usage for skill players; sacks, TFL, hurries,
tackles, and passes defended for defenders. Metrics are z-scored against *qualified* players (minimum
volume per game played, and at least 40% of team games played), combined, and adjusted for strength of
schedule (conference SP+). Rate stats are per **game played** (from per-game box scores), so injured
starters aren't penalized on rates; season-total metrics like total EPA still reflect missed time. The score is
the percentile within the position group across all of FBS.

**Position market.** Each position's pay curve (`data/position_market.csv`) comes from College Sports
Commission data (Power-4 revenue share + cleared third-party NIL, Jan–Sep 2026). Qualified starters are
treated as the top half of a paid roster: production 0th pct → CSC median, 80th → CSC 90th pct,
100th → top observed On3 valuation at the position, with log-interpolation between.
Group of 5 is scaled by ~11–12%, from the College Sports Commission's non-Power-4 figures.

**Method 1 — Position market.** Market-fair NIL = the curve at the player's production percentile.
Ratio = actual ÷ fair. Answers: *"Is he paid like a player of his production at his position?"*

**Method 2 — Expected-NIL model.** log(NIL) ~ production + position market level + Power-4 + roster
talent, fit on all tracked players. Answers: *"Does he earn more than comparable players do?"* The gap
is the **brand premium** — money driven by name, market, recruiting pedigree, or social following.

**Verdict bands (ratio).** {" · ".join(f"≥{t}× {l}" for t, l in VERDICTS[:-1])} · below → Bargain.

**Known limits.** NIL figures are third-party estimates. No public individual OL or coverage grades.
Low-volume players (injury, backups) are flagged. The model is fit on a small, star-heavy sample.
""")
