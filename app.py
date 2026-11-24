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
from nil_eval.evaluate import PCT_ANCHORS, VERDICTS, _anchors, market_fair_value
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
MONEY = lambda v: "—" if pd.isna(v) else (f"${v/1e6:.2f}M" if v >= 995e3 else f"${v/1e3:.0f}K")  # noqa: E731
LOG_TICKS = dict(tickvals=[1e5, 2.5e5, 5e5, 1e6, 2e6, 4e6, 8e6],
                 ticktext=["$100K", "$250K", "$500K", "$1M", "$2M", "$4M", "$8M"])


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

tab_eval, tab_card, tab_market, tab_finder, tab_data = st.tabs(
    ["Evaluation", "Player card", "Position market", "Bargain finder", "Data & method"])

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
    st.markdown(f"**Season totals** ({int(p['team_games'])} team games)")
    nice = lambda c: c.replace("defensive_", "").replace("interceptions_INT", "INT").replace("ppa_tot_all", "Total EPA") \
        .replace("ppa_tot_pass", "Rec EPA").replace("_", " ").title()  # noqa: E731
    st.dataframe(pd.DataFrame([{nice(c): (round(p.get(c), 2) if pd.notna(p.get(c)) else None) for c in cols}]),
                 hide_index=True)

# ---------------- Position market ----------------
with tab_market:
    st.subheader("What each position costs")
    st.caption("Power-4 pay ranges by position (low → typical → elite), with tracked players overlaid. "
               "Group of 5 programs are assumed to pay ~50% of these ranges.")
    mk = market.copy()
    mk["order"] = mk["elite_high_usd"]
    mk = mk.sort_values("order")
    fig = go.Figure()
    for _, r in mk.iterrows():
        a = _anchors(r)
        fig.add_trace(go.Scatter(x=[a[0], a[-1]], y=[r.market_position] * 2, mode="lines",
                                 line=dict(color="#b7d3f6", width=6), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r.typical_low_usd, r.typical_high_usd], y=[r.market_position] * 2,
                                 mode="lines", line=dict(color=BLUE, width=14), showlegend=False,
                                 hovertemplate=f"<b>{r.market_position}</b><br>Low: {MONEY(r.low_usd)}<br>"
                                               f"Typical starter: {MONEY(r.typical_low_usd)}–{MONEY(r.typical_high_usd)}<br>"
                                               f"Elite: {MONEY(r.elite_low_usd)}–{MONEY(r.elite_high_usd)}<extra></extra>"))
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
    st.caption("Thin bar = full range (low → elite ceiling); thick bar = typical starter range. Diamonds = players.")

    st.markdown("**Market-fair NIL by production percentile (Power 4)**")
    grid = pd.DataFrame({"Position": market["market_position"]})
    for pct in [25, 50, 75, 90, 95, 100]:
        grid[f"{pct}th pct"] = [MONEY(market_fair_value(pct, m, True, market)) for m in market["market_position"]]
    st.dataframe(grid, hide_index=True, width="stretch")
    st.dataframe(market, hide_index=True, width="stretch")

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
volume per team game), combined, and adjusted for strength of schedule (conference SP+). The score is
the percentile within the position group across all of FBS.

**Position market.** Pay ranges per position (`data/position_market.csv`) define a pay curve:
0th pct → low, 50th → typical-low, 80th → typical-high, 95th → elite, 100th → elite ceiling.
Group of 5 is scaled by {0.5:.0%}.

**Method 1 — Position market.** Market-fair NIL = the curve at the player's production percentile.
Ratio = actual ÷ fair. Answers: *"Is he paid like a player of his production at his position?"*

**Method 2 — Expected-NIL model.** log(NIL) ~ production + position market level + Power-4 + roster
talent, fit on all tracked players. Answers: *"Does he earn more than comparable players do?"* The gap
is the **brand premium** — money driven by name, market, recruiting pedigree, or social following.

**Verdict bands (ratio).** {" · ".join(f"≥{t}× {l}" for t, l in VERDICTS[:-1])} · below → Bargain.

**Known limits.** NIL figures are third-party estimates. No public individual OL or coverage grades.
Low-volume players (injury, backups) are flagged. The model is fit on a small, star-heavy sample.
""")
