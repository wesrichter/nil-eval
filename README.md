# NIL Evaluator — College Football

Compares a player's on-field **production** to his **NIL earnings**, position by position.

## What it does
1. **Production score (0–100)**: position-specific weighted metrics from CollegeFootballData
   (EPA/PPA, efficiency, per-game volume, usage; sacks/TFL/hurries/PDs for defense),
   z-scored vs. qualified FBS peers, adjusted for strength of schedule, and converted to a percentile.
2. **Position market**: pay curves per position (`data/position_market.csv`) built from College Sports
   Commission pay data (Power-4 median and 90th percentile of revenue share + cleared third-party NIL,
   Jan–Sep 2026), with the top observed On3 valuation as the ceiling. Group of 5 ≈ 11–12% of Power 4.
3. **Two verdicts**
   - **Position market**: is he paid like a player of his production at his position?
     (actual NIL ÷ market-fair NIL at his production percentile)
   - **Expected-NIL model**: does he earn more than comparable players? Ridge regression of
     log(NIL) on production, position market, Power-4 and roster talent. The residual is the **brand premium**.
4. **Transfer portal board**: pulls CFBD's portal feed for the cycle after the selected season, matches
   entrants to their production, and prices each at Power-4 and G5 rates. Before the window opens it runs
   as a watchlist of every qualified player. Includes a budget-constrained shortlist builder.
5. **Bargain finder**: every qualified FBS player with a market-fair NIL estimate.

## Setup
```bash
pip install -r requirements.txt
export CFBD_API_KEY=your_key   # free at collegefootballdata.com
streamlit run app.py
```
CLI: `python scripts/evaluate.py --year 2026` writes CSVs to `output/`.

## Data
- `data/nil_values.csv`: public NIL valuations (mostly On3's deal-based model, Sep–Nov 2026). These are
  **estimates, not confirmed contracts**. Edit in the app's *Data & method* tab or upload a CSV.
- `data/position_market.csv`: College Sports Commission position pay data (via On3/Yahoo/FootballScoop
  coverage of the Oct 2026 release) + top On3 valuation per position. Note: the CSC's NIL Go clearinghouse
  holds every reported deal, but only anonymized aggregates are public — no per-player figures.
- API responses are cached in `data/cache/` (12h).

## Known limitations
- No public individual OL grades — OL scores are team line metrics (flagged in the app) and OL are left
  unpriced on the portal board.
- No coverage/target-allowed data for DBs; defensive scores are box-score based.
- Per-game rates use team games, so injured players read low (flagged "Low volume").
- The NIL sample is small and star/QB-heavy; add more rows to make the model sharper.
