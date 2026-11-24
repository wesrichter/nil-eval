"""CLI: score a season and write the evaluation to output/.

    python scripts/evaluate.py --year 2026
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nil_eval import run  # noqa: E402

COLS = ["player", "team", "conference", "position", "market_position", "production_score", "position_rank",
        "data_confidence", "nil_value_usd", "market_fair_nil", "pay_percentile", "market_ratio", "market_verdict",
        "expected_nil", "model_ratio", "model_verdict", "brand_premium_usd", "nil_source", "nil_as_of"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    df, unmatched, coefs = run(args.year)
    out = Path(__file__).resolve().parent.parent / "output"
    out.mkdir(exist_ok=True)
    df[COLS].sort_values(["position_rank"]).to_csv(out / f"all_players_{args.year}.csv", index=False)
    ev = df[df["nil_value_usd"].notna()][COLS].sort_values("nil_value_usd", ascending=False)
    ev.to_csv(out / f"nil_evaluation_{args.year}.csv", index=False)
    print(ev[["player", "team", "market_position", "production_score", "nil_value_usd",
              "market_verdict", "model_verdict"]].to_string(index=False))
    if coefs is not None:
        print(f"\nModel: n={coefs.attrs['n']}, R2={coefs.attrs['r2']:.2f}")
    if len(unmatched):
        print("Unmatched NIL rows:", ", ".join(unmatched["player"]))
    print(f"\nWrote {out}/nil_evaluation_{args.year}.csv and all_players_{args.year}.csv")
