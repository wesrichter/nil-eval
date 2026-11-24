"""NIL evaluation: college football production vs. NIL earnings."""
from .data import build_players
from .evaluate import evaluate, load_market, load_nil
from .production import score_production


def run(year: int, nil_path=None, market_path=None):
    """Fetch, score and evaluate a season. Returns (evaluated_df, unmatched_nil, model_coefs)."""
    players = score_production(build_players(year))
    return evaluate(players, load_nil(nil_path), load_market(market_path))
