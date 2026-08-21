"""
GEO System - GEO Scoring Module (Sprint 4, part 1)
------------------------------------------------------
Combines all previously extracted signals (structure, readability,
semantic relevance, ad density, alt-text coverage) into a single
weighted 0-100 "GEO Score" -- this is the "GEO Scoring Module" named
in the report's system architecture (Figure 5) and DFD-1 (Figure 3).

This is a rule-based / weighted-average scoring approach, used as
the interpretable baseline before the ML-based citation probability
prediction (XGBoost) in Sprint 4 part 2.

Usage:
    python geo_score.py <url> ["optional target keyword"]
"""

import sys
from content_collector import analyze_url
from nlp_features import semantic_relevance_score


# Weights reflect how strongly each signal is believed to correlate with
# citation likelihood, based on the GEO literature reviewed in Chapter 2:
# semantic relevance and structure matter most; readability and
# ad-cleanliness are secondary but real factors.
DEFAULT_WEIGHTS = {
    "structure": 0.30,
    "readability": 0.15,
    "semantic_relevance": 0.40,
    "credibility": 0.15,  # author byline + meta description + structured data
}


def normalize_readability(flesch_score: float) -> float:
    """
    Flesch scores can range roughly -inf..121, but useful content
    typically falls 0-100. Clamp into that range so it's comparable
    to the other 0-100 signals.
    """
    return max(0.0, min(100.0, flesch_score))


def credibility_score(features: dict) -> float:
    """
    0-100 score based on trust/citation-readiness signals:
    author byline, meta description, structured data (schema.org).
    These are exactly the signals the report's literature review
    (Aggarwal et al.) identifies as boosting citation likelihood.
    """
    score = 0
    if features.get("has_author_byline"):
        score += 40
    if features.get("has_meta_description"):
        score += 30
    if features.get("ad_density_percent", 100) < 10:
        score += 30
    return min(score, 100)


def compute_geo_score(
    structure_score: float,
    readability_flesch: float,
    semantic_relevance: float,
    credibility: float,
    weights: dict = None,
) -> dict:
    """
    Returns the overall GEO score plus the individual weighted
    contributions, so the breakdown is transparent/explainable --
    important for the "AI-Based Recommendation Generation" module
    downstream, which needs to know WHICH signal is weakest.
    """
    weights = weights or DEFAULT_WEIGHTS
    readability_norm = normalize_readability(readability_flesch)

    contributions = {
        "structure": structure_score * weights["structure"],
        "readability": readability_norm * weights["readability"],
        "semantic_relevance": semantic_relevance * weights["semantic_relevance"],
        "credibility": credibility * weights["credibility"],
    }

    overall = round(sum(contributions.values()), 2)

    # identify the weakest signal (relative to its own 0-100 scale,
    # not its weighted contribution) -- this is what the recommendation
    # engine (Sprint 5) will target first
    raw_signals = {
        "structure": structure_score,
        "readability": readability_norm,
        "semantic_relevance": semantic_relevance,
        "credibility": credibility,
    }
    weakest_signal = min(raw_signals, key=raw_signals.get)

    return {
        "overall_geo_score": overall,
        "signals": raw_signals,
        "weighted_contributions": {k: round(v, 2) for k, v in contributions.items()},
        "weakest_signal": weakest_signal,
        "weights_used": weights,
    }


def score_url(url: str, keyword: str = "") -> dict:
    """End-to-end: analyze a URL and return its full GEO score breakdown."""
    result = analyze_url(url, keyword)
    features = result["geo_features"]
    full_text = result["full_text"]

    semantic = semantic_relevance_score(full_text, features["target_query"])
    credibility = credibility_score(features)

    geo_result = compute_geo_score(
        structure_score=features["structure_score"],
        readability_flesch=features["readability_flesch"],
        semantic_relevance=semantic,
        credibility=credibility,
    )

    return {
        "url": url,
        "target_query": features["target_query"],
        **geo_result,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python geo_score.py <url> ["optional target keyword"]')
        sys.exit(1)

    target_url = sys.argv[1]
    target_keyword = sys.argv[2] if len(sys.argv) > 2 else ""

    import json
    print(json.dumps(score_url(target_url, target_keyword), indent=2, ensure_ascii=False))
