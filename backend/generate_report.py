"""
GEO System - Visual Report Generator
--------------------------------------
Takes the output of content_collector.analyze_url() plus real
TF-IDF + SBERT features from nlp_features.py, and renders it as a
visual HTML scorecard you can open in a browser.

Usage:
    python generate_report.py <url> ["optional keyword"]

This will write geo_report.html in the current folder and you can
open it directly in your browser to see it visually.
"""

import sys
from content_collector import analyze_url
from nlp_features import compute_tfidf_keywords, semantic_relevance_score


def score_color(score: float) -> str:
    if score >= 70:
        return "#2e7d32"   # green
    if score >= 40:
        return "#f9a825"   # amber
    return "#c62828"       # red


def build_html(result: dict) -> str:
    content = result["content"]
    features = result["geo_features"]
    full_text = result["full_text"]

    structure = features["structure_score"]
    readability = features["readability_flesch"]

    # Real semantic relevance via SBERT (replaces the old word-overlap placeholder)
    semantic_score = semantic_relevance_score(full_text, features["target_query"])

    # Real TF-IDF top terms (single-document fallback until a reference
    # corpus of multiple pages is wired in -- see note in nlp_features.py)
    tfidf_terms = compute_tfidf_keywords(full_text, reference_corpus=[])

    # overall GEO score = average of structure, readability, and semantic relevance
    readability_norm = max(0, min(100, readability))
    overall = round((structure + readability_norm + semantic_score) / 3, 1)

    tfidf_html = "".join(
        f"<span class='chip'>{term} <b>{score}</b></span>" for term, score in tfidf_terms[:8]
    ) or "<span class='chip'><i>Not enough text to score</i></span>"

    headings_html = "".join(
        f"<li>{h}</li>" for h in content["headings"]["h1"] + content["headings"]["h2"]
    ) or "<li><i>No headings found</i></li>"

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>GEO Report - {content['domain']}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f8; margin: 0; padding: 40px; color: #1a1a1a; }}
  .container {{ max-width: 800px; margin: 0 auto; }}
  .header {{ background: #1f3864; color: white; padding: 24px 32px; border-radius: 10px 10px 0 0; }}
  .header h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
  .header p {{ margin: 0; opacity: 0.85; font-size: 14px; word-break: break-all; }}
  .card {{ background: white; padding: 28px 32px; border-radius: 0 0 10px 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .overall {{ text-align: center; padding: 20px 0 30px 0; }}
  .overall .score {{ font-size: 56px; font-weight: 700; color: {score_color(overall)}; }}
  .overall .label {{ font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
  .metrics {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 28px; }}
  .metric {{ background: #fafafa; border: 1px solid #eee; border-radius: 8px; padding: 16px; text-align: center; }}
  .metric .value {{ font-size: 28px; font-weight: 700; }}
  .metric .name {{ font-size: 12px; color: #666; margin-top: 4px; }}
  .section {{ margin-top: 24px; }}
  .section h3 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #1f3864; border-bottom: 2px solid #eee; padding-bottom: 6px; }}
  .chip {{ display: inline-block; background: #e8eef7; color: #1f3864; padding: 4px 10px; border-radius: 14px; font-size: 13px; margin: 3px; }}
  .meta-row {{ display: flex; justify-content: space-between; font-size: 14px; padding: 6px 0; border-bottom: 1px solid #f0f0f0; }}
  .meta-row span:first-child {{ color: #666; }}
  ul {{ margin: 8px 0; padding-left: 20px; }}
  .badge-yes {{ color: #2e7d32; font-weight: 600; }}
  .badge-no {{ color: #c62828; font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>GEO Visibility Report</h1>
    <p>{content['url']}</p>
  </div>
  <div class="card">
    <div class="overall">
      <div class="score">{overall}</div>
      <div class="label">Overall GEO Score (0-100)</div>
    </div>

    <div class="metrics">
      <div class="metric">
        <div class="value" style="color:{score_color(structure)}">{structure}</div>
        <div class="name">Structure Score</div>
      </div>
      <div class="metric">
        <div class="value" style="color:{score_color(readability_norm)}">{readability}</div>
        <div class="name">Readability (Flesch)</div>
      </div>
      <div class="metric">
        <div class="value" style="color:{score_color(semantic_score)}">{semantic_score}%</div>
        <div class="name">Semantic Relevance (SBERT)</div>
      </div>
    </div>

    <div class="section">
      <h3>Target Query</h3>
      <p><b>"{features['target_query']}"</b> &nbsp; <span class="chip">{features['target_query_source'].replace('_', ' ')}</span></p>
    </div>

    <div class="section">
      <h3>Page Info</h3>
      <div class="meta-row"><span>Title</span><span>{content['title'] or '-'}</span></div>
      <div class="meta-row"><span>Author byline</span><span class="{'badge-yes' if features['has_author_byline'] else 'badge-no'}">{'Yes' if features['has_author_byline'] else 'No'}</span></div>
      <div class="meta-row"><span>Meta description</span><span class="{'badge-yes' if features['has_meta_description'] else 'badge-no'}">{'Yes' if features['has_meta_description'] else 'No'}</span></div>
      <div class="meta-row"><span>Structured data (schema.org)</span><span class="{'badge-yes' if content['has_structured_data'] else 'badge-no'}">{'Yes' if content['has_structured_data'] else 'No'}</span></div>
      <div class="meta-row"><span>Word count</span><span>{features['word_count']}</span></div>
      <div class="meta-row"><span>Paragraphs / Lists / Tables</span><span>{content['num_paragraphs']} / {content['num_list_items']} / {content['num_tables']}</span></div>
    </div>

    <div class="section">
      <h3>Headings Found</h3>
      <ul>{headings_html}</ul>
    </div>

    <div class="section">
      <h3>Top Keywords (TF-IDF)</h3>
      <div>{tfidf_html}</div>
    </div>
  </div>
</div>
</body>
</html>
"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python generate_report.py <url> ["optional keyword"]')
        sys.exit(1)

    url = sys.argv[1]
    keyword = sys.argv[2] if len(sys.argv) > 2 else ""

    result = analyze_url(url, keyword)
    html = build_html(result)

    with open("geo_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Report saved to geo_report.html -- open it in your browser.")
