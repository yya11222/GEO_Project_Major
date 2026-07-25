"""
GEO System - Content Collection Module (Sprint 2)
---------------------------------------------------
Given a URL (and an optional target keyword), this module:
  1. Fetches and parses the webpage (Content Collection Module)
  2. Extracts raw text, metadata, and structural info
  3. Determines the target query for semantic relevance:
       - uses the user-provided keyword if given
       - otherwise falls back to the page's own title/H1
  4. Computes a first set of GEO features (readability, structure,
     keyword density) -- early piece of the GEO Feature Extraction
     Module (Sprint 3)

Usage:
    python content_collector.py <url>
    python content_collector.py <url> "target keyword phrase"
"""

import re
import sys
import json
from collections import Counter
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# ---------- 1. CONTENT COLLECTION ----------

def fetch_html(url: str, timeout: int = 10) -> str:
    """Fetch raw HTML for a URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "GEO-Research-Bot/0.1 (Himalaya College of Engineering project)"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_content(html: str, url: str) -> dict:
    """Extract text, metadata, and structural elements from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "form", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_description = meta_tag["content"].strip()

    author = ""
    author_tag = soup.find("meta", attrs={"name": "author"})
    if author_tag and author_tag.get("content"):
        author = author_tag["content"].strip()

    headings = {
        "h1": [h.get_text(strip=True) for h in soup.find_all("h1")],
        "h2": [h.get_text(strip=True) for h in soup.find_all("h2")],
        "h3": [h.get_text(strip=True) for h in soup.find_all("h3")],
    }

    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    list_items = soup.find_all("li")
    tables = soup.find_all("table")
    links = soup.find_all("a", href=True)
    json_ld_blocks = soup.find_all("script", attrs={"type": "application/ld+json"})

    full_text = " ".join(paragraphs)

    return {
        "url": url,
        "domain": urlparse(url).netloc,
        "title": title,
        "meta_description": meta_description,
        "author": author,
        "headings": headings,
        "num_paragraphs": len(paragraphs),
        "num_list_items": len(list_items),
        "num_tables": len(tables),
        "num_links": len(links),
        "has_structured_data": len(json_ld_blocks) > 0,
        "full_text": full_text,
        "word_count": len(full_text.split()),
    }


def resolve_target_query(parsed: dict, user_keyword: str = "") -> dict:
    """
    Decide what query to use for semantic relevance scoring.
    - If the user gave a keyword, use it.
    - Otherwise, fall back to the page's own H1 (or <title> if no H1).
    """
    if user_keyword and user_keyword.strip():
        return {"target_query": user_keyword.strip(), "source": "user_provided"}

    if parsed["headings"]["h1"]:
        return {"target_query": parsed["headings"]["h1"][0], "source": "derived_from_h1"}

    if parsed["title"]:
        return {"target_query": parsed["title"], "source": "derived_from_title"}

    return {"target_query": "", "source": "none_available"}


# ---------- 2. EARLY GEO FEATURE EXTRACTION ----------

def count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count, prev_was_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)

    if not sentences or not words:
        return 0.0

    syllable_count = sum(count_syllables(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllable_count / len(words)

    score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    return round(score, 2)


def keyword_density(text: str, top_n: int = 10) -> list:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is",
        "are", "was", "were", "with", "as", "by", "this", "that", "it", "be",
        "at", "from", "we", "you", "your", "our", "their", "its",
    }
    words = re.findall(r"[A-Za-z']{3,}", text.lower())
    words = [w for w in words if w not in stopwords]
    return Counter(words).most_common(top_n)


def structure_score(parsed: dict) -> float:
    score = 0
    if parsed["headings"]["h1"]:
        score += 25
    if parsed["headings"]["h2"]:
        score += 20
    if parsed["num_list_items"] > 0:
        score += 20
    if parsed["has_structured_data"]:
        score += 20
    if parsed["num_paragraphs"] >= 3:
        score += 15
    return score


def keyword_match_score(text: str, target_query: str) -> float:
    """
    Simple 0-100 score: what fraction of the target query's meaningful
    words actually appear in the page text. This is a placeholder for
    the fuller SBERT semantic relevance score (Sprint 3).
    """
    if not target_query:
        return 0.0
    query_words = set(re.findall(r"[A-Za-z']{3,}", target_query.lower()))
    text_words = set(re.findall(r"[A-Za-z']{3,}", text.lower()))
    if not query_words:
        return 0.0
    overlap = query_words & text_words
    return round(100 * len(overlap) / len(query_words), 2)


def extract_geo_features(parsed: dict, target_query_info: dict) -> dict:
    text = parsed["full_text"]
    return {
        "readability_flesch": flesch_reading_ease(text),
        "structure_score": structure_score(parsed),
        "has_author_byline": bool(parsed["author"]),
        "has_meta_description": bool(parsed["meta_description"]),
        "top_keywords": keyword_density(text),
        "target_query": target_query_info["target_query"],
        "target_query_source": target_query_info["source"],
        "keyword_match_score": keyword_match_score(text, target_query_info["target_query"]),
        "word_count": parsed["word_count"],
    }


# ---------- 3. RUN END-TO-END ----------

def analyze_url(url: str, keyword: str = "") -> dict:
    html = fetch_html(url)
    parsed = parse_content(html, url)
    target_query_info = resolve_target_query(parsed, keyword)
    features = extract_geo_features(parsed, target_query_info)
    return {
        "content": {k: v for k, v in parsed.items() if k != "full_text"},
        "geo_features": features,
        "full_text": parsed["full_text"],  # exposed for downstream NLP modules (TF-IDF/SBERT)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python content_collector.py <url> [\"optional target keyword\"]")
        sys.exit(1)

    target_url = sys.argv[1]
    target_keyword = sys.argv[2] if len(sys.argv) > 2 else ""

    result = analyze_url(target_url, target_keyword)
    print(json.dumps(result, indent=2, ensure_ascii=False))
