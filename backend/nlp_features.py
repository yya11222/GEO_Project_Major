"""
GEO System - NLP Feature Extraction Module (Sprint 3)
--------------------------------------------------------
Implements the two NLP algorithms from the report's methodology
(Sections 4.2.2 and 4.2.3):

  1. TF-IDF  - identifies which terms in the page are most
               distinctive/important relative to a small reference
               corpus of other pages.
  2. SBERT   - converts the page content and the target query into
               sentence embeddings and computes cosine similarity,
               giving a real semantic relevance score (replaces the
               placeholder word-overlap hack in content_collector.py).

Note: sentence-transformers downloads a small pretrained model
(~80MB) the first time it runs, so the first call will be slower
and needs an internet connection once.

Usage (standalone test):
    python nlp_features.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# SBERT model is loaded lazily (only when first needed) so that
# importing this file doesn't force a slow model download every time.
_sbert_model = None


def get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        # 'all-MiniLM-L6-v2' is small, fast, and good enough for this use case
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


# ---------- 1. TF-IDF ----------

def compute_tfidf_keywords(page_text: str, reference_corpus: list, top_n: int = 10) -> list:
    """
    Score keywords in `page_text` by importance relative to a small
    reference corpus of other pages (reference_corpus = list of strings).

    If no reference corpus is available yet, the page itself is used
    as a single-document corpus (falls back to plain term frequency).

    Returns: list of (term, tfidf_score) tuples, highest first.
    """
    documents = reference_corpus + [page_text] if reference_corpus else [page_text]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
    tfidf_matrix = vectorizer.fit_transform(documents)

    # The page is always the LAST document in the matrix
    page_vector = tfidf_matrix[-1]
    feature_names = vectorizer.get_feature_names_out()

    scores = page_vector.toarray()[0]
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    ranked = [(term, round(float(score), 4)) for term, score in ranked if score > 0]

    return ranked[:top_n]


# ---------- 2. SBERT semantic relevance ----------

def semantic_relevance_score(page_text: str, target_query: str) -> float:
    """
    Computes cosine similarity between the page content and the
    target query using SBERT sentence embeddings.

    Returns a 0-100 score (cosine similarity scaled up for readability).
    """
    if not target_query or not page_text.strip():
        return 0.0

    model = get_sbert_model()

    # Truncate very long pages to keep encoding fast; SBERT works on
    # sentence/short-passage level anyway, so this is a reasonable proxy.
    truncated_text = page_text[:2000]

    embeddings = model.encode([truncated_text, target_query])

    # Cosine similarity between the two embeddings
    import numpy as np
    a, b = embeddings[0], embeddings[1]
    cosine_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Cosine similarity ranges roughly -1..1; clamp and scale to 0-100
    score = max(0.0, min(1.0, cosine_sim)) * 100
    return round(score, 2)


# ---------- Combined helper for integration ----------

def extract_nlp_features(page_text: str, target_query: str, reference_corpus: list = None) -> dict:
    reference_corpus = reference_corpus or []
    return {
        "tfidf_top_terms": compute_tfidf_keywords(page_text, reference_corpus),
        "semantic_relevance_score": semantic_relevance_score(page_text, target_query),
    }


if __name__ == "__main__":
    sample_page = (
        "Generative Engine Optimization (GEO) is a new field focused on improving "
        "website visibility in AI-generated answers. Research shows that adding "
        "statistics and clear structure helps content get cited by large language "
        "models such as ChatGPT and Perplexity."
    )
    sample_query = "how to improve AI visibility for a website"

    print("TF-IDF top terms:")
    for term, score in compute_tfidf_keywords(sample_page, reference_corpus=[]):
        print(f"  {term}: {score}")

    print("\nSemantic relevance score:")
    print(semantic_relevance_score(sample_page, sample_query))
