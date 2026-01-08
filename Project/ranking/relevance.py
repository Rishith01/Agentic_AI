# ranking/relevance.py

from rank_bm25 import BM25Okapi
import re


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def score(query: str, papers):
    """
    Computes BM25 relevance scores.
    Mutates papers by setting relevance_score.
    """

    # Build corpus: title + abstract (title weighted higher)
    corpus = []
    for p in papers:
        text = (p.title + " ") * 3 + p.abstract
        corpus.append(tokenize(text))

    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Normalize scores to [0,1] for stability
    max_score = max(scores) if scores.any() else 1.0

    for p, s in zip(papers, scores):
        p.relevance_score = float(s / max_score)

    return papers
