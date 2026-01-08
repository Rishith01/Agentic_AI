# ranking/citation_score.py
import requests
import math

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def fetch_citation_counts(papers):
    """
    Mutates papers in-place by adding citation_count
    """
    for p in papers:
        try:
            params = {
                "query": p.title,
                "limit": 1,
                "fields": "citationCount"
            }
            resp = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=5)
            data = resp.json()

            if data.get("data"):
                p.citation_count = data["data"][0].get("citationCount", 0)
            else:
                p.citation_count = 0

        except Exception:
            p.citation_count = 0

    return papers


def normalize_citations(papers):
    """
    Log-normalize citation counts to [0,1]
    """
    max_cite = max((p.citation_count for p in papers), default=1)

    for p in papers:
        p.citation_score = math.log1p(p.citation_count) / math.log1p(max_cite)

    return papers
