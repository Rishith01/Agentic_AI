# ranking/relevance.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def score(query: str, papers):
    corpus = [p.abstract for p in papers] + [query]

    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(corpus)

    sims = cosine_similarity(X[-1], X[:-1]).flatten()

    for p, s in zip(papers, sims):
        p.relevance_score = float(s)

    return sorted(papers, key=lambda x: x.relevance_score, reverse=True)
