# data_sources/arxiv_client.py
import arxiv
import time
from schemas.paper import Paper

MAX_RETRIES = 3
BACKOFF_SECONDS = 2

def search_arxiv(query: str, max_results=20):
    for attempt in range(MAX_RETRIES):
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for r in search.results():
                paper = Paper(
                    title=r.title,
                    authors=[a.name for a in r.authors],
                    abstract=r.summary,
                    year=r.published.year,
                    source_id=r.entry_id.split("/")[-1]
                )
                paper.pdf_url = r.pdf_url


            return papers

        except arxiv.HTTPError as e:
            msg = str(e)

            if "429" in msg or "Too Many Requests" in msg:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_SECONDS * (attempt + 1))
                else:
                    raise RuntimeError(
                        "arXiv rate limit hit. Please wait 1–2 minutes and retry."
                    )
            else:
                raise RuntimeError(f"arXiv error: {msg}")
