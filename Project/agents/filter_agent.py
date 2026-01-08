# agents/filter_agent.py

from agents.base_agent import BaseAgent
from a2a.message import Message
from ranking.relevance import score as relevance_score
from ranking.citation_score import fetch_citation_counts, normalize_citations
from config.constants import MAX_PAPERS

ALPHA = 0.6   # relevance
BETA = 0.3    # citations
GAMMA = 0.1   # foundational boost

# agents/filter_agent.py (add near top)

def title_match_boost(query: str, title: str) -> float:
    """
    Strong boost if query appears contiguously in title
    Example: 'lstm' in 'Long Short-Term Memory'
    """
    q = query.lower().strip()
    t = title.lower()
    return 1.0 if q in t else 0.0


def age_boost(year: int) -> float:
    """
    Boost older foundational papers
    """
    if year <= 2000:
        return 1.0
    elif year <= 2010:
        return 0.6
    elif year <= 2015:
        return 0.3
    return 0.0

class FilterAgent(BaseAgent):
    def run(self, papers, query):

        self.bus.conversation_log.add(
            "[FilterAgent] Scoring relevance"
        )

        papers = relevance_score(query, papers)

        self.bus.conversation_log.add(
            "[FilterAgent] Fetching citation counts"
        )

        papers = fetch_citation_counts(papers)
        papers = normalize_citations(papers)

        self.bus.conversation_log.add(
            "[FilterAgent] Combining relevance + citations"
        )

        for p in papers:
            foundational = max(
                title_match_boost(query, p.title),
                age_boost(p.year)
            )

            p.final_score = (
                ALPHA * p.relevance_score +
                BETA * p.citation_score +
                GAMMA * foundational
            )

        papers.sort(key=lambda x: x.final_score, reverse=True)

        top = papers[:MAX_PAPERS]
        self.bus.conversation_log.add(
            "[FilterAgent] Applied foundational paper boosting"
        )

        self.bus.conversation_log.add(
            f"[FilterAgent] Selected top {len(top)} papers"
        )

        return self.bus.send(
            Message(self.name, "presentation_agent", top)
        )
