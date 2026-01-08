# agents/search_agent.py
from data_sources.arxiv_client import search_arxiv
from a2a.message import Message
from agents.base_agent import BaseAgent
class SearchAgent(BaseAgent):
    def run(self, query: str):
        self.bus.conversation_log.add(
            f"[SearchAgent] Received query: {query}"
        )
        self.bus.conversation_log.add(
            "[SearchAgent] Querying arXiv"
        )
        papers = search_arxiv(query)
        self.bus.conversation_log.add(
            f"[SearchAgent] Retrieved {len(papers)} papers"
        )
        return self.bus.send(
            Message(self.name, "filter_agent", papers)
        )
