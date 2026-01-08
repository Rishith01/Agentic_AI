# agents/filter_agent.py
from agents.base_agent import BaseAgent
from a2a.message import Message
from ranking.relevance import score
from config.constants import MAX_PAPERS

class FilterAgent(BaseAgent):
    def run(self, papers, query):
        ranked = score(query, papers)
        top = ranked[:MAX_PAPERS]
        return self.bus.send(
            Message(self.name, "presentation_agent", top)
        )
