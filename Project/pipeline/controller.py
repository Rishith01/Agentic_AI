# pipeline/controller.py

from agents.search_agent import SearchAgent
from agents.filter_agent import FilterAgent
from agents.presentation_agent import PresentationAgent
from a2a.message_bus import MessageBus


class PipelineController:
    def __init__(self, conversation_log):
        # Message bus now knows about the conversation log
        self.bus = MessageBus(conversation_log)

        # Agents
        self.search = SearchAgent("SearchAgent", self.bus)
        self.filter = FilterAgent("FilterAgent", self.bus)
        self.presenter = PresentationAgent("PresentationAgent", self.bus)

    def run(self, query: str):
        papers = self.search.run(query)
        papers = self.filter.run(papers, query)
        return self.presenter.run(papers)
