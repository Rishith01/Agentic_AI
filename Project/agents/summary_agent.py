# agents/summary_agent.py

import os
from groq import Groq

from agents.base_agent import BaseAgent
from schemas.summary import PaperSummary
from a2a.message import Message
from config.prompts import SUMMARY_PROMPT


class SummaryAgent(BaseAgent):
    def __init__(self, name, bus):
        super().__init__(name, bus)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def run(self, paper, pdf_text: str):

        self.bus.conversation_log.add(
            f"[SummaryAgent] Generating LLM summary for: {paper.title}"
        )

        prompt = SUMMARY_PROMPT.format(
            content=pdf_text[:12000]  # safety limit
        )

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        summary_text = response.choices[0].message.content

        summary = self._parse_summary(summary_text)

        self.bus.conversation_log.add(
            "[SummaryAgent] LLM summary completed"
        )

        return self.bus.send(
            Message(self.name, "presentation_agent", summary)
        )

    def _parse_summary(self, text: str) -> PaperSummary:
        """
        Very simple parser based on prompt structure.
        """
        def extract(label):
            try:
                return text.split(label + ":\n")[1].split("\n\n")[0].strip()
            except Exception:
                return "Not explicitly stated"

        return PaperSummary(
            problem=extract("Problem"),
            core_idea=extract("Core Idea"),
            methodology=extract("Methodology"),
            key_results=extract("Key Results"),
            limitations=extract("Limitations"),
            best_use_cases=extract("Best Use Cases")
        )
