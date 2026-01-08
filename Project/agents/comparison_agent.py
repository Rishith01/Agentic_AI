# agents/comparison_agent.py

import os
from dotenv import load_dotenv
from groq import Groq

from agents.base_agent import BaseAgent
from schemas.comparison import PaperComparison
from a2a.message import Message

load_dotenv()


class ComparisonAgent(BaseAgent):
    def __init__(self, name, bus):
        super().__init__(name, bus)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def run(self, papers, summaries):
        """
        papers: List[Paper]
        summaries: List[PaperSummary]
        """

        self.bus.conversation_log.add(
            "[ComparisonAgent] Comparing papers"
        )

        prompt = self._build_prompt(papers, summaries)

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        text = response.choices[0].message.content
        comparison = self._parse_output(text)

        self.bus.conversation_log.add(
            "[ComparisonAgent] Comparison completed"
        )

        return self.bus.send(
            Message(self.name, "presentation_agent", comparison)
        )

    def _build_prompt(self, papers, summaries):
        blocks = []
        for i, (p, s) in enumerate(zip(papers, summaries), start=1):
            blocks.append(
                f"""
Paper {i}: {p.title}

Problem:
{s.problem}

Core Idea:
{s.core_idea}

Methodology:
{s.methodology}

Key Results:
{s.key_results}

Limitations:
{s.limitations}
"""
            )

        return f"""
You are an academic research assistant.

Given the following paper summaries, perform a comparison.

Tasks:
1. Identify the common research theme
2. Highlight key methodological differences
3. Identify strengths and weaknesses
4. Recommend a reading order
5. Explain the recommendation

Return the result in this exact format:

Common Theme:
<text>

Key Differences:
- item
- item

Strengths:
- item
- item

Weaknesses:
- item
- item

Recommended Reading Order:
- Paper title
- Paper title

Recommendation Rationale:
<text>

Summaries:
{''.join(blocks)}
"""

    def _parse_output(self, text):
        def extract(label):
            try:
                return text.split(label + ":\n")[1].split("\n\n")[0].strip()
            except Exception:
                return ""

        return PaperComparison(
            common_theme=extract("Common Theme"),
            key_differences=extract("Key Differences").split("\n- "),
            strengths=extract("Strengths").split("\n- "),
            weaknesses=extract("Weaknesses").split("\n- "),
            recommended_reading_order=extract("Recommended Reading Order").split("\n- "),
            recommendation_rationale=extract("Recommendation Rationale"),
        )
