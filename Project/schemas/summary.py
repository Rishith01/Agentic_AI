# schemas/summary.py
from dataclasses import dataclass

@dataclass
class PaperSummary:
    problem: str
    core_idea: str
    methodology: str
    key_results: str
    limitations: str
    best_use_cases: str
