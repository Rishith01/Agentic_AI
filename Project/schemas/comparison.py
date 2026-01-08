# schemas/comparison.py
from dataclasses import dataclass
from typing import List

@dataclass
class PaperComparison:
    common_theme: str
    key_differences: List[str]
    strengths: List[str]
    weaknesses: List[str]
    recommended_reading_order: List[str]
    recommendation_rationale: str
