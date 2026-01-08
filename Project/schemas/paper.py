# schemas/paper.py
from dataclasses import dataclass
from typing import List

@dataclass
class Paper:
    title: str
    authors: List[str]
    abstract: str
    year: int
    source_id: str   # arXiv id
    relevance_score: float = 0.0
