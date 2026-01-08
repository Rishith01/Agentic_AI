# schemas/paper.py
from dataclasses import dataclass
from typing import List

@dataclass
class Paper:
    title: str
    authors: List[str]
    abstract: str
    year: int
    source_id: str
    relevance_score: float = 0.0

    # Phase 2 fields
    citation_count: int = 0
    citation_score: float = 0.0
    final_score: float = 0.0
