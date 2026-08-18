from dataclasses import dataclass

@dataclass
class SearchResult:
    score: float
    payload: dict
    point_id: str | None = None
