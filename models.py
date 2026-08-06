from dataclasses import dataclass


@dataclass
class SearchResult:

    score: float

    payload: dict