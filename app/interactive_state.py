from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from cachetools import TTLCache

from .time_parser import TimeRange


@dataclass
class InteractionState:
    session_id: str
    query: str
    stage: str  # NEED_TABLE | NEED_DIMENSION | NEED_TIME | READY
    table: str | None = None
    dimensions: list[str] = field(default_factory=list)
    time_range: TimeRange | None = None
    time_column: str | None = None
    suggested_tables: list[str] = field(default_factory=list)
    pending_sql: str | None = None
    created_at: float = field(default_factory=time.time)


class InteractionStore:
    def __init__(self, ttl_seconds: int = 1800):
        self._cache: TTLCache[str, InteractionState] = TTLCache(maxsize=4096, ttl=ttl_seconds)

    def create(self, query: str, stage: str, suggested_tables: list[str]) -> InteractionState:
        sid = str(uuid.uuid4())
        state = InteractionState(
            session_id=sid,
            query=query,
            stage=stage,
            suggested_tables=suggested_tables,
        )
        self._cache[sid] = state
        return state

    def get(self, session_id: str) -> Optional[InteractionState]:
        return self._cache.get(session_id)

    def update(self, state: InteractionState) -> None:
        self._cache[state.session_id] = state
