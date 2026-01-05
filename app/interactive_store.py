from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4


@dataclass
class SessionState:
    query: str
    chosen_table: str | None = None
    chosen_fields: list[str] = field(default_factory=list)
    chosen_metric: str | None = None
    intent_text: str | None = None
    pending_sql: str | None = None
    pending_post_risk: dict | None = None
    template: dict | None = None
    chosen_filters: list[dict] | None = None
    force_interactive: bool = False
    force_interactive: bool = False
    intent_text: str | None = None
    pending_sql: str | None = None
    pending_post_risk: dict | None = None


class InteractionStore:
    def __init__(self):
        self._lock = Lock()
        self._data: dict[str, SessionState] = {}

    def create(self, query: str) -> tuple[str, SessionState]:
        session_id = uuid4().hex
        state = SessionState(query=query)
        with self._lock:
            self._data[session_id] = state
        return session_id, state

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._data.get(session_id)

    def upsert(self, session_id: str | None, query: str) -> tuple[str, SessionState]:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                existing.query = query
                return session_id, existing
        return self.create(query)
