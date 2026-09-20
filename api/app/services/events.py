from __future__ import annotations

from queue import Empty, Full, Queue
import threading
from typing import Any


class EventHub:
    """Thread-safe fan-out for WebSocket consumers."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Queue[dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, session_id: str) -> Queue[dict[str, Any]]:
        queue: Queue[dict[str, Any]] = Queue(maxsize=100)
        with self._lock:
            self._subscribers.setdefault(session_id, []).append(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: Queue[dict[str, Any]]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(session_id, [])
            if queue in subscribers:
                subscribers.remove(queue)
            if not subscribers:
                self._subscribers.pop(session_id, None)

    def publish(self, session_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(session_id, []))
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except Full:
                try:
                    queue.get_nowait()
                except Empty:
                    pass
                try:
                    queue.put_nowait(event)
                except Full:
                    pass
