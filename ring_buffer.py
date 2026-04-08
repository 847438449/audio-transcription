"""Thread-safe ring buffer for decoupling audio capture from downstream processing."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    def __init__(self, max_items: int = 512) -> None:
        self._buf: deque[T] = deque(maxlen=max_items)
        self._cv = threading.Condition()
        self._closed = False
        self._dropped = 0

    @property
    def dropped_items(self) -> int:
        return self._dropped

    def put(self, item: T) -> None:
        with self._cv:
            if self._closed:
                return
            if len(self._buf) == self._buf.maxlen:
                self._dropped += 1
            self._buf.append(item)
            self._cv.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        end = None if timeout is None else time.time() + timeout
        with self._cv:
            while not self._buf and not self._closed:
                if timeout is None:
                    self._cv.wait()
                else:
                    remaining = end - time.time()
                    if remaining <= 0:
                        return None
                    self._cv.wait(remaining)
            if self._buf:
                return self._buf.popleft()
            return None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
