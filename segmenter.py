"""Hybrid segmenter reading from ring buffer to avoid capture discontinuity."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from audio_capture import AudioFrame
from config import SegmentParams
from ring_buffer import RingBuffer


@dataclass
class AudioSegment:
    segment_id: int
    audio: np.ndarray
    sample_rate: int
    start_ts: float
    end_ts: float


class SegmenterWorker:
    def __init__(
        self,
        input_buffer: RingBuffer[AudioFrame | None],
        output_queue: queue.Queue,
        error_queue,
        cfg: SegmentParams,
        sample_rate: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.input_buffer = input_buffer
        self.output_queue = output_queue
        self.error_queue = error_queue
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.logger = logger or logging.getLogger(__name__)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frames: list[np.ndarray] = []
        self._duration = 0.0
        self._silence = 0.0
        self._start_ts = 0.0
        self._seg_id = 1

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="SegmenterThread")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self.input_buffer.get(timeout=0.2)
            if frame is None:
                self._flush(force=True)
                self._safe_put(None)
                break

            try:
                self._consume(frame)
            except Exception as exc:
                self.logger.exception("Segmenter error")
                if self.error_queue is not None:
                    self.error_queue.put(f"分段线程异常: {exc}")

    def _safe_put(self, item) -> None:
        try:
            self.output_queue.put_nowait(item)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
                self.output_queue.put_nowait(item)
            except Exception:
                pass

    def _consume(self, frame: AudioFrame) -> None:
        if frame.is_speech or self._frames:
            if not self._frames:
                self._start_ts = frame.captured_at
                self._silence = 0.0
            self._frames.append(frame.audio)
            self._duration += frame.duration_sec
            self._silence = 0.0 if frame.is_speech else self._silence + frame.duration_sec

            if self._need_cut():
                self._flush(force=False)

    def _need_cut(self) -> bool:
        if self._duration >= self.cfg.max_segment_sec:
            return True
        if self._duration >= self.cfg.min_segment_sec and self._silence >= self.cfg.silence_end_sec:
            return True
        return False

    def _flush(self, force: bool) -> None:
        if not self._frames:
            return
        if not force and self._duration < self.cfg.min_segment_sec:
            return

        audio = np.concatenate(self._frames).astype(np.float32)
        seg = AudioSegment(
            segment_id=self._seg_id,
            audio=audio,
            sample_rate=self.sample_rate,
            start_ts=self._start_ts,
            end_ts=self._start_ts + self._duration,
        )
        self._seg_id += 1
        self._safe_put(seg)

        self._frames.clear()
        self._duration = 0.0
        self._silence = 0.0
        self._start_ts = 0.0


def sliding_windows(audio: np.ndarray, sr: int, chunk_sec: float, overlap_sec: float) -> list[np.ndarray]:
    win = int(chunk_sec * sr)
    overlap = int(overlap_sec * sr)
    step = max(1, win - overlap)
    if len(audio) <= win:
        return [audio]

    result = []
    start = 0
    while start < len(audio):
        end = min(len(audio), start + win)
        result.append(audio[start:end])
        if end >= len(audio):
            break
        start += step
    return result
