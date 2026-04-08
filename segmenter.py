"""Hybrid segmenter reading from ring buffer to avoid capture discontinuity."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
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
        self._frames_seen = 0
        self._speech_frames_seen = 0
        self._last_debug_ts = time.monotonic()
        self._pre_roll_frames: deque[np.ndarray] = deque()
        self._pre_roll_duration = 0.0

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
            frame = self.input_buffer.get(timeout=None)
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
        self._frames_seen += 1
        if frame.is_speech:
            self._speech_frames_seen += 1
        self._debug_capture_flow()

        if frame.is_speech or self._frames:
            if not self._frames:
                pre_roll_sec = self._pre_roll_duration
                if self._pre_roll_frames:
                    self._frames.extend(self._pre_roll_frames)
                    self._duration += pre_roll_sec
                    self._pre_roll_frames.clear()
                    self._pre_roll_duration = 0.0
                self._start_ts = frame.captured_at - pre_roll_sec
                self._silence = 0.0
            self._frames.append(frame.audio)
            self._duration += frame.duration_sec
            self._silence = 0.0 if frame.is_speech else self._silence + frame.duration_sec

            if self._need_cut():
                self._flush(force=False)
        else:
            self._push_pre_roll(frame.audio, frame.duration_sec)

    def _debug_capture_flow(self) -> None:
        now = time.monotonic()
        if now - self._last_debug_ts < 2.0:
            return
        ratio = 0.0 if self._frames_seen == 0 else self._speech_frames_seen / self._frames_seen
        self.logger.info(
            "segmenter_flow frames=%d speech_frames=%d speech_ratio=%.3f buffering=%s duration=%.2fs silence=%.2fs",
            self._frames_seen,
            self._speech_frames_seen,
            ratio,
            bool(self._frames),
            self._duration,
            self._silence,
        )
        self._last_debug_ts = now

    def _need_cut(self) -> bool:
        if self._duration >= self.cfg.max_segment_sec:
            return True
        if self._duration >= self.cfg.min_segment_sec and self._silence >= self.cfg.silence_end_sec:
            return True
        return False

    def _push_pre_roll(self, audio: np.ndarray, duration: float) -> None:
        target = max(0.0, self.cfg.pre_speech_sec)
        if target <= 0:
            self._pre_roll_frames.clear()
            self._pre_roll_duration = 0.0
            return
        self._pre_roll_frames.append(audio)
        self._pre_roll_duration += duration
        while self._pre_roll_frames and self._pre_roll_duration > target:
            popped = self._pre_roll_frames.popleft()
            self._pre_roll_duration -= len(popped) / self.sample_rate

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
        self.logger.info(
            "segment_emitted id=%d samples=%d duration=%.2fs force=%s",
            seg.segment_id,
            len(seg.audio),
            self._duration,
            force,
        )

        hard_cut = self._duration >= self.cfg.max_segment_sec
        if not force and hard_cut and self.cfg.max_cut_carryover_sec > 0:
            carry_samples = int(self.cfg.max_cut_carryover_sec * self.sample_rate)
            carry_samples = min(carry_samples, len(audio))
            carry = audio[-carry_samples:].astype(np.float32)
            self._frames = [carry]
            self._duration = len(carry) / self.sample_rate
            self._silence = 0.0
            self._start_ts = seg.end_ts - self._duration
            self._pre_roll_frames.clear()
            self._pre_roll_duration = 0.0
            return

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
