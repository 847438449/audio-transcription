"""Windows WASAPI loopback audio capture with non-blocking ring-buffer output."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundcard as sc

from ring_buffer import RingBuffer


@dataclass
class AudioFrame:
    audio: np.ndarray
    is_speech: bool
    duration_sec: float
    captured_at: float


class AudioCaptureError(RuntimeError):
    pass


class WasapiLoopbackCapture:
    """Capture thread is fully decoupled; never blocks on downstream slowdowns."""

    def __init__(
        self,
        output_buffer: RingBuffer[AudioFrame | None],
        error_queue,
        sample_rate: int = 16000,
        frame_seconds: float = 0.4,
        channels: int = 2,
        silence_rms_threshold: float = 0.008,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.output_buffer = output_buffer
        self.error_queue = error_queue
        self.sample_rate = sample_rate
        self.frame_seconds = frame_seconds
        self.channels = channels
        self.silence_rms_threshold = silence_rms_threshold
        self.logger = logger or logging.getLogger(__name__)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="AudioCaptureThread", daemon=True)
        self._thread.start()
        self.logger.info("Audio capture started (decoupled mode).")

    def stop(self, timeout: float = 4.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        frames_per_buffer = max(1, int(self.sample_rate * self.frame_seconds))
        try:
            speaker = sc.default_speaker()
            if speaker is None:
                raise AudioCaptureError("No default speaker found.")

            mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            if mic is None:
                raise AudioCaptureError("Unable to create WASAPI loopback microphone.")

            with mic.recorder(samplerate=self.sample_rate, channels=self.channels, blocksize=1024) as recorder:
                while not self._stop_event.is_set():
                    data = recorder.record(numframes=frames_per_buffer)
                    if data is None or len(data) == 0:
                        continue

                    chunk = np.asarray(data, dtype=np.float32)
                    if chunk.ndim > 1:
                        chunk = np.mean(chunk, axis=1)

                    rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
                    frame = AudioFrame(
                        audio=chunk,
                        is_speech=rms >= self.silence_rms_threshold,
                        duration_sec=len(chunk) / self.sample_rate,
                        captured_at=time.time(),
                    )
                    self.output_buffer.put(frame)

        except Exception as exc:
            self.logger.exception("Audio capture failed.")
            if self.error_queue is not None:
                self.error_queue.put(f"音频采集异常: {exc}")
        finally:
            self.output_buffer.put(None)
            self.output_buffer.close()
            dropped = self.output_buffer.dropped_items
            if dropped:
                self.logger.warning("Capture ring buffer dropped %d frames due downstream lag.", dropped)
