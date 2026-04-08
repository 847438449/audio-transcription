"""Two-stage transcriber with context injection, sliding windows, and GPU fallback."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Queue
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from audio_preprocess import preprocess_audio
from config import AppConfig, DecodeParams
from correction import apply_correction_layer
from hotwords import apply_hotwords
from postprocess import cleanup_text
from segmenter import AudioSegment, sliding_windows

GPU_ERROR_KEYS = ("cuda", "cudnn", "cublas", "out of memory", "device-side")


@dataclass
class TranscriptionUpdate:
    segment_id: int
    timestamp: str
    text: str
    is_final: bool


class TwoStageTranscriber:
    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        error_queue: Optional[Queue],
        cfg: AppConfig,
        hotwords: dict[str, str],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.error_queue = error_queue
        self.cfg = cfg
        self.hotwords = hotwords
        self.logger = logger or logging.getLogger(__name__)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._model: Optional[WhisperModel] = None
        self._using_cuda = False
        self._final_text_history: list[str] = []
        self._recent_segments: list[AudioSegment] = []

    def start(self) -> None:
        self._stop.clear()
        self._load_model_with_fallback()
        self._thread = threading.Thread(target=self._run, daemon=True, name="TranscriberThread")
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                seg = self.input_queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                if seg is None:
                    break
                self._handle_segment(seg)
            except Exception as exc:
                self.logger.exception("Transcriber error")
                self._push_error(f"转写异常: {exc}")
            finally:
                self.input_queue.task_done()

    def _handle_segment(self, seg: AudioSegment) -> None:
        proc = preprocess_audio(seg.audio, seg.sample_rate, self.cfg.audio)

        ts = datetime.now().strftime("[%H:%M:%S]")
        context = self._context_prompt()

        draft = self._decode_once(proc, self.cfg.realtime_decode, context)
        draft = apply_correction_layer(cleanup_text(apply_hotwords(draft, self.hotwords)))
        if draft:
            self.output_queue.put(TranscriptionUpdate(seg.segment_id, ts, draft, is_final=False))

        self._recent_segments.append(seg)
        self._trim_recent_segments(seg.end_ts)

        # quality mode: re-run with stronger decode on recent 10-20s audio
        lookback_audio = self._recent_audio()
        quality_text = self._decode_quality_with_windows(lookback_audio, context)
        quality_text = apply_correction_layer(cleanup_text(apply_hotwords(quality_text, self.hotwords)))
        if not quality_text:
            quality_text = draft

        # use tail as corrected current segment to simulate subtitle overwrite behavior
        corrected = tail_for_current_segment(quality_text, draft)
        if corrected:
            self.output_queue.put(TranscriptionUpdate(seg.segment_id, ts, corrected, is_final=True))
            self._final_text_history.append(corrected)

    def _decode_quality_with_windows(self, audio: np.ndarray, context: str) -> str:
        windows = sliding_windows(
            audio,
            self.cfg.audio.target_sample_rate,
            self.cfg.segment.chunk_length_sec,
            self.cfg.segment.overlap_seconds,
        )
        merged = ""
        for w in windows:
            part = self._decode_once(w, self.cfg.quality_decode, context)
            merged = merge_overlap_text(merged, part)
        return merged.strip()

    def _decode_once(self, audio: np.ndarray, params: DecodeParams, context: str) -> str:
        kwargs = dict(
            language=self.cfg.runtime.language,
            vad_filter=params.vad_filter,
            beam_size=params.beam_size,
            best_of=params.best_of,
            temperature=params.temperature,
            no_speech_threshold=params.no_speech_threshold,
            log_prob_threshold=params.log_prob_threshold,
            condition_on_previous_text=True,
            initial_prompt=context,
        )
        return self._transcribe_with_retry(audio, kwargs)

    def _transcribe_with_retry(self, audio: np.ndarray, kwargs: dict) -> str:
        try:
            segments, _ = self._model.transcribe(audio, **kwargs)
            return " ".join(s.text.strip() for s in segments if s.text and s.text.strip()).strip()
        except Exception as exc:
            if self._using_cuda and is_gpu_error(exc):
                self.logger.exception("GPU runtime failed, switching to CPU and retrying current audio.")
                self._push_error("GPU 推理异常，已自动切换 CPU 并重试当前段。")
                self._switch_to_cpu_model()
                segments, _ = self._model.transcribe(audio, **kwargs)
                return " ".join(s.text.strip() for s in segments if s.text and s.text.strip()).strip()
            raise

    def _load_model_with_fallback(self) -> None:
        if self.cfg.runtime.prefer_cuda:
            try:
                self._model = WhisperModel(
                    self.cfg.runtime.model_size,
                    device="cuda",
                    compute_type=self.cfg.runtime.cuda_compute_type,
                )
                self._using_cuda = True
                self.logger.info("Transcriber backend: GPU/CUDA")
                return
            except Exception as exc:
                self.logger.exception("CUDA init failed, fallback to CPU: %s", exc)
                self._push_error("CUDA 初始化失败，已自动回退 CPU。")

        self._switch_to_cpu_model()

    def _switch_to_cpu_model(self) -> None:
        self._model = WhisperModel(
            self.cfg.runtime.model_size,
            device="cpu",
            compute_type=self.cfg.runtime.cpu_compute_type,
        )
        self._using_cuda = False
        self.logger.info("Transcriber backend: CPU")

    def _context_prompt(self) -> str:
        base = (
            "これは日本語の動画音声の文字起こしです。自然な日本語として出力してください。"
            "句読点を補い、固有名詞とカタカナ語を正確に保ってください。"
        )
        if not self._final_text_history:
            return base
        ctx = " ".join(self._final_text_history[-4:])
        return f"{base}\n前文脈: {ctx[-self.cfg.runtime.context_chars:]}"

    def _recent_audio(self) -> np.ndarray:
        if not self._recent_segments:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate([s.audio for s in self._recent_segments]).astype(np.float32)

    def _trim_recent_segments(self, now_ts: float) -> None:
        lookback = self.cfg.runtime.quality_lookback_sec
        self._recent_segments = [s for s in self._recent_segments if now_ts - s.end_ts <= lookback]

    def _push_error(self, message: str) -> None:
        if self.error_queue is not None:
            self.error_queue.put(message)


def is_gpu_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in GPU_ERROR_KEYS)


def merge_overlap_text(existing: str, new: str) -> str:
    if not existing:
        return new.strip()
    if not new:
        return existing.strip()

    a = existing.split()
    b = new.split()
    max_k = min(12, len(a), len(b))
    for k in range(max_k, 0, -1):
        if a[-k:] == b[:k]:
            return " ".join(a + b[k:]).strip()

    if existing.endswith(new[:10]):
        return (existing + new[10:]).strip()
    return (existing + " " + new).strip()


def tail_for_current_segment(quality_text: str, draft_text: str) -> str:
    if not quality_text:
        return draft_text
    # use last 1-2 subtitle-friendly lines as corrected segment output
    lines = [ln.strip() for ln in quality_text.split("\n") if ln.strip()]
    if len(lines) >= 2:
        return "\n".join(lines[-2:])
    if lines:
        return lines[-1]
    return draft_text
