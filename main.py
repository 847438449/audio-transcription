"""Entrypoint wiring capture -> segmenter -> transcriber with discontinuity-safe buffering."""

from __future__ import annotations

import logging
from queue import Empty, Queue
from typing import Optional

from audio_capture import WasapiLoopbackCapture
from config import PRESETS, AppConfig
from file_writer import TranscriptFileWriter
from gui import TranscriberGUI
from hotwords import load_hotwords
from postprocess import AntiRepeatGuard
from ring_buffer import RingBuffer
from segmenter import SegmenterWorker
from transcriber import TranscriptionUpdate, TwoStageTranscriber


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")


class AppController:
    def __init__(self) -> None:
        self.logger = logging.getLogger("AppController")

        self.cfg: AppConfig = PRESETS["背景音乐场景"]

        # Capture side ring buffer: guarantees capture never blocks on ASR slowdown.
        self.frame_buffer: RingBuffer = RingBuffer(max_items=512)
        self.segment_queue: Queue = Queue(maxsize=64)
        self.update_queue: Queue = Queue(maxsize=128)
        self.error_queue: Queue = Queue()

        self.capture: Optional[WasapiLoopbackCapture] = None
        self.segmenter: Optional[SegmenterWorker] = None
        self.transcriber: Optional[TwoStageTranscriber] = None

        self.writer = TranscriptFileWriter(logging.getLogger("writer"))
        self.gui = TranscriberGUI(on_start=self.start, on_stop=self.stop)
        self.repeat_guard = AntiRepeatGuard()

        self._running = False
        self._final_map: dict[int, TranscriptionUpdate] = {}
        self._draft_map: dict[int, TranscriptionUpdate] = {}

    def start(self, txt_path: str, export_srt: bool, hotword_path: str) -> bool:
        if self._running:
            return True

        try:
            self.repeat_guard = AntiRepeatGuard()
            self._draft_map.clear()
            self._final_map.clear()
            hotwords = load_hotwords(hotword_path)
            self.writer.open(txt_path, export_srt=export_srt)

            self.capture = WasapiLoopbackCapture(
                output_buffer=self.frame_buffer,
                error_queue=self.error_queue,
                sample_rate=self.cfg.audio.target_sample_rate,
                frame_seconds=self.cfg.segment.frame_seconds,
                channels=2,
                silence_rms_threshold=self.cfg.capture.silence_rms_threshold,
                rms_smooth_alpha=self.cfg.capture.rms_smooth_alpha,
                speech_release_ratio=self.cfg.capture.speech_release_ratio,
                logger=logging.getLogger("audio_capture"),
            )
            self.segmenter = SegmenterWorker(
                input_buffer=self.frame_buffer,
                output_queue=self.segment_queue,
                error_queue=self.error_queue,
                cfg=self.cfg.segment,
                sample_rate=self.cfg.audio.target_sample_rate,
                logger=logging.getLogger("segmenter"),
            )
            self.transcriber = TwoStageTranscriber(
                input_queue=self.segment_queue,
                output_queue=self.update_queue,
                error_queue=self.error_queue,
                cfg=self.cfg,
                hotwords=hotwords,
                logger=logging.getLogger("transcriber"),
            )

            self.segmenter.start()
            self.transcriber.start()
            self.capture.start()

            self._running = True
            self.gui.set_status("状态：运行中（连续采集保护已启用）")
            self._poll()
            return True
        except Exception as exc:
            self.logger.exception("Start failed")
            self.gui.set_status(f"状态：启动失败 - {exc}")
            self.stop()
            return False

    def stop(self) -> None:
        if not self._running and not any([self.capture, self.segmenter, self.transcriber]):
            return

        try:
            if self.capture:
                self.capture.stop()
            if self.segmenter:
                self.segmenter.stop()
            if self.transcriber:
                self.transcriber.stop()
            self._drain_updates()
        finally:
            self.capture = None
            self.segmenter = None
            self.transcriber = None
            self._running = False
            self.gui.set_running_ui(False)
            self.gui.set_status("状态：已停止")

    def _poll(self) -> None:
        if not self._running:
            return

        self._drain_updates()
        self._drain_errors()
        self.gui.root.after(150, self._poll)

    def _drain_updates(self) -> None:
        try:
            while True:
                upd: TranscriptionUpdate = self.update_queue.get_nowait()
                if upd.is_final:
                    self.logger.info("update_received type=final segment_id=%d text_len=%d", upd.segment_id, len(upd.text or ""))
                    decision = self.repeat_guard.should_block(upd.text)
                    self.logger.info(
                        "anti_repeat_guard blocked_repeat=%s repeat_reason=%s normalized_text=%s",
                        decision.blocked_repeat,
                        decision.repeat_reason,
                        decision.normalized_text,
                    )
                    if decision.repeat_reason == "empty_after_normalize":
                        continue
                    if decision.blocked_repeat:
                        continue
                    draft_upd = self._draft_map.get(upd.segment_id)
                    if draft_upd and len(upd.text.strip()) < max(6, int(len(draft_upd.text.strip()) * 0.6)):
                        self.logger.info(
                            "final_shorter_than_draft use_draft segment_id=%d final_len=%d draft_len=%d",
                            upd.segment_id,
                            len(upd.text.strip()),
                            len(draft_upd.text.strip()),
                        )
                        upd = TranscriptionUpdate(upd.segment_id, upd.timestamp, draft_upd.text, is_final=True)
                    self._final_map[upd.segment_id] = upd
                    ordered = [self._final_map[k] for k in sorted(self._final_map.keys())]
                    final_content = "\n".join(f"{u.timestamp}\n{u.text}\n" for u in ordered)
                    self.gui.render_final(final_content)
                    self.writer.rewrite_all(ordered)
                else:
                    self.logger.info("update_received type=draft segment_id=%d text_len=%d", upd.segment_id, len(upd.text or ""))
                    self._draft_map[upd.segment_id] = upd
                    self.repeat_guard.update_draft(upd.text)
                    self.gui.show_draft(f"{upd.timestamp}\n{upd.text}")
        except Empty:
            pass

    def _drain_errors(self) -> None:
        try:
            while True:
                msg = self.error_queue.get_nowait()
                self.logger.error(msg)
                self.gui.set_status(f"状态：异常 - {msg}")
        except Empty:
            pass

    def run(self) -> None:
        self.gui.run()


if __name__ == "__main__":
    setup_logging()
    AppController().run()
