"""Centralized configuration and presets for the Japanese transcription system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecodeParams:
    beam_size: int
    best_of: int
    temperature: float
    vad_filter: bool
    no_speech_threshold: float
    log_prob_threshold: float


@dataclass
class SegmentParams:
    frame_seconds: float = 0.4
    min_segment_sec: float = 2.5
    max_segment_sec: float = 11.0
    silence_end_sec: float = 0.8
    chunk_length_sec: float = 7.0
    overlap_seconds: float = 1.0


@dataclass
class AudioEnhanceParams:
    target_sample_rate: int = 16000
    target_rms: float = 0.08
    hp_hz: float = 90.0
    lp_hz: float = 4200.0
    band_low_hz: float = 120.0
    band_high_hz: float = 3800.0


@dataclass
class RuntimeParams:
    model_size: str = "medium"  # switch to large-v3 for highest quality
    language: str = "ja"
    prefer_cuda: bool = True
    cuda_compute_type: str = "float16"
    cpu_compute_type: str = "int8"
    context_chars: int = 180
    quality_lookback_sec: float = 15.0


@dataclass
class AppConfig:
    runtime: RuntimeParams = field(default_factory=RuntimeParams)
    segment: SegmentParams = field(default_factory=SegmentParams)
    audio: AudioEnhanceParams = field(default_factory=AudioEnhanceParams)
    realtime_decode: DecodeParams = field(
        default_factory=lambda: DecodeParams(
            beam_size=2,
            best_of=2,
            temperature=0.2,
            vad_filter=True,
            no_speech_threshold=0.55,
            log_prob_threshold=-1.2,
        )
    )
    quality_decode: DecodeParams = field(
        default_factory=lambda: DecodeParams(
            beam_size=10,
            best_of=6,
            temperature=0.0,
            vad_filter=True,
            no_speech_threshold=0.45,
            log_prob_threshold=-1.0,
        )
    )


PRESETS: dict[str, AppConfig] = {
    "直播场景": AppConfig(),
    "培训讲话": AppConfig(
        segment=SegmentParams(min_segment_sec=2.8, max_segment_sec=12.0, silence_end_sec=0.9, chunk_length_sec=8.0, overlap_seconds=1.0),
        quality_decode=DecodeParams(beam_size=10, best_of=6, temperature=0.0, vad_filter=True, no_speech_threshold=0.45, log_prob_threshold=-1.0),
    ),
    "背景音乐场景": AppConfig(
        audio=AudioEnhanceParams(target_sample_rate=16000, target_rms=0.09, hp_hz=110.0, lp_hz=3600.0, band_low_hz=150.0, band_high_hz=3300.0),
        segment=SegmentParams(min_segment_sec=2.5, max_segment_sec=10.5, silence_end_sec=0.7, chunk_length_sec=6.5, overlap_seconds=1.1),
        quality_decode=DecodeParams(beam_size=12, best_of=8, temperature=0.0, vad_filter=True, no_speech_threshold=0.4, log_prob_threshold=-0.9),
        runtime=RuntimeParams(model_size="large-v3", language="ja", prefer_cuda=True, cuda_compute_type="float16", cpu_compute_type="int8", context_chars=220, quality_lookback_sec=18.0),
    ),
    "高精度模式": AppConfig(
        runtime=RuntimeParams(model_size="large-v3", language="ja", prefer_cuda=True, cuda_compute_type="float16", cpu_compute_type="int8", context_chars=260, quality_lookback_sec=20.0),
        quality_decode=DecodeParams(beam_size=14, best_of=8, temperature=0.0, vad_filter=True, no_speech_threshold=0.35, log_prob_threshold=-0.8),
    ),
}
