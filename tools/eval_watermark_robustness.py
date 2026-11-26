#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch

from charavoice_engine.watermark.protocol import WatermarkRecovery, recover_watermark

CHUNK_SAMPLES = 16384
DEFAULT_EXPECTED_CORE16 = 0x1234
DEFAULT_EXPECTED_EXTENSION = "beefcafedeadbeef"
DEFAULT_SEED = 20260414


@dataclass
class OffsetDetection:
    offset_samples: int
    dropped_tail_samples: int
    words: list[int]
    detect_probs: list[float]
    recovery: WatermarkRecovery

    @property
    def chunks_analyzed(self) -> int:
        return len(self.words)

    @property
    def detect_prob_mean(self) -> float:
        return float(np.mean(self.detect_probs)) if self.detect_probs else 0.0

    @property
    def detect_prob_min(self) -> float:
        return float(np.min(self.detect_probs)) if self.detect_probs else 0.0

    @property
    def detect_prob_max(self) -> float:
        return float(np.max(self.detect_probs)) if self.detect_probs else 0.0

    def rank_key(self) -> tuple[int, float, float, int, int]:
        if self.recovery.mode == "extension" and self.recovery.crc_ok:
            tier = 3
        elif self.recovery.mode == "extension":
            tier = 2
        elif self.detect_prob_mean >= 0.5:
            tier = 1
        else:
            tier = 0
        return (
            tier,
            self.recovery.score,
            self.detect_prob_mean,
            self.recovery.frame_repetitions,
            -self.offset_samples,
        )


@dataclass
class CaseResult:
    name: str
    category: str
    control: Literal["positive", "negative"]
    description: str
    audio_path: str
    duration_seconds: float
    chunks_analyzed: int
    offset_samples: int
    offsets_evaluated: int
    mode: str
    core16_hex: str
    extension_hex: str
    crc_ok: bool
    score: float
    detect_prob_mean: float
    detect_prob_min: float
    detect_prob_max: float
    frame_repetitions: int
    extension_expected_possible: bool
    core_match: bool
    extension_match: bool | None
    target_core_false_positive: bool
    target_extension_false_positive: bool
    passed: bool


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    if audio_np.ndim != 1:
        raise ValueError(f"unsupported audio shape: {audio_np.shape}")
    return audio_np, int(sample_rate)


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate)


def chunk_audio(audio: np.ndarray, chunk_samples: int, offset_samples: int = 0) -> tuple[np.ndarray, int]:
    if not 0 <= offset_samples < chunk_samples:
        raise ValueError(f"offset_samples must be in [0, {chunk_samples}), got {offset_samples}")
    audio_view = audio[offset_samples:]
    total_chunks = audio_view.shape[0] // chunk_samples
    if total_chunks <= 0:
        raise ValueError(f"audio is shorter than one chunk ({chunk_samples} samples) after offset={offset_samples}")
    usable_samples = total_chunks * chunk_samples
    dropped_samples = int(audio_view.shape[0] - usable_samples)
    return audio_view[:usable_samples].reshape(total_chunks, chunk_samples), dropped_samples


def unpack_watermark_bits(bits: np.ndarray) -> int:
    bit_array = np.asarray(bits, dtype=np.int64)
    if bit_array.shape != (16,):
        raise ValueError(f"bits must have shape (16,), got {bit_array.shape}")
    weights = (1 << np.arange(15, -1, -1, dtype=np.int64)).astype(np.int64)
    value = int((bit_array * weights).sum())
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"decoded watermark word out of range: {value}")
    return value


def detect_chunks(
    detector: torch.nn.Module,
    chunks_np: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[list[int], list[float]]:
    all_words: list[int] = []
    all_probs: list[float] = []
    with torch.inference_mode():
        for start in range(0, chunks_np.shape[0], batch_size):
            batch_np = chunks_np[start : start + batch_size]
            batch = torch.from_numpy(batch_np[:, None, :]).to(device)
            detect_prob, message = detector.detect_watermark(batch, sample_rate=16000)
            all_probs.extend(float(x) for x in detect_prob.detach().cpu().tolist())
            message_np = message.detach().cpu().numpy().astype(np.int64, copy=False)
            all_words.extend(unpack_watermark_bits(row) for row in message_np)
    return all_words, all_probs


def analyze_offset(
    audio_np: np.ndarray,
    *,
    offset_samples: int,
    chunk_samples: int,
    detector: torch.nn.Module,
    device: str,
    batch_size: int,
) -> OffsetDetection:
    chunks_np, dropped_tail_samples = chunk_audio(audio_np, chunk_samples, offset_samples=offset_samples)
    words, detect_probs = detect_chunks(detector, chunks_np, device=device, batch_size=batch_size)
    recovery = recover_watermark(words, detect_probs)
    return OffsetDetection(
        offset_samples=offset_samples,
        dropped_tail_samples=dropped_tail_samples,
        words=words,
        detect_probs=detect_probs,
        recovery=recovery,
    )


def should_skip_scan(detection: OffsetDetection) -> bool:
    if detection.recovery.mode == "extension" and detection.recovery.crc_ok:
        return True
    return bool(detection.chunks_analyzed < 4 and detection.detect_prob_mean >= 0.5)


def coarse_offsets(chunk_samples: int, coarse_step: int) -> list[int]:
    offsets = list(range(0, chunk_samples, coarse_step))
    if offsets[-1] != chunk_samples - 1:
        offsets.append(chunk_samples - 1)
    return sorted(set(offsets))


def refine_offsets(
    coarse_offsets_: list[int],
    *,
    chunk_samples: int,
    refine_radius: int,
    fine_step: int,
) -> list[int]:
    offsets: set[int] = set()
    for coarse_offset in coarse_offsets_:
        lo = max(0, coarse_offset - refine_radius)
        hi = min(chunk_samples - 1, coarse_offset + refine_radius)
        for offset in range(lo, hi + 1, fine_step):
            offsets.add(offset)
        offsets.add(coarse_offset)
    return sorted(offsets)


def scan_best_offset(
    audio_np: np.ndarray,
    *,
    chunk_samples: int,
    detector: torch.nn.Module,
    device: str,
    batch_size: int,
    coarse_step: int,
    topk: int,
    refine_radius: int,
    fine_step: int,
) -> tuple[OffsetDetection, int]:
    coarse_results = [
        analyze_offset(
            audio_np,
            offset_samples=offset,
            chunk_samples=chunk_samples,
            detector=detector,
            device=device,
            batch_size=batch_size,
        )
        for offset in coarse_offsets(chunk_samples, coarse_step)
    ]
    coarse_results.sort(key=lambda result: result.rank_key(), reverse=True)
    top_offsets = [result.offset_samples for result in coarse_results[:topk]]
    refined_results = [
        analyze_offset(
            audio_np,
            offset_samples=offset,
            chunk_samples=chunk_samples,
            detector=detector,
            device=device,
            batch_size=batch_size,
        )
        for offset in refine_offsets(
            top_offsets,
            chunk_samples=chunk_samples,
            refine_radius=refine_radius,
            fine_step=fine_step,
        )
    ]
    refined_results.sort(key=lambda result: result.rank_key(), reverse=True)
    return refined_results[0], len(coarse_results) + len(refined_results)


def detect_file(
    audio_path: Path,
    *,
    detector: torch.nn.Module,
    device: str,
    batch_size: int,
    chunk_samples: int,
    coarse_step: int,
    topk: int,
    refine_radius: int,
    fine_step: int,
) -> tuple[OffsetDetection, int, int, float]:
    audio_np, sample_rate = load_audio(audio_path)
    base_detection = analyze_offset(
        audio_np,
        offset_samples=0,
        chunk_samples=chunk_samples,
        detector=detector,
        device=device,
        batch_size=batch_size,
    )
    offsets_evaluated = 1
    if should_skip_scan(base_detection):
        best_detection = base_detection
    else:
        best_detection, scanned = scan_best_offset(
            audio_np,
            chunk_samples=chunk_samples,
            detector=detector,
            device=device,
            batch_size=batch_size,
            coarse_step=coarse_step,
            topk=topk,
            refine_radius=refine_radius,
            fine_step=fine_step,
        )
        offsets_evaluated += scanned
    duration_seconds = float(audio_np.shape[0] / sample_rate)
    return best_detection, offsets_evaluated, sample_rate, duration_seconds


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def repeat_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[0] >= length:
        return np.asarray(audio[:length], dtype=np.float32)
    repeats = math.ceil(length / audio.shape[0])
    return np.tile(audio, repeats)[:length].astype(np.float32, copy=False)


def overlay_bgm(speech: np.ndarray, bgm: np.ndarray, rel_db: float, start_sample: int) -> np.ndarray:
    bgm_loop = repeat_to_length(bgm, speech.shape[0] + start_sample)
    bgm_slice = bgm_loop[start_sample : start_sample + speech.shape[0]]
    speech_rms = rms(speech)
    bgm_rms = rms(bgm_slice)
    target_bgm_rms = speech_rms * (10.0 ** (rel_db / 20.0))
    scale = 0.0 if bgm_rms == 0.0 else target_bgm_rms / bgm_rms
    return (speech + bgm_slice * scale).astype(np.float32, copy=False)


def add_white_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(audio.shape[0], dtype=np.float32)
    audio_rms = rms(audio)
    noise_rms = rms(noise)
    target_noise_rms = audio_rms / (10.0 ** (snr_db / 20.0))
    scale = 0.0 if noise_rms == 0.0 else target_noise_rms / noise_rms
    return (audio + noise * scale).astype(np.float32, copy=False)


def mid_crop(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    crop_samples = round(seconds * sample_rate)
    if crop_samples >= audio.shape[0]:
        return audio.copy()
    start = max(0, (audio.shape[0] - crop_samples) // 2)
    return np.asarray(audio[start : start + crop_samples], dtype=np.float32)


def prepend_silence(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    silence = np.zeros(round(seconds * sample_rate), dtype=np.float32)
    return np.concatenate([silence, audio]).astype(np.float32, copy=False)


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcode_roundtrip(input_wav: Path, output_wav: Path, codec: str, bitrate: str) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wm_eval_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        if codec == "aac":
            encoded = tmpdir_path / "tmp.m4a"
            run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(input_wav),
                    "-vn",
                    "-c:a",
                    "aac",
                    "-b:a",
                    bitrate,
                    str(encoded),
                ]
            )
        elif codec == "opus":
            encoded = tmpdir_path / "tmp.ogg"
            run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(input_wav),
                    "-vn",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    bitrate,
                    "-application",
                    "audio",
                    str(encoded),
                ]
            )
        else:
            raise ValueError(f"unsupported codec: {codec}")
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(encoded),
                "-ac",
                "1",
                "-ar",
                "44100",
                "-c:a",
                "pcm_f32le",
                str(output_wav),
            ]
        )


def resample_roundtrip(input_wav: Path, output_wav: Path, mid_rate: int) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_wav),
            "-af",
            f"aresample={mid_rate},aresample=44100",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "pcm_f32le",
            str(output_wav),
        ]
    )


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    rows = payload["cases"]
    summary = payload["summary"]
    lines = [
        "# Watermark Robustness Evaluation",
        "",
        f"- source: `{payload['source_audio']}`",
        f"- expected_core16: `{payload['expected_core16_hex']}`",
        f"- expected_extension: `{payload['expected_extension_hex']}`",
        f"- required_chunks_for_extension: `{payload['required_chunks_for_extension']}`",
        f"- chunk_samples: `{payload['chunk_samples']}`",
        "",
        "## Summary",
        "",
        f"- positive_core_pass_rate: `{summary['positive_core_pass_rate']:.3f}`",
        f"- positive_extension_pass_rate_when_possible: `{summary['positive_extension_pass_rate_when_possible']:.3f}`",
        f"- negative_target_core_false_positive_rate: `{summary['negative_target_core_false_positive_rate']:.3f}`",
        f"- negative_target_extension_false_positive_rate: `{summary['negative_target_extension_false_positive_rate']:.3f}`",
        "",
        "## Cases",
        "",
        "| case | category | control | core | ext | crc | prob | score | offset | pass |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ext_cell = "-" if row["extension_match"] is None else ("1" if row["extension_match"] else "0")
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row["category"],
                    row["control"],
                    "1" if row["core_match"] else "0",
                    ext_cell,
                    "1" if row["crc_ok"] else "0",
                    f"{row['detect_prob_mean']:.3f}",
                    f"{row['score']:.3f}",
                    str(row["offset_samples"]),
                    "1" if row["passed"] else "0",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_case_result(
    *,
    name: str,
    category: str,
    control: Literal["positive", "negative"],
    description: str,
    audio_path: Path,
    detection: OffsetDetection,
    offsets_evaluated: int,
    duration_seconds: float,
    expected_core16: int,
    expected_extension_hex: str,
    required_chunks_for_extension: int,
) -> CaseResult:
    extension_expected_possible = detection.chunks_analyzed >= required_chunks_for_extension
    core_match = detection.recovery.core16 == expected_core16
    extension_match: bool | None
    if control == "positive" and extension_expected_possible:
        extension_match = detection.recovery.crc_ok and detection.recovery.extension_hex == expected_extension_hex
    else:
        extension_match = None
    target_core_false_positive = detection.recovery.core16 == expected_core16
    target_extension_false_positive = (
        detection.recovery.crc_ok and detection.recovery.extension_hex == expected_extension_hex
    )
    if control == "positive":
        passed = core_match and (True if extension_match is None else extension_match)
    else:
        passed = (not target_core_false_positive) and (not target_extension_false_positive)
    return CaseResult(
        name=name,
        category=category,
        control=control,
        description=description,
        audio_path=str(audio_path),
        duration_seconds=round(duration_seconds, 3),
        chunks_analyzed=detection.chunks_analyzed,
        offset_samples=detection.offset_samples,
        offsets_evaluated=offsets_evaluated,
        mode=detection.recovery.mode,
        core16_hex=f"0x{detection.recovery.core16:04x}",
        extension_hex=detection.recovery.extension_hex,
        crc_ok=detection.recovery.crc_ok,
        score=round(detection.recovery.score, 4),
        detect_prob_mean=round(detection.detect_prob_mean, 4),
        detect_prob_min=round(detection.detect_prob_min, 4),
        detect_prob_max=round(detection.detect_prob_max, 4),
        frame_repetitions=detection.recovery.frame_repetitions,
        extension_expected_possible=extension_expected_possible,
        core_match=core_match,
        extension_match=extension_match,
        target_core_false_positive=target_core_false_positive,
        target_extension_false_positive=target_extension_false_positive,
        passed=passed,
    )


def summarize(results: list[CaseResult]) -> dict[str, float]:
    positive = [r for r in results if r.control == "positive"]
    negative = [r for r in results if r.control == "negative"]
    positive_with_extension = [r for r in positive if r.extension_match is not None]
    return {
        "positive_core_pass_rate": (sum(1 for r in positive if r.core_match) / len(positive) if positive else 0.0),
        "positive_extension_pass_rate_when_possible": (
            sum(1 for r in positive_with_extension if r.extension_match) / len(positive_with_extension)
            if positive_with_extension
            else 0.0
        ),
        "negative_target_core_false_positive_rate": (
            sum(1 for r in negative if r.target_core_false_positive) / len(negative) if negative else 0.0
        ),
        "negative_target_extension_false_positive_rate": (
            sum(1 for r in negative if r.target_extension_false_positive) / len(negative) if negative else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run first-pass robustness evaluation for CHARAVOICE Engine AudioSeal watermark detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-audio",
        type=Path,
        default=Path("artifacts/watermark_eval/tts_long_source.wav"),
        help="Watermarked source WAV used for positive controls.",
    )
    parser.add_argument(
        "--bgm-audio",
        type=Path,
        default=Path("artifacts/watermark_eval/bgm_subtle_10s.wav"),
        help="BGM WAV used for overlay and negative-control tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/watermark_eval/robustness"),
        help="Directory for generated evaluation artifacts.",
    )
    parser.add_argument("--expected-core16", type=lambda x: int(x, 0), default=DEFAULT_EXPECTED_CORE16)
    parser.add_argument("--expected-extension", type=str, default=DEFAULT_EXPECTED_EXTENSION)
    parser.add_argument("--chunk-samples", type=int, default=CHUNK_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan-coarse-step", type=int, default=128)
    parser.add_argument("--scan-topk", type=int, default=2)
    parser.add_argument("--scan-refine-radius", type=int, default=128)
    parser.add_argument("--scan-fine-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    source_audio, source_sr = load_audio(args.source_audio)
    bgm_audio, bgm_sr = load_audio(args.bgm_audio)
    if source_sr != 44100 or bgm_sr != 44100:
        raise SystemExit("this evaluator currently expects 44.1 kHz WAV assets")

    expected_extension_hex = args.expected_extension.lower().removeprefix("0x")
    expected_extension_bytes = bytes.fromhex(expected_extension_hex)
    required_chunks_for_extension = 3 + math.ceil(len(expected_extension_bytes) / 2)

    output_audio_dir = args.output_dir / "audio"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from audioseal import AudioSeal

    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    detector.eval()
    detector.to(args.device)

    generated_cases: list[tuple[str, str, Literal["positive", "negative"], str, Path]] = []

    save_audio(output_audio_dir / "clean_full.wav", source_audio, source_sr)
    generated_cases.append(
        (
            "clean_full",
            "baseline",
            "positive",
            "Original watermarked full-length source.",
            output_audio_dir / "clean_full.wav",
        )
    )

    for seconds in (1, 2, 3, 5, 8):
        path = output_audio_dir / f"crop_mid_{seconds}s.wav"
        save_audio(path, mid_crop(source_audio, source_sr, float(seconds)), source_sr)
        generated_cases.append(
            (
                f"crop_mid_{seconds}s",
                "crop",
                "positive",
                f"Middle {seconds}s crop from the watermarked source.",
                path,
            )
        )

    prepend_path = output_audio_dir / "prepend_silence_1s.wav"
    save_audio(prepend_path, prepend_silence(source_audio, source_sr, 1.0), source_sr)
    generated_cases.append(
        (
            "prepend_silence_1s",
            "offset",
            "positive",
            "One second of prepended silence before the watermarked source.",
            prepend_path,
        )
    )

    bgm_offset = round(1.234 * source_sr)
    for rel_db in (-24.0, -18.0, -12.0):
        path = output_audio_dir / f"bgm_{int(rel_db)}db.wav"
        save_audio(path, overlay_bgm(source_audio, bgm_audio, rel_db, bgm_offset), source_sr)
        generated_cases.append(
            (
                f"bgm_{int(rel_db)}db",
                "bgm",
                "positive",
                f"Full-length source mixed with subtle BGM at {rel_db:.0f} dB relative RMS.",
                path,
            )
        )

    for snr_db in (20.0, 10.0, 5.0):
        path = output_audio_dir / f"white_noise_snr_{int(snr_db)}db.wav"
        save_audio(path, add_white_noise(source_audio, snr_db, rng), source_sr)
        generated_cases.append(
            (
                f"white_noise_snr_{int(snr_db)}db",
                "noise",
                "positive",
                f"Full-length source with additive white noise at {snr_db:.0f} dB SNR.",
                path,
            )
        )

    aac96 = output_audio_dir / "aac_96k_roundtrip.wav"
    transcode_roundtrip(output_audio_dir / "clean_full.wav", aac96, "aac", "96k")
    generated_cases.append(("aac_96k_roundtrip", "codec", "positive", "AAC 96 kbps encode/decode roundtrip.", aac96))

    aac64 = output_audio_dir / "aac_64k_roundtrip.wav"
    transcode_roundtrip(output_audio_dir / "clean_full.wav", aac64, "aac", "64k")
    generated_cases.append(("aac_64k_roundtrip", "codec", "positive", "AAC 64 kbps encode/decode roundtrip.", aac64))

    opus24 = output_audio_dir / "opus_24k_roundtrip.wav"
    transcode_roundtrip(output_audio_dir / "clean_full.wav", opus24, "opus", "24k")
    generated_cases.append(("opus_24k_roundtrip", "codec", "positive", "Opus 24 kbps encode/decode roundtrip.", opus24))

    opus16 = output_audio_dir / "opus_16k_roundtrip.wav"
    transcode_roundtrip(output_audio_dir / "clean_full.wav", opus16, "opus", "16k")
    generated_cases.append(("opus_16k_roundtrip", "codec", "positive", "Opus 16 kbps encode/decode roundtrip.", opus16))

    resample16 = output_audio_dir / "resample_16k_roundtrip.wav"
    resample_roundtrip(output_audio_dir / "clean_full.wav", resample16, 16000)
    generated_cases.append(
        (
            "resample_16k_roundtrip",
            "resample",
            "positive",
            "44.1kHz -> 16kHz -> 44.1kHz resample roundtrip.",
            resample16,
        )
    )

    bgm_only = output_audio_dir / "bgm_only_10s.wav"
    save_audio(bgm_only, repeat_to_length(bgm_audio, 10 * bgm_sr), bgm_sr)
    generated_cases.append(
        ("bgm_only_10s", "negative", "negative", "Pure BGM only, no TTS watermark payload.", bgm_only)
    )

    noise_only = output_audio_dir / "white_noise_only_10s.wav"
    save_audio(noise_only, rng.standard_normal(10 * source_sr, dtype=np.float32) * 0.03, source_sr)
    generated_cases.append(
        ("white_noise_only_10s", "negative", "negative", "White noise only negative control.", noise_only)
    )

    silence_only = output_audio_dir / "silence_10s.wav"
    save_audio(silence_only, np.zeros(10 * source_sr, dtype=np.float32), source_sr)
    generated_cases.append(("silence_10s", "negative", "negative", "Silence only negative control.", silence_only))

    results: list[CaseResult] = []
    per_case_json_dir = args.output_dir / "detections"
    per_case_json_dir.mkdir(parents=True, exist_ok=True)

    for name, category, control, description, audio_path in generated_cases:
        detection, offsets_evaluated, _sample_rate, duration_seconds = detect_file(
            audio_path,
            detector=detector,
            device=args.device,
            batch_size=args.batch_size,
            chunk_samples=args.chunk_samples,
            coarse_step=args.scan_coarse_step,
            topk=args.scan_topk,
            refine_radius=args.scan_refine_radius,
            fine_step=args.scan_fine_step,
        )
        case_result = build_case_result(
            name=name,
            category=category,
            control=control,
            description=description,
            audio_path=audio_path,
            detection=detection,
            offsets_evaluated=offsets_evaluated,
            duration_seconds=duration_seconds,
            expected_core16=args.expected_core16,
            expected_extension_hex=expected_extension_hex,
            required_chunks_for_extension=required_chunks_for_extension,
        )
        results.append(case_result)
        write_json(per_case_json_dir / f"{name}.json", asdict(case_result))

    summary = summarize(results)
    payload = {
        "source_audio": str(args.source_audio),
        "bgm_audio": str(args.bgm_audio),
        "expected_core16": args.expected_core16,
        "expected_core16_hex": f"0x{args.expected_core16:04x}",
        "expected_extension_hex": expected_extension_hex,
        "required_chunks_for_extension": required_chunks_for_extension,
        "chunk_samples": args.chunk_samples,
        "batch_size": args.batch_size,
        "device": args.device,
        "scan": {
            "coarse_step": args.scan_coarse_step,
            "topk": args.scan_topk,
            "refine_radius": args.scan_refine_radius,
            "fine_step": args.scan_fine_step,
        },
        "seed": args.seed,
        "summary": summary,
        "cases": [asdict(result) for result in results],
    }

    write_json(args.output_dir / "summary.json", payload)
    write_markdown(args.output_dir / "summary.md", payload)


if __name__ == "__main__":
    main()
