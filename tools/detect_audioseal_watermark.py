#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from charavoice_engine.watermark.common import unpack_watermark_bits
from charavoice_engine.watermark.protocol import WatermarkRecovery, recover_watermark


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

    def to_result_dict(
        self,
        *,
        audio_path: Path,
        sample_rate: int,
        chunk_samples: int,
        scan_mode: str,
        offsets_evaluated: int,
        show_chunks: bool,
    ) -> dict[str, object]:
        result = {
            "audio_path": str(audio_path),
            "sample_rate": sample_rate,
            "chunk_samples": chunk_samples,
            "chunks_analyzed": self.chunks_analyzed,
            "dropped_tail_samples": self.dropped_tail_samples,
            "offset_samples": self.offset_samples,
            "offset_seconds": round(self.offset_samples / sample_rate, 6),
            "scan_mode": scan_mode,
            "offsets_evaluated": offsets_evaluated,
            "mode": self.recovery.mode,
            "core16": self.recovery.core16,
            "core16_hex": f"0x{self.recovery.core16:04x}",
            "extension_hex": self.recovery.extension_hex,
            "crc_ok": self.recovery.crc_ok,
            "frame_len": len(self.recovery.frame_words),
            "frame_words_hex": [f"0x{word:04x}" for word in self.recovery.frame_words],
            "start_chunk": self.recovery.start_chunk,
            "frame_repetitions": self.recovery.frame_repetitions,
            "score": round(self.recovery.score, 4),
            "detect_prob_mean": round(self.detect_prob_mean, 4),
            "detect_prob_min": round(self.detect_prob_min, 4),
            "detect_prob_max": round(self.detect_prob_max, 4),
        }
        if show_chunks:
            result["chunks"] = [
                {
                    "chunk_idx": idx,
                    "chunk_start_sample": self.offset_samples + idx * chunk_samples,
                    "word": int(word),
                    "word_hex": f"0x{word:04x}",
                    "detect_prob": round(float(prob), 4),
                }
                for idx, (word, prob) in enumerate(zip(self.words, self.detect_probs, strict=True))
            ]
        return result


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim == 2:
        # Downmix stereo/5.1/etc. for detector use.
        audio_np = audio_np.mean(axis=1)
    if audio_np.ndim != 1:
        raise ValueError(f"unsupported audio shape: {audio_np.shape}")
    return audio_np, int(sample_rate)


def _chunk_audio(audio: np.ndarray, chunk_samples: int, offset_samples: int = 0) -> tuple[np.ndarray, int]:
    if not 0 <= offset_samples < chunk_samples:
        raise ValueError(f"offset_samples must be in [0, {chunk_samples}), got {offset_samples}")

    audio_view = audio[offset_samples:]
    total_chunks = audio_view.shape[0] // chunk_samples
    if total_chunks <= 0:
        raise ValueError(f"audio is shorter than one chunk ({chunk_samples} samples) after offset={offset_samples}")
    usable_samples = total_chunks * chunk_samples
    dropped_samples = int(audio_view.shape[0] - usable_samples)
    return audio_view[:usable_samples].reshape(total_chunks, chunk_samples), dropped_samples


def _detect_chunks(
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


def _analyze_offset(
    audio_np: np.ndarray,
    *,
    offset_samples: int,
    chunk_samples: int,
    detector: torch.nn.Module,
    device: str,
    batch_size: int,
) -> OffsetDetection:
    chunks_np, dropped_tail_samples = _chunk_audio(audio_np, chunk_samples, offset_samples=offset_samples)
    words, detect_probs = _detect_chunks(detector, chunks_np, device=device, batch_size=batch_size)
    recovery = recover_watermark(words, detect_probs)
    return OffsetDetection(
        offset_samples=offset_samples,
        dropped_tail_samples=dropped_tail_samples,
        words=words,
        detect_probs=detect_probs,
        recovery=recovery,
    )


def _should_skip_scan(detection: OffsetDetection) -> bool:
    if detection.recovery.mode == "extension" and detection.recovery.crc_ok:
        return True
    return bool(detection.chunks_analyzed < 4 and detection.detect_prob_mean >= 0.5)


def _coarse_offsets(chunk_samples: int, coarse_step: int) -> list[int]:
    offsets = list(range(0, chunk_samples, coarse_step))
    if offsets[-1] != chunk_samples - 1:
        offsets.append(chunk_samples - 1)
    return sorted(set(offsets))


def _refine_offsets(
    coarse_offsets: list[int],
    *,
    chunk_samples: int,
    refine_radius: int,
    fine_step: int,
) -> list[int]:
    offsets: set[int] = set()
    for coarse_offset in coarse_offsets:
        lo = max(0, coarse_offset - refine_radius)
        hi = min(chunk_samples - 1, coarse_offset + refine_radius)
        for offset in range(lo, hi + 1, fine_step):
            offsets.add(offset)
        offsets.add(coarse_offset)
    return sorted(offsets)


def _scan_best_offset(
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
        _analyze_offset(
            audio_np,
            offset_samples=offset,
            chunk_samples=chunk_samples,
            detector=detector,
            device=device,
            batch_size=batch_size,
        )
        for offset in _coarse_offsets(chunk_samples, coarse_step)
    ]
    coarse_results.sort(key=lambda result: result.rank_key(), reverse=True)
    top_offsets = [result.offset_samples for result in coarse_results[:topk]]

    refined_results = [
        _analyze_offset(
            audio_np,
            offset_samples=offset,
            chunk_samples=chunk_samples,
            detector=detector,
            device=device,
            batch_size=batch_size,
        )
        for offset in _refine_offsets(
            top_offsets,
            chunk_samples=chunk_samples,
            refine_radius=refine_radius,
            fine_step=fine_step,
        )
    ]
    refined_results.sort(key=lambda result: result.rank_key(), reverse=True)
    return refined_results[0], len(coarse_results) + len(refined_results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect CHARAVOICE Engine AudioSeal core/extension watermark from a WAV/OGG file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="Path to the audio file to inspect")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32, help="Detector batch size")
    parser.add_argument("--chunk-samples", type=int, default=16384, help="Fixed chunk size used by the server")
    parser.add_argument(
        "--scan-offsets",
        choices=["off", "auto", "on"],
        default="auto",
        help="Whether to scan offsets within one chunk to recover from arbitrary crops or prepended audio.",
    )
    parser.add_argument(
        "--scan-coarse-step",
        type=int,
        default=128,
        help="Coarse offset scan step in samples. Used before fine refinement.",
    )
    parser.add_argument(
        "--scan-topk",
        type=int,
        default=2,
        help="Number of coarse offsets to refine.",
    )
    parser.add_argument(
        "--scan-refine-radius",
        type=int,
        default=128,
        help="Fine search radius around the best coarse offsets, in samples.",
    )
    parser.add_argument(
        "--scan-fine-step",
        type=int,
        default=1,
        help="Fine search step in samples within each refinement window.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of plain text")
    parser.add_argument("--show-chunks", action="store_true", help="Include per-chunk words and detect probabilities")
    args = parser.parse_args()

    if args.scan_coarse_step <= 0:
        raise SystemExit("--scan-coarse-step must be >= 1")
    if args.scan_topk <= 0:
        raise SystemExit("--scan-topk must be >= 1")
    if args.scan_refine_radius < 0:
        raise SystemExit("--scan-refine-radius must be >= 0")
    if args.scan_fine_step <= 0:
        raise SystemExit("--scan-fine-step must be >= 1")

    audio_np, sample_rate = _load_audio(args.audio)

    from audioseal import AudioSeal

    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    detector.eval()
    detector.to(args.device)

    base_detection = _analyze_offset(
        audio_np,
        offset_samples=0,
        chunk_samples=args.chunk_samples,
        detector=detector,
        device=args.device,
        batch_size=args.batch_size,
    )

    offsets_evaluated = 1
    if args.scan_offsets == "off" or (args.scan_offsets == "auto" and _should_skip_scan(base_detection)):
        best_detection = base_detection
    else:
        best_detection, scan_evaluated = _scan_best_offset(
            audio_np,
            chunk_samples=args.chunk_samples,
            detector=detector,
            device=args.device,
            batch_size=args.batch_size,
            coarse_step=args.scan_coarse_step,
            topk=args.scan_topk,
            refine_radius=args.scan_refine_radius,
            fine_step=args.scan_fine_step,
        )
        offsets_evaluated += scan_evaluated

    result = best_detection.to_result_dict(
        audio_path=args.audio,
        sample_rate=sample_rate,
        chunk_samples=args.chunk_samples,
        scan_mode=args.scan_offsets,
        offsets_evaluated=offsets_evaluated,
        show_chunks=args.show_chunks,
    )
    result["samples"] = int(audio_np.shape[0])

    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return

    print(f"audio: {args.audio}")
    print(f"sample_rate: {sample_rate} Hz")
    print(f"offset: {best_detection.offset_samples} samples ({best_detection.offset_samples / sample_rate:.6f}s)")
    print(f"scan_mode: {args.scan_offsets} (evaluated {offsets_evaluated} offsets)")
    print(f"chunks: {best_detection.chunks_analyzed} x {args.chunk_samples} samples")
    if best_detection.dropped_tail_samples:
        print(f"dropped_tail_samples: {best_detection.dropped_tail_samples}")
    print(f"mode: {best_detection.recovery.mode}")
    print(f"core16: {best_detection.recovery.core16} (0x{best_detection.recovery.core16:04x})")
    print(f"extension: {best_detection.recovery.extension_hex or '(none)'}")
    print(f"crc_ok: {best_detection.recovery.crc_ok}")
    print(f"frame_words: {' '.join(f'0x{word:04x}' for word in best_detection.recovery.frame_words)}")
    print(
        "detect_prob: "
        f"mean={best_detection.detect_prob_mean:.4f}, "
        f"min={best_detection.detect_prob_min:.4f}, "
        f"max={best_detection.detect_prob_max:.4f}"
    )
    print(f"frame_repetitions: {best_detection.recovery.frame_repetitions}")
    print(f"recovery_score: {best_detection.recovery.score:.4f}")


if __name__ == "__main__":
    main()
