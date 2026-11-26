"""Dataset layout validation and config generation for CHARAVOICE Engine training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from charavoice_engine.config import HyperParameters, HyperParametersData
from charavoice_engine.constants import DEFAULT_STYLE, Languages

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".WAV"}


@dataclass(frozen=True)
class DatasetItem:
    wav_path: Path
    style: str
    speaker: str
    language: Languages
    text: str


@dataclass(frozen=True)
class DatasetReport:
    dataset_dir: Path
    total_items: int
    target_sampling_rate: int
    short_audio_threshold_seconds: float
    total_duration_seconds: float
    min_duration_seconds: float
    max_duration_seconds: float
    sample_rates: dict[str, int]
    speakers: list[str]
    styles: list[str]
    languages: list[str]
    counts_by_speaker: dict[str, int]
    counts_by_style: dict[str, int]
    counts_by_language: dict[str, int]
    esd_path: Path
    config_path: Path
    model_dir: Path
    assets_dir: Path


def discover_items_from_sidecars(dataset_dir: Path) -> list[DatasetItem]:
    """Discover raw wavs and read same-stem .txt transcripts."""

    raw_dir = dataset_dir / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw dataset directory not found: {raw_dir}")
    items: list[DatasetItem] = []
    for wav_path in sorted(raw_dir.rglob("*")):
        if wav_path.suffix not in SUPPORTED_AUDIO_SUFFIXES:
            continue
        style, speaker, language = _layout_parts(dataset_dir, wav_path)
        text_path = wav_path.with_suffix(".txt")
        if not text_path.is_file():
            raise FileNotFoundError(f"Transcript sidecar not found for {wav_path}: {text_path}")
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Transcript is empty: {text_path}")
        items.append(DatasetItem(wav_path.resolve(), style, speaker, language, text))
    if not items:
        raise ValueError(f"No wav files found under {raw_dir}")
    return items


def read_esd_list(dataset_dir: Path, esd_path: Path) -> list[DatasetItem]:
    """Read a 4-column esd.list and validate paths against raw layout."""

    items: list[DatasetItem] = []
    with esd_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("|")
            if len(parts) != 4:
                raise ValueError(f"{esd_path}:{line_no}: expected 4 columns wav|speaker|language|text")
            raw_wav, speaker, language_raw, text = parts
            language = Languages(language_raw)
            wav_path = _resolve_wav_path(dataset_dir, raw_wav)
            style, layout_speaker, layout_language = _layout_parts(dataset_dir, wav_path)
            if layout_speaker != speaker:
                raise ValueError(f"{esd_path}:{line_no}: speaker mismatch: list={speaker!r} layout={layout_speaker!r}")
            if layout_language != language:
                raise ValueError(
                    f"{esd_path}:{line_no}: language mismatch: list={language.value!r} layout={layout_language.value!r}"
                )
            if not text:
                raise ValueError(f"{esd_path}:{line_no}: text is empty")
            items.append(DatasetItem(wav_path.resolve(), style, speaker, language, text))
    if not items:
        raise ValueError(f"No items found in {esd_path}")
    return items


def write_esd_list(items: list[DatasetItem], output_path: Path) -> None:
    """Write an absolute-path esd.list."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item.wav_path}|{item.speaker}|{item.language.value}|{item.text}\n")


def prepare_dataset(
    dataset_dir: Path,
    model_name: str,
    assets_dir: Path,
    esd_path: Path | None = None,
    config_path: Path | None = None,
    sampling_rate: int | None = None,
    min_duration_seconds: float | None = None,
) -> DatasetReport:
    """Validate dataset layout, write esd.list when needed, and create config.json."""

    dataset_dir = dataset_dir.resolve()
    esd_path = (esd_path or dataset_dir / "esd.list").resolve()
    config_path = (config_path or dataset_dir / "config.json").resolve()
    model_dir = dataset_dir / "models"
    assets_dir = assets_dir.resolve()

    items = read_esd_list(dataset_dir, esd_path) if esd_path.is_file() else discover_items_from_sidecars(dataset_dir)

    _validate_no_duplicates(items)
    write_esd_list(items, esd_path)
    speakers = sorted({item.speaker for item in items})
    raw_styles = sorted({item.style for item in items})
    styles = [DEFAULT_STYLE] + [style for style in raw_styles if style != DEFAULT_STYLE]
    languages = sorted({item.language.value for item in items})

    data_config: dict[str, Any] = {
        "training_files": str(dataset_dir / "train.list"),
        "validation_files": str(dataset_dir / "val.list"),
        "spk2id": {speaker: idx for idx, speaker in enumerate(speakers)},
        "n_speakers": len(speakers),
        "style2id": {style: idx for idx, style in enumerate(styles)},
        "num_styles": len(styles),
    }
    if sampling_rate is not None:
        data_config["sampling_rate"] = sampling_rate
    hps = HyperParameters(
        model_name=model_name,
        data=HyperParametersData.model_validate(data_config),
    )
    short_audio_threshold_seconds = (
        min_duration_seconds if min_duration_seconds is not None else hps.train.segment_size / hps.data.sampling_rate
    )
    audio_stats = _validate_audio_files(
        items,
        target_sampling_rate=hps.data.sampling_rate,
        min_duration_seconds=short_audio_threshold_seconds,
    )
    config_path.write_text(hps.model_dump_json(indent=2), encoding="utf-8")
    model_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    report = DatasetReport(
        dataset_dir=dataset_dir,
        total_items=len(items),
        target_sampling_rate=hps.data.sampling_rate,
        short_audio_threshold_seconds=short_audio_threshold_seconds,
        total_duration_seconds=audio_stats["total_duration_seconds"],
        min_duration_seconds=audio_stats["min_duration_seconds"],
        max_duration_seconds=audio_stats["max_duration_seconds"],
        sample_rates=audio_stats["sample_rates"],
        speakers=speakers,
        styles=styles,
        languages=languages,
        counts_by_speaker=_counts(item.speaker for item in items),
        counts_by_style=_counts(item.style for item in items),
        counts_by_language=_counts(item.language.value for item in items),
        esd_path=esd_path,
        config_path=config_path,
        model_dir=model_dir,
        assets_dir=assets_dir,
    )
    _write_report(report, dataset_dir / "dataset_report.json")
    return report


def _resolve_wav_path(dataset_dir: Path, raw_wav: str) -> Path:
    path = Path(raw_wav)
    if path.is_absolute():
        wav_path = path
    elif (dataset_dir / path).is_file():
        wav_path = dataset_dir / path
    else:
        wav_path = dataset_dir / "raw" / path
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio not found: {raw_wav}")
    return wav_path.resolve()


def _layout_parts(dataset_dir: Path, wav_path: Path) -> tuple[str, str, Languages]:
    raw_dir = dataset_dir.resolve() / "raw"
    try:
        rel = wav_path.resolve().relative_to(raw_dir)
    except ValueError as exc:
        raise ValueError(f"Audio must be under {raw_dir}: {wav_path}") from exc
    if len(rel.parts) < 4:
        raise ValueError(f"Audio path must be raw/{{style}}/{{speaker}}/{{language}}/*.wav: {wav_path}")
    style, speaker, language_raw = rel.parts[:3]
    language = Languages(language_raw)
    return style, speaker, language


def _validate_no_duplicates(items: list[DatasetItem]) -> None:
    seen: set[Path] = set()
    duplicates: list[Path] = []
    for item in items:
        if item.wav_path in seen:
            duplicates.append(item.wav_path)
        seen.add(item.wav_path)
    if duplicates:
        raise ValueError(f"Duplicate wav paths in dataset: {duplicates[:20]}")


def _validate_audio_files(
    items: list[DatasetItem],
    target_sampling_rate: int,
    min_duration_seconds: float,
) -> dict:
    durations: list[float] = []
    sample_rates: dict[str, int] = {}
    sample_rate_errors: list[str] = []
    short_errors: list[str] = []
    unreadable_errors: list[str] = []

    for item in items:
        try:
            info = sf.info(str(item.wav_path))
        except Exception as exc:
            unreadable_errors.append(f"{item.wav_path}: {exc}")
            continue
        if info.frames <= 0 or info.samplerate <= 0:
            unreadable_errors.append(f"{item.wav_path}: invalid frames={info.frames} samplerate={info.samplerate}")
            continue
        sample_rates[str(info.samplerate)] = sample_rates.get(str(info.samplerate), 0) + 1
        if int(info.samplerate) != target_sampling_rate:
            sample_rate_errors.append(f"{item.wav_path}: {info.samplerate} Hz")
        duration = float(info.frames) / float(info.samplerate)
        durations.append(duration)
        if duration < min_duration_seconds:
            short_errors.append(f"{item.wav_path}: {duration:.3f}s")

    if unreadable_errors:
        raise ValueError(f"Unreadable audio files: {unreadable_errors[:20]}")
    if sample_rate_errors:
        raise ValueError(f"Audio sampling rate must be {target_sampling_rate} Hz: {sample_rate_errors[:20]}")
    if short_errors:
        raise ValueError(
            f"Audio shorter than {min_duration_seconds:.3f}s is not valid for training: {short_errors[:20]}"
        )

    return {
        "total_duration_seconds": float(sum(durations)),
        "min_duration_seconds": float(min(durations)),
        "max_duration_seconds": float(max(durations)),
        "sample_rates": sample_rates,
    }


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _write_report(report: DatasetReport, output_path: Path) -> None:
    data = asdict(report)
    for key in ("dataset_dir", "esd_path", "config_path", "model_dir", "assets_dir"):
        data[key] = str(data[key])
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
