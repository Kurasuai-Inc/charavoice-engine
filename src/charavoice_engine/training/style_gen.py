"""Generate per-utterance style vectors for CHARAVOICE Engine training."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from charavoice_engine.config import HyperParameters

logger = logging.getLogger(__name__)


class StyleVectorGenerator:
    """Thin wrapper around pyannote's Wespeaker inference model."""

    def __init__(self, device: str = "cuda") -> None:
        from pyannote.audio import Inference, Model  # pyright: ignore[reportMissingImports]

        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        if model is None:
            raise RuntimeError("Failed to load pyannote/wespeaker-voxceleb-resnet34-LM")
        self.inference = Inference(model, window="whole")
        self.inference.to(torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu"))

    def get_style_vector(self, wav_path: str | Path) -> NDArray[Any]:
        style_vec = self.inference(_load_audio_for_pyannote(wav_path))
        if not isinstance(style_vec, np.ndarray):
            raise TypeError(f"Unexpected style vector type for {wav_path}: {type(style_vec).__name__}")
        style_vec = style_vec.astype(np.float32, copy=False)
        if np.isnan(style_vec).any():
            raise ValueError(f"NaN value found in style vector: {wav_path}")
        if style_vec.shape != (256,):
            raise ValueError(f"Unexpected style vector shape for {wav_path}: {style_vec.shape}")
        return style_vec

    def save_style_vector(self, wav_path: str | Path) -> Path:
        output = Path(f"{wav_path}.npy")
        if output.exists():
            style_vec = np.load(output)
            if style_vec.shape != (256,) or np.isnan(style_vec).any():
                raise ValueError(f"Invalid existing style vector: {output}")
            return output
        style_vec = self.get_style_vector(wav_path)
        np.save(output, style_vec)
        return output


def _load_audio_for_pyannote(wav_path: str | Path) -> dict[str, Any]:
    data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "uri": str(wav_path),
    }


def _wav_paths_from_config(config_path: Path) -> list[Path]:
    hps = HyperParameters.load_from_json(config_path)
    paths: list[Path] = []
    for list_path in (hps.data.training_files, hps.data.validation_files):
        with Path(list_path).open(encoding="utf-8") as f:
            paths.extend(Path(line.split("|", 1)[0]) for line in f if line.strip())
    return paths


def generate_style_vectors(config_path: Path, device: str = "cuda", num_workers: int = 1) -> int:
    """Generate .npy style vectors for training and validation wavs."""

    wav_paths = _wav_paths_from_config(config_path)
    if not wav_paths:
        logger.info("No cleaned-list lines found for style vector generation.")
        return 0

    generator = StyleVectorGenerator(device=device)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(generator.save_style_vector, wav_paths),
                total=len(wav_paths),
                dynamic_ncols=True,
            )
        )
    logger.info("Generated or verified %d style vector files.", len(wav_paths))
    return len(wav_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CHARAVOICE Engine style vector files")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    generate_style_vectors(args.config, args.device, args.num_workers)


if __name__ == "__main__":
    main()
