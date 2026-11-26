import argparse
import json
import logging
from pathlib import Path

import numpy as np

from charavoice_engine.constants import DEFAULT_STYLE

logger = logging.getLogger(__name__)


def save_neutral_vector(
    wav_dir: Path | str,
    output_dir: Path | str,
    config_path: Path | str,
    config_output_path: Path | str,
):
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_output_path = Path(config_output_path)
    config_output_path.parent.mkdir(parents=True, exist_ok=True)
    embs = []
    for file in wav_dir.rglob("*.npy"):
        xvec = _load_style_vector(file)
        embs.append(np.expand_dims(xvec, axis=0))
    if not embs:
        raise FileNotFoundError(f"No style vector .npy files found under {wav_dir}")

    x = np.concatenate(embs, axis=0)  # (N, 256)
    mean = np.mean(x, axis=0)  # (256,)
    only_mean = np.stack([mean])  # (1, 256)
    np.save(output_dir / "style_vectors.npy", only_mean)
    logger.info(f"Saved mean style vector to {output_dir}")

    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = 1
    json_dict["data"]["style2id"] = {DEFAULT_STYLE: 0}
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved style config to {config_output_path}")


def save_styles_by_dirs(
    wav_dir: Path | str,
    output_dir: Path | str,
    config_path: Path | str,
    config_output_path: Path | str,
):
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    config_output_path = Path(config_output_path)
    config_output_path.parent.mkdir(parents=True, exist_ok=True)

    subdirs = [d for d in wav_dir.iterdir() if d.is_dir()]
    subdirs.sort()
    if len(subdirs) in (0, 1):
        logger.info(
            f"At least 2 subdirectories are required for generating style vectors with respect to them, found {len(subdirs)}."
        )
        logger.info("Generating only neutral style vector instead.")
        save_neutral_vector(wav_dir, output_dir, config_path, config_output_path)
        return

    # First get mean of all for Neutral
    embs = []
    for file in wav_dir.rglob("*.npy"):
        xvec = _load_style_vector(file)
        embs.append(np.expand_dims(xvec, axis=0))
    if not embs:
        raise FileNotFoundError(f"No style vector .npy files found under {wav_dir}")
    x = np.concatenate(embs, axis=0)  # (N, 256)
    mean = np.mean(x, axis=0)  # (256,)
    style_vectors = [mean]

    names = [DEFAULT_STYLE]
    for style_dir in subdirs:
        if style_dir.name == DEFAULT_STYLE:
            continue
        npy_files = list(style_dir.rglob("*.npy"))
        if not npy_files:
            if any(style_dir.rglob("*.wav")) or any(style_dir.rglob("*.WAV")):
                raise FileNotFoundError(f"Style directory has wav files but no style vectors: {style_dir}")
            continue
        embs = []
        for file in npy_files:
            xvec = _load_style_vector(file)
            embs.append(np.expand_dims(xvec, axis=0))

        x = np.concatenate(embs, axis=0)  # (N, 256)
        mean = np.mean(x, axis=0)  # (256,)
        style_vectors.append(mean)
        names.append(style_dir.name)

    # Stack them to make (num_styles, 256)
    style_vectors_npy = np.stack(style_vectors, axis=0)
    np.save(output_dir / "style_vectors.npy", style_vectors_npy)
    logger.info(f"Saved style vectors to {output_dir / 'style_vectors.npy'}")

    # Save style2id config to json
    style2id = {name: i for i, name in enumerate(names)}
    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(names)
    json_dict["data"]["style2id"] = style2id
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved style config to {config_output_path}")


def _load_style_vector(path: Path) -> np.ndarray:
    xvec = np.load(path)
    if xvec.shape != (256,) or np.isnan(xvec).any():
        raise ValueError(f"Invalid style vector: {path}")
    return xvec


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CHARAVOICE Engine style_vectors.npy and style config")
    parser.add_argument(
        "--wav-dir", type=Path, required=True, help="Dataset raw directory containing style subdirectories"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory where style_vectors.npy will be written"
    )
    parser.add_argument("--config", type=Path, required=True, help="Input training config.json")
    parser.add_argument("--config-output", type=Path, required=True, help="Output inference config.json")
    parser.add_argument("--neutral-only", action="store_true", help="Generate only the Neutral mean style vector")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if args.neutral_only:
        save_neutral_vector(args.wav_dir, args.output_dir, args.config, args.config_output)
    else:
        save_styles_by_dirs(args.wav_dir, args.output_dir, args.config, args.config_output)


if __name__ == "__main__":
    main()
