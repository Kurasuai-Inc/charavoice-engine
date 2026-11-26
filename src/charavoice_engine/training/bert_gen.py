"""Generate phone-level BERT feature files for cleaned training lists."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

from charavoice_engine.config import HyperParameters
from charavoice_engine.constants import Languages
from charavoice_engine.models import commons
from charavoice_engine.nlp import cleaned_text_to_sequence
from charavoice_engine.nlp.fast_bert import extract_bert_feature_fast

logger = logging.getLogger(__name__)


def process_line(line: str, add_blank: bool, device: str) -> Path:
    """Generate or verify the .bert.pt file for one cleaned-list line."""

    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone_list = phones.split(" ")
    tone_list = [int(i) for i in tone.split(" ")]
    word2ph_list = [int(i) for i in word2ph.split(" ")]
    language_enum = Languages(language_str)
    phone_ids, tone_ids, language_ids = cleaned_text_to_sequence(phone_list, tone_list, language_enum)

    if add_blank:
        phone_ids = commons.intersperse(phone_ids, 0)
        tone_ids = commons.intersperse(tone_ids, 0)
        language_ids = commons.intersperse(language_ids, 0)
        for i in range(len(word2ph_list)):
            word2ph_list[i] *= 2
        word2ph_list[0] += 1

    del tone_ids, language_ids
    expected_len = len(phone_ids)
    bert_path = Path(wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt"))

    if bert_path.exists():
        bert = torch.load(bert_path)
        if bert.shape[-1] != expected_len:
            raise ValueError(f"Existing BERT length mismatch for {wav_path}: {bert.shape[-1]} != {expected_len}")
        return bert_path

    onnx_providers: tuple[str, ...]
    if device.startswith("cuda"):
        onnx_providers = ("CUDAExecutionProvider",)
        torch_device = device if torch.cuda.is_available() else "cpu"
    else:
        onnx_providers = ("CPUExecutionProvider",)
        torch_device = "cpu"

    bert = extract_bert_feature_fast(
        text,
        word2ph_list,
        language_enum,
        torch_device,
        onnx_providers=onnx_providers,
    )
    if bert.shape[-1] != expected_len:
        raise ValueError(f"BERT length mismatch for {wav_path}: {bert.shape[-1]} != {expected_len}")
    torch.save(bert, bert_path)
    return bert_path


def generate_bert_features(config_path: Path, device: str = "cuda", num_workers: int = 1) -> int:
    """Generate BERT feature files for training and validation lists."""

    hps = HyperParameters.load_from_json(config_path)
    lines: list[str] = []
    for list_path in (hps.data.training_files, hps.data.validation_files):
        with Path(list_path).open(encoding="utf-8") as f:
            lines.extend([line for line in f if line.strip()])

    if not lines:
        logger.info("No cleaned-list lines found for BERT generation.")
        return 0

    add_blank = hps.data.add_blank
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(lambda line: process_line(line, add_blank, device), lines),
                total=len(lines),
                dynamic_ncols=True,
            )
        )
    logger.info("Generated or verified %d BERT feature files.", len(lines))
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CHARAVOICE Engine bilingual BERT feature files")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    generate_bert_features(args.config, args.device, args.num_workers)


if __name__ == "__main__":
    main()
