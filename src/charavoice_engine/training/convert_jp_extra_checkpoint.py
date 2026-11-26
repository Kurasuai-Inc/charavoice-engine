"""Prepare JP-Extra generator checkpoints for bilingual JP-Extra training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from charavoice_engine.config import HyperParameters
from charavoice_engine.models.models_bilingual_jp_extra import SynthesizerTrn
from charavoice_engine.nlp.symbols import SYMBOLS

logger = logging.getLogger(__name__)

TARGET_INITIALIZED_KEYS = {
    "emb_g.weight",
    "enc_p.style_proj.weight",
    "enc_p.style_proj.bias",
}


def read_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], int]:
    """Read a .pth/.pt/.safetensors state dict without implicit key remapping."""

    if path.suffix in {".pth", ".pt"}:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            iteration = int(checkpoint.get("iteration", 0))
            return dict(checkpoint["model"]), iteration
        if isinstance(checkpoint, dict):
            return dict(checkpoint), 0
        raise TypeError(f"Unsupported checkpoint object in {path}: {type(checkpoint)!r}")

    if path.suffix == ".safetensors":
        tensors: dict[str, torch.Tensor] = {}
        iteration = 0
        with safe_open(str(path), framework="pt", device="cpu") as f:  # type: ignore
            keys = list(f.keys())
            for key in keys:
                if key == "iteration":
                    iteration = int(f.get_tensor(key).item())
                else:
                    tensors[key] = f.get_tensor(key)
        return tensors, iteration

    raise ValueError(f"Unsupported checkpoint format: {path}")


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip DDP's module. prefix when present."""

    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def convert_generator_state_dict(
    source_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map source JP-Extra generator weights into the bilingual generator ABI."""

    source = strip_module_prefix(source_state_dict)
    converted: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    for key, target_tensor in target_state_dict.items():
        if key in source:
            tensor = source[key]
            consumed.add(key)
        elif key.startswith("enc_p.ja_bert_proj."):
            source_key = key.replace("enc_p.ja_bert_proj.", "enc_p.bert_proj.", 1)
            tensor = source[source_key]
            consumed.add(source_key)
        elif key.startswith("enc_p.en_bert_proj."):
            source_key = key.replace("enc_p.en_bert_proj.", "enc_p.bert_proj.", 1)
            tensor = source[source_key]
            consumed.add(source_key)
        elif key in TARGET_INITIALIZED_KEYS:
            tensor = target_tensor
        else:
            raise KeyError(f"Missing source tensor for target key: {key}")

        if key == "emb_g.weight" and tensor.shape != target_tensor.shape:
            if tensor.ndim != 2 or target_tensor.ndim != 2 or tensor.shape[1:] != target_tensor.shape[1:]:
                raise ValueError(
                    f"Shape mismatch for {key}: source={tuple(tensor.shape)} target={tuple(target_tensor.shape)}"
                )
            if tensor.shape[0] == 1 and target_tensor.shape[0] > 1:
                tensor = tensor.repeat(target_tensor.shape[0], 1)
            else:
                initialized = target_tensor.clone()
                rows_to_copy = min(tensor.shape[0], target_tensor.shape[0])
                initialized[:rows_to_copy] = tensor[:rows_to_copy]
                tensor = initialized
        if tensor.shape != target_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {key}: source={tuple(tensor.shape)} target={tuple(target_tensor.shape)}"
            )
        converted[key] = tensor.clone()

    allowed_unconsumed = {"enc_p.bert_proj.weight", "enc_p.bert_proj.bias"}
    unexpected = sorted(set(source) - consumed - allowed_unconsumed)
    if unexpected:
        raise KeyError(f"Unexpected source tensors not present in bilingual target: {unexpected[:20]}")
    return converted


def build_bilingual_generator(hps: HyperParameters) -> SynthesizerTrn:
    """Instantiate the target bilingual generator for validation."""

    return SynthesizerTrn(
        len(SYMBOLS),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
        use_duration_discriminator=hps.model.use_duration_discriminator,
        use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_layers_q=hps.model.n_layers_q,
        n_flow_layer=hps.model.n_flow_layer,
        n_layers_trans_flow=hps.model.n_layers_trans_flow,
        flow_share_parameter=hps.model.flow_share_parameter,
        use_transformer_flow=hps.model.use_transformer_flow,
        use_spectral_norm=hps.model.use_spectral_norm,
        gin_channels=hps.model.gin_channels,
        slm=hps.model.slm,
    )


def convert_generator_checkpoint(source_path: Path, config_path: Path, output_path: Path, half: bool = False) -> None:
    """Convert a JP-Extra generator checkpoint into strict bilingual safetensors."""

    hps = HyperParameters.load_from_json(config_path)
    target_model = build_bilingual_generator(hps)
    source_state, iteration = read_state_dict(source_path)
    converted = convert_generator_state_dict(source_state, target_model.state_dict())
    target_model.load_state_dict(converted, strict=True)
    tensors = {key: value.half() for key, value in converted.items()} if half else converted
    tensors["iteration"] = torch.LongTensor([iteration])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path)
    logger.info("Converted %s -> %s", source_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JP-Extra G checkpoint to bilingual JP-Extra")
    parser.add_argument("--source-g", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-g", type=Path, required=True)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    convert_generator_checkpoint(args.source_g, args.config, args.output_g, half=args.half)


if __name__ == "__main__":
    main()
