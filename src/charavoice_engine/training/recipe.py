"""One-command dataset preparation and training recipe for CHARAVOICE Engine."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from safetensors.torch import save_file

from charavoice_engine.constants import BASE_DIR
from charavoice_engine.training import bert_gen, default_style, preprocess_text, style_gen
from charavoice_engine.training import train as train_module
from charavoice_engine.training.convert_jp_extra_checkpoint import (
    convert_generator_checkpoint,
    read_state_dict,
    strip_module_prefix,
)
from charavoice_engine.training.dataset import DatasetReport, prepare_dataset

logger = logging.getLogger(__name__)


def run_recipe(
    dataset_dir: Path,
    model_name: str,
    assets_dir: Path,
    esd_path: Path | None = None,
    config_path: Path | None = None,
    sampling_rate: int | None = None,
    min_duration_seconds: float | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    save_epoch_interval: int | None = None,
    keep_ckpts: int | None = None,
    balance_training: bool | None = None,
    balance_fields: list[str] | None = None,
    use_wavlm_discriminator: bool | None = None,
    use_duration_discriminator: bool | None = None,
    slm_model: Path | None = None,
    fp16_run: bool = False,
    bf16_run: bool = False,
    source_jp_extra_g: Path | None = None,
    initial_g: Path | None = None,
    initial_d: Path | None = None,
    initial_dur: Path | None = None,
    initial_wd: Path | None = None,
    device: str = "cuda",
    bert_workers: int = 1,
    style_workers: int = 1,
    val_per_speaker: int = 2,
    max_val_total: int = 100,
    allow_scratch: bool = False,
    speedup: bool = False,
    skip_preprocess: bool = False,
    skip_bert: bool = False,
    skip_style: bool = False,
    skip_train: bool = False,
    skip_export: bool = False,
    skip_smoke: bool = False,
) -> DatasetReport:
    """Run the production training recipe with explicit skip switches."""

    report = prepare_dataset(
        dataset_dir,
        model_name,
        assets_dir,
        esd_path=esd_path,
        config_path=config_path,
        sampling_rate=sampling_rate,
        min_duration_seconds=min_duration_seconds,
    )
    _apply_config_overrides(
        report.config_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_epoch_interval=save_epoch_interval,
        keep_ckpts=keep_ckpts,
        balance_training=balance_training,
        balance_fields=balance_fields,
        use_wavlm_discriminator=use_wavlm_discriminator,
        use_duration_discriminator=use_duration_discriminator,
        slm_model=slm_model,
        fp16_run=fp16_run,
        bf16_run=bf16_run,
    )
    train_list = report.dataset_dir / "train.list"
    val_list = report.dataset_dir / "val.list"

    if not skip_preprocess:
        preprocess_text.preprocess(
            transcription_path=report.esd_path,
            cleaned_path=report.dataset_dir / "esd.list.cleaned",
            train_path=train_list,
            val_path=val_list,
            config_path=report.config_path,
            val_per_speaker=val_per_speaker,
            max_val_total=max_val_total,
            yomi_error="raise",
            correct_path=False,
        )
    _require_file(train_list, "train.list")
    _require_file(val_list, "val.list")

    if source_jp_extra_g is not None and initial_g is not None:
        raise ValueError("Use only one of --source-jp-extra-g or --initial-g")
    if source_jp_extra_g is not None:
        convert_generator_checkpoint(source_jp_extra_g, report.config_path, report.model_dir / "G_0.safetensors")
    elif initial_g is not None:
        _copy_generator_checkpoint(initial_g, report.config_path, report.model_dir / "G_0.safetensors")

    for source, name in (
        (initial_d, "D_0.safetensors"),
        (initial_dur, "DUR_0.safetensors"),
        (initial_wd, "WD_0.safetensors"),
    ):
        if source is not None:
            _copy_state_dict_checkpoint(source, report.model_dir / name)

    if not skip_bert:
        bert_gen.generate_bert_features(report.config_path, device=device, num_workers=bert_workers)
    if not skip_style:
        style_gen.generate_style_vectors(report.config_path, device=device, num_workers=style_workers)

    default_style.save_styles_by_dirs(
        report.dataset_dir / "raw",
        report.assets_dir,
        report.config_path,
        report.assets_dir / "config.json",
    )

    if not skip_train:
        train_module.train(
            report.config_path,
            report.model_dir,
            report.assets_dir,
            allow_scratch=allow_scratch,
            speedup=speedup,
            progress_bar=True,
        )

    if not skip_export:
        export_runtime_graphs(report.assets_dir, device=device)
    if not skip_smoke:
        validate_runtime_exports(report.assets_dir)

    return report


def export_runtime_graphs(assets_dir: Path, device: str = "cuda") -> None:
    """Export decoder, enc_p+dp, and flow ONNX graphs for the latest safetensors model."""

    from charavoice_engine.config import HyperParameters
    from charavoice_engine.convert.export_components import export_enc_p_dp_onnx, export_flow_onnx
    from charavoice_engine.convert.export_decoder import export_decoder_onnx
    from charavoice_engine.models.infer import get_net_g

    config_path = assets_dir / "config.json"
    _require_file(config_path, "inference config")
    model_path = _latest_model(assets_dir)
    hps = HyperParameters.load_from_json(config_path)
    export_decoder_onnx(
        model_path=str(model_path),
        config_path=str(config_path),
        output_path=str(assets_dir / "decoder.onnx"),
        device="cpu",
    )
    net_g = get_net_g(str(model_path), hps.version, device, hps)
    export_enc_p_dp_onnx(
        net_g=net_g,
        output_path=str(assets_dir / "enc_p_dp.onnx"),
        device=device,
        gin_channels=hps.model.gin_channels,
    )
    export_flow_onnx(
        net_g=net_g,
        output_path=str(assets_dir / "flow.onnx"),
        device=device,
        inter_channels=hps.model.inter_channels,
        gin_channels=hps.model.gin_channels,
    )


def validate_runtime_exports(assets_dir: Path) -> None:
    """Validate exported ONNX files and their public runtime ABI."""

    import onnx

    from charavoice_engine.inference.enc_dp_graph import get_enc_dp_graph_adapter

    expected_inputs = {
        "decoder.onnx": ("z", "g"),
        "enc_p_dp.onnx": get_enc_dp_graph_adapter().input_names,
        "flow.onnx": ("z_p", "y_mask", "g"),
    }
    for file_name, expected in expected_inputs.items():
        path = assets_dir / file_name
        _require_file(path, file_name)
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        actual = tuple(input_value.name for input_value in model.graph.input)
        if actual != expected:
            raise ValueError(f"{file_name} ONNX input mismatch: expected {expected}, actual {actual}")
    logger.info("Runtime ONNX smoke validation complete for %s", assets_dir)


def _apply_config_overrides(
    config_path: Path,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    save_epoch_interval: int | None,
    keep_ckpts: int | None,
    balance_training: bool | None,
    balance_fields: list[str] | None,
    use_wavlm_discriminator: bool | None,
    use_duration_discriminator: bool | None,
    slm_model: Path | None,
    fp16_run: bool,
    bf16_run: bool,
) -> None:
    from charavoice_engine.config import HyperParameters

    hps = HyperParameters.load_from_json(config_path)
    if epochs is not None:
        if epochs <= 0:
            raise ValueError("--epochs must be > 0")
        hps.train.epochs = epochs
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("--batch-size must be > 0")
        hps.train.batch_size = batch_size
    if learning_rate is not None:
        if learning_rate <= 0:
            raise ValueError("--learning-rate must be > 0")
        hps.train.learning_rate = learning_rate
    if save_epoch_interval is not None:
        if save_epoch_interval < 0:
            raise ValueError("--save-epoch-interval must be >= 0")
        hps.train.save_epoch_interval = save_epoch_interval
    if keep_ckpts is not None:
        if keep_ckpts < 0:
            raise ValueError("--keep-ckpts must be >= 0")
        hps.train.keep_ckpts = keep_ckpts
    if balance_training is not None:
        hps.train.balance_training = balance_training
    if balance_fields is not None:
        hps.train.balance_fields = balance_fields
    if use_wavlm_discriminator is not None:
        hps.model.use_wavlm_discriminator = use_wavlm_discriminator
    if use_duration_discriminator is not None:
        hps.model.use_duration_discriminator = use_duration_discriminator
    if slm_model is not None:
        if not slm_model.exists():
            raise FileNotFoundError(f"--slm-model not found: {slm_model}")
        hps.model.slm.model = _portable_repo_path(slm_model)
    if fp16_run and bf16_run:
        raise ValueError("Use only one of --fp16 or --bf16")
    if fp16_run:
        hps.train.fp16_run = True
        hps.train.bf16_run = False
    if bf16_run:
        hps.train.bf16_run = True
        hps.train.fp16_run = False
    config_path.write_text(hps.model_dump_json(indent=2), encoding="utf-8")


def _portable_repo_path(path: Path) -> str:
    """Store repo-local assets as relative paths to keep generated configs portable."""

    resolved = path.resolve()
    for base in (BASE_DIR.resolve(), Path.cwd().resolve()):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(resolved)


def _copy_state_dict_checkpoint(source: Path, output: Path) -> None:
    state_dict, iteration = read_state_dict(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    tensors = strip_module_prefix(state_dict)
    import torch

    tensors["iteration"] = torch.LongTensor([iteration])
    save_file(tensors, output)
    logger.info("Copied initial checkpoint %s -> %s", source, output)


def _copy_generator_checkpoint(source: Path, config_path: Path, output: Path) -> None:
    from charavoice_engine.config import HyperParameters
    from charavoice_engine.training.convert_jp_extra_checkpoint import (
        build_bilingual_generator,
        convert_generator_state_dict,
    )

    hps = HyperParameters.load_from_json(config_path)
    target_model = build_bilingual_generator(hps)
    state_dict, iteration = read_state_dict(source)
    converted = convert_generator_state_dict(state_dict, target_model.state_dict())
    target_model.load_state_dict(converted, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    import torch

    converted["iteration"] = torch.LongTensor([iteration])
    save_file(converted, output)
    logger.info("Prepared generator initial checkpoint %s -> %s", source, output)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} not found: {path}")


def _latest_model(assets_dir: Path) -> Path:
    models = sorted(assets_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not models:
        raise FileNotFoundError(f"No safetensors model found in {assets_dir}")
    return models[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, train, and export a bilingual JP-Extra CHARAVOICE Engine model"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--esd-path", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--sampling-rate", type=int)
    parser.add_argument("--min-duration-seconds", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--save-epoch-interval", type=int)
    parser.add_argument("--keep-ckpts", type=int)
    parser.add_argument("--no-balance-training", action="store_true")
    parser.add_argument("--balance-fields", nargs="+")
    parser.add_argument("--no-wavlm-discriminator", action="store_true")
    parser.add_argument("--duration-discriminator", action="store_true")
    parser.add_argument("--slm-model", type=Path)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--source-jp-extra-g", type=Path)
    parser.add_argument("--initial-g", type=Path)
    parser.add_argument("--initial-d", type=Path)
    parser.add_argument("--initial-dur", type=Path)
    parser.add_argument("--initial-wd", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bert-workers", type=int, default=1)
    parser.add_argument("--style-workers", type=int, default=1)
    parser.add_argument("--val-per-speaker", type=int, default=2)
    parser.add_argument("--max-val-total", type=int, default=100)
    parser.add_argument("--allow-scratch", action="store_true")
    parser.add_argument("--speedup", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-style", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_recipe(
        dataset_dir=args.dataset,
        model_name=args.model_name,
        assets_dir=args.assets_dir,
        esd_path=args.esd_path,
        config_path=args.config_path,
        sampling_rate=args.sampling_rate,
        min_duration_seconds=args.min_duration_seconds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_epoch_interval=args.save_epoch_interval,
        keep_ckpts=args.keep_ckpts,
        balance_training=False if args.no_balance_training else None,
        balance_fields=args.balance_fields,
        use_wavlm_discriminator=False if args.no_wavlm_discriminator else None,
        use_duration_discriminator=True if args.duration_discriminator else None,
        slm_model=args.slm_model,
        fp16_run=args.fp16,
        bf16_run=args.bf16,
        source_jp_extra_g=args.source_jp_extra_g,
        initial_g=args.initial_g,
        initial_d=args.initial_d,
        initial_dur=args.initial_dur,
        initial_wd=args.initial_wd,
        device=args.device,
        bert_workers=args.bert_workers,
        style_workers=args.style_workers,
        val_per_speaker=args.val_per_speaker,
        max_val_total=args.max_val_total,
        allow_scratch=args.allow_scratch,
        speedup=args.speedup,
        skip_preprocess=args.skip_preprocess,
        skip_bert=args.skip_bert,
        skip_style=args.skip_style,
        skip_train=args.skip_train,
        skip_export=args.skip_export,
        skip_smoke=args.skip_smoke,
    )


if __name__ == "__main__":
    main()
