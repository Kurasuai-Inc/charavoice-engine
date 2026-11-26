"""Bilingual JP-Extra training loop for CHARAVOICE Engine."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import platform
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from charavoice_engine.config import HyperParameters
from charavoice_engine.constants import BASE_DIR
from charavoice_engine.models import commons, utils
from charavoice_engine.models.models_bilingual_jp_extra import (
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    WavLMDiscriminator,
)
from charavoice_engine.nlp.symbols import SYMBOLS
from charavoice_engine.training.data_utils import (
    DistributedBucketSampler,
    DistributedWeightedSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from charavoice_engine.training.losses import WavLMLoss, discriminator_loss, feature_loss, generator_loss, kl_loss
from charavoice_engine.training.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logger = logging.getLogger(__name__)
GLOBAL_STEP = 0


def _setup_runtime() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("medium")
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def _distributed_context() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "gloo" if platform.system() == "Windows" or not torch.cuda.is_available() else "nccl"
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return distributed, rank, local_rank, world_size, device


def _build_generator(hps: HyperParameters, device: torch.device) -> SynthesizerTrn:
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
    ).to(device)


def _apply_freezes(net_g: SynthesizerTrn, hps: HyperParameters) -> None:
    if hps.train.freeze_JP_bert:
        logger.info("Freezing JP BERT projection")
        for param in net_g.enc_p.ja_bert_proj.parameters():
            param.requires_grad = False
    if hps.train.freeze_EN_bert:
        logger.info("Freezing EN BERT projection")
        for param in net_g.enc_p.en_bert_proj.parameters():
            param.requires_grad = False
    if hps.train.freeze_style:
        logger.info("Freezing style projection")
        for param in net_g.enc_p.style_proj.parameters():
            param.requires_grad = False
    if hps.train.freeze_decoder:
        logger.info("Freezing decoder")
        for param in net_g.dec.parameters():
            param.requires_grad = False


def _maybe_wrap_ddp(module: torch.nn.Module | None, distributed: bool, device: torch.device) -> torch.nn.Module | None:
    if module is None or not distributed:
        return module
    if device.type == "cuda":
        return DDP(module, device_ids=[device.index])
    return DDP(module)


def _load_initial_or_resume(
    model_dir: Path,
    hps: HyperParameters,
    net_g: torch.nn.Module,
    net_d: torch.nn.Module,
    net_dur_disc: torch.nn.Module | None,
    net_wd: torch.nn.Module | None,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    optim_dur_disc: torch.optim.Optimizer | None,
    optim_wd: torch.optim.Optimizer | None,
    allow_scratch: bool,
    device: torch.device,
) -> tuple[int, int]:
    """Load strict resume checkpoints or strict initial safetensors."""

    if utils.checkpoints.is_resuming(model_dir):
        _, loaded_optim_g, g_lr, epoch = utils.checkpoints.load_checkpoint(
            utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth"),
            net_g,
            optim_g,
            skip_optimizer=hps.train.skip_optimizer,
            device=device,
        )
        if loaded_optim_g is None:
            raise RuntimeError("Generator optimizer was not restored from checkpoint")
        optim_g = loaded_optim_g
        _, loaded_optim_d, d_lr, epoch_d = utils.checkpoints.load_checkpoint(
            utils.checkpoints.get_latest_checkpoint_path(model_dir, "D_*.pth"),
            net_d,
            optim_d,
            skip_optimizer=hps.train.skip_optimizer,
            device=device,
        )
        if loaded_optim_d is None:
            raise RuntimeError("Discriminator optimizer was not restored from checkpoint")
        optim_d = loaded_optim_d
        if epoch_d != epoch:
            raise ValueError(f"Generator/discriminator epoch mismatch: G={epoch}, D={epoch_d}")
        optim_g.param_groups[0].setdefault("initial_lr", g_lr)
        optim_d.param_groups[0].setdefault("initial_lr", d_lr)
        if net_dur_disc is not None:
            if optim_dur_disc is None:
                raise ValueError("optim_dur_disc is required")
            _, loaded_optim_dur, dur_lr, epoch_dur = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer,
                device=device,
            )
            if loaded_optim_dur is None:
                raise RuntimeError("Duration discriminator optimizer was not restored from checkpoint")
            optim_dur_disc = loaded_optim_dur
            if epoch_dur != epoch:
                raise ValueError(f"Duration discriminator epoch mismatch: {epoch_dur} != {epoch}")
            optim_dur_disc.param_groups[0].setdefault("initial_lr", dur_lr)
        if net_wd is not None:
            if optim_wd is None:
                raise ValueError("optim_wd is required")
            _, loaded_optim_wd, wd_lr, epoch_wd = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "WD_*.pth"),
                net_wd,
                optim_wd,
                skip_optimizer=hps.train.skip_optimizer,
                device=device,
            )
            if loaded_optim_wd is None:
                raise RuntimeError("WavLM discriminator optimizer was not restored from checkpoint")
            optim_wd = loaded_optim_wd
            if epoch_wd != epoch:
                raise ValueError(f"WavLM discriminator epoch mismatch: {epoch_wd} != {epoch}")
            optim_wd.param_groups[0].setdefault("initial_lr", wd_lr)
        step = utils.checkpoints.get_steps(utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth"))
        next_epoch = epoch + 1
        logger.info("Resumed training from epoch=%d step=%d; next epoch=%d", epoch, step, next_epoch)
        return max(next_epoch, 1), step

    required = [("G_0.safetensors", net_g), ("D_0.safetensors", net_d)]
    if net_dur_disc is not None:
        required.append(("DUR_0.safetensors", net_dur_disc))
    if net_wd is not None:
        required.append(("WD_0.safetensors", net_wd))
    missing = [name for name, _ in required if not (model_dir / name).is_file()]
    if missing:
        if not allow_scratch:
            raise FileNotFoundError(
                f"Initial checkpoint(s) missing in {model_dir}: {missing}. "
                "Create bilingual JP-Extra initial weights or pass --allow-scratch explicitly."
            )
        logger.warning("Training from scratch because --allow-scratch was provided.")
        return 1, 0

    for name, module in required:
        utils.safetensors.load_safetensors(model_dir / name, module, device=device)
    logger.info("Loaded strict initial safetensors from %s", model_dir)
    return 1, 0


def _validate_train_loader_has_batches(
    train_loader: DataLoader, train_dataset: Any, hps: HyperParameters, world_size: int
) -> None:
    if len(train_loader) == 0:
        raise ValueError(
            f"Training loader has zero batches. "
            f"train_items={len(train_dataset)}, batch_size={hps.train.batch_size}, world_size={world_size}. "
            "Increase dataset size or reduce train.batch_size."
        )


def _make_autocast(hps: HyperParameters, device: torch.device):
    enabled = hps.train.fp16_run or hps.train.bf16_run
    dtype = torch.float16 if hps.train.fp16_run else torch.bfloat16
    autocast = torch.amp.autocast  # pyright: ignore[reportPrivateImportUsage]
    return autocast(device_type=device.type, enabled=enabled, dtype=dtype)


def _resolve_slm_model_path(model: str, config_path: Path) -> str:
    """Resolve repo-relative SLM paths without preventing Hugging Face repo IDs."""

    path = Path(model)
    if path.is_absolute():
        return str(path)
    for base in (config_path.parent, BASE_DIR, Path.cwd()):
        candidate = (base / path).resolve()
        if candidate.exists():
            return str(candidate)
    return model


def train(
    config_path: Path,
    model_dir: Path,
    assets_dir: Path,
    allow_scratch: bool = False,
    speedup: bool = False,
    progress_bar: bool = True,
) -> None:
    _setup_runtime()
    distributed, rank, _local_rank, world_size, device = _distributed_context()
    hps = HyperParameters.load_from_json(config_path)
    hps.model_dir = str(model_dir)
    hps.speedup = speedup
    model_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    if hps.version != "3.0.0-Bilingual-JP-Extra":
        raise ValueError(f"Unsupported training config version: {hps.version}")
    if hps.model.use_spk_conditioned_encoder and hps.data.n_speakers <= 0:
        raise ValueError("n_speakers must be > 0 when using speaker-conditioned encoder")

    if rank == 0:
        file_handler = logging.FileHandler(model_dir / f"train_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logging.getLogger().addHandler(file_handler)

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    balance_training = hps.train.balance_training
    balance_fields = hps.train.balance_fields
    if balance_training:
        sample_weights = train_dataset.make_balanced_sample_weights(balance_fields)
        if distributed:
            train_sampler = DistributedWeightedSampler(sample_weights, num_replicas=world_size, rank=rank, shuffle=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=hps.train.batch_size,
                sampler=train_sampler,
                num_workers=1,
                pin_memory=device.type == "cuda",
                collate_fn=collate_fn,
                persistent_workers=True,
                drop_last=True,
            )
        else:
            train_sampler = WeightedRandomSampler(
                sample_weights.tolist(), num_samples=len(train_dataset), replacement=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=hps.train.batch_size,
                sampler=train_sampler,
                num_workers=0,
                pin_memory=device.type == "cuda",
                collate_fn=collate_fn,
                drop_last=True,
            )
    elif distributed:
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            num_workers=1,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=hps.train.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
            drop_last=True,
        )
    _validate_train_loader_has_batches(train_loader, train_dataset, hps, world_size)

    eval_loader = None
    writer = writer_eval = None
    validation_path = Path(hps.data.validation_files)
    has_validation = validation_path.is_file() and validation_path.read_text(encoding="utf-8").strip() != ""
    if rank == 0 and not hps.speedup and has_validation:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(model_dir))
        writer_eval = SummaryWriter(log_dir=str(model_dir / "eval"))
    elif rank == 0 and not hps.speedup:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(model_dir))
        logger.info("Validation list is empty; TensorBoard training logs are enabled without evaluation.")

    net_g = _build_generator(hps, device)
    _apply_freezes(net_g, hps)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    net_dur_disc = (
        DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).to(device)
        if hps.model.use_duration_discriminator
        else None
    )
    net_wd = (
        WavLMDiscriminator(hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel).to(device)
        if hps.model.use_wavlm_discriminator
        else None
    )

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_dur_disc = (
        torch.optim.AdamW(net_dur_disc.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
        if net_dur_disc is not None
        else None
    )
    optim_wd = (
        torch.optim.AdamW(net_wd.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
        if net_wd is not None
        else None
    )

    net_g = cast(torch.nn.Module, _maybe_wrap_ddp(net_g, distributed, device))
    net_d = cast(torch.nn.Module, _maybe_wrap_ddp(net_d, distributed, device))
    net_dur_disc = _maybe_wrap_ddp(net_dur_disc, distributed, device)
    net_wd = _maybe_wrap_ddp(net_wd, distributed, device)

    epoch_start, step_start = _load_initial_or_resume(
        model_dir,
        hps,
        net_g,
        net_d,
        net_dur_disc,
        net_wd,
        optim_g,
        optim_d,
        optim_dur_disc,
        optim_wd,
        allow_scratch,
        device,
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < hps.train.warmup_epochs:
            return float(epoch) / float(max(1, hps.train.warmup_epochs))
        return hps.train.lr_decay ** (epoch - hps.train.warmup_epochs)

    scheduler_last_epoch = epoch_start - 2
    sched_g = torch.optim.lr_scheduler.LambdaLR(optim_g, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch)
    sched_d = torch.optim.lr_scheduler.LambdaLR(optim_d, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch)
    sched_dur = (
        torch.optim.lr_scheduler.LambdaLR(optim_dur_disc, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch)
        if optim_dur_disc is not None
        else None
    )
    sched_wd = (
        torch.optim.lr_scheduler.LambdaLR(optim_wd, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch)
        if optim_wd is not None
        else None
    )
    slm_model = _resolve_slm_model_path(hps.model.slm.model, config_path)
    wavlm_loss = (
        WavLMLoss(slm_model, net_wd, hps.data.sampling_rate, hps.model.slm.sr).to(device)
        if net_wd is not None
        else None
    )
    grad_scaler = torch.amp.GradScaler  # pyright: ignore[reportPrivateImportUsage]
    scaler = grad_scaler(device.type, enabled=hps.train.fp16_run)

    global GLOBAL_STEP
    GLOBAL_STEP = step_start
    progress = (
        tqdm(total=(hps.train.epochs + 1 - epoch_start) * len(train_loader), initial=0, dynamic_ncols=True)
        if rank == 0 and progress_bar
        else None
    )
    logger.info("Start training on %s (distributed=%s)", device, distributed)

    for epoch in range(epoch_start, hps.train.epochs + 1):
        set_epoch = getattr(train_sampler, "set_epoch", None) if train_sampler is not None else None
        if callable(set_epoch):
            set_epoch(epoch)
        _train_epoch(
            rank,
            epoch,
            hps,
            device,
            [net_g, net_d, net_dur_disc, net_wd, wavlm_loss],
            [optim_g, optim_d, optim_dur_disc, optim_wd],
            scaler,
            train_loader,
            writer,
            progress,
            model_dir,
            assets_dir,
        )
        sched_g.step()
        sched_d.step()
        if sched_dur is not None:
            sched_dur.step()
        if sched_wd is not None:
            sched_wd.step()

        if rank == 0 and _should_save_epoch(hps, epoch):
            _save_all(
                model_dir,
                assets_dir,
                hps,
                epoch,
                [net_g, net_d, net_dur_disc, net_wd],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
            )
        if rank == 0 and eval_loader is not None and writer_eval is not None and not hps.speedup:
            _evaluate(hps, device, net_g, eval_loader, writer_eval)

    if progress is not None:
        progress.close()
    if writer is not None:
        writer.close()
    if writer_eval is not None:
        writer_eval.close()
    if distributed:
        dist.destroy_process_group()


def _train_epoch(
    rank: int,
    epoch: int,
    hps: HyperParameters,
    device: torch.device,
    nets: list[Any],
    optims: list[Any],
    scaler: Any,
    train_loader: DataLoader,
    writer: Any,
    progress: tqdm | None,
    model_dir: Path,
    assets_dir: Path,
) -> None:
    net_g, net_d, net_dur_disc, net_wd, wavlm_loss = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    if net_wd is not None:
        net_wd.train()
    global GLOBAL_STEP

    for batch_idx, batch in enumerate(train_loader):
        (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            _y_lengths,
            speakers,
            tone,
            language,
            ja_bert,
            en_bert,
            style_vec,
        ) = [item.to(device, non_blocking=True) if torch.is_tensor(item) else item for item in batch]

        module_g: Any = net_g.module if hasattr(net_g, "module") else net_g
        if module_g.use_noise_scaled_mas:
            current = module_g.mas_noise_scale_initial - module_g.noise_scale_delta * GLOBAL_STEP
            module_g.current_mas_noise_scale = max(current, 0.0)

        with _make_autocast(hps, device):
            (
                y_hat,
                l_length,
                _attn,
                ids_slice,
                x_mask,
                z_mask,
                (_z, z_p, m_p, logs_p, _m_q, logs_q),
                (hidden_x, logw, logw_),
                g,
            ) = net_g(x, x_lengths, spec, spec_lengths, speakers, tone, language, ja_bert, en_bert, style_vec)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc

        if net_dur_disc is not None:
            y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach(), g.detach()
            )
            loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
            optim_dur_disc.zero_grad()
            scaler.scale(loss_dur_disc).backward()
            scaler.unscale_(optim_dur_disc)
            grad_norm_dur = commons.clip_grad_value_(net_dur_disc.parameters(), None)
            scaler.step(optim_dur_disc)
        else:
            loss_dur_disc = grad_norm_dur = None
            losses_dur_disc_r = losses_dur_disc_g = []

        if net_wd is not None and wavlm_loss is not None:
            loss_slm = wavlm_loss.discriminator(y.detach().squeeze(1), y_hat.detach().squeeze(1)).mean()
            optim_wd.zero_grad()
            scaler.scale(loss_slm).backward()
            scaler.unscale_(optim_wd)
            grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
            scaler.step(optim_wd)
        else:
            loss_slm = grad_norm_wd = None

        loss_lm: torch.Tensor | None = None
        loss_lm_gen: torch.Tensor | None = None
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with _make_autocast(hps, device):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                _, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
            if net_wd is not None and wavlm_loss is not None:
                loss_lm = wavlm_loss(y.detach().squeeze(1), y_hat.squeeze(1)).mean()
                loss_lm_gen = wavlm_loss.generator(y_hat.squeeze(1))
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
            if net_dur_disc is not None:
                loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                loss_gen_all += loss_dur_gen
            else:
                loss_dur_gen = None
                losses_dur_gen = []
            if net_wd is not None and wavlm_loss is not None:
                if loss_lm is None or loss_lm_gen is None:
                    raise RuntimeError("WavLM generator losses were not computed")
                loss_gen_all += loss_lm + loss_lm_gen

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0 and writer is not None and GLOBAL_STEP % hps.train.log_interval == 0 and not hps.speedup:
            scalars: dict[str, Any] = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc_all,
                "loss/g/fm": loss_fm,
                "loss/g/mel": loss_mel,
                "loss/g/dur": loss_dur,
                "loss/g/kl": loss_kl,
                "learning_rate": optim_g.param_groups[0]["lr"],
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
            }
            scalars.update({f"loss/g/{i}": value for i, value in enumerate(losses_gen)})
            scalars.update({f"loss/d_r/{i}": value for i, value in enumerate(losses_disc_r)})
            scalars.update({f"loss/d_g/{i}": value for i, value in enumerate(losses_disc_g)})
            if loss_dur_disc is not None:
                scalars["loss/dur_disc/total"] = loss_dur_disc
                scalars["grad_norm_dur"] = grad_norm_dur
                scalars["loss/g/dur_gen"] = loss_dur_gen
                scalars.update({f"loss/dur_disc_r/{i}": v for i, v in enumerate(losses_dur_disc_r)})
                scalars.update({f"loss/dur_disc_g/{i}": v for i, v in enumerate(losses_dur_disc_g)})
                scalars.update({f"loss/g/dur_gen_{i}": v for i, v in enumerate(losses_dur_gen)})
            if loss_slm is not None:
                if loss_lm is None or loss_lm_gen is None:
                    raise RuntimeError("WavLM generator losses were not computed")
                scalars["loss/wd/total"] = loss_slm
                scalars["grad_norm_wd"] = grad_norm_wd
                scalars["loss/g/lm"] = loss_lm
                scalars["loss/g/lm_gen"] = loss_lm_gen
            utils.summarize(writer=writer, global_step=GLOBAL_STEP, scalars=scalars)

        GLOBAL_STEP += 1
        if rank == 0 and hps.train.save_interval > 0 and GLOBAL_STEP % hps.train.save_interval == 0:
            _save_all(
                model_dir,
                assets_dir,
                hps,
                epoch,
                [net_g, net_d, net_dur_disc, net_wd],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
            )
        if progress is not None:
            progress.set_description(f"Epoch {epoch} ({100.0 * batch_idx / max(1, len(train_loader)):.0f}%)")
            progress.update()


def _should_save_epoch(hps: HyperParameters, epoch: int) -> bool:
    if epoch == hps.train.epochs:
        return True
    return hps.train.save_epoch_interval > 0 and epoch % hps.train.save_epoch_interval == 0


def _save_all(
    model_dir: Path,
    assets_dir: Path,
    hps: HyperParameters,
    epoch: int,
    nets: list[Any],
    optims: list[Any],
) -> None:
    net_g, net_d, net_dur_disc, net_wd = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    utils.checkpoints.save_checkpoint(
        net_g, optim_g, hps.train.learning_rate, epoch, model_dir / f"G_{GLOBAL_STEP}.pth"
    )
    utils.checkpoints.save_checkpoint(
        net_d, optim_d, hps.train.learning_rate, epoch, model_dir / f"D_{GLOBAL_STEP}.pth"
    )
    if net_dur_disc is not None:
        if optim_dur_disc is None:
            raise RuntimeError("optim_dur_disc is required when net_dur_disc is enabled")
        utils.checkpoints.save_checkpoint(
            net_dur_disc, optim_dur_disc, hps.train.learning_rate, epoch, model_dir / f"DUR_{GLOBAL_STEP}.pth"
        )
    if net_wd is not None:
        if optim_wd is None:
            raise RuntimeError("optim_wd is required when net_wd is enabled")
        utils.checkpoints.save_checkpoint(
            net_wd, optim_wd, hps.train.learning_rate, epoch, model_dir / f"WD_{GLOBAL_STEP}.pth"
        )
    utils.safetensors.save_safetensors(
        net_g, epoch, assets_dir / f"{hps.model_name}_e{epoch}_s{GLOBAL_STEP}.safetensors"
    )
    keep_ckpts = getattr(hps.train, "keep_ckpts", 0)
    if keep_ckpts:
        utils.checkpoints.clean_checkpoints(model_dir, keep_ckpts, sort_by_time=True)
        _clean_assets_safetensors(assets_dir, hps.model_name, keep_ckpts)


def _clean_assets_safetensors(assets_dir: Path, model_name: str, keep_ckpts: int) -> None:
    files = sorted(assets_dir.glob(f"{model_name}_e*_s*.safetensors"), key=lambda path: path.stat().st_mtime)
    for checkpoint in files[:-keep_ckpts]:
        logger.info("Deleting expired safetensors checkpoint %s", checkpoint)
        checkpoint.unlink()


@torch.no_grad()
def _evaluate(
    hps: HyperParameters,
    device: torch.device,
    generator: torch.nn.Module,
    eval_loader: DataLoader,
    writer_eval: Any,
) -> None:
    generator.eval()
    audio_dict = {}
    module_g: Any = generator.module if hasattr(generator, "module") else generator
    for batch_idx, batch in enumerate(eval_loader):
        (
            x,
            x_lengths,
            spec,
            _spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            ja_bert,
            en_bert,
            style_vec,
        ) = [item.to(device, non_blocking=True) if torch.is_tensor(item) else item for item in batch]
        for use_sdp in (True, False):
            y_hat, _attn, mask, *_ = module_g.infer(
                x,
                x_lengths,
                speakers,
                tone,
                language,
                ja_bert,
                en_bert,
                style_vec,
                y=spec,
                max_len=1000,
                sdp_ratio=1.0 if use_sdp else 0.0,
            )
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
            audio_dict[f"gen/audio_{batch_idx}_{use_sdp}"] = y_hat[0, :, : y_hat_lengths[0]]
            audio_dict[f"gt/audio_{batch_idx}"] = y[0, :, : y_lengths[0]]
    utils.summarize(
        writer=writer_eval,
        global_step=GLOBAL_STEP,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bilingual JP-Extra CHARAVOICE Engine model")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--allow-scratch", action="store_true")
    parser.add_argument("--speedup", action="store_true", help="Disable TensorBoard and evaluation")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    train(
        args.config,
        args.model_dir,
        args.assets_dir,
        allow_scratch=args.allow_scratch,
        speedup=args.speedup,
        progress_bar=not args.no_progress_bar,
    )


if __name__ == "__main__":
    main()
