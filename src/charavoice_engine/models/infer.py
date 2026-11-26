"""Model loading and text processing for ONNX export.

Ported from style_bert_vits2.models.infer — only get_net_g() and get_text().
get_text() uses zero BERT embeddings because ONNX tracing only needs the
correct tensor shapes, not actual BERT values.
"""

from __future__ import annotations

import logging

import torch

from charavoice_engine.config import HyperParameters
from charavoice_engine.constants import Languages
from charavoice_engine.models import commons, utils
from charavoice_engine.models.models_bilingual_jp_extra import SynthesizerTrn
from charavoice_engine.nlp import clean_text, cleaned_text_to_sequence
from charavoice_engine.nlp.symbols import SYMBOLS

logger = logging.getLogger(__name__)


def get_net_g(model_path: str, version: str, device: str, hps: HyperParameters) -> SynthesizerTrn:
    if not version.endswith("Bilingual-JP-Extra"):
        raise ValueError(f"Unsupported model version for CHARAVOICE Engine bilingual runtime: {version}")

    logger.info("Using bilingual JP-Extra model")
    net_g = SynthesizerTrn(
        n_vocab=len(SYMBOLS),
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
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
    net_g.state_dict()
    _ = net_g.eval()
    if model_path.endswith(".pth") or model_path.endswith(".pt"):
        _ = utils.checkpoints.load_checkpoint(model_path, net_g, None, skip_optimizer=True, device=device)
    elif model_path.endswith(".safetensors"):
        _ = utils.safetensors.load_safetensors(model_path, net_g, device=device)
    else:
        raise ValueError(f"Unknown model format: {model_path}")
    return net_g


def get_text(
    text: str,
    language_str: Languages,
    hps: HyperParameters,
    device: str,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _norm_text, phone, tone, word2ph = clean_text(
        text,
        language_str,
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    # Dummy zero BERT embeddings — ONNX tracing only needs correct shapes,
    # not actual BERT values. This eliminates the PyTorch BERT dependency.
    bert_ori = torch.zeros(1024, len(phone))
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == Languages.JP:
        ja_bert = bert_ori
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == Languages.EN:
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be JP or EN")

    assert ja_bert.shape[-1] == len(phone), f"JP BERT seq len {ja_bert.shape[-1]} != {len(phone)}"
    assert en_bert.shape[-1] == len(phone), f"EN BERT seq len {en_bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return ja_bert, en_bert, phone, tone, language
