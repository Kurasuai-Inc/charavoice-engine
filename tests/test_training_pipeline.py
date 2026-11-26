"""Lightweight tests for the charavoice_engine training package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from charavoice_engine.config import HyperParameters
from charavoice_engine.training import bert_gen, default_style, preprocess_text
from charavoice_engine.training.convert_jp_extra_checkpoint import convert_generator_state_dict
from charavoice_engine.training.data_utils import (
    DistributedWeightedSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from charavoice_engine.training.dataset import prepare_dataset, read_esd_list
from charavoice_engine.training.recipe import _apply_config_overrides, validate_runtime_exports
from charavoice_engine.training.style_gen import _wav_paths_from_config
from charavoice_engine.training.train import _validate_train_loader_has_batches


def _write_wav(path: Path, seconds: float = 0.5, sampling_rate: int = 44100) -> None:
    samples = np.zeros(int(seconds * sampling_rate), dtype=np.float32)
    sf.write(path, samples, sampling_rate)


def _write_identity_onnx(path: Path, input_names: tuple[str, ...]) -> None:
    import onnx
    from onnx import TensorProto, helper

    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]) for name in input_names]
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", [input_names[0]], ["out"])
    graph = helper.make_graph([node], path.stem, inputs, [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)


def test_training_modules_import_without_optional_runtime_dependencies() -> None:
    import charavoice_engine.training.recipe as recipe_module
    import charavoice_engine.training.train as train_module

    assert hasattr(train_module, "train")
    assert hasattr(recipe_module, "run_recipe")


def test_preprocess_process_line_outputs_cleaned_seven_columns(monkeypatch) -> None:
    def fake_clean_text(*, text, language, raise_yomi_error):
        assert text == "Hello"
        assert language.value == "EN"
        assert raise_yomi_error is True
        return "hello", ["hh", "ah"], [0, 0], [1, 1]

    monkeypatch.setattr(preprocess_text, "clean_text", fake_clean_text)

    result = preprocess_text.process_line(
        "sample.wav|alice|EN|Hello",
        transcription_path=Path("/dataset/esd.list"),
        correct_path=True,
        yomi_error="raise",
    )

    assert result == "/dataset/wavs/sample.wav|alice|EN|hello|hh ah|0 0|1 1\n"


def test_bert_gen_verifies_existing_feature_length(tmp_path: Path) -> None:
    wav_path = tmp_path / "a.wav"
    bert_path = tmp_path / "a.bert.pt"
    torch.save(torch.zeros(1024, 3), bert_path)

    line = f"{wav_path}|alice|EN|a|aa|0|1"
    result = bert_gen.process_line(line, add_blank=True, device="cpu")

    assert result == bert_path


def test_bert_gen_rejects_stale_existing_feature(tmp_path: Path) -> None:
    wav_path = tmp_path / "a.wav"
    bert_path = tmp_path / "a.bert.pt"
    torch.save(torch.zeros(1024, 2), bert_path)

    line = f"{wav_path}|alice|EN|a|aa|0|1"
    try:
        bert_gen.process_line(line, add_blank=True, device="cpu")
    except ValueError as exc:
        assert "Existing BERT length mismatch" in str(exc)
    else:
        raise AssertionError("stale BERT feature was accepted")


def test_bilingual_collate_returns_ja_and_en_bert_slots() -> None:
    batch = [
        (
            torch.LongTensor([1, 2, 3]),
            torch.ones(5, 4),
            torch.ones(1, 8),
            torch.LongTensor([0]),
            torch.LongTensor([6, 7, 6]),
            torch.LongTensor([1, 1, 1]),
            torch.ones(1024, 3),
            torch.zeros(1024, 3),
            torch.full((256,), 0.25),
        ),
        (
            torch.LongTensor([4]),
            torch.ones(5, 2),
            torch.ones(1, 4),
            torch.LongTensor([1]),
            torch.LongTensor([8]),
            torch.LongTensor([2]),
            torch.zeros(1024, 1),
            torch.full((1024, 1), 2.0),
            torch.full((256,), -0.5),
        ),
    ]

    collated = TextAudioSpeakerCollate()(batch)

    assert len(collated) == 12
    assert collated[9].shape == (2, 1024, 3)
    assert collated[10].shape == (2, 1024, 3)
    assert float(collated[9][0].sum()) == 1024 * 3
    assert float(collated[10][1, :, 0].sum()) == 1024 * 2


def test_balanced_sample_weights_raise_minority_language_and_style() -> None:
    dataset = TextAudioSpeakerLoader.__new__(TextAudioSpeakerLoader)
    dataset.sample_metadata = [
        {"speaker": "alice", "language": "JP", "style": "Neutral"},
        {"speaker": "alice", "language": "JP", "style": "Neutral"},
        {"speaker": "alice", "language": "EN", "style": "happy"},
    ]

    weights = dataset.make_balanced_sample_weights(["speaker", "language", "style"])

    assert weights.shape == (3,)
    assert weights[2] > weights[0]
    assert weights[2] > weights[1]


def test_distributed_weighted_sampler_shards_replacement_samples() -> None:
    sampler = DistributedWeightedSampler(torch.DoubleTensor([1.0, 2.0, 3.0]), num_replicas=2, rank=1)
    sampler.set_epoch(3)

    indices = list(iter(sampler))

    assert len(indices) == len(sampler) == 2
    assert all(0 <= index < 3 for index in indices)


def test_style_gen_reads_training_and_validation_wav_paths(tmp_path: Path) -> None:
    train_list = tmp_path / "train.list"
    val_list = tmp_path / "val.list"
    config = tmp_path / "config.json"
    train_wav = tmp_path / "train.wav"
    val_wav = tmp_path / "val.wav"
    train_list.write_text(f"{train_wav}|alice|JP|x|a|0|1\n", encoding="utf-8")
    val_list.write_text(f"{val_wav}|alice|EN|x|aa|0|1\n", encoding="utf-8")
    config.write_text(
        json.dumps(
            {
                "version": "3.0.0-Bilingual-JP-Extra",
                "data": {
                    "training_files": str(train_list),
                    "validation_files": str(val_list),
                    "spk2id": {"alice": 0},
                    "n_speakers": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _wav_paths_from_config(config) == [train_wav, val_wav]


def test_default_style_rejects_missing_vectors_for_non_neutral_style(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    neutral_dir = raw_dir / "Neutral" / "alice" / "JP"
    happy_dir = raw_dir / "happy" / "alice" / "JP"
    neutral_dir.mkdir(parents=True)
    happy_dir.mkdir(parents=True)
    neutral_wav = neutral_dir / "a.wav"
    happy_wav = happy_dir / "b.wav"
    _write_wav(neutral_wav)
    _write_wav(happy_wav)
    np.save(f"{neutral_wav}.npy", np.zeros(256, dtype=np.float32))
    config = tmp_path / "config.json"
    config.write_text(HyperParameters().model_dump_json(indent=2), encoding="utf-8")

    try:
        default_style.save_styles_by_dirs(raw_dir, tmp_path / "assets", config, tmp_path / "assets" / "config.json")
    except FileNotFoundError as exc:
        assert "Style directory has wav files but no style vectors" in str(exc)
    else:
        raise AssertionError("missing non-neutral style vectors were accepted")


def test_convert_generator_state_dict_maps_source_bert_projection_to_both_languages() -> None:
    source = {
        "emb.weight": torch.ones(2, 2),
        "enc_p.bert_proj.weight": torch.full((2, 2, 1), 3.0),
        "enc_p.bert_proj.bias": torch.full((2,), 4.0),
    }
    target = {
        "emb.weight": torch.zeros(2, 2),
        "enc_p.ja_bert_proj.weight": torch.zeros(2, 2, 1),
        "enc_p.ja_bert_proj.bias": torch.zeros(2),
        "enc_p.en_bert_proj.weight": torch.zeros(2, 2, 1),
        "enc_p.en_bert_proj.bias": torch.zeros(2),
    }

    converted = convert_generator_state_dict(source, target)

    assert torch.equal(converted["emb.weight"], source["emb.weight"])
    assert torch.equal(converted["enc_p.ja_bert_proj.weight"], source["enc_p.bert_proj.weight"])
    assert torch.equal(converted["enc_p.ja_bert_proj.bias"], source["enc_p.bert_proj.bias"])
    assert torch.equal(converted["enc_p.en_bert_proj.weight"], source["enc_p.bert_proj.weight"])
    assert torch.equal(converted["enc_p.en_bert_proj.bias"], source["enc_p.bert_proj.bias"])


def test_convert_generator_state_dict_keeps_target_initialized_style_projection() -> None:
    source = {
        "emb.weight": torch.ones(2, 2),
    }
    target = {
        "emb.weight": torch.zeros(2, 2),
        "enc_p.style_proj.weight": torch.full((2, 2), 3.0),
        "enc_p.style_proj.bias": torch.full((2,), 4.0),
    }

    converted = convert_generator_state_dict(source, target)

    assert torch.equal(converted["emb.weight"], source["emb.weight"])
    assert torch.equal(converted["enc_p.style_proj.weight"], target["enc_p.style_proj.weight"])
    assert torch.equal(converted["enc_p.style_proj.bias"], target["enc_p.style_proj.bias"])


def test_convert_generator_state_dict_extends_multi_speaker_embedding_with_target_init() -> None:
    source = {
        "emb_g.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }
    target = {
        "emb_g.weight": torch.tensor([[0.0, 0.0], [0.0, 0.0], [9.0, 10.0]]),
    }

    converted = convert_generator_state_dict(source, target)

    assert torch.equal(
        converted["emb_g.weight"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]]),
    )


def test_convert_generator_state_dict_repeats_single_speaker_embedding() -> None:
    source = {
        "emb_g.weight": torch.tensor([[1.0, 2.0]]),
    }
    target = {
        "emb_g.weight": torch.zeros(3, 2),
    }

    converted = convert_generator_state_dict(source, target)

    assert torch.equal(
        converted["emb_g.weight"],
        torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
    )


def test_convert_generator_state_dict_initializes_missing_speaker_embedding_from_target() -> None:
    source = {
        "emb.weight": torch.ones(2, 2),
    }
    target = {
        "emb.weight": torch.zeros(2, 2),
        "emb_g.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }

    converted = convert_generator_state_dict(source, target)

    assert torch.equal(converted["emb.weight"], source["emb.weight"])
    assert torch.equal(converted["emb_g.weight"], target["emb_g.weight"])


def test_prepare_dataset_from_sidecars_writes_esd_config_and_report(tmp_path: Path) -> None:
    wav_dir = tmp_path / "raw" / "Neutral" / "alice" / "JP"
    wav_dir.mkdir(parents=True)
    wav_path = wav_dir / "a.wav"
    _write_wav(wav_path)
    wav_path.with_suffix(".txt").write_text("こんにちは", encoding="utf-8")
    assets_dir = tmp_path / "assets"

    report = prepare_dataset(tmp_path, "unit_model", assets_dir)

    assert report.total_items == 1
    assert report.target_sampling_rate == 44100
    assert report.total_duration_seconds > 0
    assert report.sample_rates == {"44100": 1}
    assert report.speakers == ["alice"]
    assert report.styles == ["Neutral"]
    assert report.languages == ["JP"]
    assert (tmp_path / "esd.list").is_file()
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "dataset_report.json").is_file()


def test_prepare_dataset_normalizes_existing_relative_esd_list(tmp_path: Path) -> None:
    wav_dir = tmp_path / "raw" / "Neutral" / "alice" / "EN"
    wav_dir.mkdir(parents=True)
    wav_path = wav_dir / "a.wav"
    _write_wav(wav_path)
    esd = tmp_path / "esd.list"
    esd.write_text("Neutral/alice/EN/a.wav|alice|EN|hello\n", encoding="utf-8")

    report = prepare_dataset(tmp_path, "unit_model", tmp_path / "assets")

    assert report.esd_path == esd.resolve()
    normalized_line = esd.read_text(encoding="utf-8").strip()
    assert normalized_line == f"{wav_path.resolve()}|alice|EN|hello"


def test_prepare_dataset_rejects_short_audio(tmp_path: Path) -> None:
    wav_dir = tmp_path / "raw" / "Neutral" / "alice" / "EN"
    wav_dir.mkdir(parents=True)
    wav_path = wav_dir / "a.wav"
    _write_wav(wav_path, seconds=0.05)
    wav_path.with_suffix(".txt").write_text("hello", encoding="utf-8")

    try:
        prepare_dataset(tmp_path, "unit_model", tmp_path / "assets")
    except ValueError as exc:
        assert "Audio shorter than" in str(exc)
    else:
        raise AssertionError("short audio was accepted")


def test_prepare_dataset_rejects_sampling_rate_mismatch(tmp_path: Path) -> None:
    wav_dir = tmp_path / "raw" / "Neutral" / "alice" / "EN"
    wav_dir.mkdir(parents=True)
    wav_path = wav_dir / "a.wav"
    _write_wav(wav_path, seconds=0.5, sampling_rate=48000)
    wav_path.with_suffix(".txt").write_text("hello", encoding="utf-8")

    try:
        prepare_dataset(tmp_path, "unit_model", tmp_path / "assets")
    except ValueError as exc:
        assert "Audio sampling rate must be 44100 Hz" in str(exc)
    else:
        raise AssertionError("sampling-rate mismatch was accepted")


def test_read_esd_list_rejects_layout_speaker_mismatch(tmp_path: Path) -> None:
    wav_dir = tmp_path / "raw" / "Neutral" / "alice" / "EN"
    wav_dir.mkdir(parents=True)
    wav_path = wav_dir / "a.wav"
    wav_path.write_bytes(b"RIFF")
    esd = tmp_path / "esd.list"
    esd.write_text(f"{wav_path}|bob|EN|hello\n", encoding="utf-8")

    try:
        read_esd_list(tmp_path, esd)
    except ValueError as exc:
        assert "speaker mismatch" in str(exc)
    else:
        raise AssertionError("speaker mismatch was accepted")


def test_preprocess_rejects_empty_training_split_without_outputs(tmp_path: Path, monkeypatch) -> None:
    def fake_clean_text(*, text, language, raise_yomi_error):
        return text, ["aa"], [0], [1]

    monkeypatch.setattr(preprocess_text, "clean_text", fake_clean_text)
    wav_path = tmp_path / "raw" / "Neutral" / "alice" / "EN" / "a.wav"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"RIFF")
    transcription = tmp_path / "esd.list"
    transcription.write_text(f"{wav_path}|alice|EN|hello\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data": {"spk2id": {}, "n_speakers": 0}}), encoding="utf-8")
    train_list = tmp_path / "train.list"
    val_list = tmp_path / "val.list"

    try:
        preprocess_text.preprocess(
            transcription_path=transcription,
            cleaned_path=tmp_path / "esd.list.cleaned",
            train_path=train_list,
            val_path=val_list,
            config_path=config,
            val_per_speaker=1,
            max_val_total=100,
            yomi_error="raise",
            correct_path=False,
        )
    except ValueError as exc:
        assert "Training list is empty" in str(exc)
    else:
        raise AssertionError("empty training split was accepted")

    assert not train_list.exists()
    assert not val_list.exists()


def test_validate_runtime_exports_rejects_single_bert_input(tmp_path: Path) -> None:
    _write_identity_onnx(tmp_path / "decoder.onnx", ("z", "g"))
    _write_identity_onnx(tmp_path / "enc_p_dp.onnx", ("x", "x_lengths", "tone", "language", "bert", "style_vec", "g"))
    _write_identity_onnx(tmp_path / "flow.onnx", ("z_p", "y_mask", "g"))

    try:
        validate_runtime_exports(tmp_path)
    except ValueError as exc:
        assert "enc_p_dp.onnx ONNX input mismatch" in str(exc)
    else:
        raise AssertionError("enc_p_dp ONNX input with a single BERT slot was accepted")


def test_train_loader_zero_batches_is_rejected() -> None:
    class EmptyLoader:
        def __len__(self) -> int:
            return 0

    class TinyDataset:
        def __len__(self) -> int:
            return 1

    hps = HyperParameters(train={"batch_size": 2})

    try:
        _validate_train_loader_has_batches(EmptyLoader(), TinyDataset(), hps, world_size=1)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Training loader has zero batches" in str(exc)
    else:
        raise AssertionError("zero-batch training loader was accepted")


def test_recipe_config_overrides_are_written(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(HyperParameters().model_dump_json(indent=2), encoding="utf-8")

    _apply_config_overrides(
        config_path,
        epochs=12,
        batch_size=4,
        learning_rate=2e-4,
        save_epoch_interval=5,
        keep_ckpts=12,
        balance_training=False,
        balance_fields=["speaker", "language"],
        use_wavlm_discriminator=False,
        use_duration_discriminator=True,
        slm_model=None,
        fp16_run=False,
        bf16_run=True,
    )

    hps = HyperParameters.load_from_json(config_path)
    assert hps.train.epochs == 12
    assert hps.train.batch_size == 4
    assert hps.train.learning_rate == 2e-4
    assert hps.train.save_epoch_interval == 5
    assert hps.train.keep_ckpts == 12
    assert hps.train.balance_training is False
    assert hps.train.balance_fields == ["speaker", "language"]
    assert hps.model.use_wavlm_discriminator is False
    assert hps.model.use_duration_discriminator is True
    assert hps.train.bf16_run is True
    assert hps.train.fp16_run is False
