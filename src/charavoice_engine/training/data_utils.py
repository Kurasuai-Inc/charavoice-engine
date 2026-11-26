import logging
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from charavoice_engine.config import HyperParametersData
from charavoice_engine.constants import Languages
from charavoice_engine.models import commons
from charavoice_engine.models.utils import load_filepaths_and_text, load_wav_to_torch
from charavoice_engine.nlp import cleaned_text_to_sequence
from charavoice_engine.training.mel_processing import mel_spectrogram_torch, spectrogram_torch

logger = logging.getLogger(__name__)
"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text: str, hparams: HyperParametersData, spec_cache: bool = True):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams
        self.spec_cache = spec_cache

        self.use_mel_spec_posterior = getattr(hparams, "use_mel_posterior_encoder", False)
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 384)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        sample_metadata = []
        skipped = 0
        logger.info("Init dataset...")
        for _id, spk, language, text, phones, tone, word2ph in tqdm(
            self.audiopaths_sid_text, file=sys.stdout, dynamic_ncols=True
        ):
            audiopath = f"{_id}"
            # if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
            phones = phones.split(" ")
            tone = [int(i) for i in tone.split(" ")]
            word2ph = [int(i) for i in word2ph.split(" ")]
            audiopaths_sid_text_new.append([audiopath, spk, language, text, phones, tone, word2ph])
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            sample_metadata.append(
                {
                    "speaker": spk,
                    "language": language,
                    "style": _style_from_audio_path(Path(audiopath)),
                }
            )
            # else:
            #     skipped += 1
        logger.info("skipped: " + str(skipped) + ", total: " + str(len(self.audiopaths_sid_text)))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        self.sample_metadata = sample_metadata

    def make_balanced_sample_weights(self, fields: list[str]) -> torch.Tensor:
        if not fields:
            return torch.ones(len(self.sample_metadata), dtype=torch.double)
        supported = {"speaker", "language", "style"}
        unknown = sorted(set(fields) - supported)
        if unknown:
            raise ValueError(f"Unsupported balance field(s): {unknown}. Supported fields: {sorted(supported)}")
        total = len(self.sample_metadata)
        counts = {field: Counter(item[field] for item in self.sample_metadata) for field in fields}
        weights = []
        for item in self.sample_metadata:
            field_weights = [total / counts[field][item[field]] for field in fields]
            weights.append(sum(field_weights) / len(field_weights))
        return torch.DoubleTensor(weights)

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text

        ja_bert, en_bert, phones, tone, language = self.get_text(text, word2ph, phones, tone, language, audiopath)

        spec, wav = self.get_audio(audiopath)
        sid = torch.LongTensor([int(self.spk_map[sid])])
        style_vec = torch.FloatTensor(np.load(f"{audiopath}.npy"))
        return (phones, spec, wav, sid, tone, language, ja_bert, en_bert, style_vec)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"{filename} {sampling_rate} SR doesn't match target {self.sampling_rate} SR")
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
        except FileNotFoundError:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            if self.spec_cache:
                torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        language_enum = Languages(language_str)
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_enum)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        bert_path = wav_path.replace(".wav", ".bert.pt")
        bert_ori = torch.load(bert_path)
        if bert_ori.shape[-1] != len(phone):
            raise ValueError(f"BERT length mismatch for {wav_path}: {bert_ori.shape[-1]} != {len(phone)}")

        if language_enum == Languages.JP:
            bert = torch.zeros(1024, len(phone))
            ja_bert = bert_ori
            en_bert = torch.zeros(1024, len(phone))
        elif language_enum == Languages.EN:
            bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = bert_ori
        else:
            raise ValueError(f"Unsupported language: {language_str}")
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        del bert
        return ja_bert, en_bert, phone, tone, language

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        ja_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        en_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        style_vec = torch.FloatTensor(len(batch), 256)

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        ja_bert_padded.zero_()
        en_bert_padded.zero_()
        style_vec.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            language = row[5]
            language_padded[i, : language.size(0)] = language

            ja_bert = row[6]
            ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

            en_bert = row[7]
            en_bert_padded[i, :, : en_bert.size(1)] = en_bert
            style_vec[i, :] = row[8]

        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
            tone_padded,
            language_padded,
            ja_bert_padded,
            en_bert_padded,
            style_vec,
        )


class DistributedBucketSampler(DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        logger.info(f"Bucket info: {self.num_samples_per_bucket}")
        # logger.info(
        #     f"Unused samples: {len(self.lengths) - sum(self.num_samples_per_bucket)}"
        # )
        # ↑マイナスになることあるし、別にこれは使われないサンプル数ではないようだ……
        # バケットの仕組みはよく分からない

        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            logger.info("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size


class DistributedWeightedSampler(torch.utils.data.Sampler[int]):
    """Weighted replacement sampler sharded across distributed ranks."""

    def __init__(
        self,
        weights: torch.Tensor,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
    ) -> None:
        if weights.numel() == 0:
            raise ValueError("weights must not be empty")
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = math.ceil(weights.numel() / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator).tolist()
        else:
            repeat_count = math.ceil(self.total_size / self.weights.numel())
            indices = (list(range(self.weights.numel())) * repeat_count)[: self.total_size]
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def _style_from_audio_path(path: Path) -> str:
    parts = path.parts
    if "raw" in parts:
        raw_index = len(parts) - 1 - parts[::-1].index("raw")
        if len(parts) > raw_index + 1:
            return parts[raw_index + 1]
    if len(parts) >= 4:
        return parts[-4]
    return "Unknown"
