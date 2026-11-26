# Third-party licenses

CHARAVOICE Engine is licensed under AGPL-3.0-only (see `LICENSE`). This file lists the third-party code and data included in this repository, the models the engine uses at runtime (not included), and the Python dependencies declared in `pyproject.toml`. Attribution texts are in `NOTICE`.

This repository contains no trained model weights, voice data, or speaker datasets.

## 1. Code and data included in this repository

| Component | Files | License | Source |
|---|---|---|---|
| Style-Bert-VITS2 (litagin02) | most of `src/charavoice_engine/models/`, `nlp/`, `constants.py`, `config.py` (marked "Ported from style_bert_vits2") | AGPL-3.0 | https://github.com/litagin02/Style-Bert-VITS2 |
| Bert-VITS2 (Fish Audio), upstream of Style-Bert-VITS2 | (through Style-Bert-VITS2) | AGPL-3.0 | https://github.com/fishaudio/Bert-VITS2 |
| VOICEVOX ENGINE | `src/charavoice_engine/nlp/japanese/mora_list.py` | LGPL-3.0 | https://github.com/VOICEVOX/voicevox_engine |
| Open JTalk mora table (Nagoya Institute of Technology, HTS Working Group), via VOICEVOX | `src/charavoice_engine/nlp/japanese/mora_list.py` (notice kept in the file header) | Modified BSD (3-clause) | http://open-jtalk.sourceforge.net/ |
| The CMU Pronouncing Dictionary 0.6 (Carnegie Mellon University) | `src/charavoice_engine/nlp/english/cmudict.rep` | Unrestricted use; acknowledgement requested | http://www.speech.cs.cmu.edu/cgi-bin/cmudict |

## 2. Models used at runtime (not included)

| Model | Use | License | Notes |
|---|---|---|---|
| `ku-nlp/deberta-v2-large-japanese-char-wwm` and its ONNX conversion `tsukumijima/deberta-v2-large-japanese-char-wwm-onnx` | Japanese BERT features | CC BY-SA 4.0 | Credit, license link and indication of changes are required; shared derivatives keep CC BY-SA 4.0. |
| `microsoft/deberta-v3-large` and `tsukumijima/deberta-v3-large-onnx` | English BERT features | MIT | |
| AudioSeal (`facebook/audioseal`, `audioseal_wm_16bits`) | Audio watermark | MIT | |
| `microsoft/wavlm-base-plus` | Training only (SLM discriminator) | Check before redistribution | Not used for inference. |
| CHARAVOICE voice models (company-trained weights) | Inference | Not published | Weights and voice data are not part of this repository. |

## 3. Python dependencies (`pyproject.toml`; versions from `uv.lock`)

These are installed separately by `uv`/`pip` and are not copied into this repository.

### Runtime (always installed)

| Package | Locked version | License | Project |
|---|---|---|---|
| cmudict | 1.1.3 | GPL-3.0-or-later | https://github.com/prosegrinder/python-cmudict |
| g2p-en | 2.1.0 | Apache-2.0 | https://github.com/Kyubyong/g2p |
| nltk | 3.8.1 | Apache-2.0 | https://www.nltk.org/ |
| num2words | 0.5.14 | LGPL | https://github.com/savoirfairelinux/num2words |
| numba | 0.64.0 | BSD | https://numba.pydata.org |
| numpy | 2.4.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |  |
| pydantic | 2.12.5 | MIT | https://github.com/pydantic/pydantic |
| pyopenjtalk-plus | 0.4.1.post7 | MIT | https://github.com/tsukumijima/pyopenjtalk-plus |
| safetensors | 0.7.0 | Apache-2.0 | https://github.com/huggingface/safetensors |
| setuptools | 70.3.0 | MIT | https://github.com/pypa/setuptools |
| transformers | 4.57.6 | Apache-2.0 | https://github.com/huggingface/transformers |
| torch | 2.10.0+cu126 | BSD-3-Clause |  |
| onnxruntime-gpu | 1.24.2 | MIT |  |
| soundfile | 0.13.1 | BSD | https://github.com/bastibe/python-soundfile |
| psutil | 7.2.2 | BSD-3-Clause | https://github.com/giampaolo/psutil |
| accelerate | 1.12.0 | Apache-2.0 | https://github.com/huggingface/accelerate |
| onnx | 1.19.0 | Apache-2.0 | https://onnx.ai/ |
| onnxsim | 0.6.2 | MIT AND (Apache-2.0 OR BSD-2-Clause) | https://github.com/daquexian/onnx-simplifier |

### Extra `server`

| Package | Locked version | License | Project |
|---|---|---|---|
| fastapi | 0.133.1 | MIT | https://github.com/fastapi/fastapi |
| uvicorn | 0.41.0 | BSD-3-Clause | https://github.com/Kludex/uvicorn |
| av | 17.0.0 | BSD-3-Clause |  |
| prometheus-client | 0.24.1 | Apache-2.0 AND BSD-2-Clause | https://github.com/prometheus/client_python |
| aiohttp | 3.13.3 | Apache-2.0 AND MIT | https://github.com/aio-libs/aiohttp |

### Extra `tensorrt` (optional)

| Package | Locked version | License | Project |
|---|---|---|---|
| tensorrt-cu12 | 10.15.1.29 | NVIDIA proprietary (TensorRT SDK license) | https://github.com/nvidia/tensorrt |
| tensorrt-cu12-bindings | 10.15.1.29 | NVIDIA proprietary (TensorRT SDK license) | https://github.com/nvidia/tensorrt |
| tensorrt-cu12-libs | 10.15.1.29 | NVIDIA proprietary (TensorRT SDK license) | https://github.com/nvidia/tensorrt |

### Extra `convert`

| Package | Locked version | License | Project |
|---|---|---|---|
| torchaudio | 2.10.0+cu126 | BSD-2-Clause |  |
| accelerate | 1.12.0 | Apache-2.0 | https://github.com/huggingface/accelerate |
| onnx | 1.19.0 | Apache-2.0 | https://onnx.ai/ |
| onnxscript | 0.5.0 | MIT | https://microsoft.github.io/onnxscript/ |
| onnxsim | 0.6.2 | MIT AND (Apache-2.0 OR BSD-2-Clause) | https://github.com/daquexian/onnx-simplifier |
| onnxconverter-common | 1.16.0 | MIT | https://github.com/microsoft/onnxconverter-common |

### Extra `train`

| Package | Locked version | License | Project |
|---|---|---|---|
| librosa | 0.11.0 | ISC | https://github.com/librosa/librosa |
| matplotlib | 3.11.0 | PSF | https://matplotlib.org |
| pyannote.audio | (not locked) | MIT |  |
| scipy | 1.17.1 | BSD |  |
| tensorboard | 2.20.0 | Apache-2.0 | https://github.com/tensorflow/tensorboard |
| torchaudio | 2.10.0+cu126 | BSD-2-Clause |  |
| tqdm | 4.67.3 | MPL-2.0 AND MIT |  |

### Extra `dev` (development only)

| Package | Locked version | License | Project |
|---|---|---|---|
| aiohttp | 3.13.3 | Apache-2.0 AND MIT | https://github.com/aio-libs/aiohttp |
| pytest | 9.0.2 | MIT | https://github.com/pytest-dev/pytest |
| pytest-asyncio | 1.3.0 | Apache-2.0 | https://github.com/pytest-dev/pytest-asyncio |
| ruff | 0.15.2 | MIT | https://docs.astral.sh/ruff |
| pyright | 1.1.408 | MIT | https://github.com/RobertCraigie/pyright-python |

### Extra `watermark`

| Package | Locked version | License | Project |
|---|---|---|---|
| audioseal | 0.2.0 | MIT | https://github.com/facebookresearch/audioseal |
| onnxscript | 0.5.0 | MIT | https://microsoft.github.io/onnxscript/ |

Notes:

- `cmudict` (Python package) is GPL-3.0-or-later, and `num2words` is LGPL; both are compatible with AGPL-3.0 when used as dependencies.
- The optional `tensorrt` extra installs NVIDIA TensorRT under NVIDIA's proprietary license. It is not included in this repository; anyone redistributing an image that contains TensorRT must follow NVIDIA's terms.
- Licenses were read from PyPI metadata for the locked versions on 2026-11-16 (torch, torchaudio, onnxruntime-gpu and pyannote.audio from their PyPI/project pages). Check upstream for the authoritative terms.
