from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from charavoice_engine.models.utils import checkpoints, safetensors  # noqa: F401

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
_matplotlib_loaded = False


def summarize(
    writer: SummaryWriter,
    global_step: int,
    scalars: dict[str, Any] | None = None,
    histograms: dict[str, Any] | None = None,
    images: dict[str, Any] | None = None,
    audios: dict[str, Any] | None = None,
    audio_sampling_rate: int = 44100,
) -> None:
    """Write TensorBoard summaries."""

    for key, value in (scalars or {}).items():
        writer.add_scalar(key, value, global_step)
    for key, value in (histograms or {}).items():
        writer.add_histogram(key, value, global_step)
    for key, value in (images or {}).items():
        writer.add_image(key, value, global_step, dataformats="HWC")
    for key, value in (audios or {}).items():
        writer.add_audio(key, value, global_step, audio_sampling_rate)


def load_wav_to_torch(full_path: str | Path) -> tuple[torch.FloatTensor, int]:
    """Load a wav file as float tensor and return (audio, sample_rate)."""

    from scipy.io.wavfile import read  # pyright: ignore[reportMissingImports]

    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), int(sampling_rate)


def load_filepaths_and_text(filename: str | Path, split: str = "|") -> list[list[str]]:
    """Load delimited metadata rows from a text file."""

    with Path(filename).open(encoding="utf-8") as f:
        return [line.strip().split(split) for line in f if line.strip()]


def plot_spectrogram_to_numpy(spectrogram: NDArray[Any]) -> NDArray[Any]:
    """Render a spectrogram image for TensorBoard."""

    global _matplotlib_loaded
    if not _matplotlib_loaded:
        import matplotlib  # pyright: ignore[reportMissingImports]

        matplotlib.use("Agg")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        _matplotlib_loaded = True
    import matplotlib.pylab as plt  # pyright: ignore[reportMissingImports]

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    canvas = cast(Any, fig.canvas)
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment: NDArray[Any], info: str | None = None) -> NDArray[Any]:
    """Render an alignment image for TensorBoard."""

    global _matplotlib_loaded
    if not _matplotlib_loaded:
        import matplotlib  # pyright: ignore[reportMissingImports]

        matplotlib.use("Agg")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        _matplotlib_loaded = True
    import matplotlib.pylab as plt  # pyright: ignore[reportMissingImports]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()
    canvas = cast(Any, fig.canvas)
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
