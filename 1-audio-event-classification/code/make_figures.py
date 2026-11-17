# Create figs/examples.pdf and figs/confusion.pdf for the report
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

from audio_loader import load_audio
from audio_filter import filter_audio

project = Path(__file__).resolve().parent.parent
data_dir = project / "data"
out_dir = project / "report" / "figs"
out_dir.mkdir(parents=True, exist_ok=True)

def representative_clip(category):
    # Pick the clip whose RMS is closest to the median RMS of its category
    files = sorted((data_dir / category).glob("*.wav"))
    rms = []
    for path in files:
        sr, audio = load_audio(path)
        rms.append(np.sqrt(np.mean(filter_audio(audio, sr) ** 2)))
    return files[int(np.argmin(np.abs(np.array(rms) - np.median(rms))))]


examples = {name: representative_clip(name)
            for name in ("stirring", "coffee_machine", "kettle")}
print({k: v.name for k, v in examples.items()})

fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.0), sharey="row")
for col, (name, path) in enumerate(examples.items()):
    sr, audio = load_audio(path)
    audio = filter_audio(audio, sr)
    t = np.arange(len(audio)) / sr
    axes[0, col].plot(t, audio, lw=0.3, color="#1f4e79")
    axes[0, col].set_xlim(0, 4.5)            # same time scale
    axes[0, col].set_ylim(-1, 1)
    axes[0, col].set_title(name.replace("_", " "), fontsize=8)
    axes[0, col].set_xlabel("time (s)", fontsize=7)

    window = np.hanning(len(audio))
    spec = np.abs(rfft(audio * window)) / len(audio)
    freq = rfftfreq(len(audio), 1 / sr)
    axes[1, col].plot(freq / 1000, 20 * np.log10(spec + 1e-9), lw=0.3, color="#9c3d0a")
    axes[1, col].set_xlim(0, 10)             # same frequency scale
    axes[1, col].set_ylim(-110, -20)
    axes[1, col].set_xlabel("frequency (kHz)", fontsize=7)
axes[0, 0].set_ylabel("amplitude", fontsize=7)
axes[1, 0].set_ylabel("magnitude (dB)", fontsize=7)
for ax in axes.ravel():
    ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(out_dir / "examples.pdf")

# Confusion matrix of the final choice (summed over all folds)
cm = pd.read_csv(project / "data" / "processed" / "confusion_matrix_KNN_(k=3).csv", index_col=0)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.imshow(cm.values, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels([c.replace("_", "\n") for c in cm.columns], fontsize=7)
ax.set_yticks(range(3)); ax.set_yticklabels(cm.index, fontsize=7)
ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("actual", fontsize=8)
for i in range(3):
    for j in range(3):
        ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", fontsize=9,
                color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
fig.tight_layout()
fig.savefig(out_dir / "confusion.pdf")
print("figures written to", out_dir)
