
# Load and validate WAV audio files

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def load_audio(file_path):
    # Check whether the input file exists and has a WAV extension
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if file_path.suffix.lower() != ".wav":
        raise ValueError("Only WAV files are supported.")

    # Read the audio file
    sample_rate, audio = wavfile.read(file_path)

    # Validate the audio channels and samples
    if audio.ndim != 1:
        raise ValueError("Audio file must be mono.")

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    # Convert integer samples to floating-point values
    if np.issubdtype(audio.dtype, np.integer):
        if np.issubdtype(audio.dtype, np.unsignedinteger):
            max_value = np.iinfo(audio.dtype).max
            midpoint = (max_value + 1) / 2
            audio = (audio.astype(np.float32) - midpoint) / midpoint
        else:
            max_value = max(
                abs(np.iinfo(audio.dtype).min),
                np.iinfo(audio.dtype).max
            )
            audio = audio.astype(np.float32) / max_value

    elif np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32)

    else:
        raise TypeError("Unsupported audio data type.")

    # Check for invalid sample values
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numerical values.")

    # Calculate and validate the audio duration
    duration = len(audio) / sample_rate

    if not 3 <= duration <= 8:
        raise ValueError(
            f"Audio duration must be between 3 and 8 seconds. "
            f"Current duration: {duration:.2f} seconds."
        )

    return sample_rate, audio
