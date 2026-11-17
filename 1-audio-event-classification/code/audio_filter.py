
# Apply a Butterworth band-pass filter to an audio signal

import numpy as np
from scipy.signal import butter, sosfiltfilt


def filter_audio(audio, sample_rate, lowcut=100, highcut=8000, order=4):
    # Validate the input audio signal
    audio = np.asarray(audio)

    if audio.ndim != 1:
        raise ValueError("Audio signal must be mono.")

    if audio.size == 0:
        raise ValueError("Audio signal is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numerical values.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    # Validate the filter parameters
    nyquist_frequency = sample_rate / 2

    if not 0 < lowcut < highcut < nyquist_frequency:
        raise ValueError(
            "Cutoff frequencies must satisfy "
            "0 < lowcut < highcut < Nyquist frequency."
        )

    if not isinstance(order, int) or order < 1:
        raise ValueError("Filter order must be a positive integer.")

    # Design the Butterworth band-pass filter
    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sample_rate,
        output="sos"
    )

    # Apply zero-phase filtering
    filtered_audio = sosfiltfilt(sos, audio)

    return filtered_audio.astype(np.float32)
