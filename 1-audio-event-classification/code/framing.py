
# Divide an audio signal into overlapping frames

import numpy as np


def frame_audio(audio, sample_rate, frame_duration=1.0, overlap=0.5):
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

    # Validate the framing parameters
    if frame_duration <= 0:
        raise ValueError("Frame duration must be positive.")

    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be between 0 and 1.")

    # Calculate frame length and hop length in samples
    frame_length = int(round(frame_duration * sample_rate))
    hop_length = int(round(frame_length * (1 - overlap)))

    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("Invalid frame or hop length.")

    if len(audio) < frame_length:
        raise ValueError("Audio is shorter than one complete frame.")

    # Calculate the number of complete frames
    num_frames = 1 + (len(audio) - frame_length) // hop_length

    # Extract frames and their start times
    frames = np.empty((num_frames, frame_length), dtype=np.float32)
    start_times = np.empty(num_frames, dtype=np.float64)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length

        frames[i] = audio[start:end]
        start_times[i] = start / sample_rate

    return frames, start_times
