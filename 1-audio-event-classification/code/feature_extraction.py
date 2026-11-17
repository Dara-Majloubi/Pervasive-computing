
# Extract time-domain, frequency-domain, and MFCC features from an audio frame

import numpy as np
from scipy.fft import dct


def hz_to_mel(frequencies):
    # Convert frequencies from Hz to the Slaney mel scale
    frequencies = np.asarray(frequencies, dtype=np.float64).copy()

    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    mel_frequencies = frequencies / f_sp
    log_region = frequencies >= min_log_hz

    mel_frequencies[log_region] = (
        min_log_mel
        + np.log(frequencies[log_region] / min_log_hz) / logstep
    )

    return mel_frequencies


def mel_to_hz(mel_frequencies):
    # Convert frequencies from the Slaney mel scale to Hz
    mel_frequencies = np.asarray(
        mel_frequencies, dtype=np.float64
    ).copy()

    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    frequencies = mel_frequencies * f_sp
    log_region = mel_frequencies >= min_log_mel

    frequencies[log_region] = (
        min_log_hz
        * np.exp(logstep * (mel_frequencies[log_region] - min_log_mel))
    )

    return frequencies


def create_mel_filter_bank(
    sample_rate,
    n_fft,
    n_mels=128,
    fmin=0.0,
    fmax=None
):
    # Validate the filter bank parameters
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if n_fft < 2:
        raise ValueError("FFT size must be at least 2.")

    if not isinstance(n_mels, int) or n_mels < 1:
        raise ValueError("Number of Mel filters must be a positive integer.")

    if fmax is None:
        fmax = sample_rate / 2

    if not 0 <= fmin < fmax <= sample_rate / 2:
        raise ValueError("Invalid Mel filter frequency range.")

    # Create triangular filters spaced on the Slaney mel scale
    mel_limits = hz_to_mel(np.array([fmin, fmax]))
    mel_points = np.linspace(mel_limits[0], mel_limits[1], n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fft_frequencies = np.fft.rfftfreq(
        n_fft,
        d=1.0 / sample_rate
    )

    filter_bank = np.zeros(
        (n_mels, len(fft_frequencies)),
        dtype=np.float64
    )

    for i in range(n_mels):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]

        lower_slope = (fft_frequencies - lower) / (center - lower)
        upper_slope = (upper - fft_frequencies) / (upper - center)

        filter_bank[i] = np.maximum(
            0.0,
            np.minimum(lower_slope, upper_slope)
        )

    # Apply Slaney area normalization
    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    filter_bank *= enorm[:, np.newaxis]

    return filter_bank


def calculate_mfcc(
    frame,
    sample_rate,
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
    n_mels=128
):
    # Validate the input frame and MFCC parameters
    frame = np.asarray(frame, dtype=np.float64)

    if frame.ndim != 1:
        raise ValueError("Audio frame must be mono.")

    if not np.all(np.isfinite(frame)):
        raise ValueError("Audio frame contains invalid numerical values.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if not isinstance(n_fft, int) or n_fft < 2:
        raise ValueError("FFT size must be an integer of at least 2.")

    if not isinstance(hop_length, int) or hop_length < 1:
        raise ValueError("Hop length must be a positive integer.")

    if not isinstance(n_mels, int) or n_mels < 1:
        raise ValueError("Number of Mel filters must be a positive integer.")

    if not isinstance(n_mfcc, int) or n_mfcc < 1:
        raise ValueError("Number of MFCC coefficients must be a positive integer.")

    if n_mfcc > n_mels:
        raise ValueError("Number of MFCC coefficients cannot exceed n_mels.")

    if len(frame) < n_fft:
        raise ValueError("Audio frame must be at least n_fft samples long.")

    # Create overlapping analysis frames
    frame_count = 1 + (len(frame) - n_fft) // hop_length
    frames = np.empty((frame_count, n_fft), dtype=np.float64)

    for i in range(frame_count):
        start = i * hop_length
        frames[i] = frame[start:start + n_fft]

    # Apply a periodic Hann window
    window = np.hanning(n_fft + 1)[:-1]
    windowed_frames = frames * window[np.newaxis, :]

    # Calculate the power spectrum of each analysis frame
    spectrum = np.abs(
        np.fft.rfft(windowed_frames, axis=1)
    ) ** 2

    # Create the Mel filter bank and calculate Mel energies
    mel_filters = create_mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels
    )

    mel_energies = spectrum @ mel_filters.T

    # Convert Mel energies to decibels
    amin = 1e-10
    log_mel = 10.0 * np.log10(np.maximum(amin, mel_energies))

    # Apply the 80 dB dynamic range limit
    log_mel = np.maximum(log_mel, np.max(log_mel) - 80.0)

    # Apply the type-II discrete cosine transform
    mfcc_values = np.asarray(
        dct(
            log_mel,
            type=2,
            axis=1,
            norm="ortho"
        ),
        dtype=np.float64
    )

    return mfcc_values[:, :n_mfcc].T


def extract_features(frame, sample_rate, rolloff_percent=0.85, n_mfcc=13):
    # Validate the input frame
    frame = np.asarray(frame, dtype=np.float64)

    if frame.ndim != 1:
        raise ValueError("Audio frame must be mono.")

    if frame.size < 2:
        raise ValueError("Audio frame must contain at least two samples.")

    if not np.all(np.isfinite(frame)):
        raise ValueError("Audio frame contains invalid numerical values.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if not 0 < rolloff_percent <= 1:
        raise ValueError("Rolloff percentage must be between 0 and 1.")

    if not isinstance(n_mfcc, int) or n_mfcc < 1:
        raise ValueError("Number of MFCC coefficients must be positive.")

    # Calculate time-domain features
    rms_energy = np.sqrt(np.mean(frame ** 2))

    zero_crossings = np.sum(np.diff(np.signbit(frame)) != 0)
    zero_crossing_rate = zero_crossings / (len(frame) - 1)

    # Apply a Hann window before calculating the FFT
    window = np.hanning(len(frame))
    windowed_frame = frame * window

    # Calculate the one-sided FFT and frequency bins
    fft_result = np.fft.rfft(windowed_frame)
    frequencies = np.fft.rfftfreq(
        len(frame),
        d=1.0 / sample_rate
    )

    magnitude_spectrum = np.abs(fft_result)
    power_spectrum = magnitude_spectrum ** 2

    # Calculate spectral centroid
    total_magnitude = np.sum(magnitude_spectrum)

    if total_magnitude > 0:
        spectral_centroid = (
            np.sum(frequencies * magnitude_spectrum) / total_magnitude
        )
    else:
        spectral_centroid = 0.0

    # Calculate spectral rolloff
    total_power = np.sum(power_spectrum)

    if total_power > 0:
        cumulative_power = np.cumsum(power_spectrum)
        rolloff_threshold = rolloff_percent * total_power
        rolloff_index = np.searchsorted(cumulative_power, rolloff_threshold)
        rolloff_index = min(rolloff_index, len(frequencies) - 1)
        spectral_rolloff = frequencies[rolloff_index]
    else:
        spectral_rolloff = 0.0

    # Calculate spectral flatness
    if np.any(power_spectrum > 0):
        epsilon = np.finfo(np.float64).eps
        positive_power = power_spectrum + epsilon
        geometric_mean = np.exp(np.mean(np.log(positive_power)))
        arithmetic_mean = np.mean(positive_power)
        spectral_flatness = geometric_mean / arithmetic_mean
    else:
        spectral_flatness = 0.0

    # Calculate relative energy in frequency bands
    band_edges = [0, 250, 1000, 4000, sample_rate / 2]
    band_energies = []

    for i in range(len(band_edges) - 1):
        band_mask = (
            (frequencies >= band_edges[i])
            & (frequencies < band_edges[i + 1])
        )
        band_power = np.sum(power_spectrum[band_mask])

        if total_power > 0:
            relative_energy = band_power / total_power
        else:
            relative_energy = 0.0

        band_energies.append(relative_energy)

    # Calculate MFCC coefficients using the manual implementation
    mfcc_values = calculate_mfcc(
        frame=frame,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    # Calculate the mean and standard deviation of each MFCC coefficient
    mfcc_means = np.mean(mfcc_values, axis=1)
    mfcc_stds = np.std(mfcc_values, axis=1)

    # Combine all features into an ordered dictionary
    features = {
        "rms_energy": float(rms_energy),
        "zero_crossing_rate": float(zero_crossing_rate),
        "spectral_centroid": float(spectral_centroid),
        "spectral_rolloff": float(spectral_rolloff),
        "spectral_flatness": float(spectral_flatness),
        "low_band_energy": float(band_energies[0]),
        "low_mid_band_energy": float(band_energies[1]),
        "high_mid_band_energy": float(band_energies[2]),
        "high_band_energy": float(band_energies[3])
    }

    # Add MFCC mean and standard deviation features
    for i in range(n_mfcc):
        features[f"mfcc_{i + 1:02d}_mean"] = float(mfcc_means[i])
        features[f"mfcc_{i + 1:02d}_std"] = float(mfcc_stds[i])

    return features
