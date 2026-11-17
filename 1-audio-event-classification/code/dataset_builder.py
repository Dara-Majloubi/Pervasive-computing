
# Build a feature dataset from categorized audio recordings

import re
from pathlib import Path

import pandas as pd

from audio_loader import load_audio
from audio_filter import filter_audio
from framing import frame_audio
from feature_extraction import extract_features


def get_recording_id(category, audio_file):
    return f"{category}/{audio_file.stem}"


def get_source_id(category, audio_file):
    # Segments such as kettle_1000ml_007.wav were cut (with overlap) from ONE
    # long recording. All segments of that recording share one source_id, so
    # cross-validation never puts near-identical audio on both sides.
    stem = re.sub(r"_\d{3}$", "", audio_file.stem)
    return f"{category}/{stem}"


def build_dataset(
    data_dir,
    output_path=None,
    frame_duration=1.0,
    overlap=0.5,
    lowcut=100,
    highcut=8000,
    filter_order=4,
    categories=("stirring", "coffee_machine", "kettle")
):
    # Validate the dataset directory
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {data_dir}")

    if len(categories) < 3:
        raise ValueError("At least three categories must be specified.")

    if "stirring" not in categories:
        raise ValueError("The required stirring category must be included.")

    if len(set(categories)) != len(categories):
        raise ValueError("Category names must be unique.")

    dataset_rows = []
    reference_sample_rate = None

    # Process only the specified category folders
    for category in categories:
        category_dir = data_dir / category

        if not category_dir.exists():
            raise FileNotFoundError(
                f"Category folder not found: {category_dir}"
            )

        if not category_dir.is_dir():
            raise NotADirectoryError(
                f"Expected a category directory: {category_dir}"
            )

        # Find WAV recordings in the category folder
        audio_files = sorted(
            path for path in category_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".wav"
        )

        if not audio_files:
            raise ValueError(
                f"No WAV recordings found in category: {category}"
            )

        # Process each recording
        for audio_file in audio_files:
            # Load and validate the original recording
            sample_rate, audio = load_audio(audio_file)

            # Check sample rate consistency
            if reference_sample_rate is None:
                reference_sample_rate = sample_rate
            elif sample_rate != reference_sample_rate:
                raise ValueError(
                    f"Inconsistent sample rate in {audio_file}. "
                    f"Expected {reference_sample_rate} Hz, "
                    f"received {sample_rate} Hz."
                )

            # Apply band-pass filtering
            filtered_audio = filter_audio(
                audio,
                sample_rate,
                lowcut=lowcut,
                highcut=highcut,
                order=filter_order
            )

            # Divide the filtered signal into overlapping frames
            frames, start_times = frame_audio(
                filtered_audio,
                sample_rate,
                frame_duration=frame_duration,
                overlap=overlap
            )

            # Assign a unique ID to each original WAV recording
            recording_id = get_recording_id(category, audio_file)

            # Extract features from each frame
            for frame_index, (frame, start_time) in enumerate(
                zip(frames, start_times)
            ):
                features = extract_features(frame, sample_rate)

                row = {
                    **features,
                    "label": category,
                    "recording_id": recording_id,
                    "source_id": get_source_id(category, audio_file),
                    "frame_index": frame_index,
                    "start_time": float(start_time)
                }

                dataset_rows.append(row)

    # Create the final feature dataset
    if not dataset_rows:
        raise ValueError("No feature rows were generated.")

    dataset = pd.DataFrame(dataset_rows)

    # Store the reference sample rate
    dataset.attrs["sample_rate"] = reference_sample_rate

    # Display independent recording counts
    print("\nIndependent Recordings per Category:")

    recordings_per_class = (
        dataset.groupby("label")["recording_id"].nunique()
    )

    print(recordings_per_class.to_string())

    print(
        f"\nTotal Independent Recordings: "
        f"{dataset['recording_id'].nunique()}"
    )

    print("\nIndependent Source Recordings per Category:")
    print(dataset.groupby("label")["source_id"].nunique().to_string())

    # Save the dataset when an output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)

    return dataset
