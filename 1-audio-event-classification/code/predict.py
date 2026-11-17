
# Classify all WAV recordings in the test directory

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from audio_loader import load_audio
from audio_filter import filter_audio
from framing import frame_audio
from feature_extraction import extract_features


def get_file_number(path, valid_pattern):
    # Extract the numeric identifier from a valid filename
    match = valid_pattern.fullmatch(path.name)

    if match is None:
        raise ValueError(f"Invalid test filename: {path.name}")

    return int(match.group(1))


def prepare_test_files(test_dir):
    # Find all WAV files in the test directory
    audio_files = sorted(
        path for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )

    # Identify files that follow the naming pattern
    valid_pattern = re.compile(r"test_(\d+)\.wav", re.IGNORECASE)

    valid_files = []
    invalid_files = []

    for audio_file in audio_files:
        if valid_pattern.fullmatch(audio_file.name):
            valid_files.append(audio_file)
        else:
            invalid_files.append(audio_file)

    # Find the next available file number
    used_numbers = []

    for audio_file in valid_files:
        used_numbers.append(
            get_file_number(audio_file, valid_pattern)
        )

    next_number = max(used_numbers, default=0) + 1

    # Rename files with invalid names
    renamed_files = {}

    for audio_file in invalid_files:
        while True:
            new_name = f"test_{next_number:02d}.wav"
            new_path = test_dir / new_name
            next_number += 1

            if not new_path.exists():
                break

        print(
            f"WARNING: Invalid filename detected: "
            f"{audio_file.name}"
        )

        audio_file.rename(new_path)

        renamed_files[new_path.name] = audio_file.name

        print(f"Renamed to: {new_path.name}")
        print()

        valid_files.append(new_path)

    # Sort files by their numeric identifiers
    valid_files.sort(
        key=lambda path: get_file_number(path, valid_pattern)
    )

    return valid_files, renamed_files


def predict_single_file(audio_path, model_artifact):
    # Load the trained model and metadata
    model = model_artifact["model"]
    feature_columns = model_artifact["feature_columns"]

    frame_duration = model_artifact["frame_duration"]
    overlap = model_artifact["overlap"]
    lowcut = model_artifact["lowcut"]
    highcut = model_artifact["highcut"]
    filter_order = model_artifact["filter_order"]
    expected_sample_rate = model_artifact["sample_rate"]

    # Load the audio recording
    sample_rate, audio = load_audio(audio_path)

    # Validate the sample rate
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Sample rate mismatch. Expected "
            f"{expected_sample_rate} Hz, "
            f"received {sample_rate} Hz."
        )

    # Apply the band-pass filter
    filtered_audio = filter_audio(
        audio,
        sample_rate,
        lowcut=lowcut,
        highcut=highcut,
        order=filter_order
    )

    # Divide the audio into overlapping frames
    frames, start_times = frame_audio(
        filtered_audio,
        sample_rate,
        frame_duration=frame_duration,
        overlap=overlap
    )

    if len(frames) == 0:
        raise ValueError("No valid audio frames were generated.")

    # Extract features from each frame
    feature_rows = []

    for frame in frames:
        features = extract_features(frame, sample_rate)
        feature_rows.append(features)

    # Arrange features in the training feature order
    feature_dataset = pd.DataFrame(
        feature_rows,
        columns=feature_columns
    )

    X = feature_dataset.to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(X)):
        raise ValueError("Extracted features contain invalid values.")

    # Predict the class of each frame
    frame_predictions = model.predict(X)

    # Count predictions for each class
    prediction_counts = (
        pd.Series(frame_predictions)
        .value_counts()
        .reindex(model_artifact["classes"], fill_value=0)
    )

    # Determine the final prediction
    final_prediction = prediction_counts.idxmax()

    # Calculate prediction percentages
    prediction_percentages = (
        prediction_counts / len(frame_predictions) * 100
    )

    return {
        "sample_rate": sample_rate,
        "frames": len(frames),
        "start_times": start_times,
        "frame_predictions": frame_predictions,
        "prediction_counts": prediction_counts,
        "prediction_percentages": prediction_percentages,
        "final_prediction": final_prediction
    }


def run_batch_prediction(test_dir, model_path):
    # Validate the input directories
    test_dir = Path(test_dir)
    model_path = Path(model_path)

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    if not test_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a test directory: {test_dir}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    # Load the trained model
    model_artifact = joblib.load(model_path)

    if not isinstance(model_artifact, dict):
        raise ValueError("Invalid model artifact format.")

    required_keys = [
        "model",
        "feature_columns",
        "classes",
        "sample_rate",
        "frame_duration",
        "overlap",
        "lowcut",
        "highcut",
        "filter_order"
    ]

    missing_keys = [
        key for key in required_keys
        if key not in model_artifact
    ]

    if missing_keys:
        raise ValueError(
            f"Model artifact is missing required fields: {missing_keys}"
        )

    if model_artifact["sample_rate"] <= 0:
        raise ValueError("Invalid sample rate in model artifact.")

    # Prepare and rename test files
    print("Preparing test files...")
    print()

    audio_files, renamed_files = prepare_test_files(test_dir)

    if not audio_files:
        print("No WAV files found in the test directory.")
        return

    print(f"Total WAV Files: {len(audio_files)}")
    print()

    # Store batch results
    batch_results = []

    # Process each audio file
    for file_index, audio_path in enumerate(audio_files, start=1):
        original_name = renamed_files.get(
            audio_path.name,
            audio_path.name
        )

        print(
            f"[{file_index}/{len(audio_files)}] "
            f"Processing: {audio_path.name}"
        )

        try:
            result = predict_single_file(
                audio_path,
                model_artifact
            )

            print(f"Sample Rate: {result['sample_rate']} Hz")
            print(f"Number of Frames: {result['frames']}")
            print()
            print("Frame Predictions:")

            for frame_index, prediction in enumerate(
                result["frame_predictions"]
            ):
                print(
                    f"Frame {frame_index + 1}: {prediction} "
                    f"(Start: "
                    f"{result['start_times'][frame_index]:.2f}s)"
                )

            print()
            print("Prediction Summary")

            for category in result["prediction_counts"].index:
                count = result["prediction_counts"][category]
                percentage = result["prediction_percentages"][category]

                print(
                    f"{category}: {count} frames "
                    f"({percentage:.1f}%)"
                )

            print()
            print(
                f"Final Prediction: "
                f"{result['final_prediction']}"
            )

            batch_results.append({
                "original_filename": original_name,
                "filename": audio_path.name,
                "predicted_class": result["final_prediction"],
                "frames": result["frames"],
                "status": "Success",
                "error": ""
            })

        except Exception as error:
            print(f"ERROR: {error}")

            batch_results.append({
                "original_filename": original_name,
                "filename": audio_path.name,
                "predicted_class": "",
                "frames": 0,
                "status": "Error",
                "error": str(error)
            })

        print()

    # Create the batch results DataFrame
    results_df = pd.DataFrame(batch_results)

    # Save the results in the test directory
    report_path = test_dir / "test_results.csv"

    results_df.to_csv(
        report_path,
        index=False
    )

    # Calculate batch statistics
    successful_results = results_df[
        results_df["status"] == "Success"
    ]

    failed_results = results_df[
        results_df["status"] == "Error"
    ]

    # Display the batch summary
    print("BATCH TEST SUMMARY:")
    print()
    print(f"Total Files: {len(results_df)}")
    print(f"Successfully Processed: {len(successful_results)}")
    print(f"Processing Errors: {len(failed_results)}")
    print(f"Renamed Files: {len(renamed_files)}")
    print()

    print("Class Predictions:")

    if not successful_results.empty:
        class_counts = (
            successful_results["predicted_class"]
            .value_counts()
        )

        for category, count in class_counts.items():
            print(f"{category}: {count}")
    else:
        print("No successful predictions.")

    print()
    print(f"Results File: {report_path}")


def main():
    # Define project paths
    project_dir = Path(__file__).resolve().parent.parent

    default_test_dir = project_dir / "new_test"
    default_model_path = project_dir / "models" / "knn_model.joblib"

    # Define command-line arguments
    parser = argparse.ArgumentParser(
        description="Classify all WAV files in a test directory."
    )

    parser.add_argument(
        "--test-dir",
        default=str(default_test_dir),
        help="Directory containing test WAV files."
    )

    parser.add_argument(
        "--model",
        default=str(default_model_path),
        help="Path to the trained model."
    )

    parser.add_argument(
        "--file",
        default=None,
        help="Classify ONE wav file and print its category (no renaming)."
    )

    args = parser.parse_args()

    if args.file is not None:
        artifact = joblib.load(args.model)
        result = predict_single_file(Path(args.file), artifact)
        print(result["final_prediction"])
        return

    run_batch_prediction(
        test_dir=args.test_dir,
        model_path=args.model
    )


if __name__ == "__main__":
    main()
