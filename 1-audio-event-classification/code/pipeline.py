
# Run the complete audio classification pipeline

from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from dataset_builder import build_dataset
from classifier import create_classifiers
from evaluation import evaluate_classifiers


def main():
    # Define the project directories
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    output_dir = data_dir / "processed"
    model_dir = project_dir / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Define the pipeline configuration
    frame_duration = 1.0
    overlap = 0.5
    lowcut = 100
    highcut = 8000
    filter_order = 4
    n_splits = 4
    random_state = 42

    # Display the pipeline configuration
    print("Audio Classification Pipeline")
    print("=============================")
    print(f"Data Directory: {data_dir}")
    print(f"Frame Duration: {frame_duration} seconds")
    print(f"Frame Overlap: {overlap * 100:.0f}%")
    print(f"Filter Range: {lowcut}-{highcut} Hz")
    print(f"Filter Order: {filter_order}")
    print(f"Cross-Validation Folds: {n_splits}")
    print(f"Random State: {random_state}")
    print()

    # Build the feature dataset
    print("Step 1: Building the feature dataset...")

    dataset = build_dataset(
        data_dir=data_dir,
        output_path=output_dir / "features.csv",
        frame_duration=frame_duration,
        overlap=overlap,
        lowcut=lowcut,
        highcut=highcut,
        filter_order=filter_order
    )

    # Validate the generated dataset
    if dataset["label"].nunique() < 3:
        raise ValueError("At least three audio categories are required.")

    if "stirring" not in dataset["label"].unique():
        raise ValueError("The required stirring category is missing.")

    # Retrieve the reference sample rate
    reference_sample_rate = dataset.attrs.get("sample_rate")

    if reference_sample_rate is None or reference_sample_rate <= 0:
        raise ValueError("A valid reference sample rate was not found.")

    print(f"Reference Sample Rate: {reference_sample_rate} Hz")
    print()

    # Select the feature columns
    metadata_columns = [
        "label",
        "recording_id",
        "source_id",
        "frame_index",
        "start_time"
    ]

    feature_columns = [
        column for column in dataset.columns
        if column not in metadata_columns
    ]

    if not feature_columns:
        raise ValueError("No feature columns were found.")

    # Validate feature values
    X = dataset[feature_columns].to_numpy(dtype=np.float64)
    y = dataset["label"].to_numpy()

    if not np.all(np.isfinite(X)):
        raise ValueError("The feature dataset contains invalid values.")

    print(f"Feature Rows: {len(dataset)}")
    print(f"Number of Features: {len(feature_columns)}")
    print(f"Number of Categories: {dataset['label'].nunique()}")
    print(f"Number of Recordings: {dataset['recording_id'].nunique()}")
    print()

    # Display the category distribution
    print("Category Distribution:")
    print(dataset["label"].value_counts().to_string())
    print()

    # Create the classifiers
    print("Step 2: Creating the classifiers...")

    classifiers = create_classifiers(
        random_state=random_state
    )

    for name in classifiers:
        print(f"Created: {name}")

    print()

    # Evaluate all classifiers
    print("Step 3: Evaluating the classifiers...")

    results = evaluate_classifiers(
        dataset=dataset,
        classifiers=classifiers,
        n_splits=n_splits,
        random_state=random_state
    )

    # Save the fold-level evaluation results
    fold_results_path = output_dir / "fold_results.csv"

    results["fold_results"].to_csv(
        fold_results_path,
        index=False
    )

    # Save the evaluation summary
    summary = results["summary"].copy()
    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]

    summary_path = output_dir / "evaluation_summary.csv"

    summary.to_csv(summary_path)

    # Save the confusion matrix for each classifier
    for name, matrix in results["confusion_matrices"].items():
        matrix_df = pd.DataFrame(
            matrix,
            index=results["classes"],
            columns=results["classes"]
        )

        matrix_df.index.name = "Actual"

        safe_name = name.replace(" ", "_")

        matrix_path = (
            output_dir / f"confusion_matrix_{safe_name}.csv"
        )

        matrix_df.to_csv(matrix_path)

    # Display the evaluation summary
    print()
    print("Evaluation Summary")
    print("------------------")
    print(summary.round(4).to_string())

    # Display all confusion matrices
    print()
    print("Confusion Matrices")

    for name, matrix in results["confusion_matrices"].items():
        print()
        print(f"{name}:")
        print(
            pd.DataFrame(
                matrix,
                index=results["classes"],
                columns=results["classes"]
            ).to_string()
        )

    # Train the selected KNN model on the complete dataset
    print()
    print("Step 4: Training the final KNN model...")

    final_model = classifiers["KNN (k=3)"]
    final_model.fit(X, y)

    # Save the trained model and its metadata
    model_artifact = {
        "model": final_model,
        "feature_columns": feature_columns,
        "classes": results["classes"].tolist(),
        "sample_rate": reference_sample_rate,
        "frame_duration": frame_duration,
        "overlap": overlap,
        "lowcut": lowcut,
        "highcut": highcut,
        "filter_order": filter_order,
        "classifier": "KNN",
        "n_neighbors": 3
    }

    model_path = model_dir / "knn_model.joblib"

    joblib.dump(model_artifact, model_path)

    print(f"Final Model: {model_path}")
    print("Final KNN model trained and saved successfully.")

    # Display the saved output files
    print()
    print("Saved Output Files")
    print("------------------")
    print(f"Feature Dataset: {output_dir / 'features.csv'}")
    print(f"Fold Results: {fold_results_path}")
    print(f"Evaluation Summary: {summary_path}")

    for name in results["confusion_matrices"]:
        safe_name = name.replace(" ", "_")
        print(
            f"Confusion Matrix ({name}): "
            f"{output_dir / f'confusion_matrix_{safe_name}.csv'}"
        )

    print(f"Trained Model: {model_path}")

    print()
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
