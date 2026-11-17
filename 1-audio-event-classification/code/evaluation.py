
# Evaluate classifiers using grouped, stratified cross-validation

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)


def grouped_stratified_folds(y, groups, n_splits):
    # Deterministic grouped, stratified folds (same result on every machine).
    # For each class, whole source recordings are assigned greedily (largest
    # first) to the fold that currently holds the fewest frames of that class.
    # Every fold therefore contains each class, and no source is ever split.
    fold_of_group = {}

    for label in np.unique(y):
        mask = y == label
        names, counts = np.unique(groups[mask], return_counts=True)
        order = sorted(zip(names, counts), key=lambda item: (-item[1], item[0]))
        load = np.zeros(n_splits)

        for name, count in order:
            fold = int(np.argmin(load))
            fold_of_group[name] = fold
            load[fold] += count

    fold_ids = np.array([fold_of_group[g] for g in groups])

    for fold in range(n_splits):
        test = np.where(fold_ids == fold)[0]
        train = np.where(fold_ids != fold)[0]
        yield train, test


def evaluate_classifiers(
    dataset,
    classifiers,
    n_splits=4,
    random_state=42,
    group_column="source_id"
):
    # Validate the input dataset
    required_columns = [
        "label",
        group_column
    ]

    for column in required_columns:
        if column not in dataset.columns:
            raise ValueError(f"Missing required column: {column}")

    if dataset.empty:
        raise ValueError("Dataset is empty.")

    if not classifiers:
        raise ValueError("No classifiers were provided.")

    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("Number of folds must be an integer of at least 2.")

    if dataset["label"].isna().any():
        raise ValueError("Dataset contains missing labels.")

    if dataset[group_column].isna().any():
        raise ValueError("Dataset contains missing group IDs.")

    # Identify feature columns
    excluded_columns = {
        "label",
        "recording_id",
        "source_id",
        "frame_index",
        "start_time"
    }

    feature_columns = [
        column for column in dataset.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No feature columns found.")

    X = dataset[feature_columns].to_numpy(dtype=np.float64)
    y = dataset["label"].to_numpy()
    groups = dataset[group_column].to_numpy()

    if not np.all(np.isfinite(X)):
        raise ValueError("Feature dataset contains invalid values.")

    # Validate the number of independent recordings per class
    recordings_per_class = (
        dataset.groupby("label")[group_column].nunique()
    )

    if (recordings_per_class < n_splits).any():
        raise ValueError(
            f"Each class must contain at least {n_splits} "
            "independent recordings."
        )

    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            "The number of independent recordings is smaller "
            "than the number of folds."
        )

    classes = np.unique(y)

    # Store fold results and out-of-fold predictions
    fold_results = []

    predictions = {
        name: np.full(len(y), None, dtype=object)
        for name in classifiers
    }

    evaluation_counts = np.zeros(len(y), dtype=int)

    # Evaluate each fold
    for fold_index, (train_indices, test_indices) in enumerate(
        grouped_stratified_folds(y, groups, n_splits),
        start=1
    ):
        # Separate training and testing data
        X_train = X[train_indices]
        X_test = X[test_indices]

        y_train = y[train_indices]
        y_test = y[test_indices]

        groups_train = groups[train_indices]
        groups_test = groups[test_indices]

        # Verify that training and testing groups do not overlap
        if set(groups_train).intersection(set(groups_test)):
            raise RuntimeError("Data leakage detected between groups.")

        # Verify that all classes are present in both sets
        if not np.array_equal(np.unique(y_train), classes):
            raise RuntimeError(
                f"Some classes are missing from training fold {fold_index}."
            )

        if not np.array_equal(np.unique(y_test), classes):
            raise RuntimeError(
                f"Some classes are missing from testing fold {fold_index}."
            )

        # Track the number of times each sample is evaluated
        evaluation_counts[test_indices] += 1

        # Calculate the majority class baseline
        majority_class = pd.Series(y_train).value_counts().idxmax()

        baseline_predictions = np.full(
            len(y_test),
            majority_class
        )

        baseline_accuracy = accuracy_score(
            y_test,
            baseline_predictions
        )

        baseline_f1 = f1_score(
            y_test,
            baseline_predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )

        fold_results.append({
            "fold": fold_index,
            "classifier": "Majority Baseline",
            "accuracy": baseline_accuracy,
            "macro_f1": baseline_f1
        })

        # Train and evaluate each classifier
        for name, model in classifiers.items():
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            predictions[name][test_indices] = y_pred

            fold_accuracy = accuracy_score(
                y_test,
                y_pred
            )

            fold_f1 = f1_score(
                y_test,
                y_pred,
                labels=classes,
                average="macro",
                zero_division=0
            )

            fold_results.append({
                "fold": fold_index,
                "classifier": name,
                "accuracy": fold_accuracy,
                "macro_f1": fold_f1
            })

    # Verify that every sample was evaluated exactly once
    if not np.all(evaluation_counts == 1):
        raise RuntimeError(
            "Each sample must receive exactly one out-of-fold prediction."
        )

    # Convert fold results into a DataFrame
    fold_results = pd.DataFrame(fold_results)

    # Calculate mean and standard deviation for each classifier
    summary = (
        fold_results
        .groupby("classifier")[["accuracy", "macro_f1"]]
        .agg(["mean", "std"])
    )

    # Calculate confusion matrices from all out-of-fold predictions
    confusion_matrices = {}

    for name, y_pred in predictions.items():
        if any(prediction is None for prediction in y_pred):
            raise RuntimeError(
                f"Some recordings were not evaluated for {name}."
            )

        confusion_matrices[name] = confusion_matrix(
            y,
            y_pred,
            labels=classes
        )

    return {
        "fold_results": fold_results,
        "summary": summary,
        "confusion_matrices": confusion_matrices,
        "classes": classes,
        "predictions": predictions
    }
