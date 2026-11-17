import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_FILE = "data/imu_trials.csv"
MODEL_FILE = "imu_model.joblib"

WINDOW_SIZE = 10
STEP_SIZE = 5

SENSOR_COLUMNS = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z"
]


def extract_features(window):
    features = []

    for column in SENSOR_COLUMNS:
        values = window[column].to_numpy()

        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.sqrt(np.mean(values ** 2))
        ])

    return features


def create_windows(data):
    X = []
    y = []

    for _, trial in data.groupby("trial_id"):
        trial = trial.reset_index(drop=True)

        for start in range(0, len(trial) - WINDOW_SIZE + 1, STEP_SIZE):
            window = trial.iloc[start:start + WINDOW_SIZE]

            features = extract_features(window)
            label = window["label"].mode()[0]

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


def main():
    data = pd.read_csv(DATA_FILE)

    train_data = data[data["split"] == "train"]
    test_data = data[data["split"] == "test"]

    X_train, y_train = create_windows(train_data)
    X_test, y_test = create_windows(test_data)

    print("Training windows:", len(X_train))
    print("Testing windows:", len(X_test))
    print("Features per window:", X_train.shape[1])

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    labels = ["idle", "raise", "rotate", "translate"]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=labels))

    joblib.dump(model, MODEL_FILE)

    print("\nModel saved to:", MODEL_FILE)


if __name__ == "__main__":
    main()