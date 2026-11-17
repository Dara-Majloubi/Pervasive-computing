import time
import joblib
import numpy as np

from collections import deque
from sense_hat import SenseHat


MODEL_FILE = "imu_model.joblib"

WINDOW_SIZE = 10
STEP_SIZE = 5
SAMPLE_INTERVAL = 0.05

SEQUENCE = ["raise", "rotate", "translate"]
TIMEOUT = 20

CONFIRMATION_WINDOWS = 3

SENSOR_COLUMNS = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z"
]


def extract_features(window):
    features = []

    for i in range(6):
        values = window[:, i]

        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.sqrt(np.mean(values ** 2))
        ])

    return np.array(features).reshape(1, -1)


def main():
    sense = SenseHat()
    model = joblib.load(MODEL_FILE)

    samples = deque(maxlen=WINDOW_SIZE)
    predictions = deque(maxlen=CONFIRMATION_WINDOWS)

    stage = 0
    waiting_for_idle = False
    last_activity_time = None

    sample_count = 0
    last_displayed = None

    print("Live Movement Recognition Started")
    print("Classes: Idle, Raise, Rotate, Translate")
    print("Required sequence: Raise -> Rotate -> Translate")
    print("Return to Idle between movements.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            accel = sense.get_accelerometer_raw()
            gyro = sense.get_gyroscope_raw()

            sample = [
                accel["x"], accel["y"], accel["z"],
                gyro["x"], gyro["y"], gyro["z"]
            ]

            samples.append(sample)
            sample_count += 1

            if len(samples) == WINDOW_SIZE and sample_count % STEP_SIZE == 0:

                window = np.array(samples)
                features = extract_features(window)

                prediction = model.predict(features)[0]
                predictions.append(prediction)

                if prediction != last_displayed:
                    print("Detected:", prediction)
                    last_displayed = prediction

                if len(predictions) == CONFIRMATION_WINDOWS:

                    stable_prediction = max(
                        set(predictions),
                        key=predictions.count
                    )

                    if predictions.count(stable_prediction) == CONFIRMATION_WINDOWS:

                        current_time = time.time()

                        if (
                            stage > 0
                            and last_activity_time is not None
                            and current_time - last_activity_time > TIMEOUT
                        ):
                            print("\nSequence timeout. Resetting.")
                            stage = 0
                            waiting_for_idle = False
                            last_activity_time = None

                        if stable_prediction == "idle":

                            if waiting_for_idle:
                                waiting_for_idle = False
                                print("Idle confirmed. Ready for next movement.")

                        elif stable_prediction in SEQUENCE:

                            if waiting_for_idle:
                                continue

                            expected = SEQUENCE[stage]

                            if stable_prediction == expected:

                                stage += 1
                                waiting_for_idle = True
                                last_activity_time = current_time

                                print(
                                    f"\nStep {stage} completed: "
                                    f"{stable_prediction}"
                                )

                                if stage == len(SEQUENCE):
                                    print("\nSUCCESS! Sequence completed.")
                                    print("Raise -> Rotate -> Translate\n")

                                    stage = 0
                                    waiting_for_idle = False
                                    last_activity_time = None

                            else:
                                print(
                                    f"\nWrong movement: {stable_prediction}. "
                                    "Sequence reset."
                                )

                                stage = 0
                                waiting_for_idle = False
                                last_activity_time = None

                        predictions.clear()

            time.sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        print("\nLive recognition stopped.")


if __name__ == "__main__":
    main()