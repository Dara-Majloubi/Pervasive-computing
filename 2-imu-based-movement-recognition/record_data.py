import os
import csv
import time
from sense_hat import SenseHat

sense = SenseHat()

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "imu_trials.csv")

CLASSES = {
    "idle": 0,
    "raise": 1,
    "rotate": 2,
    "translate": 3
}

DURATION = 5
SAMPLE_INTERVAL = 0.05
TRAIN_TRIALS = 8
TEST_TRIALS = 4


def record_trial(label, class_id, trial_id, split):
    input(f"\nPrepare {label} trial {trial_id}. Press Enter to start.")

    print("Starting in 3 seconds...")
    time.sleep(3)

    rows = []
    start_time = time.monotonic()

    print(f"Recording {label}...")

    while time.monotonic() - start_time < DURATION:
        timestamp = time.time()
        elapsed = time.monotonic() - start_time

        accel = sense.get_accelerometer_raw()
        gyro = sense.get_gyroscope_raw()

        rows.append([
            timestamp,
            elapsed,
            accel["x"], accel["y"], accel["z"],
            gyro["x"], gyro["y"], gyro["z"],
            label,
            class_id,
            trial_id,
            split
        ])

        time.sleep(SAMPLE_INTERVAL)

    os.makedirs(DATA_DIR, exist_ok=True)

    file_exists = os.path.exists(DATA_FILE)

    with open(DATA_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp", "elapsed_s",
                "accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z",
                "label", "class_id", "trial_id", "split"
            ])

        writer.writerows(rows)

    print(f"Saved {len(rows)} samples for {label}, trial {trial_id}.")


def collect_data(split, trials_per_class):
    for label, class_id in CLASSES.items():
        for trial in range(1, trials_per_class + 1):
            trial_id = f"{split}_{label}_{trial:02d}"
            record_trial(label, class_id, trial_id, split)


if __name__ == "__main__":
    print("IMU Data Collection")
    print("1. Training data")
    print("2. Testing data")

    choice = input("Select mode (1/2): ")

    if choice == "1":
        collect_data("train", TRAIN_TRIALS)
    elif choice == "2":
        collect_data("test", TEST_TRIALS)
    else:
        print("Invalid selection.")