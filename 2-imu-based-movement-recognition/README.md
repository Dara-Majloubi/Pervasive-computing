# IMU-Based Movement Recognition

A real-time movement recognition project using the accelerometer and gyroscope on a Raspberry Pi Sense HAT. A Random Forest classifier recognizes four classes: **idle**, **raise**, **rotate**, and **translate**.

## Hardware

- Raspberry Pi 4
- Raspberry Pi Sense HAT
- Raspberry Pi OS

## Project Structure

```text
2-imu-based-movement-recognition/
├── record_data.py
├── train_model.py
├── live_detection.py
├── requirements.txt
├── imu_model.joblib
└── data/
    └── imu_trials.csv
```

## Dataset

Sensor recordings are stored in `data/imu_trials.csv`. Each sample includes three accelerometer channels and three gyroscope channels, along with a movement label and trial metadata.

The dataset contains 48 trials across four classes:

- 8 training trials per class
- 4 test trials per class
- Approximately 5 seconds per trial

The recording script samples the Sense HAT sensors and appends the results to the CSV file.

## Feature Extraction and Model

The training script groups sensor readings into overlapping windows of 10 samples, advancing by 5 samples at a time. For each window, it calculates five statistics for each of the six sensor channels:

- Mean
- Standard deviation
- Minimum
- Maximum
- Root mean square (RMS)

This produces 30 features per window. A Random Forest classifier with 100 trees is trained on the recorded data.

The current project documentation reports **88.19% test accuracy** on separately recorded test trials. Performance can vary with movement speed and how the device is held.

## Installation

Run these commands from the project folder on the Raspberry Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Record Sensor Data

Run:

```bash
python3 record_data.py
```

Choose `1` to record training trials or `2` to record test trials. Follow the prompts for each movement. The script saves the readings to `data/imu_trials.csv`.

## Train the Model

Run:

```bash
python3 train_model.py
```

The script trains and evaluates the classifier using the training and test trials, prints the evaluation results, and saves the trained model as `imu_model.joblib`.

## Run Live Recognition

Run:

```bash
python3 live_detection.py
```

The program reads the Sense HAT sensors and displays the predicted movement. To complete the required sequence, perform:

```text
Raise → Idle → Rotate → Idle → Translate
```

A movement is confirmed after three consecutive matching predictions. Return to idle between movements. An incorrect movement resets the sequence, and an incomplete sequence times out after 20 seconds.

Press `Ctrl+C` to stop the program.
