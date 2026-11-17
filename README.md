# Pervasive Computing Projects

This repository contains projects developed for the Pervasive Computing course. The projects explore how software can use sensor data and machine learning to recognize sounds and physical activity.

## Projects

### 1. Audio Event Classification

**Folder:** [`1-audio-event-classification`](./1-audio-event-classification/)

Classifies short recordings of three everyday sounds:

- Coffee machine
- Kettle
- Stirring

The project processes audio recordings, extracts handcrafted time and frequency features, and compares KNN, Decision Tree, and MLP classifiers. Evaluation uses grouped cross-validation to keep segments from the same source recording together. The saved KNN model can classify new audio recordings.

### 2. IMU-Based Movement Recognition

**Folder:** [`2-imu-based-movement-recognition`](./2-imu-based-movement-recognition/)

Uses accelerometer and gyroscope readings from a Raspberry Pi Sense HAT to recognize four movement classes:

- Idle
- Raise
- Rotate
- Translate

A Random Forest classifier predicts movement from sensor readings. The live application also checks for the required sequence:

```text
Raise → Idle → Rotate → Idle → Translate
```

## Repository Structure

```text
Pervasive-computing/
├── 1-audio-event-classification/
├── 2-imu-based-movement-recognition/
└── README.md
```

Each project folder contains its own source code, dependencies, data or model files, and project-specific instructions. See the README inside each folder for setup and usage details.

## Technologies

- Python
- NumPy, SciPy, and pandas
- scikit-learn
- Raspberry Pi and Sense HAT sensors
- Audio and image/sensor data processing
