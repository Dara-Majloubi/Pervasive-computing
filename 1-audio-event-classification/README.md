# Audio Event Classification

An end-to-end Python project for classifying short audio recordings of everyday kitchen sounds. It recognizes three classes: **stirring**, **coffee machine**, and **kettle**.

The dataset contains 120 labeled clips, with 40 clips per class. The clips come from 28 independent source recordings. The included recordings are mono WAV files sampled at 44.1 kHz.

## How It Works

The pipeline:

1. Loads and validates WAV recordings.
2. Applies a fourth-order Butterworth band-pass filter from 100 Hz to 8,000 Hz.
3. Splits each recording into one-second frames with 50% overlap.
4. Extracts 35 handcrafted features from each frame.
5. Compares classifiers using four-fold, grouped, stratified cross-validation.
6. Trains and saves the final KNN model.

Cross-validation groups clips by their original source recording, so segments from the same take do not appear in both the training and test folds.

## Audio Features

Each frame is represented by 35 features:

- RMS energy and zero-crossing rate
- Spectral centroid, rolloff, and flatness
- Relative energy in four frequency bands
- The mean and standard deviation of 13 MFCC coefficients

## Model Evaluation

Results below are mean ± standard deviation across four grouped folds.

| Classifier | Accuracy | Macro F1 |
|---|---:|---:|
| MLP (64, 32) | 91.3% ± 7.7% | 90.9% ± 7.0% |
| KNN (k=3) | 90.5% ± 10.0% | 90.5% ± 9.7% |
| Decision Tree | 81.0% ± 15.5% | 78.1% ± 14.3% |
| Majority baseline | 23.2% ± 12.2% | 12.1% ± 5.8% |

The MLP achieved the highest mean score. KNN with three neighbors was selected as the final model for its simplicity and easier interpretation. It is saved as `models/knn_model.joblib`.

## Project Structure

```text
audio-event-classification/
├── code/
│   ├── audio_loader.py
│   ├── audio_filter.py
│   ├── framing.py
│   ├── feature_extraction.py
│   ├── dataset_builder.py
│   ├── classifier.py
│   ├── evaluation.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── make_figures.py
│   └── requirements.txt
├── data/
│   ├── coffee_machine/
│   ├── kettle/
│   ├── stirring/
│   └── processed/
├── models/
│   └── knn_model.joblib
├── new_test/
├── report/
│   └── figs/
└── README.md
```

## Requirements

- Python 3.13
- Dependencies listed in `code/requirements.txt`

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r code/requirements.txt
```

## Train and Evaluate

Place the labeled WAV recordings in the corresponding `data` folders:

```text
data/
├── coffee_machine/
├── kettle/
└── stirring/
```

Then run:

```bash
python code/pipeline.py
```

The pipeline writes the extracted features, fold results, evaluation summary, and confusion matrices to `data/processed/`. It also trains the final KNN model and saves it to `models/knn_model.joblib`.

## Predict Audio

To classify one WAV file:

```bash
python code/predict.py --file path/to/audio.wav
```

The file must be mono WAV, 3–8 seconds long, and use the same sample rate as the training recordings.

To classify every WAV file in `new_test/`:

```bash
python code/predict.py
```

Batch predictions are saved to `new_test/test_results.csv`. The batch script renames WAV files that do not match the `test_<number>.wav` naming pattern. Keep a copy of the test files if you need to preserve their original filenames.

## Generate Figures

After running the pipeline, generate the report figures with:

```bash
python code/make_figures.py
```

The figures are saved in `report/figs/`.
