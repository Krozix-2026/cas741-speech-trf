# cas741-speech-trf

**Developer:** Xiao Shao  
**Project start date:** January 14, 2026

## Overview

This project provides a research pipeline for **model-to-brain alignment** using speech representations and MEG data.

The framework supports:

- training deep learning speech models
- extracting hidden-state representations as predictors
- converting predictors into formats compatible with Eelbrain
- running downstream mTRF / encoding-model analysis on MEG data

This repository is designed as a research framework rather than a packaged end-user application, and it is intended to be extensible for future models and experiments.

---

## Repository Structure

```text
docs/       Project documentation
refs/       Reference materials, including documents
src/        Source code
src/test/   Test cases
```

Project documentation is available at:

- [GitHub Pages Documentation](https://krozix-2026.github.io/cas741-speech-trf/)

Repository:

- [GitHub Repository](https://github.com/Krozix-2026/cas741-speech-trf/tree/main)

---

## Environment Requirements

The main Python dependencies are listed in:

- `src/requirements.txt`

The current project environment assumes:

- **Python:** `3.14.2`
- **CUDA:** `13.1`

Please make sure your Python, PyTorch, and CUDA environment are configured correctly before running the code.

A typical setup is:

```bash
cd src
pip install -r requirements.txt
```

> **Note:** This project is a research framework with environment-dependent components.  
> You may need to adjust package versions depending on your local GPU, CUDA, and PyTorch installation.

---

## Datasets

### 1. LibriSpeech

This project requires the **LibriSpeech** dataset for deep model training.

Official download page:

- [LibriSpeech (OpenSLR SLR12)](https://www.openslr.org/12)

For the default configuration, the most relevant subsets are:

- `train-clean-100`
- `dev-clean`
- `test-clean`

After downloading and extracting the dataset, update the dataset paths in:

- `src/config/paths_local.py`

### 2. Appleseed

The **Appleseed** dataset is **not provided in this repository**, because it has not yet been formally published.

If you already have local access to the Appleseed data, update the corresponding paths in your configuration files before running downstream analysis.

---

## Quick Start

### Step 1. Configure dataset and project paths

Edit:

- `src/config/paths_local.py`

and set the correct local paths for your datasets, manifests, checkpoints, and output directories.

### Step 2. Choose a training preset

Training presets are defined in:

- `src/config/presets.py`

By default, the framework is set up to run an **LSTM** model.

### Step 3. Train a deep learning model

Run:

```bash
cd src
python train_once.py
```

This starts training according to the preset selected in `config/presets.py`.

---

## Adding a New Deep Learning Model

To add a new model:

1. Add the model implementation under:

```text
src/network/
```

2. Add the corresponding preset/configuration in:

```text
src/config/presets.py
```

After that, the new model can be trained and evaluated through the same pipeline.

---

## Predictor Extraction Pipeline

After training a deep learning model, generate predictors for downstream MEG analysis using the following steps.

### Step 1. Convert pickle files to NPZ

```bash
python utils/0export_gammatone_pickle_to_npz.py
```

This converts pickle files into `.npz` format for easier processing in Python.

### Step 2. Extract hidden states from the trained model

Use one of the following scripts:

```bash
python utils/1gpu_make_predictors_from_npz.py
python utils/1gpu_make_predictors_from_npz_sem.py
```

These scripts load the trained model and extract hidden states as predictor `.npz` files.

### Step 3. Convert NPZ predictors back to pickle

```bash
python utils/2wrap_predictors_npz_to_pickle.py
```

This converts the generated predictor `.npz` files into `.pickle` files so that they can be used in Eelbrain.

---

## Running Eelbrain / mTRF Analysis

After generating predictor `.pickle` files, place them in the `predictors` folder of your local Appleseed dataset directory.

You can then use Eelbrain for downstream mTRF / encoding-model analysis.

Official documentation:

- [Eelbrain Documentation](https://eelbrain.readthedocs.io/en/stable/)

---

## Expected Workflow Summary

The typical workflow is:

1. Download and prepare **LibriSpeech**
2. Configure local paths in `src/config/paths_local.py`
3. Choose a training preset in `src/config/presets.py`
4. Run `train_once.py` to train a deep learning model
5. Convert data and generate predictors:
   - `utils/0export_gammatone_pickle_to_npz.py`
   - `utils/1gpu_make_predictors_from_npz.py`
   - `utils/1gpu_make_predictors_from_npz_sem.py`
   - `utils/2wrap_predictors_npz_to_pickle.py`
6. Place predictor files into the Appleseed `predictors` folder
7. Run downstream Eelbrain-based mTRF analysis

---

## Notes

- This repository is intended primarily for research use.
- Some scripts assume a specific local directory structure.
- You will likely need to adapt paths and preset settings to match your own environment.
- Appleseed-related analysis requires local access to the corresponding MEG dataset.
- The current framework is command-line/configuration based rather than GUI based.

---

## Troubleshooting

### 1. Dataset path errors

Make sure all dataset paths in `src/config/paths_local.py` are correct.

### 2. CUDA / PyTorch issues

Check that your installed PyTorch version is compatible with your CUDA installation.

### 3. Predictor conversion issues

Make sure each stage of the predictor pipeline completes successfully before moving to the next one:

- pickle → npz
- hidden-state extraction
- npz → pickle

### 4. Eelbrain analysis issues

Verify that predictor `.pickle` files are placed in the correct directory and that the Appleseed-related paths are configured properly.

---

## Documentation

Project documentation is maintained at:

- [https://krozix-2026.github.io/cas741-speech-trf/](https://krozix-2026.github.io/cas741-speech-trf/)

---

## License / Usage

This repository is currently intended for academic and research use.

Please make sure to also cite the original datasets, tools, and papers used in your experiments where appropriate, including:

- LibriSpeech
- Eelbrain
- any corresponding model- or dataset-specific references used in your workflow
