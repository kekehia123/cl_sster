# Contrastive Learning for Shared Spatiotemporal EEG Representations (CL-SSTER)

CL-SSTER aligns the neural representations from different subjects by minimizing the discrepancy of neural representations when the subjects receive the same stimuli. It regards a pair of samples from two different subjects when they receive the same stimuli as the positive sample pair, and samples corresponding to different stimuli as negative sample pairs.

## Introduction

This implementation focuses on "Contrastive Learning of Shared Spatiotemporal EEG Representations Across Individuals for Naturalistic Neuroscience." It demonstrates how contrastive learning can be leveraged to align neural representations from different subjects, facilitating a more generalized understanding of neural responses to stimuli.

## Getting Started

Follow the steps below to reproduce the results presented in the paper.

### Environment Setup

First, reproduce the running environment using Anaconda:

```bash
conda env create -n cl_sster_env -f cl_sster_env.yaml
conda activate cl_sster_env
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### Running the Notebooks

To reproduce the analysis and results for different datasets, run the provided Jupyter notebooks:

- **Simulated Data**: `example_simulatedData.ipynb`
- **Broderick Dataset**: `example_speech.ipynb`
- **FACED Dataset**: `example_video.ipynb`

Execute each cell in the notebooks to reproduce the results for simulated data, the Broderick dataset, and the FACED dataset, respectively.

## System Requirements

This codebase was developed and tested with the following system configuration:

- **Operating System**: Windows
- **Graphics Processing Unit (GPU)**: NVIDIA GeForce RTX 4080
- **Memory**: 32GB RAM
- **Processor**: AMD Ryzen 9 7900X 12-Core Processor
- **GPU Driver Version**: 528.49
- **CUDA Version**: 11.8
- **cuDNN Version**: 8700

Ensure your system meets these specifications to replicate the results accurately.

## `cl_sster` Class Documentation

The `cl_sster` class encapsulates the functionality for contrastive learning with EEG data. Below is an overview of its parameters, attributes, and methods.

### Parameters

- `n_folds`: int (default: 10) - Number of folds for cross-validation.
- `timeLen`: int (default: 30) - The duration of one sample in seconds.
- `weight_decay`: float (default: 0.1) - Weight decay factor for contrastive learning.
- `epochs_pretrain`: int (default: 50) - Number of pretraining epochs.
- `timeFilterLen`: int (default: 30) - Length of temporal filters.
- `avgPoolLen`: int (default: 15) - Kernel size for average pooling.
- `device`: str (default: 'cuda') - Specifies the device for computation ('cuda' for GPU).
- `gpu_index`: int (default: 0) - Index of the GPU to use.
- `randSeed`: int (default: 7) - Seed for random number generation.
- `data_type`: str (default: 'simulation') - Type of data to process.
- `fs`: int (default: 128) - Sampling rate of the EEG data.

### Attributes

- `save_dir`: str - Directory path to save training history and model parameters.

### Methods

#### `load_data`

Loads the EEG data and prepares it for model training.

- **Parameters**:
  - `data`: (array-like) The EEG dataset.
  - `n_points`: (array-like) Number of data points per trial.
- **Returns**:
  - The object itself, updated with the loaded data.

#### `train_cl_sster`

Trains the contrastive learning model using the loaded data.

- **Parameters**: None.
- **Returns**:
  - The object itself, updated with training outcomes.

#### `get_hidden`

Extracts hidden representations from the trained model.

- **Parameters**:
  - `fold`: (int) The cross-validation fold to use.
  - `isNorm`: (bool, default: False) Whether to normalize the hidden representations.
- **Returns**:
  - Hidden representations of the EEG data.







