# Contrastive Learning for Shared Spatiotemporal EEG Representations (CL-SSTER)

## Introduction

CL-SSTER aligns the neural representations from different subjects by minimizing the discrepancy of neural representations when the subjects receive the same stimuli. It regards a pair of samples from two different subjects when they receive the same stimuli as the positive sample pair, and samples corresponding to different stimuli as negative sample pairs.

This implementation focuses on reproduction of the paper "Contrastive Learning of Shared Spatiotemporal EEG Representations Across Individuals for Naturalistic Neuroscience." It demonstrates how contrastive learning can be leveraged to align neural representations from different subjects, facilitating a more generalized understanding of neural responses to stimuli.

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

## `cl_sster` Class Documentation

The `cl_sster` class encapsulates the functionality for contrastive learning with EEG data. Below is an overview of its parameters, attributes, and methods.

### Parameters

- `n_folds`: int (default: 10) - Number of folds for cross-validation.
- `timeLen`: int (default: 30) - The duration of one sample in seconds.
- `weight_decay`: float (default: 0.1) - Weight decay factor for contrastive learning.
- `epochs_pretrain`: int (default: 50) - Number of pretraining epochs.
- `timeFilterLen`: int (default: 30) - Length of temporal filters.
- `avgPoolLen`: int (default: 15) - Kernel size for average pooling.
- `device`: str (default: 'cuda') - Specifies the device for computation ('cuda' for GPU). Note: the cpu version has not been tested.
- `gpu_index`: int (default: 0) - Index of the GPU to use.
- `randSeed`: int (default: 7) - Seed for random number generation.
- `data_type`: str (default: 'simulation') - Type of data to process. It will create a subdirectory in 'results' with the name as 'data_type'.
- `fs`: int (default: 128) - Sampling rate of the EEG data.

### Attributes

- `save_dir`: str - Directory path to save training history and model parameters.

### Methods

#### `load_data`

Loads the EEG data and prepares it for model training.

- **Parameters**:
  - `data`: array-like of shape `(n_subjects, n_timepoints, n_channels)` - The input EEG data.
  - `n_points`: array-like of shape `(n_trials,)` - Number of data points per trial.
- **Returns**:
  - The object itself, updated with the loaded data.

#### `train_cl_sster`

Trains the contrastive learning model using the loaded data. Save the trained parameters ('[fold_number]/checkpoint_*.pth.tar') and training history ('results.pkl') to self.save_dir.

- **Parameters**: None.
- **Returns**:
  - The object itself, updated with training outcomes.

### `get_hidden`

Extracts hidden representations from the trained model.

- **Parameters**:
  - `fold`: int - Specifies the cross-validation fold to use.
  - `isNorm`: bool (default: False) - Determines whether to normalize the hidden representations.
- **Returns**:
  - `out`: array-like of shape `(n_subjects, n_latent_dimensions, n_timepoints_latent)` - The hidden representations of the data.
  - `n_points_cum`: array-like of shape `(n_trials,)` - The cumulative sum of the number of points for each trial in the latent space, adjusted for the time filter length and average pooling.

### `get_hidden_psd`

Calculates the power spectral density (PSD) of the hidden representations.

- **Parameters**:
  - `fold`: int - Specifies the cross-validation fold to use.
  - `inds_sel`: list of int - Indices of the selected dimensions in the hidden representations for PSD calculation.
  - `isNorm`: bool (default: False) - Determines whether to normalize the hidden representations before PSD calculation.
- **Returns**:
  - `psd`: array-like of shape `(n_subjects, n_selected_dimensions, n_trials)` - The PSD of the selected dimensions of the hidden representations for each trial.

### `calc_psd`

Calculates the power spectral density of a signal.

- **Parameters**:
  - `signal`: Array-like of shape `(n_dimensions, n_timepoints)` - The input signal.
- **Returns**:
  - `psd`: float - The average power spectral density of the signal within the frequency range of interest (0.5 Hz to 40 Hz).

### `get_hidden_nopool`

Gets the hidden representations without applying average pooling.

- **Parameters**:
  - `fold`: int - Specifies the cross-validation fold to use.
  - `inds_sel`: list of int - Indices of the selected dimensions in the hidden representations.
  - `isNorm`: bool (default: False) - Determines whether to normalize the hidden representations.
- **Returns**:
  - `out`: array-like of shape `(n_subjects, n_latent_dimensions, n_timepoints_latent)` - The hidden representations without average pooling.
  - `n_points_cum`: array-like of shape `(n_trials,)` - The cumulative sum of the number of points for each trial in the latent space, adjusted for the time filter length.

### `check_nonzero_dims`

Checks and returns the dimensions with non-zero variance across all trials and subjects.

- **Parameters**:
  - `self`: object.
- **Returns**:
  - `nonzero_dims`: Array-like of shape `(n_nonzero_dimensions)` - Indices of the dimensions with non-zero variance.

### `calc_out_corr_dims`

Calculates the correlation between dimensions of the output representations.

- **Parameters**:
  - `self`: object.
- **Returns**:
  - `out_corr_dims_mean`: array-like of shape `(n_nonzero_dimensions, n_nonzero_dimensions)` - The mean correlation matrix across subjects and trials.

### `get_correspond_dims`

Finds corresponding dimensions in the hidden representations of cross-validation model that match specified dimensions in the hidden representations of the model trained on all data based on correlation.

- **Parameters**:
  - `n_folds`: int - Number of folds in cross-validation.
  - `out`: array-like of shape `(n_subjects, n_latent_dimensions, n_timepoints_latent)` - The hidden representations of the model trained on all data.
  - `calc_dims`: list of int - The specified dimensions to match in the hidden representations (`out`).
  - `isNorm`: bool (default: False) - Determines whether to normalize the hidden representations before finding corresponding dimensions.
  - `isPool`: bool (default: True) - Indicates whether the hidden representations were obtained with pooling.
- **Returns**:
  - `correspondDims_fold`: array-like of shape `(n_folds, n_selected_dimensions)` - The corresponding dimensions found in each fold.
  - `corr_mean_fold`: array-like of shape `(n_folds, n_selected_dimensions)` - The mean correlation of the corresponding dimensions across folds.

### `get_correspond_dims_memEffi`

A memory-efficient version of `get_correspond_dims`.

- **Parameters**:
  - `n_folds`: int - Number of folds in cross-validation.
  - `out_sel`: aarray-like of shape `(n_subjects, n_selected_dimensions, n_timepoints_latent)` - A subset of the hidden representations for finding corresponding dimensions.
  - `isNorm`: bool (default: False) - Determines whether to normalize the hidden representations before finding corresponding dimensions.
  - `isPool`: bool (default: True) - Indicates whether the hidden representations were obtained with pooling.
- **Returns**:
  - correspondDims_fold: array-like of shape `(n_folds, n_selected_dimensions)` - The corresponding dimensions found in each fold.
  - corr_mean_fold: array-like of shape `(n_folds, n_selected_dimensions)` - The mean correlation of the corresponding dimensions across folds.









