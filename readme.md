The code is the implementation for "Contrastive Learning for Shared SpatioTemporal EEG Representations (CL-SSTER)".

CL-SSTER aligns the neural representations from different subjects by minimizing the discrepancy of neural representations when the subjects receive the same stimuli.

It regards a pair of samples from two different subjects when they receive the same stimuli as the positive sample pair, and samples corresponding to different stimuli as negative sample pairs.


Please follow the steps to reproduce the results in the paper: "Contrastive Learning of Shared Spatiotemporal EEG Representations Across Individuals for Naturalistic Neuroscience":

1. Reproduce the running environment with Anaconda.

conda env create -n cl_sster_env -f cl_sster_env.yaml

conda activate cl_sster_env

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

2. Run each cell in example_simulatedData.ipynb, example_speech.ipynb and example_video.ipynb to reproduce the results for simulated data, Broderick dataset and FACED dataset, respectively.


The code was originally implemented on a windows operation system with an NVIDIA GeForce RTX 4080 GPU, 32G RAM and an AMD Ryzen 9 7900X 12-Core Processor.

The GPU driver version is 528.49. The cuda version is 11.8. The cudnn version is 8700. 


Documentation for the class cl_sster:
Parameters:
    n_folds: int. Number of folds in cross-validation. If 1, the model will be trained with all data. Default: 10
    timeLen: int. The time length of one sample (in seconds). Default: 30
    weight_decay: float. Weight decay in contrastive learning. Default: 0.1
    epochs_pretrain: int. The training epochs in contrastive learning. Default: 50
    timeFilterLen: int. The length of temporal filters. Default: 30
    avgPoolLen: int. The kernel size of average pooling. Default: 15
    device: str. Use gpu if set as 'cuda'. Default: 'cuda'. Note: the cpu version has not been tested.
    gpu_index: int. Which gpu index to use. Default: 0
    randSeed: int. Set the random seed. Default: 7
    data_type: str. What data type to use. It will create a subdirectory in 'results' with the name as data_type. Default: 'simulation'
    fs: int. The sampling rate of the data. Default: 128

Attributes:
    save_dir: str. The directory to save the training history and model parameters.

Methods:
    load_data: load data and define the number of samples in each trial for contrastive learning.
        Parameters:
            data: array-like of shape (n_subjects, n_timepoints, n_channels). The input data.
            n_points: array-like of shape (n_trials,). The number of sampling points of EEG data in each trial.
        Return:
            self: object. Loaded data and number of samples in each trial.

    train_cl_sster: implement data loader and model training. Save the trained parameters ('[fold_number]/checkpoint_*.pth.tar') and training history ('results.pkl') to self.save_dir.
        Parameters:
            self: object. 
        Return:
            self: object. 

    get_hidden: get the hidden representations of the model.
        Parameters:
            fold: int. Which fold to use.
            isNorm: bool. Whether to normalize the hidden representations in the feature dimension. Default: False.
        Return:
            out: array-like of shape (n_subjects, n_latent_dimensions, n_timepoints_latent). The hidden representations of the data.
            n_points_cum: array-like of shape (n_trials,). The cumulative sum of the number of points for each trial in the latent space (adjusted for the time filter length and average pooling).

    get_hidden_psd: calculate the power spectral density (PSD) of the hidden representations.
        Parameters:
            fold: int. The fold number of the cross-validation.
            inds_sel: list of int. Indices of the selected dimensions in the hidden representations for PSD calculation.
            isNorm: bool. Whether to normalize the hidden representations before PSD calculation. Default: False.
        Return:
            psd: array-like of shape (n_subjects, n_selected_dimensions, n_trials). The PSD of the selected dimensions of the hidden representations for each trial.

    calc_psd: calculate the power spectral density of a signal.
        Parameters:
            signal: array-like of shape (n_dimensions, n_timepoints). The input signal.
        Return:
            psd: float. The average power spectral density of the signal within the frequency range of interest (0.5 Hz to 40 Hz).

    get_hidden_nopool: get the hidden representations without applying average pooling.
        Parameters:
            fold: int. The fold number of the cross-validation.
            inds_sel: list of int. Indices of the selected dimensions in the hidden representations.
            isNorm: bool. Whether to normalize the hidden representations. Default: False.
        Return:
            out: array-like of shape (n_subjects, n_latent_dimensions, n_timepoints_latent). The hidden representations without average pooling.
            n_points_cum: array-like of shape (n_trials,). The cumulative sum of the number of points for each trialin the latent space (adjusted for the time filter length).

    check_nonzero_dims: check and return the dimensions with non-zero variance across all trials and subjects.
        Parameters:
            self: object.
        Return:
            nonzero_dims: array-like of shape (n_nonzero_dimensions). Indices of the dimensions with non-zero variance.

    calc_out_corr_dims: calculate the correlation between dimensions of the output representations.
        Parameters:
            self: object.
        Return:
            out_corr_dims_mean: array-like of shape (n_nonzero_dimensions, n_nonzero_dimensions). The mean correlation matrix across subjects and trials.

    get_correspond_dims: find corresponding dimensions in the hidden representations of cross-validation model that match specified dimensions in the hidden representations of the model trained on all data based on correlation.
        Parameters:
            n_folds: int. Number of folds in cross-validation.
            out: array-like of shape (n_subjects, n_latent_dimensions, n_timepoints_latent). The hidden representations of the model trained on all data.
            calc_dims: list of int. The specified dimensions to match in the hidden representations (out).
            isNorm: bool. Whether to normalize the hidden representations before finding corresponding dimensions. Default: False.
            isPool: bool. Whether the hidden representations were obtained with pooling. Default: True.
        Return:
            correspondDims_fold: array-like of shape (n_folds, n_selected_dimensions). The corresponding dimensions found in each fold.
            corr_mean_fold: array-like of shape (n_folds, n_selected_dimensions). The mean correlation of the corresponding dimensions across folds.

    get_correspond_dims_memEffi: a memory-efficient version of get_correspond_dims.
        Parameters:
            n_folds: int. Number of folds in cross-validation.
            out_sel: array-like of shape (n_subjects, n_selected_dimensions, n_timepoints_latent). A subset of the hidden representations for finding corresponding dimensions.
            isNorm: bool. Whether to normalize the hidden representations before finding corresponding dimensions. Default: False.
            isPool: bool. Whether the hidden representations were obtained with pooling. Default: True.
        Return:
            correspondDims_fold: array-like of shape (n_folds, n_selected_dimensions). The corresponding dimensions found in each fold.
            corr_mean_fold: array-like of shape (n_folds, n_selected_dimensions). The mean correlation of the corresponding dimensions across folds.





