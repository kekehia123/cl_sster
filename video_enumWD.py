# import packages
from cl_sster import cl_sster
import numpy as np
import os
import scipy.io as sio
from postprocessing_utils import calc_isc, calc_isc_train, calc_corr
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from postprocessing_utils import plot_dendrogram

# parameters
fs = 125
epochs_pretrain = 50
timeLen = 5

# Load video data
datadir = r'D:\Data\Emotion\FACED\Processed_data_filter_epoch_0.50_40_manualRemove_ica'
print(datadir)
video_len = [81,63,73,78,69,90,56,60,105,45,60,81,35,44,38,43,55,69,73,129,77,75,34,37,67,63,54,77]
n_points = np.array(video_len).astype(int) * fs

n_vids = len(video_len)
data_paths = os.listdir(datadir)
data_paths.sort()
n_subs = 123
chn = 30
count = 0
data = np.zeros((n_subs, np.sum(n_points), chn))
for idx, path in enumerate(data_paths):
    if path[:3] == 'sub':
        data[count,:,:] = sio.loadmat(os.path.join(datadir, path))['data_all_cleaned'].transpose()
        count += 1
print(data.shape)

# Normalization without outliers
print('Normalizing')
for sub in range(n_subs):
    thr = 30 * np.median(abs(data[sub]))
    data[sub] = (data[sub] - np.mean(data[sub][data[sub] < thr])) / np.std(data[sub][data[sub] < thr])
    
# Train with all data
for wd in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
    cl_model_allData = cl_sster(n_folds=1, epochs_pretrain=50, timeLen=timeLen, fs=fs, data_type='video', weight_decay=wd) # If n_folds == 1, it will use all data to train the model
    cl_model_allData.load_data(data, n_points) # fs: sampling rate
    cl_model_allData.train_cl_sster() # Train the model
    
    out, n_points_cum = cl_model_allData.get_hidden(fold=0) # The function will load the trained model and get hidden representations
    out_all_corr_mean = calc_isc_train(out.transpose(2,1,0), n_points_cum)

    # Get the clusters 
    nonzero_dims = cl_model_allData.check_nonzero_dims()
    out_corr_dims_mean = cl_model_allData.calc_out_corr_dims() # Only used nonzero dims here
    affinity_mat = 1 - np.abs(out_corr_dims_mean)

    cluster_model = AgglomerativeClustering(distance_threshold=0.9, n_clusters=None, metric='precomputed', linkage='average')
    cluster_model = cluster_model.fit(affinity_mat)
    c_labels = cluster_model.labels_
    n_clusters = len(np.unique(c_labels))
    
    # Get the diemensions with best ISC in each cluster (trained with all data)
    isc_mean = np.mean(out_all_corr_mean, axis=1)
    isc_mean_sel = isc_mean[nonzero_dims]
    inds_order = np.arange(len(nonzero_dims))
    inds_cluster_max = np.zeros(n_clusters).astype(int)
    for i in range(n_clusters):
        tmp = np.argmax(isc_mean_sel[c_labels==i])
        inds_cluster_max[i] = nonzero_dims[int(inds_order[c_labels==i][tmp])]
    print(inds_cluster_max)
    print(isc_mean[inds_cluster_max])
    
    sio.savemat(os.path.join(cl_model_allData.save_dir, 'inds_cluster_max.mat'), {'inds_cluster_max':inds_cluster_max})
    
    # Train the cross-validation model
    cl_model = cl_sster(n_folds=10, epochs_pretrain=epochs_pretrain, timeLen=timeLen, fs=fs, data_type='video', weight_decay=wd)
    cl_model.load_data(data, n_points) # fs: sampling rate
    cl_model.train_cl_sster()    
    
    # Get the corresponding dimensions in each training fold
    correspondDims_fold, corr_mean_fold = cl_model.get_correspond_dims(n_folds=10, out=out, calc_dims=inds_cluster_max)
    
    # Calculate training and validation ISC in each fold
    n_folds = 10
    n_per = int(n_subs // n_folds)
    isc_train_folds = np.zeros((n_folds, n_subs, n_clusters)) - 2.
    isc_val_folds = np.zeros((n_subs, n_clusters))

    for val_fold in range(n_folds):
        if val_fold < n_folds-1:
            val_sub = np.arange(val_fold*n_per, (val_fold+1)*n_per)
        else:
            val_sub = np.arange(val_fold*n_per, n_subs)
        train_sub = list(set(np.arange(n_subs)) - set(val_sub))
        sub_order = np.concatenate((train_sub, val_sub))
        
        out, n_points_cum = cl_model.get_hidden(val_fold)
        out = out[sub_order,:,:]
        out_train_corr_mean, out_val_corr_mean = calc_isc(out.transpose(2,1,0), n_points_cum, len(train_sub))
        
        isc_train_folds[val_fold, train_sub, :] = out_train_corr_mean[correspondDims_fold[val_fold],:].transpose()
        isc_val_folds[val_sub, :] = out_val_corr_mean[correspondDims_fold[val_fold]].transpose()
    
    isc_train_mean = np.zeros((n_subs, n_clusters))
    for sub in range(n_subs):
        for dim in range(n_clusters):
            isc_train_mean[sub,dim] = np.mean(isc_train_folds[:,sub,dim][isc_train_folds[:,sub,dim]!=-2], axis=0)
    isc_train_std = np.std(isc_train_mean, axis=0)
    isc_train_grandmean = np.mean(isc_train_mean, axis=0)
    print('isc train mean, std: ')
    print(isc_train_grandmean, isc_train_std)

    isc_val_mean = np.mean(isc_val_folds, axis=0)
    isc_val_std = np.std(isc_val_folds, axis=0)
    print('isc val mean, std: ')
    print(isc_val_mean, isc_val_std)
    
    sio.savemat(os.path.join(cl_model.save_dir, 'isc.mat'), {'isc_train': isc_train_mean, 'isc_val': isc_val_folds})