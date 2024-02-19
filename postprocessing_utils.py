from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
import numpy as np

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples_dend = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples_dend:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples_dend]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    R = dendrogram(linkage_matrix, **kwargs)
    return R

def calc_isc(Y, n_points_cum, n_train_subs):
    T, n_dims, n_subs = Y.shape
    n_trials = len(n_points_cum) - 1
    corr_mat = np.zeros((n_subs, n_subs, n_trials, n_dims))
    for dim in range(n_dims):
        for tr in range(n_trials):
            corr_mat[:,:,tr,dim] = np.corrcoef(Y[n_points_cum[tr]: n_points_cum[tr+1], dim, :].transpose(1,0))
    
    n_val_subs = n_subs - n_train_subs
    out_train_corr_mean = np.zeros((n_dims, n_train_subs))
    for sub in range(n_train_subs):
        for dim in range(n_dims):
            other_subs = list(set(list(np.arange(n_train_subs))) - set([sub]))
            out_train_corr_mean[dim, sub] = np.mean(corr_mat[other_subs,:,:,dim][:,sub,:])
    
    out_val_corr_mean = np.zeros((n_dims, n_val_subs))
    for sub in range(n_val_subs):
        other_subs = list(set(list(np.arange(n_subs))) - set([n_train_subs+sub]))
        for dim in range(n_dims):
            out_val_corr_mean[dim, sub] = np.mean(corr_mat[other_subs,:,:,dim][:,n_train_subs+sub,:])
    return out_train_corr_mean, out_val_corr_mean


def calc_isc_train(Y, n_points_cum):
    T, n_dims, n_subs = Y.shape
    n_trials = len(n_points_cum) - 1
    corr_mat = np.zeros((n_subs, n_subs, n_trials, n_dims))
    for dim in range(n_dims):
        for tr in range(n_trials):
            corr_mat[:,:,tr,dim] = np.corrcoef(Y[n_points_cum[tr]: n_points_cum[tr+1], dim, :].transpose(1,0))
    
    out_train_corr_mean = np.zeros((n_dims, n_subs))
    for sub in range(n_subs):
        for dim in range(n_dims):
            other_subs = list(set(list(np.arange(n_subs))) - set([sub]))
            out_train_corr_mean[dim, sub] = np.mean(corr_mat[other_subs,:,:,dim][:,sub,:])
    return out_train_corr_mean

# X: (N*D1), Y: (N*D2) 
# return corrmat: (D1*D2)
def calc_corr(X, Y):
    x_mean = np.mean(X, axis=0)
    y_mean = np.mean(Y, axis=0)
    cov = np.matmul((X - x_mean).transpose(), Y - y_mean) # (D1, D2)
    x_sqrmean = np.sqrt(np.sum((X - x_mean)**2, axis=0))
    y_sqrmean = np.sqrt(np.sum((Y - y_mean)**2, axis=0))
    x_grid, y_grid = np.meshgrid(x_sqrmean, y_sqrmean)
    corrmat = cov / (x_grid.transpose() * y_grid.transpose())
    return corrmat