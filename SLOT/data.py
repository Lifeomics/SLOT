import pandas as pd
import scanpy as sc
import numpy as np


def norm_data(data_path, res_path, data_type="Protein", min_gene=200, min_cell=2):
    data = pd.read_csv(data_path)
    if data_type == "Protein":
        adata = sc.AnnData(data.T.iloc[2:, :], dtype=np.float32)
        adata.obs_names = [i for i in data.T.iloc[2:, :].index]  # colname as cell name
        adata.var_names = list(data.iloc[:, 1])
    elif data_type == "mRNA":
        adata = sc.AnnData(data.T.iloc[1:, :], dtype=np.float32)
        adata.obs_names = [f"section_{i}" for i in data.T.iloc[1:, :].index]
        adata.var_names = list(data.iloc[:, 0])
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=min_gene)
    sc.pp.filter_genes(adata, min_cells=min_cell)
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    if data_type == "Protein":
        sc.pp.normalize_total(adata, target_sum=4e8)  # transcriot 1e6
    else:
        sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    print(f"save data to {res_path}")
    adata.write(res_path)

def bin_matrix(X, num_bins=10):
    """
    Convert input matrix X (each row represents a vector) into probability distributions,
    then divide each row into num_bins intervals and sum the values in each interval,
    returning a new matrix.

    Parameters:
    X: numpy array, shape (n_samples, spatial_dim)
        Input data matrix where each row is a feature vector
    num_bins: int, default=10
        Number of bins to divide each row into

    Returns:
    X_binned: numpy array, shape (n_samples, num_bins)
        Binned matrix where each row contains sums of binned values
    """
    # X: numpy array, shape (n_genes, spatial_dim)
    X = np.array(X)  
    row_sum = X.sum(axis=1, keepdims=True)
    X_prob = X / row_sum

    def bin_row(row):
        bins = np.array_split(row, num_bins)
        return np.array([b.sum() for b in bins])
    
    X_binned = np.apply_along_axis(bin_row, 1, X_prob)
    return X_binned
