from scipy.stats import wasserstein_distance
from sklearn.neighbors import KNeighborsTransformer
from natsort import natsorted
import igraph as ig
import leidenalg
import numpy as np
import scanpy as sc
import pandas as pd


def slot_dist_metric(x, y, location):
    # location = np.arange(len(x))
    return wasserstein_distance(location, location, x, y)


def slot_neighbors(adata, feature='prob_matrix', n_neighbors=5, metric=slot_dist_metric):
    """
    Calculates the neighbors and connectivity matrices for the given AnnData object 
    using the provided distance metric (defaults to slot_dist_metric).
    
    Parameters:
    - adata: AnnData object that contains the dataset.
    - feature: The feature from adata to use for computing the distances (default is 'prob_matrix').
    - n_neighbors: The number of nearest neighbors to consider.
    - metric: The distance metric to use for computing distances (defaults to slot_dist_metric).
    
    Adds the neighbors' distances and connectivity matrices to `adata.obsp` and 
    stores the metric information in `adata.uns`.
    """
    # check adata need to be raw (gene) x col (cell)
    print("The AnnData object is expected to be in the format: (genes) x (cells).")
    
    try:
        # Ensure that the feature exists in adata.obsm
        if feature not in adata.obsm:
            raise KeyError(f"Feature '{feature}' not found in adata.obsm.")
        
        # Set metric parameters
        metric_para = {"location": np.arange(adata.obsm[feature].shape[1])}
       
        # Create the KNeighborsTransformer for calculating distances and connectivity
        transformer = KNeighborsTransformer(
            n_neighbors=n_neighbors,
            mode="distance",
            metric=metric,
            metric_params=metric_para,
        )
        connect_transformer = KNeighborsTransformer(
            n_neighbors=n_neighbors,
            mode="connectivity",
            metric=metric,
            metric_params=metric_para,
        )

        # Calculate the distance matrix and connectivity matrix
        adata.obsp["slot_distances"] = transformer.fit_transform(adata.obsm[feature])
        adata.obsp["slot_connectivities"] = connect_transformer.fit_transform(
            adata.obsm[feature]
        )
        
        # Store neighbor information
        adata.uns["slot_neighbors"] = {"metric": "slot_metric", "connectivities_key": "slot_connectivities"}
        adata.uns['slot_neighbors']['distances_key'] = 'slot_distances'
        adata.uns["slot_neighbors"]['params'] = {'n_neighbors': n_neighbors,
                                                'method': 'umap',
                                                'random_state': 0,
                                                'metric': 'euclidean',}
        print("Neighbor matrices added to adata.obsp")

    except KeyError as e:
        print(f"KeyError in slot_neighbors: {e}")
        raise
    except Exception as e:
        print(f"Error in slot_neighbors: {e}")
        raise ValueError("Failed to compute neighbors.")


def gaussian_transform(distances, sigma):
    """
    Applies a Gaussian transformation to the distances.
    
    Parameters:
    - distances: Array of distances between neighbors.
    - sigma: Standard deviation for the Gaussian transformation.
    
    Returns:
    - Transformed distances using Gaussian function.
    """
    try:
        # Return Gaussian-transformed distances
        return np.exp(-(distances**2) / (2 * sigma**2))
    except Exception as e:
        print(f"Error in gaussian_transform: {e}")
        raise ValueError("Failed to transform distances.")


def leiden(
    adata, n_iterations=-1, weight=True, sigma=1.0, resolution=1.0, random_seed=123
):
    """
    Performs clustering using the Leiden algorithm on the neighbor graph.

    Parameters:
    - adata: AnnData object containing the dataset.
    - n_iterations: Number of iterations for Leiden algorithm. Default is -1 for automatic.
    - weight: Whether to use weights in graph edges. Default is True.
    - sigma: Parameter controlling the Gaussian transformation. Default is 1.0.
    - resolution: Resolution parameter for clustering. Default is 1.0.
    - random_seed: Random seed for reproducibility. Default is 123.
    
    Stores clustering results in `adata.obs['slot_pattern']`.
    """
    try:
        # Construct graph
        print("Constructing neighbor graph")
        sources, targets = adata.obsp["slot_distances"].nonzero()
        dist = adata.obsp["slot_distances"][sources, targets]
        weights = gaussian_transform(dist.A1, sigma=sigma)
        
        # Initialize graph
        g = ig.Graph(directed=True)
        g.add_vertices(adata.obsp["slot_distances"].shape[0])
        g.add_edges(list(zip(sources, targets)))
        
        # Clustering parameters
        partition_kwargs = {
            "n_iterations": n_iterations,
            "seed": random_seed,
            "resolution_parameter": resolution
        }
        if weight:
            partition_kwargs["weights"] = np.array(weights).astype(np.float64)
        
        # Perform clustering
        print("Clustering...")
        partition_type = leidenalg.RBConfigurationVertexPartition
        part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
        
        # Store the results into adata.obs
        groups = np.array(part.membership)
        adata.obs["slot_pattern"] = pd.Categorical(
            values=groups.astype("U"),
            categories=natsorted(map(str, np.unique(groups))),
        )
        print("Clustering completed and stored in adata.obs['slot_pattern']")
    
    except Exception as e:
        print(f"Error in leiden clustering: {e}")
        raise ValueError("Failed to perform Leiden clustering.")


def slot_neighbors2edges(adata, key="slot_connectivity"):
    """
    Converts the neighbor connectivity matrix to a list of edges.

    Parameters:
    - adata: AnnData object containing the dataset.
    - key: The key in `adata.obsp` that stores the connectivity matrix (default is 'slot_connectivity').
    
    Returns:
    - List of protein edges, where each edge is represented as a frozenset of two protein names.
    """
    try:
        # Check if the connectivity matrix exists
        if key not in adata.obsp:
            raise KeyError(f"Connectivity matrix '{key}' not found in adata.obsp.")
        
        # Get connectivity matrix
        protein_knn_matrix = adata.obsp[key].toarray()
        protein_edges_list = []
        row = len(protein_knn_matrix)
        
        # Extract edges from the connectivity matrix
        for i in range(row):
            for j in range(row):
                if protein_knn_matrix[i][j] == 1 and i != j:
                    edge = frozenset([adata.obs_names[i], adata.obs_names[j]])
                    if edge not in protein_edges_list:
                        protein_edges_list.append(edge)
        
        return protein_edges_list

    except KeyError as e:
        print(f"KeyError in slot_neighbors2edges: {e}")
        raise
    except Exception as e:
        print(f"Error in slot_neighbors2edges: {e}")
        raise ValueError("Failed to convert neighbors to edges.")
