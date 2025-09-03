import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import math
import pandas as pd

def gaussian_splines_norm(
        arr: np.ndarray, 
        sigma: float, 
        n_splines: int
    ) -> np.ndarray:
    """
    Apply Gaussian smoothing and cubic spline interpolation to a 1D array, then return the normalized result.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array to be smoothed and interpolated.
    sigma : float
        Standard deviation for Gaussian kernel (controls smoothing strength).
    n_splines : int
        Number of points for cubic spline interpolation. If None or 0, skip interpolation.

    Returns
    -------
    np.ndarray
        The processed (smoothed, interpolated) array.
    """
    # Step 1: Gaussian smoothing
    arr = gaussian_filter(arr, sigma=sigma)

    # Step 2: Cubic spline interpolation (if requested)
    if n_splines:
        x = np.arange(arr.shape[0])
        cs = CubicSpline(x, arr, bc_type='natural')  # 'natural' boundary for smoothness
        x_new = np.linspace(x[0], x[-1], n_splines)
        arr = cs(x_new)

    # Step 3: (Optional) Normalization to [-1, 1] can be added if needed
    # min_val = arr.min()
    # max_val = arr.max()
    # arr = (arr - min_val) / (max_val - min_val) * 2 - 1

    return arr

def plot_3d(protein, adata, sigma: float = 3,
            n_splines: int = 100, save_dir=None, dpi=300, cbar_label='Location probability', elev=10, azim=225):
    """
    Create a 3D spherical surface plot for protein expression data.
    
    Parameters
    ----------
    protein : str
        Name of the protein/gene to visualize from adata.var_names.
    adata : AnnData
        AnnData object containing expression data with samples as observations.
    sigma : float, default=3
        Standard deviation for Gaussian smoothing kernel. Higher values create smoother surfaces.
    n_splines : int, default=100
        Number of points for cubic spline interpolation. Controls surface resolution.
    save_dir : str or None, default=None
        Directory path to save the plot. If None, displays the plot instead of saving.
    dpi : int, default=300
        Resolution (dots per inch) for saved figure. Higher values create sharper images.
    cbar_label : str, default='Location probability'
        Label text for the colorbar legend.
    elev : int, default=10
        Elevation angle for 3D view in degrees. Controls vertical viewing angle.
    azim : int, default=225
        Azimuth angle for 3D view in degrees. Controls horizontal viewing angle.
    
    Returns
    -------
    None
        Displays or saves the 3D plot.
    """
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    arr = adata[:,protein].X.reshape(-1)
    normalized_arr = gaussian_splines_norm(arr, sigma, n_splines)

    sample_num = normalized_arr.shape[0]

    # Make data
    u = np.linspace(0, 2 * np.pi, sample_num)
    v = np.linspace(0, np.pi, sample_num)
    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

    color_vector = np.tile(normalized_arr, sample_num).reshape(sample_num, sample_num)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection='3d')

    norm = plt.Normalize(color_vector.min(), color_vector.max())
    color_map = plt.cm.Blues
    colors = color_map(norm(color_vector))

    # Plot the surface
    surf = ax.plot_surface(x, y, z, 
                          facecolors=colors,
                          rstride=1, cstride=1)

    # Set an equal aspect ratio
    ax.set_aspect('equal')
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim)


    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    mappable.set_array(color_vector)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    if save_dir:
        plt.savefig(save_dir, bbox_inches='tight', dpi=dpi)
    else:
        plt.show()
    return

def plot_2d(protein, adata, 
            value_type='slot',
            cbar_label='Location probability',
            save_dir=None, 
            dpi=300,
            cmap=plt.cm.Blues,
            sigma: float = 3,
            n_splines: int = 100,
            ):
    """
    Create a 2D circular heatmap visualization for protein expression data.
    
    Parameters
    ----------
    protein : str
        Name of the protein/gene to visualize from adata.var_names.
    adata : AnnData
        AnnData object containing expression data with samples as observations.
    value_type : str, default='slot'
        Type of values being visualized (used for documentation purposes).
    cbar_label : str, default='Location probability'
        Label text for the colorbar legend.
    save_dir : str or None, default=None
        Directory path to save the plot. If None, displays the plot instead of saving.
        When provided, saves the plot as '{protein}.png' in the specified directory.
    dpi : int, default=300
        Resolution (dots per inch) for saved figure. Higher values create sharper images.
    cmap : matplotlib.colors.Colormap, default=plt.cm.Blues
        Colormap for the heatmap visualization. Controls the color scheme of the plot.
    sigma : float, default=3
        Standard deviation for Gaussian smoothing kernel. Higher values create smoother surfaces.
    n_splines : int, default=100
        Number of points for cubic spline interpolation. Controls surface resolution.
    
    Returns
    -------
    None
        Displays or saves the 2D circular heatmap plot.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    import matplotlib.patches as patches
    from matplotlib.patches import Circle
    
    # Data nomalization
    arr = adata[:,protein].X.reshape(-1)
    normalized_arr = gaussian_splines_norm(arr, sigma, n_splines)
    sample_num = normalized_arr.shape[0]

    # Plot
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    path = Path([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 1]])
    patch = PathPatch(path, facecolor='none')
    
    color_vector = np.tile(normalized_arr, sample_num).reshape(sample_num, sample_num).T
    im = ax.imshow(color_vector, 
                cmap=cmap,
                )
    patch = patches.Circle((sample_num//2, sample_num//2), 
                        radius=sample_num//2-1, 
                        transform=ax.transData,
                        )
    circle = Circle((sample_num//2, sample_num//2),  
                    radius=sample_num//2-1.5,    
                    edgecolor='lightgrey',  
                    linewidth=0,      
                    facecolor='none', 
                    transform=ax.transData
                    )  
    im.set_clip_path(patch)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{protein}', fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.1)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(cbar_label, fontsize=10)
    
    # Save the figure
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{protein}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()
    return


def plot_cluster_rep_dist(adata, 
                          category_field='slot_pattern',
                          sorting_field='mean_counts',
                          distribution_key='prob_10bins',
                          top_n=5,
                          save_path='cluster_representative_distribution.pdf'):
    """
    Plot representative distributions for each cluster.
    
    For each cluster, the function selects the top_n samples sorted by the sorting_field in descending order,
    computes the average distribution from the distribution matrix, and plots these distributions as a line chart.
    
    Parameters:
    - adata: AnnData object containing sample metadata in adata.obs and distribution matrix in adata.obsm.
    - category_field: Key in adata.obs containing cluster labels (default 'slot_pattern').
    - sorting_field: Key in adata.obs used for sorting samples in descending order (default 'mean_counts').
    - distribution_key: Key in adata.obsm containing the distribution matrix (default 'prob_10bins').
    - top_n: Number of top samples per cluster to average (default 5).
    - save_path: File path to save the resulting plot.
    """
    # Get unique clusters from the specified category field
    clusters = adata.obs[category_field].unique()
    
    # Dictionaries to store the average distribution and average sorting_field value for each cluster
    cluster_avg_dist = {}
    cluster_avg_sorting = {}
    
    # Iterate over each cluster
    for cat in clusters:
        # Select samples in the current cluster
        cat_df = adata.obs[adata.obs[category_field] == cat]
        # Sort the samples by the sorting_field in descending order and select the top_n samples
        top_cat_samples = cat_df.sort_values(by=sorting_field, ascending=False).head(top_n)
        selected_indices = top_cat_samples.index.tolist()  # List of sample names
        
        # Get index positions for these samples in the AnnData object
        indices_positions = adata.obs.index.get_indexer(selected_indices)
        # Extract the corresponding distribution matrix rows
        sub_matrix = adata.obsm[distribution_key][indices_positions, :]
        
        # Calculate the average distribution for the current cluster
        avg_distribution = np.mean(sub_matrix, axis=0)
        cluster_avg_dist[cat] = avg_distribution
        
        # Calculate the average sorting_field value for these top samples
        avg_sorting = top_cat_samples[sorting_field].mean()
        cluster_avg_sorting[cat] = avg_sorting
    
    # Sort clusters based on the average sorting_field value in descending order
    sorted_clusters = sorted(cluster_avg_sorting, key=cluster_avg_sorting.get, reverse=True)
    
    # Create the line chart
    plt.figure(figsize=(10, 6))
    # Define x-axis based on the number of bins in the distribution matrix
    x = np.arange(adata.obsm[distribution_key].shape[1])
    
    # Plot the average distribution for each cluster in sorted order
    for cat in sorted_clusters:
        plt.plot(x, cluster_avg_dist[cat], marker='o', 
                 label=f'Cluster {cat}')
    
    plt.xlabel('Bins AN -> VG')
    plt.ylabel('Average Distribution')
    plt.title('Representative Distribution per Cluster')
    plt.legend()
    plt.tight_layout()
    # Save the plot with a transparent background
    plt.savefig(save_path, dpi=300, transparent=True)
    plt.close()
    
import numpy as np
import matplotlib.pyplot as plt

def plot_cluster_rep_hist(adata, 
                          category_field='slot_pattern',
                          sorting_field='mean_counts',
                          distribution_key='prob_10bins',
                          top_n=5,
                          save_path='cluster_representative_hist.png'):
    """
    Plot representative distributions for each cluster as histograms.
    
    For each cluster, the function selects the top_n samples sorted by the sorting_field in descending order,
    computes the average distribution from the distribution matrix, and displays these distributions as bar charts.
    Each cluster is shown in a separate subplot.
    
    Parameters:
    - adata: AnnData object containing sample metadata in adata.obs and distribution matrix in adata.obsm.
    - category_field: Key in adata.obs containing cluster labels (default 'slot_pattern').
    - sorting_field: Key in adata.obs used for sorting samples in descending order (default 'mean_counts').
    - distribution_key: Key in adata.obsm containing the distribution matrix (default 'prob_10bins').
    - top_n: Number of top samples per cluster to average (default 5).
    - save_path: File path to save the resulting plot.
    """
    # Get unique clusters from the specified category field
    clusters = adata.obs[category_field].unique()
    
    # Dictionaries to store the average distribution and average sorting_field value for each cluster
    cluster_avg_dist = {}
    cluster_avg_sorting = {}
    
    # Iterate over each cluster to compute average distributions
    for cat in clusters:
        # Select samples belonging to the current cluster
        cat_df = adata.obs[adata.obs[category_field] == cat]
        # Sort samples by the sorting_field in descending order and select the top_n samples
        top_cat_samples = cat_df.sort_values(by=sorting_field, ascending=False).head(top_n)
        selected_indices = top_cat_samples.index.tolist()  # List of sample names
        
        # Get the index positions of these samples in the AnnData object
        indices_positions = adata.obs.index.get_indexer(selected_indices)
        # Extract the corresponding rows from the distribution matrix
        sub_matrix = adata.obsm[distribution_key][indices_positions, :]
        
        # Compute the average distribution for the current cluster
        avg_distribution = np.mean(sub_matrix, axis=0)
        cluster_avg_dist[cat] = avg_distribution
        
        # Compute the average sorting_field value for these top samples
        avg_sorting = top_cat_samples[sorting_field].mean()
        cluster_avg_sorting[cat] = avg_sorting
    
    # Sort clusters based on the average sorting_field value in descending order
    sorted_clusters = sorted(cluster_avg_sorting, key=cluster_avg_sorting.get, reverse=True)
    
    n_clusters = len(sorted_clusters)
    # Create subplots (one subplot per cluster)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 4 * n_clusters))
    if n_clusters == 1:
        axes = [axes]
    
    # Define x-axis based on the number of bins in the distribution matrix
    x = np.arange(adata.obsm[distribution_key].shape[1])
    
    # Plot a bar chart (histogram) for each cluster
    for ax, cat in zip(axes, sorted_clusters):
        ax.bar(x, cluster_avg_dist[cat])
        ax.set_xlabel('Bins')
        ax.set_ylabel('Average Distribution')
        ax.set_title(f'Cluster {cat}')
    
    plt.tight_layout()
    # Save the plot with a transparent background
    plt.savefig(save_path, dpi=300, transparent=True)
    plt.close()
    
def plot_cluster_rep_heatmap(adata, 
                             category_field='slot_pattern',
                             sorting_field='mean_counts',
                             distribution_key='prob_10bins',
                             top_n=4,
                             save_dir='./',
                             color_key=None):
    """
    Plot representative heatmaps for each cluster.
    
    For each cluster (specified in the category_field), the function selects the top_n samples 
    sorted by the sorting_field in descending order, extracts the corresponding distribution matrix 
    (from obsm), transposes it, and visualizes it as a heatmap with annotations. The colormap is 
    customized to transition from white to a cluster-specific color, with values below a threshold 
    shown in light grey.
    
    Parameters:
    - adata: AnnData object containing sample metadata in .obs and the distribution matrix in .obsm.
    - category_field: The key in .obs that contains cluster labels (default 'slot_pattern').
    - sorting_field: The key in .obs used for sorting samples in descending order (default 'mean_counts').
    - distribution_key: The key in .obsm for the distribution matrix (default 'prob_10bins').
    - top_n: Number of top samples per cluster to select (default 4).
    - save_dir: Directory path where the plots will be saved.
    - color_key: Key in .uns for the cluster colors. If None, defaults to '{category_field}_colors'.
    """
    # Ensure the category field is a categorical variable (to maintain the order matching the color list)
    if not pd.api.types.is_categorical_dtype(adata.obs[category_field]):
        adata.obs[category_field] = pd.Categorical(adata.obs[category_field])
    
    # Get categories in order and corresponding colors
    categories = adata.obs[category_field].cat.categories
    if color_key is None:
        color_key = f"{category_field}_colors"
    cluster_colors = adata.uns[color_key]
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Iterate over each category
    for i, cat in enumerate(categories):
        # ------------------ Select Representative Samples ------------------
        # Filter samples in the current cluster and sort them by the sorting_field in descending order,
        # then select the top_n samples.
        cat_df = adata.obs[adata.obs[category_field] == cat]
        top_cat_samples = cat_df.sort_values(by=sorting_field, ascending=False).head(top_n)
        selected_indices = top_cat_samples.index.tolist()
        
        # ------------------ Extract Corresponding Matrix ------------------
        indices_positions = adata.obs.index.get_indexer(selected_indices)
        sub_matrix = adata.obsm[distribution_key][indices_positions, :]
        
        # Transpose the matrix so that each column corresponds to a sample (original rows correspond to proteins)
        sub_matrix = sub_matrix.T
        
        # ------------------ Construct Custom Colormap ------------------
        # Create a colormap that transitions from white to the cluster-specific color,
        # with values below the threshold displayed as light grey.
        cluster_color = cluster_colors[i]
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['white', cluster_color])
        cmap.set_under('lightgrey')
        
        # ------------------ Create Annotation Matrix ------------------
        # Annotate values greater than 0.1 with two decimal places; otherwise, leave empty.
        annot_labels = np.empty(sub_matrix.shape, dtype=object)
        for r in range(sub_matrix.shape[0]):
            for c in range(sub_matrix.shape[1]):
                if sub_matrix[r, c] > 0.1:
                    annot_labels[r, c] = f"{sub_matrix[r, c]:.2f}"
                else:
                    annot_labels[r, c] = ""
        
        # ------------------ Plot Heatmap ------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            sub_matrix,
            ax=ax,
            cmap=cmap,
            linewidths=2,
            linecolor='white',
            cbar=False,
            square=True,
            vmin=0.0001,  # values below this threshold will be displayed in light grey
            annot=annot_labels,
            fmt=''
        )
        
        # Set x-axis: center ticks on each column (sample) and display sample names rotated 45 degrees at the top.
        num_samples = sub_matrix.shape[1]
        ax.set_xticks(np.arange(num_samples) + 0.5)
        ax.set_xticklabels(selected_indices, rotation=45, fontsize=10)
        ax.xaxis.tick_top()  # move x-axis labels to the top
        
        # Adjust axis lines: hide the bottom spine and customize the top spine.
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['top'].set_color('black')
        
        # Use seaborn's despine without removing the top spine.
        sns.despine(ax=ax, left=True, bottom=False, top=False, right=True)
        
        # Remove y-axis ticks.
        ax.set_yticks([])
        
        # Save the figure with a filename that includes the category name.
        file_path = os.path.join(save_dir, f"cluster_heatmap_{cat}.pdf")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
def cluster_rep_heatmap_group(adata, 
                                    category_field='slot_pattern',
                                    sorting_field='mean_counts',
                                    distribution_key='prob_10bins',
                                    top_n=4,
                                    save_path='cluster_heatmap_single.pdf',
                                    color_key=None,
                                    grid_linewidth=2,
                                    grid_color='white',
                                    annot_threshold=0.1):
    """
    Plot representative heatmaps for each cluster as one single combined image, without using Matplotlib ticks.
    
    For each cluster (specified in category_field), the function:
      1. Selects the top_n samples sorted by sorting_field in descending order.
      2. Extracts the corresponding distribution matrix from adata.obsm[distribution_key], transposes it,
         and plots it as a segment of a larger heatmap with cell annotations.
      3. Draws a horizontal color bar (line) above each cluster segment, matching the cluster's color.
      4. Places sample labels manually above the heatmap, with labels centered over each column.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with sample metadata in .obs and distribution matrix in .obsm.
    category_field : str
        Key in .obs that contains cluster labels (default 'slot_pattern').
    sorting_field : str
        Key in .obs used for sorting samples in descending order (default 'mean_counts').
    distribution_key : str
        Key in .obsm for the distribution matrix (default 'prob_10bins').
    top_n : int
        Number of top samples per cluster to select (default 4).
    save_path : str
        File path to save the final combined figure.
    color_key : str or None
        Key in .uns for cluster colors. If not found, a default palette is generated.
    grid_linewidth : float
        Width of the grid lines between cells (default 2).
    grid_color : str
        Color of the grid lines (default 'white').
    annot_threshold : float
        Only annotate cell values greater than this threshold (default 0.1).
    """
    # Ensure the category field is categorical to preserve ordering
    if not pd.api.types.is_categorical_dtype(adata.obs[category_field]):
        adata.obs[category_field] = pd.Categorical(adata.obs[category_field])
    
    # Get cluster categories in order
    categories = adata.obs[category_field].cat.categories
    
    # Determine cluster colors: use provided color_key if available; otherwise default to '{category_field}_colors'
    if color_key is None:
        color_key = f"{category_field}_colors"
    if color_key in adata.uns:
        cluster_colors = adata.uns[color_key]
    else:
        # Generate a default color palette using seaborn's tab10 palette
        n_categories = len(categories)
        cluster_colors = sns.color_palette("tab10", n_categories)
    
    # Number of bins (height of the heatmap) from the distribution matrix
    n_bins = adata.obsm[distribution_key].shape[1]
    
    # We'll keep track of the x-centers for each sample label
    x_positions = []
    x_labels = []
    
    # Keep track of the horizontal extents for each cluster to draw the color bar
    cluster_extents = []  # List of tuples: (x_start, x_end, cluster_color)

    total_columns = 0

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # ------------------ Main loop: draw each cluster's heatmap segment ------------------
    for i, cat in enumerate(categories):
        # 1. Select top_n samples by sorting_field
        cat_df = adata.obs[adata.obs[category_field] == cat]
        top_cat_samples = cat_df.sort_values(by=sorting_field, ascending=False).head(top_n)
        selected_indices = top_cat_samples.index.tolist()
        
        # 2. Extract and transpose the distribution sub-matrix
        indices_positions = adata.obs.index.get_indexer(selected_indices)
        sub_matrix = adata.obsm[distribution_key][indices_positions, :]
        sub_matrix = sub_matrix.T  # shape: [n_bins, n_samples]
        n_samples = sub_matrix.shape[1]
        
        # 3. Build a custom colormap that transitions from white to cluster_color
        cluster_color = cluster_colors[i]
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['white', cluster_color])
        cmap.set_under('lightgrey')
        
        # 4. Plot the segment with imshow
        x_start = total_columns
        x_end = total_columns + n_samples
        extent = [x_start, x_end, 0, n_bins]
        
        ax.imshow(
            sub_matrix,
            cmap=cmap,
            aspect='auto',
            interpolation='none',
            vmin=0.0001,
            extent=extent,
            origin='upper'
        )
        
        # 5. Annotate each cell above annot_threshold
        for row in range(n_bins):
            for col in range(n_samples):
                val = sub_matrix[row, col]
                if val > annot_threshold:
                    y_coord = n_bins - (row + 0.5)
                    ax.text(x_start + col + 0.5, y_coord, f"{val:.2f}", ha='center', va='center', fontsize=9)
        
        # Record x-centers and labels for each sample
        for col in range(n_samples):
            x_positions.append(x_start + col + 0.5)
            x_labels.append(selected_indices[col])
        
        # Save the cluster's horizontal extent for drawing the color bar
        cluster_extents.append((x_start, x_end, cluster_color))
        
        # Update for next cluster
        total_columns += n_samples
    
    # ------------------ Draw grid lines for the entire image ------------------
    for x in np.arange(0, total_columns + 1, 1):
        line = ax.axvline(x=x, color=grid_color, linewidth=grid_linewidth, zorder=5)
        line.set_antialiased(False)
        line.set_snap(True)
    for y in np.arange(0, n_bins + 1, 1):
        line = ax.axhline(y=y, color=grid_color, linewidth=grid_linewidth, zorder=5)
        line.set_antialiased(False)
        line.set_snap(True)
    
    # Set axis limits and extend y-axis to make room for color bar and labels
    ax.set_xlim(0, total_columns)
    ax.set_ylim(0, n_bins)
    
    # Remove standard ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # ------------------ Draw the horizontal color line for each cluster ------------------
    # Place these lines at y = n_bins + 1.0 (just above the heatmap)
    cluster_line_y = n_bins 
    for (x_start, x_end, c_color) in cluster_extents:
        ax.hlines(y=cluster_line_y, xmin=x_start, xmax=x_end, 
                  color=c_color, linewidth=4, zorder=6)
    
    # ------------------ Manually place sample labels above each column ------------------
    # Place them at y = n_bins + 2.0 so they appear above the colored lines
    label_y_pos = n_bins + 0.2
    for x, lab in zip(x_positions, x_labels):
        ax.text(
            x, label_y_pos, lab,
            rotation=45, ha='center', va='bottom', fontsize=12, fontweight='bold',
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    
def plot_gene_spatial(adata, gene_name, bins=None, save_path=None, plot_type='line'):
    """
    Visualize gene expression across samples with optional smoothing by binning.
    
    Parameters:
    - adata: AnnData object with samples in obs and genes in var
    - gene_name: Name of the gene to visualize
    - bins: Number of bins to smooth the expression profile (None means no binning)
    - save_path: Path to save the plot (None means show plot)
    - plot_type: 'line' for line plot, 'bar' for bar plot, or 'scatter' for scatter plot
    
    Returns:
    - None (displays or saves plot)
    """
    # Check if gene exists
    if gene_name not in adata.var_names:
        raise ValueError(f"Gene {gene_name} not found in adata.var_names")
    
    # Get expression values in original sample order
    expr_values = adata[:, gene_name].X.toarray().flatten()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    if bins is not None:
        # Calculate bin averages and positions
        bin_size = len(expr_values) / bins
        bin_means = []
        bin_ranges = []  # Store (start, end) for each bin
        
        for i in range(bins):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            if i == bins - 1:  # Ensure last bin includes all remaining samples
                end = len(expr_values)
            
            bin_means.append(np.mean(expr_values[start:end]))
            bin_ranges.append((start, end))
        
        # Create x-axis positions and labels
        x_positions = np.arange(bins)  # Use bin indices as x positions
        x_labels = [f"{start}-{end-1}" for start, end in bin_ranges]
        
        # Plot binned data
        if plot_type == 'bar':
            plt.bar(x_positions, bin_means, width=0.8, alpha=0.7)
            plt.xticks(x_positions, x_labels, rotation=45)
        elif plot_type == 'scatter':
            plt.scatter(x_positions, bin_means, s=50, alpha=0.7)
            plt.xticks(x_positions, x_labels, rotation=45)
        else:  # line plot
            plt.plot(x_positions, bin_means, '-o', markersize=4)
            plt.xticks(x_positions, x_labels, rotation=45)
    else:
        # Plot all samples without binning
        if plot_type == 'bar':
            plt.bar(range(len(expr_values)), expr_values, width=0.8)
        elif plot_type == 'scatter':
            plt.scatter(range(len(expr_values)), expr_values, s=20, alpha=0.7)
        else:  # line plot
            plt.plot(expr_values)
    
    plt.xlabel('Sample Position')
    plt.ylabel('Expression Level')
    plt.title(f'Expression of {gene_name} Across Samples')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_gene_scatter_with_band(adata, gene_name, bins=None, save_path=None, ci=0.95):
    """
    Plot gene expression scatter with fitted trend line and confidence band,
    clipping negative y-values.
    """
    # Get expression data (non-negative)
    expr = np.clip(adata[:, gene_name].X.toarray().flatten(), 0, None)
    x = np.arange(len(expr))
    
    plt.figure(figsize=(10, 5))
    
    if bins:
        # Bin the data
        bin_size = len(expr) / bins
        x_binned = []
        y_binned = []
        
        for i in range(bins):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            if i == bins - 1:
                end = len(expr)
            
            x_binned.append(np.mean(x[start:end]))
            y_binned.append(np.mean(expr[start:end]))
        
        x_plot = np.array(x_binned)
        y_plot = np.array(y_binned)
        plt.scatter(x_binned, y_binned, s=50, alpha=0.7)
    else:
        x_plot = x
        y_plot = expr
        plt.scatter(x, expr, s=20, alpha=0.5)
    
    # Fit polynomial trend line (constrained to non-negative)
    from scipy.optimize import curve_fit
    def fit_func(x, a, b, c):
        return np.maximum(a * x**2 + b * x + c, 0)  # Clip at 0
    
    popt, _ = curve_fit(fit_func, x_plot, y_plot, p0=[0, 0, 0])
    y_trend = fit_func(x_plot, *popt)
    
    # Calculate confidence band (also clipped)
    residuals = y_plot - y_trend
    std_err = np.std(residuals)
    n = len(x_plot)
    from scipy import stats
    t_val = stats.t.ppf(1 - (1-ci)/2, n-2)
    ci_band = t_val * std_err * np.sqrt(1/n + (x_plot - np.mean(x_plot))**2 / np.sum((x_plot - np.mean(x_plot))**2))
    
    # Plot trend line and band (automatically clipped by ylim)
    plt.plot(x_plot, y_trend, 'r-', lw=2, label='Trend line')
    plt.fill_between(x_plot, 
                    np.maximum(y_trend - ci_band, 0),  # Clip lower band at 0
                    y_trend + ci_band,
                    color='red', alpha=0.2, label=f'{int(ci*100)}% CI')
    
    # Set y-axis minimum to 0
    plt.ylim(bottom=0)
    
    plt.xlabel('Sample Position')
    plt.ylabel('Expression Level')
    plt.title(f'{gene_name} Expression with Trend Band')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_cluster_pattern(adata, target_class, n_bins=10, color="darkgreen",
                         figsize=(3, 3), dpi=300, save_prefix="mrna_cluster"):
    """
    Plot the spatial distribution curves for a specified cluster (slot_pattern).
    
    Parameters
    ----------
    adata : AnnData
        AnnData object (n_obs × n_vars).
    target_class : str
        The cluster label (string) corresponding to adata.obs['slot_pattern'].
    n_bins : int, default=10
        Number of bins to split the positions into.
    color : str, default="darkgreen"
        Color for the target cluster.
    figsize : tuple, default=(3,3)
        Figure size.
    dpi : int, default=300
        Resolution of the figure.
    save_prefix : str, default="mrna_cluster"
        Prefix for the saved file name.
    """

    # Display index should be +1 because slot_pattern is 0-based
    display_class = int(target_class) + 1

    # 1. Extract proteins belonging to the target cluster
    mask = adata.obs['slot_pattern'] == target_class
    subset = adata[mask]

    # 2. Split positions into bins
    n_positions = adata.shape[1]
    bins = np.array_split(np.arange(n_positions), n_bins)

    # 3. Compute bin averages
    def bin_means(matrix, bins):
        """Compute mean values of each bin for every protein (rows)."""
        return np.vstack([matrix[:, b].mean(axis=1) for b in bins]).T

    subset_binned = bin_means(subset.X, bins)
    all_binned = bin_means(adata.X, bins)

    # 4. Compute mean curves
    subset_mean = subset_binned.mean(axis=0)
    all_mean = all_binned.mean(axis=0)

    # 5. Normalize to percentages (sum = 100)
    subset_binned = subset_binned / subset_binned.sum(axis=1, keepdims=True) * 100
    subset_mean = subset_mean / subset_mean.sum() * 100
    all_mean = all_mean / all_mean.sum() * 100

    # 6. Plot
    plt.figure(figsize=figsize, dpi=dpi)

    # Thin lines: individual proteins in the cluster
    for row in subset_binned:
        plt.plot(range(1, n_bins+1), row, color=color, alpha=0.1, linewidth=0.1)

    # Bold line: mean of the cluster
    plt.plot(range(1, n_bins+1), subset_mean, color=color, linewidth=2.5,
             label=f"mean (pattern {display_class})")

    # Dashed line: global mean
    plt.plot(range(1, n_bins+1), all_mean, linestyle="--", color="black", linewidth=1.5,
             label="global mean")

    # Labels and legend
    plt.xlabel("Bins (AN ---> VG)")
    plt.ylabel("Normalized localization probability (%)")
    plt.title(f"Pattern {display_class} (n={subset.shape[0]})")
    plt.legend()

    sns.despine()
    plt.tight_layout()

    # Save figure
    save_name = f"{save_prefix}_{display_class}.png"
    plt.savefig(save_name)
    plt.show()

    print(f"Figure saved as {save_name}")