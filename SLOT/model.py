import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import wasserstein_distance, mannwhitneyu
import scipy.stats as stats
import statsmodels.stats.multitest as mt
from joblib import Parallel, delayed
from tqdm import tqdm

class SLOT_model:
    """
    SLOT_model wraps an AnnData object and provides methods to:
      - construct a binned spatial probability matrix,
      - compute SLOT scores,
      - perform permutation-based Wasserstein tests,
      - compare localization halves,
      - and define polar proteins with a Mann–Whitney U test.
    """
    
    def __init__(self, adata):
        """
        Initialize with an AnnData object.
        
        Parameters
        ----------
        adata : scanpy.AnnData
            Annotated data matrix with cells × genes (or proteins).
        """
        self.adata = adata

    def construct_prob_matrix(self, step=5):
        """
        Bin the spatial signal into non‐overlapping windows of size `step`
        and store the resulting probability matrix in adata.varm["prob_matrix"].
        
        Also compute a uniform prior over cells and a per‐gene scale factor.
        """
        # Sum expression per gene across all cells
        factor = self.adata.X.sum(axis=0)
        # Compute cell‐normalized expression probabilities, shape (genes, cells)
        x_prob = self.adata.X.T / factor[:, np.newaxis]
        # Uniform probability over cells
        uni_array = np.ones(len(self.adata)) / len(self.adata)
        # Bin the uniform prior in chunks of size `step`
        uni_prob = np.add.reduceat(uni_array, range(0, len(uni_array), step))
        # Bin the expression probabilities in the same way
        result_matrix = np.add.reduceat(x_prob,
                                        range(0, x_prob.shape[1], step),
                                        axis=1)
        
        # Store results in AnnData
        self.adata.varm["prob_matrix"] = result_matrix
        self.adata.uns["uni_prob"] = uni_prob
        # Scale factor per gene = total counts ÷ mean total counts
        scale_factor = factor / factor.mean()
        self.adata.uns["scale_factor"] = scale_factor
        
        print("Added spatial probability matrix to adata.varm")
        self.adata.uns["step"] = step

    def slot_metric(self, dist="uni", scale=False):
        """
        Compute the normalized Wasserstein distance (SLOT score) of each gene's
        distribution against a reference 'dist', which can be:
          - "uni": uniform prior,
          - or a user‐provided vector of length K.
        
        If `scale=True`, multiply by the per‐gene scale factor.
        """
        # Choose reference distribution
        prob_dist = self.adata.uns["uni_prob"] if dist == "uni" else dist
        prob_matrix = self.adata.varm["prob_matrix"]
        location = np.arange(len(prob_dist))
        distance = []

        # Maximum possible distance (uniform vs. maximally polarized)
        max_slot_prob = np.zeros(len(prob_dist))
        max_slot_prob[-1] = 1
        max_slot_score = wasserstein_distance(location,
                                              location,
                                              max_slot_prob,
                                              prob_dist)

        # Compute normalized Wasserstein distance per gene
        for row in range(len(prob_matrix)):
            wd = wasserstein_distance(location,
                                      location,
                                      prob_matrix[row],
                                      prob_dist)
            distance.append(wd / max_slot_score)

        # Store raw and optionally scaled SLOT scores
        self.adata.var["slot_score"] = distance
        if scale:
            self.adata.var["slot_score_scale"] = (
                np.array(distance) * self.adata.uns["scale_factor"]
            )
        
        print("Added SLOT score to adata.var")

    def slot_pattern_matching_test(self,
                                pattern="desc",
                                save_name='an',
                                N=2000,
                                n_jobs=1,
                                distance_scale=1):
        """
        Perform a permutation‐based test of each gene's distance to a
        pattern distribution ("desc", "asc", or "mid") and apply FDR correction.
        
        Parameters
        ----------
        pattern : str
            Pattern type: "desc", "asc", or "mid"
        save_name : str
            Prefix for saving results
        N : int
            Number of permutations for null distribution
        n_jobs : int
            Number of parallel jobs
        distance_scale : float, default=1.0
            Scale factor for distance calculation. Values < 1.0 make the test
            more lenient (easier to reject null), values > 1.0 make it stricter.
        
        Returns arrays of:
          - observed distances,
          - raw p‐values,
          - BH‐adjusted p‐values,
          - boolean mask of rejections (p_adj < 0.05).
        """
        prob_matrix = self.adata.varm["prob_matrix"]
        num_genes, K = prob_matrix.shape
        loc = np.arange(K)

        def make_pattern(p):
            """Generate the target pattern distribution q for a single vector p."""
            if pattern == "desc":
                return np.sort(p)[::-1]
            elif pattern == "asc":
                return np.sort(p)
            elif pattern == "mid":
                return np.array(self.mid_permu(p))
            else:
                raise ValueError("pattern must be 'desc', 'asc', or 'mid'")

        def test_gene(p):
            """
            For one gene's vector p:
              1. Compute observed Wasserstein distance D_obs to q.
              2. Permute p N times to build null distances D_i.
              3. Calculate p‐value using:
                 p = (1 + # of null D_i ≤ D_obs) / (N + 1)
                 — the +1 correction prevents zero p‐values in a finite test.
            """
            q = make_pattern(p)
            # Apply distance scaling to make test more/less stringent
            D_obs = wasserstein_distance(loc, loc, p, q) * distance_scale
            # Null distribution by permutation (also scaled)
            D_null = np.array([
                wasserstein_distance(loc, loc, np.random.permutation(p), q) * distance_scale
                for _ in range(N)
            ])
            # Phipson–Smyth correction: include the observed as one of N+1 trials
            p_val = (1 + np.sum(D_null <= D_obs)) / (N + 1)
            return D_obs, p_val

        # Parallelize across genes with progress bar
        results = Parallel(n_jobs=n_jobs)(
            delayed(test_gene)(prob_matrix[i])
            for i in tqdm(range(num_genes), desc="Pattern matching test")
        )
        D_obs_array = np.array([r[0] for r in results])
        p_values    = np.array([r[1] for r in results])

        # Benjamini–Hochberg FDR correction
        rejected, pvals_corrected, _, _ = mt.multipletests(
            p_values, alpha=0.05, method="fdr_bh"
        )

        # Store raw and adjusted p‐values
        self.adata.var[f"{save_name}_p_value_raw"] = p_values
        self.adata.var[f"{save_name}_p_value_adj"] = pvals_corrected
        self.adata.var[f"{save_name}_pattern_dist"] = D_obs_array
        print(f"Added {save_name} pattern_dist, p-values and p-value_adj to adata.var")
        print(f"Using distance_scale={distance_scale} for pattern matching test")

    @staticmethod
    def mid_permu(arr):
        """
        Rearrange a sorted array into a 'middle‐peaked' pattern:
        largest in the center, then alternating to left/right.
        """
        sorted_arr = sorted(arr, reverse=True)
        n = len(sorted_arr)
        result = [0] * n
        mid = n // 2
        result[mid] = sorted_arr[0]
        for i in range(1, n):
            if i % 2 == 1:
                result[mid - (i // 2 + 1)] = sorted_arr[i]
            else:
                result[mid + (i // 2)] = sorted_arr[i]
        return result

    def compare_halves_matrix(self, fc=2):
        """
        Label each gene as 'AN' or 'VG' if one half's sum exceeds
        fc × the other; otherwise 'UN' (undetermined).
        """
        matrix = self.adata.varm['prob_matrix']
        rows, cols = matrix.shape
        mid = cols // 2
        first_half = matrix[:, :mid].sum(axis=1)
        second_half = matrix[:, mid:].sum(axis=1)
        labels = np.full(rows, "UN", dtype=object)
        labels[first_half >= fc * second_half] = "AN"
        labels[second_half >= fc * first_half] = "VG"
        self.adata.var['orientation'] = labels

    def define_polar_proteins(self, score='slot_score', percentile=95):
        """
        Call proteins 'polar' if their SLOT score is in the top `percentile`%.
        Then perform a Mann–Whitney U test comparing polar vs. nonpolar scores.
        """
        slot_scores = self.adata.var[score]
        threshold = np.percentile(slot_scores, percentile)
        self.adata.uns['Threshold'] = threshold
        self.adata.var["polar_protein"] = slot_scores >= threshold
        
        polar = slot_scores[self.adata.var["polar_protein"]]
        nonpolar = slot_scores[~self.adata.var["polar_protein"]]
        
        # Test whether polar scores are significantly greater
        stat, p_value = mannwhitneyu(polar, nonpolar, alternative='greater')
        self.adata.uns['p_value_mwu'] = p_value
        print(f"Mann–Whitney U statistic: {stat}, p-value: {p_value}")
        return threshold, stat, p_value

    def compute_var_distance_matrix(self, feature="prob_matrix", n_jobs=-1):
        """
        Compute pairwise Wasserstein distance (W1) between all variables (genes/proteins)
        and store the distance matrix in adata.varm['var_distance'].
        
        Parameters
        ----------
        feature : str
            Data source to use for distance calculation:
            - "prob_matrix": use adata.varm["prob_matrix"] (default)
            - "X": use adata.X (transposed to genes x cells)
        n_jobs : int
            Number of parallel jobs to use (-1 means using all processors)
        """
        if feature == "prob_matrix":
            prob_matrix = self.adata.varm["prob_matrix"]
        elif feature == "X":
            # Convert to probability distribution by normalizing each gene
            prob_matrix = self.adata.X.T / self.adata.X.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("feature must be either 'prob_matrix' or 'X'")
        
        n_var = prob_matrix.shape[0]
        loc = np.arange(prob_matrix.shape[1])
        
        # Initialize distance matrix
        dist_matrix = np.zeros((n_var, n_var))
        
        # Function to compute distance for one pair
        def _compute_distance(i, j):
            return wasserstein_distance(loc, loc, prob_matrix[i], prob_matrix[j])
        
        # Parallel computation of upper triangle
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_distance)(i, j)
            for i in tqdm(range(n_var), desc="Computing distances")
            for j in range(i+1, n_var)
        )
        
        # Fill upper triangle
        idx = 0
        for i in range(n_var):
            for j in range(i+1, n_var):
                dist_matrix[i, j] = results[idx]
                idx += 1
        
        # Make symmetric by copying upper to lower triangle
        dist_matrix = dist_matrix + dist_matrix.T
        
        # Store result
        self.adata.varm['var_distance'] = dist_matrix
        print("Added pairwise variable distance matrix to adata.varm['var_distance']")