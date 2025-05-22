from typing import List
import numpy as np
import pandas as pd
from ete3 import Tree
from joblib import Parallel, delayed, parallel_backend, Memory
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from typing import List, Tuple, Set, Dict
# from numba import jit

def simulate_glrates_bit(self):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch
    For each trait only simulates on branches beyond a certain distance from the root, 
    This threshold is chosen as the first branch where a trait arises in the ancestral trait reconstruction
    """
    from tree_simulator import TreeSimulator
    assert(type(self) == TreeSimulator)
    # Preprocess and setup
    all_nodes = list(self.tree.traverse()) # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    bits = 128
    nptype = np.uint128
    sim = np.zeros((num_nodes, num_traits), dtype=nptype)
    self.NUM_TRIALS = bits

    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    gain_rates = self.gains / (self.gain_subsize * self.MULTIPLIER)
    loss_rates = self.losses / (self.loss_subsize * self.MULTIPLIER)
    np.nan_to_num(gain_rates, copy = False)
    np.nan_to_num(loss_rates, copy = False)

    # Distance calculations
    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse(): # type: ignore
        if node in node_dists: continue
        node_dists[node] = node_dists[node.up] + node.dist

    print("Simulating Trees...")
    for node in self.tree.traverse(): # type: ignore
        if node.up == None:
            root = self.root_states > 0
            sim[node_map[node],root] = 2^self.NUM_TRIALS-1
            continue
        
        parent = sim[node_map[node.up], :]
        node_dist_multiplier = node.dist * self.MULTIPLIER
        node_total_dist = node_dists[node]  # Total distance from the root to the current node

        # Zero out gain and loss rates for traits where the node's distance is less than the specified threshold
        applicable_traits_gains = node_total_dist >= self.dists
        applicable_traits_losses = node_total_dist >= self.loss_dists
        gain_events = np.zeros((num_traits), dtype=nptype)
        loss_events = np.zeros((num_traits), dtype=nptype)
        gain_events[applicable_traits_gains] = np.packbits((np.random.binomial(node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis], (applicable_traits_gains.sum(), self.NUM_TRIALS)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(np.uint64).flatten()
        loss_events[applicable_traits_losses] = np.packbits((np.random.binomial(node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis], (applicable_traits_losses.sum(), self.NUM_TRIALS)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(np.uint64).flatten()

        # Handle event cancellation
        # canceled_events = np.logical_and(gain_events, loss_events)
        # gain_events[canceled_events] = False
        # loss_events[canceled_events] = False

        updated_state = np.bitwise_or(parent, gain_events)  # Gain new traits
        updated_state = np.bitwise_and(updated_state, np.bitwise_not(loss_events))  # Remove lost traits
        # Store updated node state
        sim[node_map[node], :] = updated_state

        # print(f"Node {node.name} Completed")

    print("Completed Tree Simulation Sucessfully")
    lineages = sim[[node_map[node] for node in self.tree], :]
    # return lineages
    res = compile_results_KDE_bit_async(self, lineages, bits = bits, nptype = nptype)

    return res


def compile_results_KDE_bit(self, sim, obspairs=[], batch_size=1000, bits = 128, nptype = np.uint128):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction", 
                               "p-value_ant", "p-value_syn", "p-value", "significant", 
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

    def circular_bitshift_right(arr: np.ndarray, k: int) -> np.ndarray:
        """Perform a circular right bit shift on all np.uint64 entries in an array."""
        k %= bits
        return np.bitwise_or(np.right_shift(arr, k), np.left_shift(arr, bits - k))
    
    def sum_all_bits(arr: np.ndarray) -> np.ndarray:
        """Compute the sum of 1s for all 64 bit positions in an array of uint64 values."""
        bit_sums = np.zeros((bits, arr.shape[-1]), dtype=np.float64)
        for i in range(bits):
            bit_sums[i] = np.sum((arr >> i) & 1, dtype=nptype, axis=0)
        return bit_sums

    def compute_bitwise_cooc(tp, tq):
        """Compute bitwise co-occurrence statistics for a batch."""
        cooc_batch = []
        for k in range(bits):
            shifted = circular_bitshift_right(tq, k)
            a = sum_all_bits(tp & shifted) + 0.1
            b = sum_all_bits(tp & ~shifted) + 0.1
            c = sum_all_bits(~tp & shifted) + 0.1
            d = sum_all_bits(~tp & ~shifted) + 0.1
            cooc_batch.append(np.log((a * d) / (b * c)))
        return np.vstack(cooc_batch)  # Shape: (64, batch_size)

    print("Computing Co-occurrence Scores...")
    all_medians, all_iqrs, all_pvals_ant, all_pvals_syn = [], [], [], []

    for index in range(0, len(pairs), batch_size):
        print(f"Processing Batch {index}-{min(index+batch_size,len(pairs))}")
        pair_batch = pairs[index: index + batch_size]
        tp = sim[:, pair_batch[:, 0]]
        tq = sim[:, pair_batch[:, 1]]

        # Compute bitwise co-occurrence in batches
        batch_cooc = compute_bitwise_cooc(tp,tq).T

        # Add small noise
        noised_batch_cooc = batch_cooc + np.random.normal(0, 1e-9, size=batch_cooc.shape)

        def compute_kde_stats(i):
            kde = gaussian_kde(noised_batch_cooc[i])
            cdf_func = lambda x: kde.integrate_box_1d(-np.inf, x)
            kde_pval_ant = cdf_func(obspairs[index + i])  # P(X ≤ observed)
            kde_pval_syn = 1 - kde_pval_ant  # P(X ≥ observed)
            med = np.median(batch_cooc[i])
            q75, q25 = np.percentile(batch_cooc[i], [75, 25])
            iqr = q75 - q25
            return kde_pval_ant, kde_pval_syn, med, iqr

        print("Computing P-Values for Batch...")
        results = Parallel(n_jobs=-1, verbose=10, batch_size=25)(
            delayed(compute_kde_stats)(i) for i in range(len(pair_batch))
        )

        kde_pvals_ant, kde_pvals_syn, medians, iqrs = map(np.array, zip(*results))

        all_medians.extend(medians.tolist())
        all_iqrs.extend(iqrs.tolist())
        all_pvals_ant.extend(kde_pvals_ant.tolist())
        all_pvals_syn.extend(kde_pvals_syn.tolist())

        # Store results efficiently
        res["pair"].extend([tuple(p) for p in pair_batch])
        res["first"].extend(pair_batch[:, 0].tolist())
        res["second"].extend(pair_batch[:, 1].tolist())
        res["num_pair_trials"].extend([sim.shape[1] ** 2] * len(pair_batch))
        res["o_occ"].extend(obspairs[index: index + len(pair_batch)].tolist())
        res["e_occ"].extend(medians.tolist())
        print(f"Completed Batch {index}-{min(index+batch_size,len(pairs))}")

    # Compute p-values, directionality, significance, and effect sizes
    pvals = np.minimum(all_pvals_syn, all_pvals_ant)
    directions = np.where(np.array(all_pvals_ant) < np.array(all_pvals_syn), -1, 1)
    significants = pvals < 0.05
    effects = (np.array(all_medians) - obspairs) / np.maximum(np.array(all_iqrs), 1)

    # Final results storage
    res["p-value_ant"], res["p-value_syn"], res["p-value"] = all_pvals_ant, all_pvals_syn, pvals.tolist()
    res["direction"], res["significant"] = directions.tolist(), significants.tolist()
    res["median"], res["iqr"], res["effect size"] = all_medians, all_iqrs, effects.tolist()

    return pd.DataFrame.from_dict(res)



def compile_results_KDE_bit_async(
    self, 
    sim: np.ndarray, 
    obspairs: List[float] = [], 
    batch_size: int = 1000,
    bits = 128,
    nptype = np.uint128
) -> pd.DataFrame:
    """
    Compile KDE results asynchronously using parallel batch processing, optimizing `sim` memory handling.

    :param sim: Large NumPy array storing simulation data.
    :param obspairs: Observed pairs statistics.
    :param batch_size: Size of each batch for processing.
    :return: DataFrame with compiled results.
    """
    # Use Joblib Memory to avoid redundant copies
    memory = Memory(location=None, verbose=0)  # No disk caching, just memory optimization
    res: Dict[str, List] = {
        "pair": [], "first": [], "second": [], "num_pair_trials": [], 
        "direction": [], "p-value_ant": [], "p-value_syn": [], "p-value": [], 
        "significant": [], "e_occ": [], "o_occ": [], "median": [], "iqr": [], 
        "effect size": []
    }

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

    # Convert sim to read-only memory-mapped array to reduce memory duplication
    sim = np.asarray(sim, order="C")  # Ensure contiguous memory
    sim.setflags(write=False)  # Set as read-only to avoid unintended copies

    def circular_bitshift_right(arr: np.ndarray, k: int) -> np.ndarray:
        """Perform a circular right bit shift on all np.uint64 entries in an array."""
        k %= bits
        return np.bitwise_or(np.right_shift(arr, k), np.left_shift(arr, bits - k))
    
    def sum_all_bits(arr: np.ndarray) -> np.ndarray:
        """Compute the sum of 1s for all 64 bit positions in an array of uint64 values."""
        bit_sums = np.zeros((bits, arr.shape[-1]), dtype=np.float64)
        for i in range(bits):
            bit_sums[i] = np.sum((arr >> i) & 1, axis=0, dtype=nptype)
        return bit_sums

    def compute_bitwise_cooc(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        """Compute bitwise co-occurrence statistics for a batch."""
        cooc_batch = []
        for k in range(bits):
            shifted = circular_bitshift_right(tq, k)
            a = sum_all_bits(tp & shifted) + 0.1
            b = sum_all_bits(tp & ~shifted) + 0.1
            c = sum_all_bits(~tp & shifted) + 0.1
            d = sum_all_bits(~tp & ~shifted) + 0.1
            cooc_batch.append(np.log((a * d) / (b * c)))
        return np.vstack(cooc_batch).T  # Shape: (batch_size, bits)

    def compute_kde_stats(observed_value: float, simulated_values: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute KDE statistics for a single pair."""
        kde = gaussian_kde(simulated_values)
        cdf_func = kde.integrate_box_1d
        kde_pval_ant = cdf_func(-np.inf, observed_value)  # P(X ≤ observed)
        kde_pval_syn = 1 - kde_pval_ant  # P(X ≥ observed)
        med = np.median(simulated_values)
        q75, q25 = np.percentile(simulated_values, [75, 25])
        iqr = q75 - q25
        return kde_pval_ant, kde_pval_syn, med, iqr

    def process_batch(index: int, sim_readonly: np.ndarray) -> Dict[str, List]:
        """Process a single batch of data, ensuring memory-efficient sim access."""
        pair_batch = pairs[index: index + batch_size]
        tp = sim_readonly[:, pair_batch[:, 0]]
        tq = sim_readonly[:, pair_batch[:, 1]]

        # Compute bitwise co-occurrence in batches
        batch_cooc = compute_bitwise_cooc(tp, tq)

        # Add small noise
        noised_batch_cooc = batch_cooc + np.random.normal(0, 1e-9, size=batch_cooc.shape)

        # Compute KDE statistics in parallel
        results = Parallel(n_jobs=-1, verbose=0, batch_size=25)(
            delayed(compute_kde_stats)(obspairs[index + i], noised_batch_cooc[i])
            for i in range(len(pair_batch))
        )

        kde_pvals_ant, kde_pvals_syn, medians, iqrs = map(np.array, zip(*results))

        batch_res = {
            "pair": [tuple(p) for p in pair_batch],
            "first": pair_batch[:, 0].tolist(),
            "second": pair_batch[:, 1].tolist(),
            "num_pair_trials": [sim_readonly.shape[1] ** 2] * len(pair_batch),
            "o_occ": obspairs[index: index + len(pair_batch)].tolist(),
            "e_occ": medians.tolist(),
            "median": medians.tolist(),
            "iqr": iqrs.tolist(),
            "p-value_ant": kde_pvals_ant.tolist(),
            "p-value_syn": kde_pvals_syn.tolist(),
            "p-value": np.minimum(kde_pvals_syn, kde_pvals_ant).tolist(),
            "direction": np.where(kde_pvals_ant < kde_pvals_syn, -1, 1).tolist(),
            "significant": (np.minimum(kde_pvals_syn, kde_pvals_ant) < 0.05).tolist(),
            "effect size": ((medians - obspairs[index: index + len(pair_batch)]) / np.maximum(iqrs, 1)).tolist(),
        }

        return batch_res

    num_pairs = len(pairs)
    batch_indices = range(0, num_pairs, batch_size)

    print("Processing Batches")

    # Run batches in parallel, passing a read-only copy of sim
    batch_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_batch)(index, sim) for index in batch_indices
    )

    print("Aggregating Results...")

    # Merge batch results
    for batch_res in batch_results:
        for key in res.keys():
            res[key].extend(batch_res[key])

    return pd.DataFrame.from_dict(res)
