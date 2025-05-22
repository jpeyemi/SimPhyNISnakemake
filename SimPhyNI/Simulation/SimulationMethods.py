from typing import List
import numpy as np
import pandas as pd
from ete3 import Tree
from joblib import Parallel, delayed, parallel_backend, Memory
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from typing import List, Tuple, Set, Dict

def simulate_events(self):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml
    Sums gains and losses to a total number of events then alocates each event to a 
    branch on the tree with probability propotional to the length of the branch
    For each trait only simulates on branches beyond a certain distance from the root, 
    This thereshold is chosen as the first branch where a trait arises in the ancestral trait reconsrtuction
    """
    from tree_simulator import TreeSimulator
    assert(type(self) == TreeSimulator)
    # Preprocess and setup
    all_nodes = list(self.tree.traverse()) # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    node_map_r = {ind: node for ind, node in enumerate(all_nodes)}
    total_events = self.gains #+ self.losses
    losses = self.losses

    # bl = sum([i.dist for i in self.tree.traverse()]) #type: ignore
    # div = self.subsize / bl
    # total_events = np.floor(total_events/div)

    # Distance calculations
    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse(): # type: ignore
        if node in node_dists: continue
        node_dists[node] = node_dists[node.up] + node.dist
    node_df = pd.DataFrame.from_dict(node_dists, orient = 'index', columns = ['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    def get_nodes(dist):
        nodes = node_df[node_df['total_dist'] >= dist]
        # nodes.loc[:,'used_dist'] = np.minimum(nodes['total_dist'] - dist, nodes['dist'])
        nodes = nodes.assign(used_dist=np.minimum(nodes['total_dist'] - dist, nodes['dist']))
        bl = nodes['used_dist'].sum()
        p = nodes['used_dist'] / bl
        if any(p.isna()):
            return (None, None)
        node_index = [node_map[node] for node in nodes.index]
        return node_index, p

    def sim_events(trait, total_events, ap):
        a, p = ap
        if not a: return (None,trait)
        event_locs = np.apply_along_axis(lambda x: np.random.choice(a, size=x.shape[0], replace=False, p=p),
                                          arr=np.zeros((int(np.ceil(total_events[trait])), self.NUM_TRIALS)), 
                                          axis=0)
        return (event_locs, trait)
    
    if self.parallel:
        # Simulating events
        branch_probabilities = [get_nodes(self.dists[trait]) for trait, events in enumerate(total_events)]
        all_event_locs = Parallel(n_jobs=-1, batch_size=100)(delayed(sim_events)(trait, total_events, branch_probabilities[trait]) for trait in range(num_traits)) # type: ignore
        for event_locs, trait in all_event_locs: # type: ignore
            if event_locs is not None:
                sim[:, trait, :][event_locs.flatten("F"), np.repeat(np.arange(self.NUM_TRIALS), int(np.ceil(total_events[trait])))] = True
    else:
        for trait, events in enumerate(total_events):
            a, p = get_nodes(self.dists[trait])
            if not a:
                print(trait)
                continue
            # event_locs = np.random.choice(a, size=(int(np.ceil(events)), self.NUM_TRIALS), replace=True, p=p)
            # event_locs = np.random.choice(list(node_df.index.map(lambda x: node_map[x])), size=(int(np.ceil(events)), self.NUM_TRIALS), replace=True, p=node_df['dist']/node_df['dist'].sum())
            # event_locs = np.array([np.random.choice(a, size=(int(np.ceil(events))), replace=False, p=p) for _ in range (self.NUM_TRIALS)]).T
            event_locs = np.apply_along_axis(lambda x: np.random.choice(a, size=x.shape[0], replace=False, p=p), arr = np.zeros((int(np.ceil(events)), self.NUM_TRIALS)), axis = 0) # type: ignore
            sim[:,trait,:][event_locs.flatten("F"),np.repeat(np.arange(self.NUM_TRIALS),int(np.ceil(events)))] = True
        

    # Lineage calculations
    for node in all_nodes:
        if node.up == None:
            root = self.root_states > 0
            sim[node_map[node],root,:] = True
            continue
        parent = sim[node_map[node.up],:,:]
        curr = sim[node_map[node],:,:]
        sim[node_map[node],:,:] = np.logical_xor(parent,curr)

    lineages = sim[[node_map[node] for node in self.tree],:,:]
    #gain & not losses
    # plot_histograms(self,sim)

    # Results compilation
    res = compile_results(self,lineages)

    # Trait data calculation
    trait_data = calculate_trait_data(self,lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [pd.DataFrame(sim[:,:,i], index = [node.name for node in all_nodes], columns = [self.mapping[str(j)] for j in range(num_traits)]).loc[self.tree.get_leaf_names()] for i in range(num)] # type: ignore
    
    return res, trait_data, get_simulated_trees(5)

def compile_results(self,sim,obspairs = []):
    """
    compiles simulation results from give filled simulation np array
    : param sim: A filled simulation matrix from a simulation method
    : param obspairs: obsered values for each pair considered, defaults to `self.obspairs`
    """
    if self.kde:
        if self.parallel:
            return compile_results_KDE_async(self,sim,obspairs)
        return compile_results_KDE(self,sim,obspairs)
    elif self.parallel:
        return compile_results_async(self,sim,obspairs)
    else:
        return compile_results_sync(self,sim,obspairs)

def compile_results_KDE(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction", 
                               "p-value_ant", "p-value_syn", "p-value", "significant", 
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    medians = np.zeros(len(pairs))
    iqrs = np.zeros(len(pairs))
    kde_pvals_ant = np.zeros(len(pairs))
    kde_pvals_syn = np.zeros(len(pairs))

    all_cooc = []

    for roll in range(tq.shape[-1]):
        rolled_tq = np.roll(tq, roll, axis=-1)
        rolled_cooc = self.pair_statistic(tp, rolled_tq)
        all_cooc.append(rolled_cooc)
    
    # Stack all trials into one array for KDE
    all_cooc = np.concatenate(all_cooc, axis=-1)

    for i in range(len(pairs)):
        
        noised = all_cooc[i] + np.random.normal(0, 1e-9, size=len(all_cooc[i]))
        # print(noised)
        kde = gaussian_kde(noised,bw_method='silverman')
        cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
        kde_syn = gaussian_kde(-1*noised, bw_method='silverman')
        cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf,-x)
        # Compute KDE-based p-values
        kde_pvals_ant[i] = cdf_func_ant(obspairs[i])  # P(X ≤ observed)
        kde_pvals_syn[i] = cdf_func_syn(obspairs[i])  # P(X ≥ observed)

        # Compute statistics
        medians[i] = np.median(all_cooc[i])
        q75, q25 = np.percentile(all_cooc[i], [75, 25])
        iqrs[i] = q75 - q25

        # if i % 1 == 0:
        #     disc_pvals_ant = np.sum(all_cooc[i] <= obspairs[i]) / len(all_cooc[i])
        #     disc_pvals_syn = np.sum(all_cooc[i] >= obspairs[i]) / len(all_cooc[i])
        #     plt.rcParams.update({'font.size': 14})
        #     # --------- FIRST PLOT: Histogram with KDE ----------
        #     plt.figure(figsize=(5, 4))

        #     # Plot histogram with KDE
        #     ax = sns.histplot(noised, bins=50, kde=True, stat="density", color="cornflowerblue", alpha=0.5)
        #     sns.kdeplot(noised, color="orange", alpha=0.8)

        #     # plt.legend()
        #     plt.title(f"Null Distribution for Pair {pairs[i]}")
        #     plt.xlabel("Association Score")
        #     plt.ylabel("Probability Density")

        #     plt.show()
        if i % 1 == 0:
            disc_pvals_ant = np.sum(all_cooc[i] <= obspairs[i]) / len(all_cooc[i])
            disc_pvals_syn = np.sum(all_cooc[i] >= obspairs[i]) / len(all_cooc[i])
            plt.rcParams.update({'font.size': 14})
            # --------- FIRST PLOT: Histogram with KDE ----------
            plt.figure(figsize=(4, 3))

            # Plot histogram with KDE
            ax = sns.histplot(noised, bins=50, kde=True, stat="density", color="cornflowerblue", alpha=0.5)
            sns.kdeplot(noised, color="orange", alpha=0.8)

            # Add observed value line
            plt.axvline(obspairs[i], color="black", linestyle="dashed", label=f"Observed: {obspairs[i]:.2f}")

            # plt.legend()
            # plt.title(f"Null Distribution for Pair {pairs[i]}")
            plt.xlabel("Association Score")
            plt.ylabel("Probability Density")
            if i == 2:
                plt.savefig('fig.svg', format='svg')
            plt.show()

            # --------- SECOND PLOT: Text Summary ----------
            plt.figure(figsize=(4, 2))

            # Create a text box with the relevant statistics
            textstr = (
                f'KDE P(X≤obs): {kde_pvals_ant[i]:.3e}\n'
                f'KDE P(X≥obs): {kde_pvals_syn[i]:.3e}\n'
                f'Disc P(X≤obs): {disc_pvals_ant:.3e}\n'
                f'Disc P(X≥obs): {disc_pvals_syn:.3e}'
            )

            # Add the text box to the plot
            plt.gca().axis('off')  # Turn off axes
            plt.text(0.5, 0.5, textstr,
                    fontsize=12,
                    ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

            plt.title(f"Statistical Summary for Pair {pairs[i]}")
            plt.show()


    # Compute p-values, directionality, significance, and effect sizes
    pvals = np.minimum(kde_pvals_syn, kde_pvals_ant)
    directions = np.where(kde_pvals_ant < kde_pvals_syn, -1, 1)
    significants = pvals < 0.05
    effects = (medians - obspairs) / np.maximum(iqrs, 1)

    # Store results
    res['pair'] = [(p, q) for p, q in pairs]
    res['first'] = pairs[:, 0].tolist()
    res['second'] = pairs[:, 1].tolist()
    res['num_pair_trials'] = [tq.shape[-1] ** 2] * len(pairs)
    res['o_occ'] = obspairs.tolist()
    res['e_occ'] = medians.tolist()
    res['p-value_ant'] = kde_pvals_ant.tolist()
    res['p-value_syn'] = kde_pvals_syn.tolist()
    res['p-value'] = pvals.tolist()
    res['direction'] = directions.tolist()
    res['significant'] = significants.tolist()
    res['median'] = medians.tolist()
    res['iqr'] = iqrs.tolist()
    res['effect size'] = effects.tolist()

    return pd.DataFrame.from_dict(res)

def compile_results_KDE_async(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction", 
                               "p-value_ant", "p-value_syn", "p-value", "significant", 
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    # Generate rolled co-occurrence matrices in parallel
    def compute_rolled_cooc(roll):
        rolled_tq = np.roll(tq, roll, axis=-1)
        return self.pair_statistic(tp, rolled_tq)

    all_cooc = Parallel(n_jobs=-1,verbose=10, batch_size= 10)(delayed(compute_rolled_cooc)(roll) for roll in range(tq.shape[-1]))
    all_cooc = np.vstack(all_cooc)  # Efficient stacking

    def compute_pair_stats(i):
        noised = all_cooc[i] + np.random.normal(0, 1e-9, size=len(all_cooc[i]))

        kde = gaussian_kde(noised,bw_method='silverman')
        cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
        kde_syn = gaussian_kde(-1*noised, bw_method='silverman')
        cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf,-x)

        kde_pval_ant = cdf_func_ant(obspairs[i])
        kde_pval_syn = cdf_func_syn(obspairs[i])
        med = np.median(all_cooc[i])
        q75, q25 = np.percentile(all_cooc[i], [75, 25])
        iqr = q75 - q25

        # Visualization every 100 pairs (optional)
        if i % 100 == 0:
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(noised, bins=50, kde=False, stat="density", color="blue", alpha=0.5)
            plt.axvline(obspairs[i], color="black", linestyle="dashed", label=f"Observed: {obspairs[i]:.2f}")
            plt.text(obspairs[i], ax.get_ylim()[1], 
                    f'KDE P(X≤obs): {kde_pval_ant:.3e}\n'
                    f'KDE P(X≥obs): {kde_pval_syn:.3e}',
                    fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7))
            plt.legend()
            plt.title(f"Histogram for Pair {i}")
            
            plt.figure(figsize=(6, 4))
            sns.kdeplot(noised, color="red", alpha=0.5)
            plt.axvline(obspairs[i], color="black", linestyle="dashed", label=f"Observed: {obspairs[i]:.2f}")
            plt.legend()
            plt.title(f"KDE for Pair {i}")

        return (i, kde_pval_ant, kde_pval_syn, med, iqr)

    # Run computations in parallel
    results = Parallel(n_jobs=-1, batch_size=25, return_as='generator',verbose=10)(delayed(compute_pair_stats)(i) for i in range(len(pairs)))

    # Convert results into structured output
    for i, kde_pval_ant, kde_pval_syn, med, iqr in results:
        pval = min(kde_pval_syn, kde_pval_ant)
        direction = -1 if kde_pval_ant < kde_pval_syn else 1
        significant = pval < 0.05
        effect_size = (med - obspairs[i]) / max(iqr, 1)

        res["pair"].append(tuple(pairs[i]))
        res["first"].append(pairs[i, 0])
        res["second"].append(pairs[i, 1])
        res["num_pair_trials"].append(tq.shape[-1] ** 2)
        res["o_occ"].append(obspairs[i])
        res["e_occ"].append(med)
        res["p-value_ant"].append(kde_pval_ant)
        res["p-value_syn"].append(kde_pval_syn)
        res["p-value"].append(pval)
        res["direction"].append(direction)
        res["significant"].append(significant)
        res["median"].append(med)
        res["iqr"].append(iqr)
        res["effect size"].append(effect_size)

    return pd.DataFrame.from_dict(res)


def compile_results_sync(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials","direction", "p-value_ant", "p-value_syn", "p-value", "significant", "e_occ", "o_occ", "median", "iqr", "effect size"]}
    
    # Gather the pairs from the object
    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)
    # Extracting the relevant slices for pairs
    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    # Initialize arrays to store results
    syn = np.zeros(len(pairs))
    ant = np.zeros(len(pairs))
    means = np.zeros(len(pairs))
    medians = np.zeros(len(pairs))
    iqrs = np.zeros(len(pairs))

    for roll in range(tq.shape[-1]):
        rolled_tq = np.roll(tq, roll, axis=-1)  # Roll along the last axis
        rolled_cooc = self.pair_statistic(tp, rolled_tq)

        syn += np.sum(rolled_cooc >= obspairs[:, np.newaxis], axis=-1)
        ant += np.sum(rolled_cooc <= obspairs[:, np.newaxis], axis=-1)

        # Accumulate statistics for means, medians, and IQRs
        means += np.mean(rolled_cooc, axis=-1)
        medians += np.median(rolled_cooc, axis=-1)
        q75, q25 = np.percentile(rolled_cooc, [75, 25], axis=-1)
        iqrs += (q75 - q25)

    sim_trials = tq.shape[-1] ** 2

    # Finalize the statistics by dividing by the number of rolls
    means /= tq.shape[-1]
    medians /= tq.shape[-1]
    iqrs /= tq.shape[-1]

    pvals_ant = ant / sim_trials
    pvals_syn = syn / sim_trials
    pvals = np.minimum(pvals_syn, pvals_ant)
    directions = np.where(pvals_ant < pvals_syn, -1, 1)
    significants = pvals < 0.05
    effects = (medians - obspairs) / np.maximum(iqrs, 1)

    res['pair'] = [(p, q) for p, q in pairs]
    res['first'] = pairs[:, 0].tolist()
    res['second'] = pairs[:, 1].tolist()
    res['num_pair_trials'] = [sim_trials] * len(pairs)
    res['o_occ'] = obspairs.tolist()
    res['e_occ'] = means.tolist()
    res['p-value_ant'] = pvals_ant.tolist()
    res['p-value_syn'] = pvals_syn.tolist()
    res['p-value'] = pvals.tolist()
    res['direction'] = directions.tolist()
    res['significant'] = significants.tolist()
    res['median'] = medians.tolist()
    res['iqr'] = iqrs.tolist()
    res['effect size'] = effects.tolist()

    return pd.DataFrame.from_dict(res)

    # Populate the result dictionary
    # for i, (p, q) in enumerate(pairs):
    #     update_result_dict2(res, p, q, sim_trials, obspairs[i], syn[i], ant[i], means[i], medians[i], iqrs[i])

    # return pd.DataFrame.from_dict(res)

def compile_results_async2(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction", "p-value_ant", "p-value_syn", "p-value", "significant", "e_occ", "o_occ", "median", "iqr", "effect size"]}
    
    # Gather the pairs from the object
    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)
    # Extracting the relevant slices for pairs
    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    # Helper function to process each roll
    def process_roll(roll):
        rolled_tq = np.roll(tq, roll, axis=-1)  # Roll along the last axis
        rolled_cooc = self.pair_statistic(tp, rolled_tq)

        syn = np.sum(rolled_cooc >= obspairs[:, np.newaxis], axis=-1)
        ant = np.sum(rolled_cooc <= obspairs[:, np.newaxis], axis=-1)
        mean = np.mean(rolled_cooc, axis=-1)
        median = np.median(rolled_cooc, axis=-1)
        q75, q25 = np.percentile(rolled_cooc, [75, 25], axis=-1)
        iqr = (q75 - q25)
        
        return syn, ant, mean, median, iqr

    # Run the process in parallel
    with parallel_backend('loky', n_jobs=-1):
        results = Parallel(batch_size=10, return_as = 'generator')(delayed(process_roll)(roll) for roll in range(tq.shape[-1]))

    # results = Parallel(n_jobs=-1, batch_size=10)(delayed(process_roll)(roll) for roll in range(tq.shape[-1])) #type: ignore
    # Aggregate results
    syn = np.sum([r[0] for r in results], axis=0)  #type: ignore
    ant = np.sum([r[1] for r in results], axis=0) #type: ignore
    means = np.mean([r[2] for r in results], axis=0) #type: ignore
    medians = np.mean([r[3] for r in results], axis=0) #type: ignore
    iqrs = np.mean([r[4] for r in results], axis=0) #type: ignore

    sim_trials = tq.shape[-1] ** 2

    pvals_ant = ant / sim_trials
    pvals_syn = syn / sim_trials
    pvals = np.minimum(pvals_syn, pvals_ant)
    directions = np.where(pvals_ant < pvals_syn, -1, 1)
    significants = pvals < 0.05
    effects = (medians - obspairs) / np.maximum(iqrs, 1)

    res['pair'] = [(p, q) for p, q in pairs]
    res['first'] = pairs[:, 0].tolist()
    res['second'] = pairs[:, 1].tolist()
    res['num_pair_trials'] = [sim_trials] * len(pairs)
    res['o_occ'] = obspairs.tolist()
    res['e_occ'] = means.tolist()
    res['p-value_ant'] = pvals_ant.tolist()
    res['p-value_syn'] = pvals_syn.tolist()
    res['p-value'] = pvals.tolist()
    res['direction'] = directions.tolist()
    res['significant'] = significants.tolist()
    res['median'] = medians.tolist()
    res['iqr'] = iqrs.tolist()
    res['effect size'] = effects.tolist()

    return pd.DataFrame.from_dict(res)

def compile_results_async(self, sim, obspairs = []):

    def process_pair(ind, p, q, obs):
        tp, tq = sim[:, p, :], sim[:, q, :]
        ant, syn = 0,0
        means = []
        medians = []
        iqrs = []
        for roll in range(tq.shape[1]):
            cooc = self.pair_statistic(tp,np.roll(tq,roll, axis = 1))
            syn += np.sum(cooc >= obs)
            ant += np.sum(cooc <= obs)
            means.append(np.mean(cooc))
            medians.append(np.median(cooc))
            q75, q25 = np.percentile(cooc, [75, 25])
            iqr = q75 - q25
            iqrs.append(iqr)

        sim_trials = tq.shape[1] ** 2
        return (p, q, sim_trials, obs, syn, ant, means, medians, iqrs)

    obspairs = obspairs or self.obspairs
    pair_stats = Parallel(n_jobs=-1, batch_size=10, return_as='generator',verbose=10)(delayed(process_pair)(ind, p, q, obs) for ind, ((p, q), obs) in enumerate(zip(self.pairs, obspairs))) # type: ignore

    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials","direction", "p-value_ant", "p-value_syn", "p-value", "significant", "e_occ", "o_occ", "median", "iqr", "effect size"]} # type: ignore
   
    for pair in pair_stats: 
        p, q, sim_trials, obs, syn, ant, means, medians, iqrs = pair # type: ignore
        update_result_dict2(res, p, q, sim_trials, obs, syn, ant, means, medians, iqrs) 
        
    return pd.DataFrame.from_dict(res)

def calculate_trait_data(self, sim, num_traits):
    """
    Formats and returns simulation statistics for each trait in the simulation run of `self`
    """
    sums = (sim > 0).sum(axis=0)
    median_vals = np.median(sums, axis=1)
    q75, q25 = np.percentile(sums, [75, 25], axis=1)
    iqr_vals = q75 - q25

    return pd.DataFrame({
        "trait": np.arange(num_traits),
        "mean": sums.mean(axis=1),
        "std": sums.std(axis=1),
        "iqr": iqr_vals,
        "median": median_vals
    })

def update_result_dict(res: dict[str,list], p, q, sim_trials, obs, cooc, cooccur_bool):
    """
    updates a result dictionary, only to be used within compile results
    """
    res['pair'].append((p, q))
    res['first'].append(p)
    res['second'].append(q)
    res['num_pair_trials'].append(sim_trials)
    res['o_occ'].append(obs)
    res['e_occ'].append(np.mean(cooc[~np.isnan(cooc)]))
    pval_ant = np.sum(cooc <= obs) / sim_trials 
    pval_syn = np.sum(cooc >= obs) / sim_trials 
    res['p-value_ant'].append(pval_ant)
    res['p-value_syn'].append(pval_syn)
    res['p-value'].append(min(pval_syn,pval_ant))
    res['direction'].append(-1 if pval_ant < pval_syn else 1)
    res['significant'].append(res['p-value'][-1] < .05)

def update_result_dict2(res: dict[str,list], p, q, sim_trials, obs, syn, ant, means, medians, iqrs):
    """
    updates a result dictionary, only to be used within compile results
    """
    res['pair'].append((p, q))
    res['first'].append(p)
    res['second'].append(q)
    res['num_pair_trials'].append(sim_trials)
    res['o_occ'].append(obs)
    res['e_occ'].append(np.mean(means))
    pval_ant = ant / sim_trials 
    pval_syn = syn / sim_trials 
    res['p-value_ant'].append(pval_ant)
    res['p-value_syn'].append(pval_syn)
    res['p-value'].append(min(pval_syn,pval_ant))
    res['direction'].append(-1 if pval_ant < pval_syn else 1)
    res['significant'].append(res['p-value'][-1] < .05)
    median = np.mean(medians)
    iqr = np.mean(iqrs)
    res['median'].append(median)
    res['iqr'].append(iqr)
    res['effect size'].append((median - obs)/max(iqr,1))


def plot_histograms(self,sim):
    """
    Generates histograms of the counts for each trait where each data value is the count for one trial.
    
    Parameters:
    sim (numpy array): A 3D numpy array with dimensions (num_nodes, num_traits, num_trials) containing
                       the simulation results.
    """
    num_nodes, num_traits, num_trials = sim.shape
    
    # Count the number of events for each trait across all nodes for each trial
    trait_counts = np.sum(sim, axis=0)
    
    # Plot a histogram for each trait
    for trait in range(num_traits):
        plt.figure()
        plt.hist(trait_counts[trait], bins=20, edgecolor='black')
        plt.title(f'Histogram of Event Counts for Trait {self.mapping[str(trait)]}')
        plt.xlabel('Number of Events')
        plt.ylabel('Frequency')
        plt.show()

def mask_simulate_subtree(self):
    return mask_simulate(self, method = "subtree")

def mask_simulate_dropout(self):
    return mask_simulate(self, method = "dropout")

def mask_simulate(self, method = "dropout"):
    from tree_simulator import TreeSimulator
    assert(type(self) == TreeSimulator)

    all_nodes = list(self.tree.traverse()) # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    total_events = self.gains + self.losses

    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse(): # type: ignore
        if node in node_dists: continue
        node_dists[node] = node_dists[node.up] + node.dist
    node_df = pd.DataFrame.from_dict(node_dists, orient = 'index', columns = ['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    def get_nodes(dist):
        nodes = node_df[node_df['total_dist'] >= dist]
        bl = nodes['dist'].sum()
        p = nodes['dist'] / bl
        node_index = [node_map[node] for node in nodes.index]
        return node_index, p

    for trait, events in enumerate(total_events):
        a, p = get_nodes(self.dists[trait])
        if not a:
            print(trait)
            continue
        event_locs = np.random.choice(a, size=(int(np.ceil(events)), self.NUM_TRIALS), replace=True, p=p)
        sim[:,trait,:][event_locs.flatten("F"),np.repeat(np.arange(self.NUM_TRIALS),events)] = True

    for node in all_nodes:
        if node.up == None:
            continue
        parent = sim[node_map[node.up],:,:]
        curr = sim[node_map[node],:,:]
        sim[node_map[node],:,:] = np.logical_xor(parent,curr)


    mask_level = 1
    def range_from_level(level: int) -> tuple[int,int]:
        num_entries = 2**level
        start_entry = sum(2**i for i in range(level)) or 0
        return (start_entry, start_entry + num_entries)
    masks_init = [all_nodes[i] for i in range(*range_from_level(mask_level))]

    def isValidMask(node: Tree,level: int):
        return 1.1 * len(self.tree)/(2**level) >= len(node.get_leaves()) 
    
    def make_level_mask(nodes: list[Tree], level: int) -> list:
        res = []
        for i in nodes:
            if isValidMask(i,level):
                if(not i.is_leaf()): res.append(i)
            else:
                res += make_level_mask(i.get_children()[:1], level)
                res += make_level_mask(i.get_children()[1:], level)
        return res
    masks = make_level_mask(masks_init, mask_level)
    print(len(masks))
    print(masks)


    masked_res = []
    for mask in masks:
        to_prune = set(mask.get_descendants())
        if method == "dropout":
            lineages = sim[[node_map[node] for node in self.tree if node not in to_prune],:,:]
            masked_df = self.obsdf_modified.drop([n.name for n in to_prune if n.name in self.obsdf_modified])
        elif method == "subtree":
            lineages = sim[[node_map[node] for node in self.tree if node in to_prune],:,:]
            masked_df = self.obsdf_modified.drop([n.name for n in self.tree.get_leaves() if n not in to_prune and n.name in self.obsdf_modified])
        masked_pairs, masked_obs_pairs_raw = self._get_pair_data(masked_df, masked_df , prevalence_threshold = 0)
        all_pairs = set(self.pairs)
        masked_obs_pairs = [j[1] for j in zip(masked_pairs,masked_obs_pairs_raw) if j[0] in all_pairs]
        masked_res.append(compile_results(self,lineages,obspairs= masked_obs_pairs))
    res = pd.DataFrame()
    res['pair'] = masked_res[0]['pair']
    res['first'] = masked_res[0]['first']
    res['second'] = masked_res[0]['second']
    res['p-value'] = np.sum([np.array(i['p-value']) for i in masked_res], axis= 0)/len(masked_res)
    res['significant'] = res['p-value'] < .05
    res['votes'] = np.sum([np.array(i['significant']) for i in masked_res], axis = 0)/len(masked_res)
    res['num_masks'] = [len(masked_res) for i in range(len(res))]

    lineages = sim[[node_map[node] for node in self.tree],:,:]

    # Trait data calculation
    trait_data = calculate_trait_data(self,lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [pd.DataFrame(sim[:,:,i], index = [node.name for node in all_nodes], columns = [self.mapping[str(j)] for j in range(num_traits)]).loc[self.tree.get_leaf_names()] for i in range(num)]
    
    return res, trait_data, get_simulated_trees(5)

def simulate_glrates(self):
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
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
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
    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    for node in self.tree.traverse(): # type: ignore
        if node.up == None:
            root = self.root_states > 0
            sim[node_map[node],root,:] = True
            continue
        
        parent = sim[node_map[node.up], :, :]
        node_dist_multiplier = node.dist * self.MULTIPLIER
        node_total_dist = node_dists[node]  # Total distance from the root to the current node

        # Zero out gain and loss rates for traits where the node's distance is less than the specified threshold
        applicable_traits_gains = node_total_dist >= self.dists
        applicable_traits_losses = node_total_dist >= self.loss_dists
        gain_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)
        loss_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)

        # Simulate gain and loss events
        gain_events[applicable_traits_gains] = np.random.binomial(node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis], (applicable_traits_gains.sum(), self.NUM_TRIALS)) > 0
        loss_events[applicable_traits_losses] = np.random.binomial(node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis], (applicable_traits_losses.sum(), self.NUM_TRIALS)) > 0

        gain_mask = np.logical_not(np.logical_xor(parent,gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent,loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node],:,:] = parent.copy()
        sim[node_map[node],:,:][gain_events] = True
        sim[node_map[node],:,:][loss_events] = False

    lineages = sim[[node_map[node] for node in self.tree], :, :]

    # Results compilation
    res = compile_results(self, lineages)

    # Trait data calculation
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [pd.DataFrame(sim[:, :, i], index=[node.name for node in all_nodes], columns=[self.mapping[str(j)] for j in range(num_traits)]).loc[self.tree.get_leaf_names()] for i in range(num)]
    
    return res, trait_data, get_simulated_trees(5)

def synth_mutual_4state_nosim(dir, t, mod):
    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []


    with open('../kde_model.pkl', 'rb') as f:
        kde = pickle.load(f)

    samples = kde.resample(2)
    gains = 10**samples[0]
    losses = 10**samples[1]
    root_state = np.rint(samples[2]).astype(int)
    root_state = int(f'{root_state[1]:0b}{root_state[0]:0b}',2)



    # Normalize gain/loss rates by total branch length
    # bl = sum(sorted([i.dist for i in t.traverse()])[:-3])
    # gain_rates = gains * 6.019839999999989 / bl# changing branch length units form ecoli tree to current tree
    # loss_rates = losses * 6.019839999999989 / bl#/ (bl * MULTIPLIER)

    # median branch length scaling:
    bl = np.median(np.array([i.dist for i in t.traverse()]))
    gain_rates = gains * 0.00103 / bl
    loss_rates = losses * 0.00103 / bl

    # Define gain/loss modifiers

    sim = np.zeros((num_nodes, 1), dtype=int)
    sim[node_map[t]] = root_state

    for node in t.traverse(strategy='preorder'):
        if node.up is None:
            continue

        parent_state = sim[node_map[node.up], 0]
        curr_rates = rates[parent_state]

        dist = node.dist
        total_rate = sum(curr_rates.values())
        prob_change = 1 - np.exp(-total_rate * dist)

        if np.random.rand() < prob_change and total_rate > 0:
            next_states = list(curr_rates.keys())
            probs = np.array([curr_rates[s] for s in next_states]) / total_rate
            new_state = np.random.choice(next_states, p=probs)
        else:
            new_state = parent_state

        sim[node_map[node], 0] = new_state
        if node.is_leaf():
            leaves.append(node)

    # Decode states to binary traits
    trait1 = (sim[:, 0] & 1).astype(np.int16)
    trait2 = ((sim[:, 0] & 2) >> 1).astype(np.int16)

    lineages = np.stack([trait1[[node_map[l] for l in leaves]],
                         trait2[[node_map[l] for l in leaves]]], axis=1)
    prev = lineages.mean(axis=0)

    gains_out = gains.tolist()
    losses_out = losses.tolist()
    dists = np.zeros(2)

    return lineages, prev, gains_out, losses_out, dists, leaves

def simulate_glrates_ctmp_vectorized(self): 
    #Not funtional
    """
    Vectorized CTMP simulation of trait evolution with gain/loss rates on a tree.
    Gains and losses occur as a Poisson process. Multiple events per branch possible.
    """
    from tree_simulator import TreeSimulator
    assert isinstance(self, TreeSimulator)

    all_nodes = list(self.tree.traverse())
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: idx for idx, node in enumerate(all_nodes)}

    # Normalize rates
    gain_rates = np.nan_to_num(self.gains / self.gain_subsize)
    loss_rates = np.nan_to_num(self.losses / self.loss_subsize)

    # Root initialization
    sim[node_map[self.tree]] = (self.root_states[:, None] > 0)

    # Compute total distance from root for each node
    node_dists = {self.tree: 0}
    for node in self.tree.traverse():  # type: ignore
        if node not in node_dists:
            node_dists[node] = node_dists[node.up] + node.dist
    total_dists = np.array([node_dists[node] for node in all_nodes])
    node_dists_arr = np.array([node.dist for node in all_nodes])

    # Vectorized simulation
    for node in self.tree.traverse():  # type: ignore
        if node.is_root():
            continue

        parent_idx = node_map[node.up]
        curr_idx = node_map[node]
        dist = node.dist
        curr_total_dist = total_dists[curr_idx]

        # Broadcast parent state
        sim[curr_idx] = sim[parent_idx]

        # Vectorized rate calculation
        gain_mask = curr_total_dist > self.dists
        loss_mask = curr_total_dist > self.loss_dists
        g_rates = np.where(gain_mask, gain_rates, 0.0)
        l_rates = np.where(loss_mask, loss_rates, 0.0)

        # For each trait, get current state and simulate transitions in bulk
        parent_states = sim[parent_idx]  # shape: (num_traits, num_trials)
        flat_states = parent_states.reshape(num_traits * self.NUM_TRIALS)
        g_repeat = np.repeat(g_rates, self.NUM_TRIALS)
        l_repeat = np.repeat(l_rates, self.NUM_TRIALS)

        # Initial time
        t = np.zeros_like(flat_states, dtype=float)
        state = flat_states.copy()

        while np.any(t < dist):
            current_rates = np.where(state, l_repeat, g_repeat)
            current_rates[current_rates == 0] = np.inf  # Avoid zero division
            waits = np.random.exponential(1.0 / current_rates)
            waits[current_rates == 0] = np.inf
            t_next = t + waits
            flips = t_next <= dist
            state[flips] = ~state[flips]
            t = t_next

        # Assign result
        sim[curr_idx] = state.reshape(num_traits, self.NUM_TRIALS)

    lineages = sim[[node_map[node] for node in self.tree], :, :]
    res = compile_results(self, lineages)
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [
            pd.DataFrame(sim[:, :, i], index=[n.name for n in all_nodes], columns=[self.mapping[str(j)] for j in range(num_traits)])
              .loc[self.tree.get_leaf_names()]
            for i in range(num)
        ]

    return res, trait_data, get_simulated_trees(5)

def simulate_glrates_ctmp(self):
    """
    CTMP simulation of trait evolution with gain/loss rates on a tree.
    Gains and losses occur as a Poisson process, governed by rates inferred from pastML.
    Multiple events per branch possible.
    """
    from tree_simulator import TreeSimulator
    assert isinstance(self, TreeSimulator)

    all_nodes = list(self.tree.traverse())
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: idx for idx, node in enumerate(all_nodes)}

    gain_rates = self.gains / (self.gain_subsize)
    loss_rates = self.losses / (self.loss_subsize)
    gain_rates = np.nan_to_num(gain_rates)
    loss_rates = np.nan_to_num(loss_rates)

    # root states initialization
    sim[node_map[self.tree], :, :] = self.root_states[:, np.newaxis] > 0

    # Distance calculations
    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse(): # type: ignore
        if node in node_dists: continue
        node_dists[node] = node_dists[node.up] + node.dist
    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    for node in self.tree.traverse(): # type: ignore
        if node.is_root():
            continue

        parent_idx = node_map[node.up] # type: ignore
        curr_idx = node_map[node]
        dist = node.dist

        # Start from parent state
        sim[curr_idx] = sim[parent_idx]

        # Simulate each trait and trial independently
        for trait in range(num_traits):
            g = gain_rates[trait] if node_dists[node] > self.dists[trait] else 0
            l = loss_rates[trait] if node_dists[node] > self.loss_dists[trait] else 0
            for trial in range(self.NUM_TRIALS):
                state = sim[parent_idx, trait, trial]
                t = 0.0

                while t < dist:
                    rate = g if not state else l
                    if rate == 0:
                        break
                    wait_time = np.random.exponential(1 / rate)
                    if t + wait_time > dist:
                        break
                    t += wait_time
                    state = not state  # flip state

                sim[curr_idx, trait, trial] = state
 
    lineages = sim[[node_map[node] for node in self.tree], :, :]

    res = compile_results(self, lineages)
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [pd.DataFrame(sim[:, :, i], index=[n.name for n in all_nodes], columns=[self.mapping[str(j)] for j in range(num_traits)]).loc[self.tree.get_leaf_names()] for i in range(num)]

    return res, trait_data, get_simulated_trees(5)


def simulate_glrates_nodist(self):
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
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    node_map_r = {ind: node for ind, node in enumerate(all_nodes)}
    bl = sum(i.dist for i in self.tree.traverse()) # type: ignore
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
    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    for node in self.tree.traverse(): # type: ignore
        if node.up == None:
            root = self.root_states > 0
            sim[node_map[node],root,:] = True
            continue
        
        parent = sim[node_map[node.up], :, :]
        node_dist_multiplier = node.dist * self.MULTIPLIER
        node_total_dist = node_dists[node]  # Total distance from the root to the current node

        # Zero out gain and loss rates for traits where the node's distance is less than the specified threshold
        applicable_traits = node_total_dist >= np.zeros_like(self.dists)
        gain_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)
        loss_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)

        # Simulate gain and loss events
        gain_events[applicable_traits] = np.random.binomial(node_dist_multiplier, gain_rates[applicable_traits, np.newaxis], (applicable_traits.sum(), self.NUM_TRIALS)) > 0
        loss_events[applicable_traits] = np.random.binomial(node_dist_multiplier, loss_rates[applicable_traits, np.newaxis], (applicable_traits.sum(), self.NUM_TRIALS)) > 0

        gain_mask = np.logical_not(np.logical_xor(parent,gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent,loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node],:,:] = parent.copy()
        sim[node_map[node],:,:][gain_events] = True
        sim[node_map[node],:,:][loss_events] = False

    lineages = sim[[node_map[node] for node in self.tree], :, :]

    # Results compilation
    res = compile_results(self, lineages)

    # Trait data calculation
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [pd.DataFrame(sim[:, :, i], index=[node.name for node in all_nodes], columns=[self.mapping[str(j)] for j in range(num_traits)]).loc[self.tree.get_leaf_names()] for i in range(num)]
    
    return res, trait_data, get_simulated_trees(5)


def simulate_distnorm(self):
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
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    bl = 2*len(self.tree)-1

    gain_rates = self.gains / bl
    loss_rates = self.losses / bl
    np.nan_to_num(gain_rates, copy = False)
    np.nan_to_num(loss_rates, copy = False)

    for node in self.tree.traverse(): # type: ignore
        if node.up == None:
            prev = self.obsdf_modified.mean()
            high_prev = list(prev[prev >= .5].index.astype(int))
            sim[node_map[node],high_prev,:] = True
            continue
        
        parent = sim[node_map[node.up], :, :]

        gain_events = np.random.binomial(1, gain_rates[:, np.newaxis], (len(gain_rates), self.NUM_TRIALS)) > 0
        loss_events = np.random.binomial(1, loss_rates[:, np.newaxis], (len(loss_rates), self.NUM_TRIALS)) > 0

        gain_mask = np.logical_not(np.logical_xor(parent,gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent,loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node],:,:] = parent.copy()
        sim[node_map[node],:,:][gain_events] = True
        sim[node_map[node],:,:][loss_events] = False

    lineages = sim[[node_map[node] for node in self.tree], :, :]

    # Results compilation
    res = compile_results(self, lineages)

    # Trait data calculation
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [pd.DataFrame(sim[:, :, i], index=[node.name for node in all_nodes], columns=[self.mapping[str(j)] for j in range(num_traits)]).loc[self.tree.get_leaf_names()] for i in range(num)]
    
    return res, trait_data, get_simulated_trees(5)


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
    bits = 64
    nptype = np.uint64
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
        gain_events[applicable_traits_gains] = np.packbits((np.random.binomial(node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis], (applicable_traits_gains.sum(), self.NUM_TRIALS)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()
        loss_events[applicable_traits_losses] = np.packbits((np.random.binomial(node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis], (applicable_traits_losses.sum(), self.NUM_TRIALS)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()

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

def simulate_glrates_bit_norm(self):
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
    bits = 64
    nptype = np.uint64
    sim = np.zeros((num_nodes, num_traits), dtype=nptype)
    self.NUM_TRIALS = bits

    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    gain_rates = self.gains / len(all_nodes)
    loss_rates = self.losses / len(all_nodes)
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
        node_total_dist = node_dists[node]  # Total distance from the root to the current node

        # Zero out gain and loss rates for traits where the node's distance is less than the specified threshold
        applicable_traits_gains = node_total_dist >= self.dists
        applicable_traits_losses = node_total_dist >= self.loss_dists
        gain_events = np.zeros((num_traits), dtype=nptype)
        loss_events = np.zeros((num_traits), dtype=nptype)
        gain_events[applicable_traits_gains] = np.packbits((np.random.binomial(1, gain_rates[applicable_traits_gains, np.newaxis], (applicable_traits_gains.sum(), self.NUM_TRIALS)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()
        loss_events[applicable_traits_losses] = np.packbits((np.random.binomial(1, loss_rates[applicable_traits_losses, np.newaxis], (applicable_traits_losses.sum(), self.NUM_TRIALS)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()

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


def compile_results_KDE_bit(self, sim, obspairs=[], batch_size=1000, bits = 64, nptype = np.uint64):
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
            a = sum_all_bits(tp & shifted) + 0.01
            b = sum_all_bits(tp & ~shifted) + 0.01
            c = sum_all_bits(~tp & shifted) + 0.01
            d = sum_all_bits(~tp & ~shifted) + 0.01
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
            kde = gaussian_kde(noised_batch_cooc[i],bw_method='silverman')
            cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
            kde_syn = gaussian_kde(-1*noised_batch_cooc[i], bw_method='silverman')
            cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf,-x)
            kde_pval_ant = cdf_func_ant(obspairs[index + i])  # P(X ≤ observed)
            kde_pval_syn = cdf_func_syn(obspairs[index + i])  # P(X ≥ observed)
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
    effects = (obspairs - np.array(all_medians)) / np.maximum(np.array(all_iqrs) * 1.349, 1)

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
    bits = 64,
    nptype = np.uint64
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
            a = sum_all_bits(tp & shifted) + 1e-2
            b = sum_all_bits(tp & ~shifted) + 1e-2
            c = sum_all_bits(~tp & shifted) + 1e-2
            d = sum_all_bits(~tp & ~shifted) + 1e-2
            cooc_batch.append(np.log((a * d) / (b * c)))
        return np.vstack(cooc_batch).T  # Shape: (batch_size, bits)

    def compute_kde_stats(observed_value: float, simulated_values: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute KDE statistics for a single pair."""
        kde = gaussian_kde(simulated_values, bw_method='silverman')
        cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
        kde_syn = gaussian_kde(-1*simulated_values, bw_method='silverman')
        cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf,-x)
        kde_pval_ant = cdf_func_ant(observed_value)  # P(X ≤ observed)
        kde_pval_syn = cdf_func_syn(observed_value)  # P(X ≥ observed)
            
        # cdf_func = kde.integrate_box_1d
        # kde_pval_ant = cdf_func(-np.inf, observed_value)  # P(X ≤ observed)
        # kde_pval_syn = 1 - kde_pval_ant  # P(X ≥ observed)
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
        noised_batch_cooc = batch_cooc + np.random.normal(0, 1e-12, size=batch_cooc.shape)

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
            "effect size": ((obspairs[index: index + len(pair_batch)]-medians) / np.maximum(iqrs * 1.349, 1)).tolist(),
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


def simulate_glrates_manual(tree, gains, losses, gain_subsize, loss_subsize, MULTIPLIER, NUM_TRIALS, 
                     dists, loss_dists, root_states):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml.
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch.
    For each trait, only simulates on branches beyond a certain distance from the root.
    This threshold is chosen as the first branch where a trait arises in the ancestral trait reconstruction.
    """
    from tree_simulator import TreeSimulator
    import pandas as pd
    import numpy as np

    # Preprocess and setup
    all_nodes = list(tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(gains)
    sim = np.zeros((num_nodes, num_traits, NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    # Compute total branch length and tree range
    bl = sum(i.dist for i in tree.traverse())  # type: ignore
    tree_range = tree.get_farthest_leaf()[1]
    assert isinstance(tree_range, float)

    # Compute gain and loss modifiers
    gain_mod = np.nan_to_num(tree_range / dists, nan=1)
    gain_mod[gain_mod > 1] = 1
    loss_mod = np.nan_to_num(tree_range / loss_dists, nan=1)
    loss_mod[loss_mod > 1] = 1

    # Compute gain and loss rates
    gain_rates = gains / (gain_subsize * MULTIPLIER)
    loss_rates = losses / (loss_subsize * MULTIPLIER)
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    # Compute node distances
    node_dists = {tree: tree.dist or 0}
    for node in tree.traverse():  # type: ignore
        if node not in node_dists:
            node_dists[node] = node_dists[node.up] + node.dist

    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    # Simulate traits across the tree
    for node in tree.traverse():  # type: ignore
        if node.up is None:
            root = root_states > 0
            sim[node_map[node], root, :] = True
            continue

        parent = sim[node_map[node.up], :, :]
        node_dist_multiplier = node.dist * MULTIPLIER
        node_total_dist = node_dists[node]

        # Apply distance-based thresholds
        applicable_traits_gains = node_total_dist >= dists
        applicable_traits_losses = node_total_dist >= loss_dists
        gain_events = np.zeros((num_traits, NUM_TRIALS), dtype=bool)
        loss_events = np.zeros((num_traits, NUM_TRIALS), dtype=bool)

        # Simulate gain and loss events
        gain_events[applicable_traits_gains] = np.random.binomial(
            node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis], 
            (applicable_traits_gains.sum(), NUM_TRIALS)
        ) > 0
        loss_events[applicable_traits_losses] = np.random.binomial(
            node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis], 
            (applicable_traits_losses.sum(), NUM_TRIALS)
        ) > 0

        # Handle event cancellation
        gain_mask = np.logical_not(np.logical_xor(parent, gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent, loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node], :, :] = parent.copy()
        sim[node_map[node], :, :][gain_events] = True
        sim[node_map[node], :, :][loss_events] = False

    # Extract lineages
    lineages = sim[[node_map[node] for node in tree], :, :]
    
    return lineages

def simulate_glrates_manual_full_tree(tree, gains, losses, gain_subsize, loss_subsize, MULTIPLIER, NUM_TRIALS, 
                     dists, loss_dists, root_states):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml.
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch.
    For each trait, only simulates on branches beyond a certain distance from the root.
    This threshold is chosen as the first branch where a trait arises in the ancestral trait reconstruction.
    """
    from tree_simulator import TreeSimulator
    import pandas as pd
    import numpy as np

    # Preprocess and setup
    all_nodes = list(tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(gains)
    sim = np.zeros((num_nodes, num_traits, NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    # Compute total branch length and tree range
    bl = sum(i.dist for i in tree.traverse())  # type: ignore
    tree_range = tree.get_farthest_leaf()[1]
    assert isinstance(tree_range, float)

    # Compute gain and loss modifiers
    gain_mod = np.nan_to_num(tree_range / dists, nan=1)
    gain_mod[gain_mod > 1] = 1
    loss_mod = np.nan_to_num(tree_range / loss_dists, nan=1)
    loss_mod[loss_mod > 1] = 1

    # Compute gain and loss rates
    gain_rates = gains / (gain_subsize * MULTIPLIER)
    loss_rates = losses / (loss_subsize * MULTIPLIER)
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    # Compute node distances
    node_dists = {tree: tree.dist or 0}
    for node in tree.traverse():  # type: ignore
        if node not in node_dists:
            node_dists[node] = node_dists[node.up] + node.dist

    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    # Simulate traits across the tree
    for node in tree.traverse():  # type: ignore
        if node.up is None:
            root = root_states > 0
            sim[node_map[node], root, :] = True
            continue

        parent = sim[node_map[node.up], :, :]
        node_dist_multiplier = node.dist * MULTIPLIER
        node_total_dist = node_dists[node]

        # Apply distance-based thresholds
        applicable_traits_gains = node_total_dist >= dists
        applicable_traits_losses = node_total_dist >= loss_dists
        gain_events = np.zeros((num_traits, NUM_TRIALS), dtype=bool)
        loss_events = np.zeros((num_traits, NUM_TRIALS), dtype=bool)

        # Simulate gain and loss events
        gain_events[applicable_traits_gains] = np.random.binomial(
            node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis], 
            (applicable_traits_gains.sum(), NUM_TRIALS)
        ) > 0
        loss_events[applicable_traits_losses] = np.random.binomial(
            node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis], 
            (applicable_traits_losses.sum(), NUM_TRIALS)
        ) > 0

        # Handle event cancellation
        gain_mask = np.logical_not(np.logical_xor(parent, gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent, loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node], :, :] = parent.copy()
        sim[node_map[node], :, :][gain_events] = True
        sim[node_map[node], :, :][loss_events] = False

    
    return sim


# from numba import njit, prange
# import numpy as np

# @njit
def simulate_branch_ctmp(state_bits, gain_rates, loss_rates, gain_mask, loss_mask, node_dist):
    num_traits, num_trials = state_bits.shape
    bits = num_trials

    # Initial rates
    rates = np.where(state_bits, loss_rates[:, None], gain_rates[:, None])
    rates = rates * (gain_mask | loss_mask)[:, None]
    rates[rates == 0] = np.inf

    wait_times = np.random.exponential(1 / rates)
    flat_waits = wait_times.flatten()
    flat_indices = np.arange(flat_waits.size)

    order = np.argsort(flat_waits)
    sorted_waits = flat_waits[order]
    sorted_indices = flat_indices[order]

    dist_remaining = node_dist
    event_ptr = 0

    while event_ptr < sorted_waits.size and sorted_waits[event_ptr] < dist_remaining:
        wait_time = sorted_waits[event_ptr]
        dist_remaining -= wait_time

        linear_idx = sorted_indices[event_ptr]
        trait_idx = linear_idx // bits
        trial_idx = linear_idx % bits

        # Flip bit
        state_bits[trait_idx, trial_idx] ^= 1

        # Resample
        new_rate = loss_rates[trait_idx] if state_bits[trait_idx, trial_idx] else gain_rates[trait_idx]
        if (gain_mask[trait_idx] or loss_mask[trait_idx]) and new_rate > 0:
            new_wait = np.random.exponential(1 / new_rate)

            # Insert into sorted array
            insert_pos = event_ptr + 1
            while insert_pos < sorted_waits.size and sorted_waits[insert_pos] < wait_time + new_wait:
                insert_pos += 1

            sorted_waits = np.insert(sorted_waits, insert_pos, wait_time + new_wait)
            sorted_indices = np.insert(sorted_indices, insert_pos, linear_idx)

        event_ptr += 1

    return state_bits


def simulate_glrates_bit_ctmp_numba(self):
    from tree_simulator import TreeSimulator
    assert isinstance(self, TreeSimulator)

    all_nodes = list(self.tree.traverse())
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    bits, nptype = 64, np.uint64
    sim = np.zeros((num_nodes, num_traits), dtype=nptype)
    self.NUM_TRIALS = bits

    node_map = {node: i for i, node in enumerate(all_nodes)}

    gain_rates = self.gains / (self.gain_subsize * self.MULTIPLIER)
    loss_rates = self.losses / (self.loss_subsize * self.MULTIPLIER)
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    node_dists = {self.tree: self.tree.dist or 0}
    for node in self.tree.traverse():
        if node not in node_dists:
            node_dists[node] = node_dists[node.up] + node.dist

    print("Simulating CTMP with Numba acceleration...")

    for node in self.tree.traverse():
        idx = node_map[node]

        if node.up is None:
            root_mask = self.root_states > 0
            sim[idx, root_mask] = (2 ** self.NUM_TRIALS - 1)
            continue

        parent_state = sim[node_map[node.up]].copy()
        node_dist = node.dist * self.MULTIPLIER
        node_total_dist = node_dists[node]

        gain_mask = node_total_dist >= self.dists
        loss_mask = node_total_dist >= self.loss_dists

        state_bits = np.unpackbits(
            parent_state.view(np.uint8).reshape(num_traits, 8),
            axis=1, bitorder='little'
        )

        # Call Numba accelerated branch simulator
        state_bits = simulate_branch_ctmp(state_bits, gain_rates, loss_rates, gain_mask, loss_mask, node_dist)

        packed_state = np.packbits(
            state_bits, axis=1, bitorder='little'
        ).reshape(num_traits, 8).view(nptype).flatten()

        sim[idx, :] = packed_state

    print("Completed CTMP Simulation Successfully")
    lineages = sim[[node_map[node] for node in self.tree], :]

    res = compile_results_KDE_bit_async(self, lineages, bits=bits, nptype=nptype)
    return res
