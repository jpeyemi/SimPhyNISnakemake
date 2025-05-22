#!/usr/bin/env python

import sys
import os
import pickle
from pathlib import Path
import argparse

import numpy as np
from scipy import stats
import statsmodels.stats.multitest as sm
from sklearn.metrics import classification_report, accuracy_score

# Local imports
sys.path.insert(0, "PhyloSim/Simulation")
# sys.path.insert(0, "PhyloSim/Data")
from tree_simulator import TreeSimulator
from SimulationMethods import simulate_glrates
from PairStatistics import PairStatistics

# ----------------------
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser(description="Run SimPhyNI KDE-based trait simulation.")
parser.add_argument("-p", "--pastml", required=True, help="Path to PastML output CSV")
parser.add_argument("-s", "--systems", required=True, help="Path to input traits CSV")
parser.add_argument("-t", "--tree", required=True, help="Path to rooted Newick tree")
parser.add_argument("-o", "--outdir", required=True, help="Output path to save the Sim object")
parser.add_argument("-r", "--runtype", type=int, choices=[0, 1], default=0,
                    help="1 for single trait mode (e.g., IBD), 0 for multi-trait [default: 0]")

args = parser.parse_args()
single_trait = bool(args.runtype)

# ----------------------
# Simulation Setup
# ----------------------
Sim = TreeSimulator(
    tree=args.tree,
    pastmlfile=args.pastml,
    obsdatafile=args.systems
)

Sim.set_trials(64)
print("Initializing SimPhyNI...")

Sim.initialize_simulation_parameters(
    pair_statistic=PairStatistics._log_odds_ratio_statistic,
    collapse_theshold=0.001,
    single_trait=single_trait,
    prevalence_threshold=0.00,
    kde=True
)

# ----------------------
# Run Simulation
# ----------------------
print("Running SimPhyNI analysis...")
Sim.run_simulation(
    parallel=True,
    simulation_function=simulate_glrates,
    bit=True,
    norm=True
)

# ----------------------
# Save Outputs
# ----------------------
output_dir = Path(args.outdir)
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'sim_norm_kde.pkl', 'wb') as f:
    pickle.dump(Sim, f)

Sim.get_top_results(top = 2**63 - 1).to_csv(output_dir / 'sr.csv')
print("Simulation completed.")
