import pickle
from tree_simulator import TreeSimulator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys

# Load simulation
with open("sim_kde.pkl", "rb") as f:
    Sim: TreeSimulator = pickle.load(f)

# Plot Results
Sim.plot_results(correction = 'fdr_tsbky', figure_size=10, prevalence_range=[0.05, 0.95])

# Plot Effect Sizes
Sim.plot_effect_size(correction = 'fdr_tsbky', prevalence_range=[0.05, 0.95])

# Top Significant Results
res = Sim.get_top_results(
    correction='fdr_tsbky',
    prevalence_range=[0.05, 0.95],
    top=50,
    direction=0,
    by="p-value",
    alpha=0.05,
)
