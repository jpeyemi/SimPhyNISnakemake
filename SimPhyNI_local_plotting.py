import pickle
import sys
sys.path.insert(0,"SimPhyNI/Simulation")
sys.path.insert(0,"SimPhyNI")
from tree_simulator import TreeSimulator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys

simphyni_object = 'sim_kde.pkl'
# Load simulation
with open(simphyni_object, "rb") as f:
    Sim: TreeSimulator = pickle.load(f)

# Plot Results Heatmap (All against All)
Sim.plot_results(correction = 'fdr_bhc', figure_size=10, prevalence_range=[0.05, 0.95])

# Plot Effect Sizes
Sim.plot_effect_size(correction = 'fdr_bhc', prevalence_range=[0.05, 0.95])

# Top Significant Results
res = Sim.get_top_results(
    correction='fdr_bhc',
    prevalence_range=[0.05, 0.95],
    top=50,
    direction=0,
    by="p-value",
    alpha=0.05,
)
