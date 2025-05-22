
#%% Import necessary libraries
import pandas as pd
import numpy as np
from ete3 import Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from itertools import combinations

#%% Load the Newick file and the trait CSV
def load_tree(nwk_file):
    """
    Load the phylogenetic tree from a Newick file.
    """
    tree = Tree(nwk_file, format=1)  # Format=1 expects internal node names
    return tree

def load_traits(csv_file):
    """
    Load binary trait data from a CSV file.
    Each row is a taxon (tip) and columns are binary traits.
    """
    traits = pd.read_csv(csv_file, index_col=0)  # Assuming first column is the taxon ID
    return traits

#%% Compute pairwise phylogenetic distances
def compute_phylogenetic_distance(tree):
    """
    Compute pairwise phylogenetic distances between all tips in the tree.
    """
    taxa = tree.get_leaf_names()
    dist_matrix = pd.DataFrame(index=taxa, columns=taxa)
    
    for i, taxon1 in enumerate(taxa):
        for j, taxon2 in enumerate(taxa):
            dist_matrix.loc[taxon1, taxon2] = (tree&taxon1).get_distance(taxon2)
    
    return dist_matrix.astype(float)

#%% Compute pairwise trait statistics
def compute_pairwise_trait_stats(traits):
    """
    Compute pairwise co-occurrence and mutual exclusivity statistics for traits.
    """
    trait_pairs = list(combinations(traits.columns, 2))
    stats = []

    for trait1, trait2 in trait_pairs:
        # Co-occurrence counts
        both_present = np.sum((traits[trait1] == 1) & (traits[trait2] == 1))
        both_absent = np.sum((traits[trait1] == 0) & (traits[trait2] == 0))
        only_first = np.sum((traits[trait1] == 1) & (traits[trait2] == 0))
        only_second = np.sum((traits[trait1] == 0) & (traits[trait2] == 1))
        
        stats.append({
            'trait1': trait1,
            'trait2': trait2,
            'both_present': both_present,
            'both_absent': both_absent,
            'only_first': only_first,
            'only_second': only_second
        })

    return pd.DataFrame(stats)

#%% Label trait associations as positive, negative, or neutral
def label_associations(stats):
    """
    Label trait associations as positive (1), negative (-1), or neutral (0).
    """
    labels = []
    
    for _, row in stats.iterrows():
        if row['both_present'] > 10:  # Arbitrary threshold for positive association
            labels.append(1)
        elif row['only_first'] > 10 or row['only_second'] > 10:  # Arbitrary threshold for negative association
            labels.append(-1)
        else:
            labels.append(0)
    
    stats['association'] = labels
    return stats

#%% Prepare features for the random forest model
def prepare_features(traits, tree):
    """
    Prepare the features for the random forest model.
    Combine trait statistics with phylogenetic distances.
    """
    # Get pairwise trait statistics
    trait_stats = compute_pairwise_trait_stats(traits)
    
    # Compute phylogenetic distance matrix
    dist_matrix = compute_phylogenetic_distance(tree)
    
    # For each trait pair, extract the distance between the taxa
    features = []
    
    for _, row in trait_stats.iterrows():
        # Average phylogenetic distance for all tips with this pair of traits
        taxa_with_both = traits.index[(traits[row['trait1']] == 1) & (traits[row['trait2']] == 1)]
        
        if len(taxa_with_both) > 1:
            avg_dist = np.mean([dist_matrix.loc[taxa1, taxa2] for taxa1, taxa2 in combinations(taxa_with_both, 2)])
        else:
            avg_dist = 0  # If no co-occurrence, distance is irrelevant
        
        features.append({
            'trait1': row['trait1'],
            'trait2': row['trait2'],
            'avg_dist': avg_dist,
            'both_present': row['both_present'],
            'both_absent': row['both_absent'],
            'only_first': row['only_first'],
            'only_second': row['only_second']
        })
    
    return pd.DataFrame(features)

#%% Train and evaluate the random forest model
def train_random_forest(features, labels):
    """
    Train a random forest model on the features and labels.
    """
    X = features[['avg_dist', 'both_present', 'both_absent', 'only_first', 'only_second']]
    y = labels['association']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print(classification_report(y_test, y_pred))

    return clf

#%% Run the full pipeline
def run_pipeline(nwk_file, trait_csv):
    tree = load_tree(nwk_file)
    traits = load_traits(trait_csv)
    
    # Prepare features and labels
    trait_stats = compute_pairwise_trait_stats(traits)
    labeled_stats = label_associations(trait_stats)
    features = prepare_features(traits, tree)
    
    # Train Random Forest
    clf = train_random_forest(features, labeled_stats)
    
    return clf

#%% Example usage
if __name__ == "__main__":
    nwk_file = "Data/Sepi/sepi_cluster_by_phylogroup.newick"
    trait_csv = "Data/Sepi/defense_systems_pivot.csv"
    
    model = run_pipeline(nwk_file, trait_csv)

# %%
