"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np

def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    if len(labels) == 0:
        return 0.0
    
    # np.unique returns a tuple: (unique_classes, counts)
    _, counts = np.unique(labels, return_counts=True)
    
    # Calculate probabilities
    props = counts / len(labels)
    
    # Gini impurity formula: 1 - sum(p_i^2)
    G = 1.0 - np.sum(props**2)
    return G

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    col = features[:, feature_index]
    mask = col <= threshold
    left_features, left_labels = features[mask], labels[mask]
    right_features, right_labels = features[~mask], labels[~mask]
    return left_features, left_labels, right_features, right_labels

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    p_imp, left_imp, right_imp = impurity(parent_labels), impurity(left_labels), impurity(right_labels)
    gain = p_imp - (left_imp*len(left_labels)/len(right_labels) + right_imp*len(right_labels)/len(parent_labels))
    return gain

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    out = {"feature_index": None, "threshold":None, "score":0.0}
    for fi in feature_indices:
        u = np.unique(features[:, fi])
        mids = (u[:-1] + u[1:]) / 2
        for t in mids:
            lf, ll, rf, rl = split_dataset(features, labels, fi, t)
            s = split_score(labels, ll, rl)
            if s > out["score"]:
                out["score"] = s 
                out["feature_index"] = fi 
                out["threshold"] = t 
    return out

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # Fixed the syntax error by removing the extra "if" keywords
    if depth >= max_depth or len(labels) < min_samples_split or len(np.unique(labels)) == 1:
        return True
    else:
        return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    labels = np.array(labels, dtype=int)
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

