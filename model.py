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

# Step 7 - build_tree
import numpy as np

def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    """Recursively grow a decision tree, returning a nested dict of leaf/internal nodes."""
    
    # 1. Check if we should stop and return a leaf node
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {"leaf": True, "prediction": leaf_prediction(labels)}
    
    # 2. Build the candidate feature list
    if feature_subset is None:
        candidates = range(features.shape[1])
    else:
        candidates = list(feature_subset)
        
    # 3. Find the best split
    best = best_split(features, labels, candidates)
    
    # Fall back to a leaf if no usable split was found
    if best["feature_index"] is None:
        return {"leaf": True, "prediction": leaf_prediction(labels)}
        
    # 4. Partition the dataset
    fi = best["feature_index"]
    t = best["threshold"]
    left_features, left_labels, right_features, right_labels = split_dataset(features, labels, fi, t)
    
    # 5. If either child side is empty, emit a leaf instead of an invalid split
    if len(left_labels) == 0 or len(right_labels) == 0:
        return {"leaf": True, "prediction": leaf_prediction(labels)}
        
    # 6. Otherwise, recurse on both halves and structure as an internal node dict
    return {
        "leaf": False,
        "feature_index": fi,
        "threshold": t,
        "left": build_tree(left_features, left_labels, max_depth, min_samples_split, feature_subset, depth + 1),
        "right": build_tree(right_features, right_labels, max_depth, min_samples_split, feature_subset, depth + 1)
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    node = tree 
    while node["leaf"] is False:
        
        j, t = node["feature_index"], node["threshold"]
        v = example[j]
        if v<= t:
            node = node["left"]
        else:
            node = node["right"]
            
    return int(node["prediction"])

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.
    return np.array([predict_example_tree(tree, features[i]) for i in range(len(features))])

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # TODO: draw a bootstrap sample of rows (with replacement) using rng.
    n = features.shape[0]
    indicies = rng.integers(0, n, size=n)
    return features[indicies], labels[indicies]

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # TODO: return num_to_pick distinct random feature indices from range(num_features) using rng.
    return rng.choice(num_features, size=num_to_pick, replace=False)

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    """
    Grow num_trees decision trees on bootstrap samples with random feature subsets.
    """
    # 1. Build a single generator ONCE outside the loop to ensure reproducible variance
    rng = np.random.default_rng(random_state)
    
    # 2. Compute the per-split feature count
    num_features = features.shape[1]
    if num_features_per_split is None:
        # Default to max(1, round(sqrt(d)))
        num_to_pick = max(1, int(round(np.sqrt(num_features))))
    else:
        num_to_pick = num_features_per_split
        
    forest = []
    
    # 3. Loop num_trees times
    for _ in range(num_trees):
        # Draw a bootstrap sample of rows (sampled with replacement)
        X_boot, y_boot = bootstrap_sample(features, labels, rng)
        
        # Pick a random subset of columns (no replacement)
        f_subset = feature_subset(num_features, num_to_pick, rng)
        
        # Fit the tree
        tree = build_tree(
            X_boot, 
            y_boot, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            feature_subset=f_subset
        )
        
        # Append the tree and its designated features to the ensemble
        forest.append({
            "tree": tree, 
            "feature_indices": f_subset
        })
        
    return forest

# Step 13 - combine_predictions
import numpy as np

def combine_predictions(tree_predictions):
    """Aggregate the per-tree predictions of an ensemble into one prediction per example."""
    
    # Convert to a 2D array and transpose. 
    # Shape goes from (num_trees, num_examples) -> (num_examples, num_trees)
    preds = np.array(tree_predictions).T
    
    # Helper to calculate the mode (majority vote) for a single example
    def majority_vote(example_preds):
        values, counts = np.unique(example_preds, return_counts=True)
        return values[np.argmax(counts)]
        
    # Apply the majority vote across all examples
    return np.array([majority_vote(row) for row in preds])

# Step 14 - predict_forest
import numpy as np

def predict_forest(forest, features):
    """Predict classes for a dataset using the whole trained forest."""
    tree_predictions = []
    
    # 1. Collect predictions from every tree
    for t in forest:
        # Extract the actual tree dictionary from the forest ensemble list
        tree_structure = t["tree"]
        
        # Get this specific tree's predictions for all examples
        p = predict_tree(tree_structure, features)
        tree_predictions.append(p)
        
    # 2. Return the combined majority-vote predictions
    return combine_predictions(tree_predictions)

# Step 15 - accuracy
def accuracy(predictions, labels):
    # TODO: compute the fraction of entries where predictions equals labels
    return float(np.mean(predictions == labels))

