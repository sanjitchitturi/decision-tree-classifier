import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Utility functions
def entropy(y):
    """Calculate entropy of a label array y."""
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def gini(y):
    """Calculate Gini impurity of a label array y."""
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

def information_gain(y, y_left, y_right, criterion="entropy"):
    """Compute information gain for a split."""
    if criterion == "entropy":
        measure = entropy
    else:
        measure = gini

    H_before = measure(y)
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    H_after = (n_left / n) * measure(y_left) + (n_right / n) * measure(y_right)
    return H_before - H_after
  
# Decision Tree Node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 

    def is_leaf(self):
        return self.value is not None

# Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, criterion="entropy"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # stopping conditions
        if (depth >= self.max_depth
            or num_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # find best split
        best_feat, best_thresh = self._best_split(X, y, n_features)

        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # split
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_idx = X[:, feat] <= thresh
                right_idx = ~left_idx

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                gain = information_gain(y, y[left_idx], y[right_idx], self.criterion)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh

        return split_idx, split_thresh

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def print_tree(self, feature_names=None):
        def _print(node, depth=0):
            indent = "  " * depth
            if node.is_leaf():
                print(f"{indent}Leaf: Class={node.value}")
            else:
                feat_name = feature_names[node.feature] if feature_names is not None else f"X[{node.feature}]"
                print(f"{indent}{feat_name} <= {node.threshold}")
                _print(node.left, depth + 1)
                _print(node.right, depth + 1)
        _print(self.root)

    def plot_2d(self, X, y, feature_idx=(0, 1), h=0.02):
        """Visualize decision boundary for 2D features using matplotlib."""
        x_min, x_max = X[:, feature_idx[0]].min() - 1, X[:, feature_idx[0]].max() + 1
        y_min, y_max = X[:, feature_idx[1]].min() - 1, X[:, feature_idx[1]].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        X_grid = np.c_[xx.ravel(), yy.ravel()]
        preds = self.predict(np.insert(X_grid, 2, 0, axis=1)[:, :X.shape[1]])
        preds = preds.reshape(xx.shape)

        plt.contourf(xx, yy, preds, alpha=0.3)
        plt.scatter(X[:, feature_idx[0]], X[:, feature_idx[1]], c=y, edgecolor='k')
        plt.xlabel(f"Feature {feature_idx[0]}")
        plt.ylabel(f"Feature {feature_idx[1]}")
        plt.title("Decision Tree Decision Boundary")
        plt.show()



# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=5, criterion="entropy")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nDecision Tree:")
    clf.print_tree(feature_names=data.feature_names)

    # Plot decision boundary for two features (sepal length, sepal width)
    clf.plot_2d(X_train[:, :2], y_train, feature_idx=(0, 1))
