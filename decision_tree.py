import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

def gini(y):
    values, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def information_gain(y, y_left, y_right, criterion="entropy"):
    measure = entropy if criterion == "entropy" else gini
    before = measure(y)
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    after = (n_left / n) * measure(y_left) + (n_right / n) * measure(y_right)
    return before - after

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, criterion="entropy"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_features_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))

        best_feat, best_thresh = self._best_split(X, y, n_features)
        if best_feat is None:
            return Node(value=self._most_common_label(y))

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
                if left_idx.sum() == 0 or right_idx.sum() == 0:
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
        X = np.asarray(X)
        if X.ndim == 1:
            return self._traverse_tree(X, self.root)
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
                name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
                print(f"{indent}{name} <= {node.threshold}")
                _print(node.left, depth + 1)
                _print(node.right, depth + 1)

        _print(self.root)

    def plot_2d(self, X, y, feature_idx=(0, 1), h=0.02):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.n_features_ is None:
            raise ValueError("Call fit(...) before plot_2d(...)")

        f0, f1 = feature_idx
        x_min, x_max = X[:, f0].min() - 1, X[:, f0].max() + 1
        y_min, y_max = X[:, f1].min() - 1, X[:, f1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        n_points = xx.ravel().shape[0]
        grid = np.tile(np.mean(X, axis=0), (n_points, 1))
        grid[:, f0] = xx.ravel()
        grid[:, f1] = yy.ravel()

        preds = self.predict(grid).reshape(xx.shape)

        plt.contourf(xx, yy, preds, alpha=0.3)
        plt.scatter(X[:, f0], X[:, f1], c=y, edgecolor="k")
        plt.xlabel(f"Feature {f0}")
        plt.ylabel(f"Feature {f1}")
        plt.title("Decision Boundary")
        plt.show()

if __name__ == "__main__":
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=5, criterion="entropy")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)

    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=data.target_names))

    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

    print("\nDecision tree:")
    clf.print_tree(feature_names=data.feature_names)

    clf.plot_2d(X_train, y_train, feature_idx=(0, 1))
