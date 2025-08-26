# Decision Tree Classifier

This project implements a **Decision Tree Classifier** entirely from scratch in Python. It demonstrates how decision trees work internally, without relying on machine learning libraries like scikit-learn for the algorithm itself.

---

## What it Does
- Builds a decision tree using **recursive binary splits**
- Supports two criteria for splitting:
  - **Entropy** (Information Gain)
  - **Gini Impurity**
- Stopping conditions:
  - Maximum depth
  - Minimum samples per split
- Can **predict new samples** by traversing the tree
- Provides a method to **print the tree structure** in a readable format
- Includes a **2D decision boundary visualization** with matplotlib

---

## Example Workflow
1. Load a dataset (Iris dataset in the demo).
2. Train the decision tree classifier.
3. Print the tree structure.
4. Evaluate accuracy on a test split.
5. Visualize decision boundaries for two features.
   
---

## Example Output

**Printed Tree Example:**
```

sepal length (cm) <= 5.4
sepal width (cm) <= 3.3
Leaf: Class=0
...

```

**Decision Boundary Visualization:**
A matplotlib plot showing classification regions with training points overlaid.

---

## Purpose
This implementation is meant for **educational purposes**, to show:
- How decision trees choose splits based on information gain.
- How recursion is used to grow the tree.
- How predictions are made by traversing nodes.
- How decision boundaries can be visualized.

---

## Results
On the **Iris dataset**, the classifier achieves around **95% accuracy**.

---
