# Chapter 10: Decision Trees

> **Why this matters for ML**: Decision Trees mirror human decision-making using a flowchart of `if-else` rules. They are intuitive, highly interpretable, and form the building blocks for modern powerhouse algorithms like Random Forest and XGBoost.

---

## 1. Anatomy of a Decision Tree

```
                       [ Age <= 30? ]             <── ROOT NODE
                        /          \
                      Yes           No
                      /              \
             [ Income <= 50k? ]     [ Class: BUY (Yes) ] <── LEAF NODE
               /            \
             Yes             No
             /                \
   [ Class: NO ]         [ Class: BUY ]            <── LEAF NODES (Predictions)
```

- **Root Node**: The top-most decision node with the entire starting dataset.
- **Decision Nodes**: Internal splits based on feature tests ($x_j \le \text{threshold}$).
- **Leaf (Terminal) Nodes**: Final output nodes that hold the class label or predicted value (no further splits).

---

## 2. How the Tree Decides Where to Split (Purity Metrics)

The tree evaluates all possible splits across all features and chooses the split that creates the **purest child nodes** (nodes containing samples of only one class).

### 1. Entropy (Measure of Randomness / Disorder)
$$H(S) = - \sum_{i=1}^C p_i \log_2(p_i)$$
- If a node is **100% pure** (all Cat): $H = - (1 \log_2 1) = \mathbf{0}$.
- If a node is **completely mixed** (50% Cat, 50% Dog): $H = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) = \mathbf{1.0}$ (Maximum Chaos).

### 2. Information Gain ($IG$)
The reduction in entropy achieved by splitting on feature $A$:
$$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
The algorithm picks the feature with the **highest Information Gain** (ID3 / C4.5).

### 3. Gini Impurity (Default in Scikit-Learn CART)
$$Gini(S) = 1 - \sum_{i=1}^C p_i^2$$
- Pure node: $Gini = 1 - (1)^2 = \mathbf{0}$.
- 50/50 split: $Gini = 1 - (0.5^2 + 0.5^2) = \mathbf{0.5}$.
- *Why CART prefers Gini over Entropy*: Gini doesn't require calculating logarithms ($\log_2$), making training much faster on large datasets.

---

## 3. The Overfitting Problem & Tree Pruning

- **Problem**: If allowed to grow unchecked, a Decision Tree will keep splitting until every single training sample has its own private leaf node. This leads to **100% training accuracy but disastrous test accuracy (Severe Overfitting)**.
- **Solution (Hyperparameter Pruning)**:
  - `max_depth`: Limits the maximum depth/height of the tree (e.g., `max_depth=4`).
  - `min_samples_split`: Minimum samples required in a node to attempt a split (e.g., `min_samples_split=10`).
  - `min_samples_leaf`: Minimum samples required in a terminal leaf node.

---

## 4. Python Scikit-Learn Implementation

```python
from sklearn.tree import DecisionTreeClassifier, export_text

# 1. Initialize tree with depth constraint to prevent overfitting
dt_model = DecisionTreeClassifier(
    criterion='gini',    # or 'entropy'
    max_depth=3,         # Pruning
    random_state=42
)

# 2. Fit and predict
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

# 3. Print human-readable if-else rules of the tree
tree_rules = export_text(dt_model, feature_names=list(X.columns))
print(tree_rules)
```

---

## 🎯 Quick B.Tech Viva Questions
1. **Q: Does Decision Tree require feature scaling (normalization/standardization)?**  
   *A:* **No**. Tree splits only evaluate whether a feature value is greater than or less than a threshold ($x_i \le \theta$). Monotonic scaling does not change the order or location of optimal splits.
2. **Q: What is the maximum value of Gini Impurity for a 2-class problem?**  
   *A:* $\mathbf{0.5}$ (when both classes have equal $50\%$ probability).

