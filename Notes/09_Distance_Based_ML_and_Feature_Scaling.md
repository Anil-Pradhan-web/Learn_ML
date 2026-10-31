# Chapter 9: Distance-Based ML & Feature Scaling

> **Why this matters for ML**: Algorithms like $K$-Nearest Neighbors (KNN) and $K$-Means rely on calculating geometric distances between points. Without proper feature scaling, features with large numbers (like Salary in \$50,000s) completely wipe out smaller features (like Age in 20s).

---

## 1. $K$-Nearest Neighbors (KNN)

### Core Concept (Simple English)
KNN is a **non-parametric, lazy-learning algorithm**. It does not learn an explicit mathematical equation during training. Instead, when a new query point arrives, it calculates the distance to all points in the dataset and takes a **majority vote among the $K$ closest neighbors**.

```
Query Point (?)
   │
   ├── Finds 3 closest training samples (K = 3)
   ├── 2 neighbors are Class Blue (🟦)
   └── 1 neighbor is Class Orange (🟧)
   ──► Majority Vote = Class Blue (🟦)
```

> 💡 **Hinglish Intuition**:  
> KNN ka simple rule hai: *"Jaisa sang, waisa rang"*. Agar tumhare aas-pass ke 5 sabse kareebi dost computer science padhte hain, toh high probability hai ki tum bhi computer science student ho.

---

## 2. Distance Metrics

Given two points $\mathbf{p} = (p_1, p_2)$ and $\mathbf{q} = (q_1, q_2)$:

1. **Euclidean Distance ($L_2$ Norm)** (Straight line distance):
   $$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$
2. **Manhattan Distance ($L_1$ Norm / City Block)**:
   $$d(\mathbf{p}, \mathbf{q}) = \sum_{i=1}^n |p_i - q_i|$$

---

## 3. How to Choose $K$?
- **If $K = 1$**: Overfitting (extremely sensitive to noisy individual points).
- **If $K$ is too large (e.g., $K = 100$)**: Underfitting (majority class in dataset will always win).
- **Rule of Thumb**: Choose $K = \sqrt{N}$ (square root of sample count). Always pick an **odd number** for binary classification to avoid tie-breakers!

---

## 4. Why Feature Scaling is STRICTLY MANDATORY

Imagine two features:
- **Age**: Ranges from $20$ to $60$ (Difference $\approx 10$)
- **Salary**: Ranges from $\$20,000$ to $\$100,000$ (Difference $\approx 20,000$)

When calculating Euclidean distance:
$$d = \sqrt{(\Delta \text{Age})^2 + (\Delta \text{Salary})^2} = \sqrt{(10)^2 + (20000)^2} \approx \sqrt{100 + 400,000,000} \approx 20,000$$
The Salary feature completely dominates distance by $99.99\%$, making Age completely irrelevant.

---

## 5. Feature Scaling Techniques

| Method | Formula | Range | When to Use |
|---|---|---|---|
| **Standardization (Z-Score)** | $Z = \frac{X - \mu}{\sigma}$ | Centered at Mean $= 0$, Std $= 1$ | Best for algorithms assuming normal distribution (Linear Reg, Logistic Reg, PCA). Robust to moderate outliers. |
| **Normalization (Min-Max)** | $X_{\text{new}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$ | Strictly bounded in $[0, 1]$ | Best for KNN, Neural Networks, or bounded image pixels. Sensitive to extreme outliers. |

---

## 6. Python Scikit-Learn Code

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# It is best practice to combine scaling and KNN inside a Pipeline
pipeline = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, metric='euclidean')
)

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print("KNN Test Accuracy:", accuracy)
```

---

## 🎯 Quick B.Tech Viva Questions
1. **Q: Why is KNN called a "Lazy Learner"?**  
   *A:* Because it does no training computations during `.fit()`. It simply stores the training data and performs all heavy distance calculations during test/inference time (`.predict()`).
2. **Q: Do Tree-based models (like Decision Trees) require Feature Scaling?**  
   *A:* **No**. Decision trees make horizontal/vertical split decisions on individual features independently ($x_i \ge \text{threshold}$); monotonic scaling changes nothing in tree splits.

