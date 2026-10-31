# Chapter 12: Unsupervised Learning & K-Means Clustering

> **Why this matters for ML**: In the real world, 90% of data has no labels ($y$). Companies have millions of customer transaction logs or website clicks without human annotations. Unsupervised learning discovers hidden natural groupings inside unlabelled data.

---

## 1. Supervised vs Unsupervised Learning

- **Supervised Learning**: Data has inputs $X$ and ground-truth targets $y$. Model learns to predict $y$.
- **Unsupervised Learning**: Only inputs $X$ are provided; **NO target labels $y$ exist**. The model discovers intrinsic geometric structures, patterns, or clusters on its own.

```
Supervised:    [Cat Picture] ──► Label: "Cat"
               [Dog Picture] ──► Label: "Dog"

Unsupervised:  [10,000 Unlabelled Animal Photos] ──► Discovers 2 distinct clusters:
                                                     Group A (pointed ears, whiskers)
                                                     Group B (floppy ears, snout)
```

---

## 2. The $K$-Means Clustering Algorithm

$K$-Means partitions $N$ unlabeled data points into $K$ distinct, non-overlapping clusters.

### Mathematical Objective: Minimize Inertia (WCSS)
Minimize the **Within-Cluster Sum of Squares (WCSS)**:
$$J = \sum_{k=1}^K \sum_{\mathbf{x} \in C_k} ||\mathbf{x} - \boldsymbol{\mu}_k||^2$$
where $\boldsymbol{\mu}_k$ is the center (centroid) of cluster $C_k$.

### Step-by-Step Algorithm:
1. **Choose $K$**: Select the number of clusters to find.
2. **Initialize Centroids**: Randomly place $K$ centroids in the feature space (or using the smarter `k-means++` heuristic).
3. **Assignment Step**: Assign each data point to its closest centroid using Euclidean distance.
4. **Update Step**: Calculate the mathematical mean of all points assigned to each cluster and move the centroid to this new mean position:
   $$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x} \in C_k} \mathbf{x}$$
5. **Convergence**: Repeat Steps 3 and 4 until centroids stop moving or reach maximum iterations.

---

## 3. How to Choose the Optimal $K$? (The Elbow Method)

Since there are no labels, how do we know whether to choose $K = 2, 3,$ or $5$?
- We compute **Inertia (WCSS)** for different values of $K$ ($K=1$ to $10$) and plot them.
- As $K$ increases, WCSS naturally drops because clusters become smaller and tighter.
- We look for the **"Elbow Point"**: The bend in the graph where the rate of drop drastically slows down. That point is the optimal $K$.

```
WCSS (Inertia)
  ▲
  │ \
  │  \
  │   \  <── Steep drop
  │    \
  │     ●  <── THE ELBOW POINT (Optimal K = 3)
  │      \───────-─-─-─-─-─-─ <── Minimal drop after this
  └────────────────────────────► Number of Clusters (K)
        1   2   3   4   5   6
```

---

## 4. Python Scikit-Learn Implementation

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Finding optimal K via Elbow Method
wcss = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 2. Fit K-Means with chosen optimal K (e.g., K=3)
optimal_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
cluster_labels = optimal_kmeans.fit_predict(X)
centroids = optimal_kmeans.cluster_centers_

print("Assigned Cluster Labels (First 10):", cluster_labels[:10])
print("Coordinates of Centroids:\n", centroids)
```

---

## 5. Course Completion Summary & What Comes Next

```
[Prerequisites & Stats] ──► [Linear & Non-Linear Reg] ──► [Errors & Bias-Variance]
           │                                                        │
           ▼                                                        ▼
[Distance ML & Scaling] ◄── [Confusion Matrix & Metrics] ◄── [Train-Test Split & Logistic]
           │
           ▼
[Decision Trees] ──► [Random Forest & Ensembles] ──► [Unsupervised K-Means Clustering]
```

---

## 🎯 Quick B.Tech Viva Questions
1. **Q: Why is $K$-Means sensitive to feature scaling?**  
   *A:* $K$-Means calculates Euclidean distances to assign points to centroids. Just like KNN, unscaled features with large values dominate cluster assignments.
2. **Q: What is the advantage of `k-means++` initialization over standard random initialization?**  
   *A:* Standard random initialization can place centroids too close together, leading to suboptimal local minima. `k-means++` spreads initial centroids far apart from each other, guaranteeing faster and superior convergence.

