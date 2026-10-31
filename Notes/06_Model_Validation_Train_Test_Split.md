# Chapter 6: Model Validation & Train-Test Split

> **Why this matters for ML**: If a teacher gives students the exact same exam questions they practiced in homework, everyone scores 100%, but no one knows if they actually learned the subject. Model validation ensures our ML models perform well on brand-new, unseen data.

---

## 1. Why We Never Evaluate ONLY on Training Data

If you test a model on the same data it trained on:
- An overfitted model can achieve **100% accuracy** simply by acting as a lookup table (memorization).
- You get a **falsely optimistic evaluation** that will crash horribly when deployed to real production users.

---

## 2. The Train-Test Split (Standard Workflow)

We randomly partition the dataset $\mathcal{D}$ into two non-overlapping subsets:
1. **Training Set ($70\% - 80\%$)**: Used by the algorithm to adjust parameters and learn patterns.
2. **Testing Set ($20\% - 30\%$)**: Kept hidden in a vault. Used only once at the very end to evaluate generalization.

```
Total Dataset (N = 1000 samples)
┌──────────────────────────────────────────────────────────┬──────────────────────┐
│                  Training Set (80% = 800)                │  Testing Set (200)   │
│             Used to fit weights: model.fit(X, y)         │   model.predict(X)   │
└──────────────────────────────────────────────────────────┴──────────────────────┘
```

### The Role of `random_state`
- `random_state=42` sets a fixed seed for the random number generator.
- It guarantees that every time you or your professor runs the script, the exact same rows end up in the training and testing sets, making the experiment **100% reproducible**.

---

## 3. Data Leakage (The Deadly Sin of ML)

> ⚠️ **Exam Question: "Why should the test set never be used to fit the model?"**  
> **Answer**: Fitting on the test set causes **Data Leakage (Snooping Bias)**. The model peeks at future test patterns before predicting them, resulting in artificially high test scores that completely fail in the real world.

---

## 4. The Scikit-Learn Input Shape Error Trap

In Scikit-Learn, all estimators (`LinearRegression`, `LogisticRegression`, etc.) expect:
- **Feature Matrix $X$**: Must be a **2-Dimensional array/DataFrame** of shape `(n_samples, n_features)`.
- **Target Vector $y$**: Must be a **1-Dimensional array/Series** of shape `(n_samples,)`.

```python
# ❌ INCORRECT (Throws ValueError: Expected 2D array, got 1D array instead):
X = df["Area"]  # 1D Series of shape (1000,)

# ✅ CORRECT OPTION 1 (Double brackets to retain 2D DataFrame):
X = df[["Area"]]  # 2D DataFrame of shape (1000, 1)

# ✅ CORRECT OPTION 2 (NumPy reshape):
X = df["Area"].values.reshape(-1, 1)  # 2D array of shape (1000, 1)
```

---

## 5. Limitation of a Single Split & $K$-Fold Cross Validation

- **Limitation of Single Split**: It is vulnerable to luck. If all hard or outlier samples accidentally end up in the test set, the score drops drastically.
- **Solution ($K$-Fold Cross-Validation)**:
  - Divide data into $K$ equal folds (e.g., $K=5$).
  - Train on $4$ folds, test on the $1$ remaining fold.
  - Repeat $5$ times so every sample gets tested once.
  - Final score is the **average of all 5 runs**.

```
Fold 1: [ Test ] [Train] [Train] [Train] [Train] ──► Score 1
Fold 2: [Train] [ Test ] [Train] [Train] [Train] ──► Score 2
Fold 3: [Train] [Train] [ Test ] [Train] [Train] ──► Score 3
Fold 4: [Train] [Train] [Train] [ Test ] [Train] ──► Score 4
Fold 5: [Train] [Train] [Train] [Train] [ Test ] ──► Score 5
                                                     ───────
                                            Average Score
```

---

## 6. Python Scikit-Learn Code

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

# 1. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 2. Fit ONLY on training data
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Predict on unseen test data
y_pred = model.predict(X_test)

# 4. Optional: 5-Fold Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Average CV Score:", cv_scores.mean())
```

---

## 🎯 Quick B.Tech Viva Questions
1. **Q: If a dataset has 500 samples and `test_size=0.25`, how many samples are in train and test?**  
   *A:* Test = $500 \times 0.25 = 125$ samples, Train = $500 - 125 = 375$ samples.
2. **Q: When should we use Stratified Train-Test Split?**  
   *A:* In classification problems with imbalanced classes (e.g., 95% non-fraud, 5% fraud) so both train and test have the exact same class proportions.

