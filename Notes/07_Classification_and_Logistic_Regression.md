# Chapter 7: Classification & Logistic Regression

> **Why this matters for ML**: Despite having the word "Regression" in its name, Logistic Regression is the most widely used fundamental algorithm for **Binary Classification** (predicting Yes/No, 0/1, True/False).

---

## 1. Classification vs Regression

- **Regression**: Predicts a continuous quantity (e.g., House price = \$250,000, Salary = \$75,000).
- **Classification**: Predicts a discrete categorical class label (e.g., Tumor = Malignant vs Benign, Email = Spam vs Ham).

---

## 2. Why Linear Regression FAILS for Classification

If we try to fit a straight line $y = wx + b$ to predict probabilities ($0$ or $1$):
1. **Unbounded Outputs**: Straight lines extend to $-\infty$ and $+\infty$. It produces senseless "probabilities" like $-0.4$ or $1.8$.
2. **Outlier Sensitivity**: A distant legitimate sample will violently drag the slope of the regression line, shifting the decision threshold and misclassifying normal points.

```
Linear Regression (Flawed for classification):
Prob
  ▲                 /  <── Predicted prob > 1 (Invalid!)
 1│               /
  │             /
  │           /
 0│─────────/──────────► Feature X
  │       / <── Predicted prob < 0 (Invalid!)
```

---

## 3. The Logistic (Sigmoid) Function

To fix this, we take the linear score $z = \mathbf{w}^T \mathbf{x} + b$ and pass it through the **Sigmoid function** $\sigma(z)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```
Sigmoid Function (S-Curve):
P(y=1)
  ▲
 1├───────────────────────-─-─-─-─-─-─-─  (Asymptote at 1)
  │                     . · ´
  │                  . ·
0.5├───────────────+ (z = 0)
  │            . ·
  │      . · ´
 0├-─-─-─-─-─-─-────────────────────────  (Asymptote at 0)
  └──────────────────┼─────────────────► z = w^T x + b
                    z=0
```

### Key Mathematical Properties of Sigmoid:
1. **Strictly Bounded**: For any input $z \in (-\infty, +\infty)$, $\sigma(z)$ always lies strictly inside the open interval **$(0, 1)$**.
2. **At $z = 0$**: $\sigma(0) = \frac{1}{1 + e^0} = \frac{1}{2} = \mathbf{0.5}$.
3. **Monotonically Increasing**: As $z$ increases, probability $\hat{p}$ increases smoothly.

> 💡 **Hinglish Intuition**:  
> Sigmoid ek "compressor" ki tarah hai. Linear line kitna bhi bada number de de ($+1000$ ya $-5000$), Sigmoid usko daba ke $0$ se $1$ ke beech ka genuine probability bana deta hai.

---

## 4. Converting Probabilities to Class Labels (Decision Threshold)

By default, we set a classification threshold $\tau = 0.5$:
$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\mathbf{x}) \ge 0.5 \\ 0 & \text{if } P(y=1|\mathbf{x}) < 0.5 \end{cases}$$

### When should you change the threshold?
- **Lower Threshold (e.g., 0.3)**: When missing a positive case is catastrophic (e.g., Cancer / Disease diagnosis). We prefer false alarms over letting a sick patient go home undetected (**High Recall**).
- **Higher Threshold (e.g., 0.8)**: When false alarms are annoying or expensive (e.g., marking emails as Spam or accusing someone of credit card fraud). We only flag when super confident (**High Precision**).

---

## 5. Python Scikit-Learn Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Initialize and train
clf = LogisticRegression()
clf.fit(X_train, y_train)

# 2. Predict raw probabilities [P(y=0), P(y=1)]
y_prob = clf.predict_proba(X_test)[:, 1]

# 3. Predict class labels (using default 0.5 threshold)
y_pred = clf.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 🎯 Quick B.Tech Viva Questions
1. **Q: What is the Loss function used in Logistic Regression?**  
   *A:* **Binary Cross-Entropy Loss (Log-Loss)**, not MSE. MSE makes the loss surface non-convex (trapping gradient descent in local minima), while Log-Loss is convex.
2. **Q: What is the Decision Boundary of Logistic Regression?**  
   *A:* The hyper-plane where $z = \mathbf{w}^T \mathbf{x} + b = 0$, which corresponds to probability $P(y=1) = 0.5$.

