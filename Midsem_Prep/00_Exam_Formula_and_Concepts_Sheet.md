# 🎯 CSE 3967: ML Concepts 1 — Midsem Formula & Quick Revision Sheet

> **Exam Focus**: 2-Mark Definitions, Key Formulas, Pitfalls & Hinglish Intuitions  
> **Syllabus Coverage**: Python/NumPy/Pandas Basics $\rightarrow$ Statistics $\rightarrow$ Regression $\rightarrow$ Errors & Bias-Variance $\rightarrow$ Train-Test Split $\rightarrow$ Logistic Regression

---

## 1. Classical Programming vs Machine Learning (2 Marks)

| Parameter | Classical (Traditional) Programming | Machine Learning (ML) |
|---|---|---|
| **Inputs** | Rules (Logic/Code) + Data | Data + Output (Labels / Answers) |
| **Output** | Answers / Output | Rules / Model ($y = f(x)$) |
| **Adaptability**| Rigid; programmer manually writes `if-else`. | Adaptive; learns patterns automatically from historical data. |

> 💡 **Hinglish Intuition**:  
> Traditional coding me hum bolte hain: *"Agar temperature > 30 hai toh fan on karo"* (Rules humne likhe).  
> ML me hum computer ko 10,000 din ka data aur result dete hain, computer khud formula dhoondhta hai ki kab fan on hona chahiye.

---

## 2. Descriptive Statistics & Central Tendency (2 Marks)

### Mean, Median, Mode
- **Mean ($\mu$ or $\bar{x}$)**: $\frac{\sum x_i}{N}$ — **Highly sensitive to outliers**.
- **Median**: Middle value in sorted data. If $N$ is odd: middle item. If even: average of two middle items. **Robust to outliers**.
- **Mode**: Most frequently occurring value.

> ⚠️ **Exam Trap**: *"If an extreme outlier (e.g., 100 in [35, 42, 44...]) is added, which metric changes more?"*  
> **Answer**: **Mean** changes drastically because it sums all values. **Median** changes very little because it only cares about order/position.

### Dispersion: Standard Deviation ($\sigma$) & Variance ($\sigma^2$)
$$\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{N}}$$
- **Meaning**: Standard deviation tells us how spread out the data points are around the mean.
- A **low standard deviation** means data points are clustered close to the average; a **high standard deviation** means data is widely scattered.

---

## 3. Regression Error Metrics (Formulas & Penalties) (2 Marks)

Let $e_i = y_i - \hat{y}_i$ (Error = Actual $-$ Predicted), and $n$ be the number of samples.

| Metric | Formula | Key Characteristic / Exam Answer |
|---|---|---|
| **MAE** (Mean Absolute Error) | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | Treats all errors linearly. Does not overly penalize single large mistakes. |
| **MSE** (Mean Squared Error) | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | **Penalizes large errors heavily** because errors are squared. Unit is squared (e.g., $(\text{dollars})^2$). |
| **RMSE** (Root Mean Squared Error)| $\sqrt{\text{MSE}}$ | Penalizes large errors heavily, but unit matches original target $y$. **$\text{RMSE} \ge \text{MAE}$ always.** |
| **$R^2$ Score** (Coefficient of Determination) | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ | Explains percentage of variance explained by model. $R^2 \approx 1$ means excellent fit. $R^2 = 0$ means model is no better than predicting average $\bar{y}$. |

> 💡 **Hinglish Intuition**:  
> Agar tumne 1 prediction me 10 rupaye ka mistake kiya, toh MAE me penalty = 10, par MSE me penalty = $10^2 = 100$! Isiliye **MSE aur RMSE large errors ko buri tarah punish karte hain**.

---

## 4. Sources of Model Error (2 Marks)

Total Error = **$\text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$**

1. **Sub-optimal Parameters**: Model structure is fine, but gradient descent hasn't converged or learning rate is wrong.
2. **Noisy Data**: Random human/measurement errors in target or features (Irreducible error $\epsilon$).
3. **Omitted Features (Under-specification)**: Missing crucial inputs (e.g., trying to predict house price without knowing square feet).
4. **Model Inadequacy (High Bias)**: Using a straight line for quadratic data ($y = x^2$).

---

## 5. Underfitting vs Overfitting (Bias-Variance Tradeoff) (2 Marks)

```
        High Error
           │   \                      /
           │    \   High Bias        /  High Variance
           │     \ (Underfitting)   /  (Overfitting)
           │      \                /
           │       \              /  <-- Test Error (U-shaped)
           │        \____________/
           │         \          
           │          \__________    <-- Training Error (Decreases continuously)
           │
           └───────────────────────────> Model Complexity (Degree of polynomial)
```

| State | Training Error | Test / Validation Error | Symptoms | Remedy / Fix |
|---|---|---|---|---|
| **Underfitting** (High Bias) | **High** | **High** | Model is too simple; cannot capture patterns. | Increase complexity, add polynomial features, add more features. |
| **Overfitting** (High Variance) | **Very Low** ($R^2 \approx 0.99$) | **High** ($R^2 \ll \text{Train}$) | Model memorized training noise; fails on unseen test data. | Get more data, regularization (L1/L2), reduce features, simplify model. |
| **Optimal Fit** | Low | Low & close to Train | Generalizes well to new data. | Balanced bias & variance. |

---

## 6. Model Validation & Train-Test Split (2 Marks)

### Train-Test Split Rules
- Formula: If $N = 1000$ and `test_size=0.2`, then $\text{Test samples} = 1000 \times 0.2 = 200$, $\text{Train samples} = 800$.
- **Why never evaluate ONLY on training data?**  
  A model can simply memorize training data (100% accuracy) but miserably fail in production. Test set acts as an unbiased evaluation.
- **Why never fit on test data?**  
  Causes **Data Leakage**. Model pehle se test data ke patterns dekh lega, jisse evaluation fake aur over-optimistic ho jayegi.
- **Role of `random_state`**: Sets the random seed so that the data shuffle and split are 100% reproducible across runs.
- **Limitation of single split**: The split might accidentally put all hard samples into test set (high variance in estimate). Remedy: **$K$-Fold Cross Validation**.

### Input Matrix Shape Trap (`Scikit-Learn`)
- `X` must ALWAYS be a **2D array/matrix** of shape `(n_samples, n_features)`.
- `y` is a **1D vector** of shape `(n_samples,)`.
- `df["area"]` produces a 1D Pandas Series $\rightarrow$ **Throws shape error `ValueError: Expected 2D array, got 1D array instead`**.
- **Fix**: Use `df[["area"]]` (double brackets) or `df["area"].values.reshape(-1, 1)`.

---

## 7. Classification & Logistic Regression (2 Marks)

### Regression vs Classification
- **Regression**: Predicts a continuous numerical value (e.g., House Price, Salary).
- **Classification**: Predicts a discrete categorical label (e.g., Anemic / Non-Anemic, Spam / Not Spam).

### Sigmoid Function (Logistic Function)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Input $z$**: Any real number from $-\infty$ to $+\infty$ ($z = w^T x + b$).
- **Output Range**: Strictly bounded between $(0, 1)$.
- **Purpose**: Compresses any linear prediction into a valid probability $P(y=1|x)$.
- **Decision Threshold (Default = 0.5)**:
  - If $\hat{P} \ge 0.5 \implies \text{Class } 1$
  - If $\hat{P} < 0.5 \implies \text{Class } 0$
- **Why Linear Regression Fails for Classification**:
  1. Linear regression outputs $(-\infty, +\infty)$, which can be $< 0$ or $> 1$ (invalid probabilities).
  2. Highly sensitive to extreme outliers that shift the decision boundary incorrectly.

