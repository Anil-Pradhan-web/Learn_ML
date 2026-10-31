# Chapter 3: Prediction Problems Using Linear Regression

> **Why this matters for ML**: Linear Regression is the "Hello World" of Machine Learning. It models the simplest possible relationship: a straight-line trend between input features and a continuous numeric target.

---

## 1. Problem Formulation

### Core Concept (Simple English)
Given $p$ independent features $\mathbf{x} = [x_1, x_2, \dots, x_p]^T$ and one continuous dependent target $y$, linear regression assumes that $y$ is linearly related to the features plus random noise $\epsilon$:
$$y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p + \epsilon$$
- **Simple Linear Regression (1 Feature)**: $y = mx + c$
  - $m$ (or $w_1$): **Slope / Weight** (how much $y$ changes when $x$ increases by 1 unit).
  - $c$ (or $w_0$): **Intercept / Bias** (value of $y$ when $x = 0$).

> 💡 **Hinglish Intuition**:  
> Agar study hours $x$ hain aur exam marks $y$ hain:  
> $\text{Marks} = 6.5 \times \text{Hours} + 35$  
> Yahan $35$ base marks hain jo bina padhe bhi milte (Intercept), aur har 1 ghanta padhne se $6.5$ marks badhte hain (Slope).

---

## 2. The Cost Function (How We Measure Error)

For each sample $i$, the model makes a prediction $\hat{y}_i$. The difference is the residual error $e_i = y_i - \hat{y}_i$.  
We use the **Mean Squared Error (MSE)** as the cost function $J(w)$:
$$J(w) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

### Why square the errors?
1. Prevents negative and positive errors from canceling each other out (e.g., $+5$ and $-5$ summing to $0$).
2. Heavily penalizes large, catastrophic prediction errors.

---

## 3. Finding the Best-Fit Line (Two Approaches)

1. **Ordinary Least Squares (OLS / Closed-form Analytical Solution)**:
   For simple regression:
   $$m = \frac{\text{Cov}(x, y)}{\text{Var}(x)} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad c = \bar{y} - m\bar{x}$$
   In matrix form for multiple features: $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$ (Normal Equation).
2. **Gradient Descent (Iterative Optimization)**:
   Takes small steps proportional to negative gradient:
   $$w_j \leftarrow w_j - \alpha \frac{\partial J}{\partial w_j}$$
   where $\alpha$ is the **learning rate**.

---

## 4. Evaluation Metrics (Must-Know for Exams)

| Metric | Formula | What It Means |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | Average absolute magnitude of errors (linear scale). |
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | **Penalizes large errors heavily** (unit is squared). |
| **RMSE** | $\sqrt{\text{MSE}}$ | Penalizes large errors heavily, but unit is back in original scale of $y$. ($\text{RMSE} \ge \text{MAE}$). |
| **$R^2$ Score** | $1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ | **Coefficient of Determination**. Percentage of variance explained by model ($0 \le R^2 \le 1$). |

---

## 5. Python Scikit-Learn Implementation

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# X MUST be a 2D array/DataFrame, y is 1D series
X = df[['Area_sqft']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

y_pred = model.predict(X)
print("MAE:", mean_absolute_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))
```

---

## 🎯 Quick B.Tech Viva Questions
1. **Q: Why is RMSE always greater than or equal to MAE?**  
   *A:* Because RMSE squares errors before averaging. By Jensen's Inequality, squaring gives higher weight to larger errors, making $\text{RMSE} \ge \text{MAE}$.
2. **Q: What does $R^2 = 0$ mean?**  
   *A:* The model is performing no better than simply predicting the average $\bar{y}$ for every sample.

