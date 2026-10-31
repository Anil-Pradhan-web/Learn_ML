# Chapter 4: Non-Linear Models & Feature Engineering

> **Why this matters for ML**: Real-world relationships are rarely straight lines. When data curves, bends, or accelerates, a simple straight line fails. Feature engineering and non-linear transforms allow linear algorithms to learn curved patterns.

---

## 1. Why Straight Lines Fail (Non-Linearity)

Many real-world physics and economic processes have accelerating or diminishing curves:
- Braking distance of a car is proportional to velocity squared: $d \propto v^2$.
- Salary vs Experience grows fast initially, then plateaus.

If you fit a straight line ($y = mx + c$) to curved data, you get **High Bias (Underfitting)**.

---

## 2. Polynomial Regression

### The Core Concept (Simple English)
Polynomial regression models a non-linear relationship by creating higher-order polynomial powers of the original features:
$$y = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \dots + w_d x^d$$
where $d$ is the **degree of the polynomial**.

### ⚠️ The #1 Exam Trap: "Why is it still called Linear Regression?"
- **Answer**: Linearity in machine learning refers to **linearity in the parameters/coefficients ($w_0, w_1, w_2$)**, NOT linearity in the feature variables $x$.
- If we substitute $z_1 = x, z_2 = x^2, z_3 = x^3$, the equation becomes:
  $$y = w_0 + w_1 z_1 + w_2 z_2 + w_3 z_3$$
- This is mathematically identical to a standard **Multiple Linear Regression** equation in a new transformed feature space!

> 💡 **Hinglish Intuition**:  
> Humne algorithm ko change nahi kiya! Humne feature ki power calculate karke ek naya column bana diya ($x^2$). Algorithm ke liye wo bas ek doosra feature column hai, isiliye Scikit-learn ka standard `LinearRegression()` hi use hota hai.

---

## 3. Feature Engineering Techniques

Feature engineering is the process of creating new informative features from raw data using domain knowledge.

| Technique | Example | Why do it? |
|---|---|---|
| **Ratios / Interaction** | $\text{Price per Sqft} = \frac{\text{Price}}{\text{Area}}$ | Captures relative value rather than absolute scale. |
| **Statistical Aggregation**| $\text{Mean Temp} = \frac{\text{Morning Temp} + \text{Evening Temp}}{2}$ | Smooths out measurement noise. |
| **Date/Time Extraction** | Extract `DayOfWeek` or `Is_Weekend` from `Timestamp` | Exposes periodic human behavior patterns to the model. |

### Benefits vs Risks (Exam Question)
- **Benefit**: Exposes non-linear patterns directly to simple models without needing complex deep learning.
- **Risk**: Creating too many redundant features causes **Multicollinearity**, slows down computation, and causes **Overfitting**.

---

## 4. Scikit-Learn Code Implementation

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4]])
y = np.array([5.5, 10.0, 15.5, 22.0])

# 1. Transform features to degree 2 (adds x and x^2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# 2. Fit standard Linear Regression on transformed features
model = LinearRegression()
model.fit(X_poly, y)

print("Coefficients [w1, w2]:", model.coef_)       # Output: [3. , 0.5]
print("Intercept [w0]:", model.intercept_)         # Output: 2.0
# Equation: y = 2.0 + 3.0*x + 0.5*x^2
```

---

## 🎯 Quick B.Tech Viva Questions
1. **Q: What happens if you set polynomial degree too high (e.g., degree = 20)?**  
   *A:* The model fits every single training point perfectly but wiggles wildly between points, causing severe **Overfitting**.
2. **Q: Does `PolynomialFeatures(degree=2)` create interaction terms between multiple features?**  
   *A:* Yes. For two features $x_1$ and $x_2$, it generates $[x_1, x_2, x_1^2, x_1 x_2, x_2^2]$.

