# 📝 Midsem Sample Questions 1 to 9 (Fully Solved)

> **Format**: Question $\rightarrow$ **Simple English 2-Mark Solution** $\rightarrow$ **Code / Math Steps** $\rightarrow$ **Hinglish Concept Breakdown**

---

## Question 1

### (a) Difference between Classical Programming and Machine Learning
- **Classical Programming**: The human programmer writes explicit rules and logic ($Rules + Data \rightarrow Output$).
- **Machine Learning**: The algorithm automatically learns the mapping logic/rules from historical examples ($Data + Output \rightarrow Model/Rules$).
- **Role of Historical Data in Prediction**: Historical data provides past feature-outcome pairs. By identifying patterns across these examples, the model estimates a function $\hat{y} = f(x)$ that generalizes to unseen future inputs.
- **Real-World Example**: 
  - *Spam Filter*: In classical coding, you write thousands of manual keyword rules (`if "lottery" in email: spam`). In ML, you feed 50,000 past labeled emails; the model automatically learns probabilistic word associations to classify new incoming emails.

> 💡 **Hinglish Intuition**:  
> Classical me programmer "cook" hai jo recipe (rules) khud likhta hai. ML me computer pehle se bane 1000 dishes (data) dekh ke khud recipe create kar leta hai.

---

### (b) NumPy Column Mean & Column Maximum
```python
import numpy as np

data = np.array([
    [10, -20, 30],
    [40, -50, 60],
    [70,  80, -90]
])

# Column-wise operations require axis=0 (along vertical rows)
col_mean = np.mean(data, axis=0)
col_max = np.max(data, axis=0)

print("Column Mean:", col_mean)  # [40.0, 3.33333333, 0.0]
print("Column Max:", col_max)    # [70, 80, 60]
```

> ⚠️ **Exam Trap**: `axis=0` means column-wise (vertical collapse), `axis=1` means row-wise (horizontal collapse). Teachers check this axis value!

---

### (c) Random Number Generation & Random Seed
```python
# 1. Set random seed for reproducibility
np.random.seed(42)

# 2. 50 samples from Normal Distribution (mean=0, std=1)
# Parameters: loc=mean, scale=standard deviation, size=sample count
normal_samples = np.random.normal(loc=0.0, scale=1.0, size=50)

# 3. 50 samples from Uniform Distribution (range [0, 1))
# Parameters: low=minimum, high=maximum, size=sample count
uniform_samples = np.random.uniform(low=0.0, high=1.0, size=50)
```
- **Role of Random Seed (`np.random.seed`)**: Pseudo-random number generators use mathematical algorithms starting from a base seed. Fixing the seed (e.g., `42`) ensures the exact same sequence of random numbers is generated every time the script runs, making experiments and splits reproducible.

---

## Question 2

### (a) Count Null Values in Every Column
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Marks': [80, None, 70, None],
    'Age': [20, 21, None, 22]
})

null_counts = df.isnull().sum()
print(null_counts)
# Name     0
# Marks    2
# Age      1
```

---

### (b) Impute Missing `Marks` with Mean of Non-Missing Values
```python
marks_mean = df['Marks'].mean()  # Automatically ignores nulls: (80 + 70)/2 = 75.0
df['Marks'] = df['Marks'].fillna(marks_mean)
```

---

### (c) Drop Missing Rows & Trade-off between Deletion and Imputation
```python
df_complete = df.dropna()
```
- **Trade-off**:
  - **Deletion (`dropna`)**:
    - *Advantage*: Ensures data integrity; avoids introducing artificial or distorted values.
    - *Disadvantage*: Loss of valuable sample volume; can introduce severe selection bias if data is not missing completely at random.
  - **Imputation (`fillna` with mean/median)**:
    - *Advantage*: Preserves full dataset sample size and statistical power.
    - *Disadvantage*: Reduces variance artificially, shrinks standard deviation, and may distort true correlations between features.

---

## Question 3

### (a) Mean, Median, and Mode of Daily Customer Counts
Given observations: $[42, 35, 51, 48, 35, 44, 60]$  
Number of observations $n = 7$.

1. **Mean ($\bar{x}$)**:
   $$\bar{x} = \frac{42 + 35 + 51 + 48 + 35 + 44 + 60}{7} = \frac{315}{7} = 45.0$$
2. **Median**:
   - Sort observations in ascending order: $[35, 35, 42, 44, 48, 51, 60]$
   - Position = $\frac{n+1}{2} = \frac{7+1}{2} = 4\text{th value}$.
   - **Median = 44**.
3. **Mode**:
   - The value that appears with the highest frequency is **35** (appears twice; all others appear once).  
   - **Mode = 35**.

---

### (b) Effect of Adding New Outlier Value $100$
- New data: $[35, 35, 42, 44, 48, 51, 60, 100]$ ($n = 8$).
- **Mean is affected much more than Median**.
  - *Calculation*: New Mean = $\frac{315 + 100}{8} = \frac{415}{8} = 51.875$ (jumped by $+6.875$).
  - New Median = Average of 4th and 5th values $\frac{44 + 48}{2} = 46$ (jumped by only $+2.0$).
- *Reason*: Mean is calculated using the algebraic sum of every single data point, making it non-resistant to extreme values. Median depends only on ordinal positional ranking.

---

### (c) What Standard Deviation Tells Us About Dispersion
- Standard deviation ($\sigma$) quantifies the average spread or variability of individual observations around the mean.
- A **small standard deviation** indicates that data points are clustered closely around the mean (high consistency / low dispersion).
- A **large standard deviation** indicates wide dispersion and high variability across observations.

---

## Question 4

### (a) & (b) Matplotlib Line Plot and Annotated Bar Chart
```python
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
consumption = [210, 230, 195, 250, 280, 265]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Line Plot
ax1.plot(months, consumption, marker='o', color='b', linestyle='-')
ax1.set_title('Monthly Electricity Consumption Trend')
ax1.set_xlabel('Month')
ax1.set_ylabel('Consumption (kWh)')
ax1.grid(True)

# Bar Chart with Annotations
bars = ax2.bar(months, consumption, color='teal')
ax2.set_title('Monthly Electricity Consumption Comparison')
ax2.set_xlabel('Month')
ax2.set_ylabel('Consumption (kWh)')

# Annotating each bar with value
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 3, str(yval), ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

---

### (c) Histogram vs Line Plot Information
```python
plt.hist(consumption, bins=4, edgecolor='black')
plt.title('Distribution of Electricity Consumption')
plt.xlabel('Consumption Range')
plt.ylabel('Frequency')
plt.show()
```
- **Line plot** displays **temporal trends and continuity** (how consumption fluctuates chronologically over time).
- **Histogram** reveals the **underlying probability distribution, frequency count, and density** (e.g., how many months consumption fell in 200–250 vs 250–300, and whether data is skewed or multimodal), completely disregarding time order.

---

## Question 5

### (a) Calculate Correlations with Target `distance`
```python
correlation_matrix = df[['acc', 'time', 'distance']].corr()
correlations_with_target = correlation_matrix['distance'].drop('distance')
print(correlations_with_target)
```

---

### (b) Visualize Distance against Acc and Time (1×2 Grid)
```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(df['acc'], df['distance'], color='blue')
axes[0].set_title('Distance vs Acceleration')
axes[0].set_xlabel('acc')
axes[0].set_ylabel('distance')

axes[1].scatter(df['time'], df['distance'], color='green')
axes[1].set_title('Distance vs Time')
axes[1].set_xlabel('time')
axes[1].set_ylabel('distance')

plt.tight_layout()
plt.show()
```

---

### (c) Why Strong Correlation Does Not Guarantee Predictive Usefulness
1. **Correlation Measures Only Linear Association**: Pearson correlation measures strict linear relationships. If the underlying relationship is non-linear (e.g., $y = x^2$ or sinusoidal), the linear correlation may be near 0 even though predictability is 100%, and vice-versa.
2. **Confounding Variables & Spurious Correlation**: Two variables may show strong correlation due to a lurking third variable or random chance without having any causal or predictive mechanism.
3. **Multicollinearity**: If two features are both highly correlated with the target and with each other, adding the second feature provides zero new predictive information.

---

## Question 6

### (a) Load, Inspect, and Feature Selection
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('housing_data.csv')
print("Shape of dataset:", df.shape)
print("Missing values:\n", df.isnull().sum())

# Must select X as 2D DataFrame: df[['area_sqft']]
X = df[['area_sqft']]
y = df['price']
```

---

### (b) Train-Test Split (80/20) and Linear Regression Prediction
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

### (c) Why Test Set Must NEVER Be Used to Fit the Model
- Fitting on the test set causes **Data Leakage (Snooping Bias)**.
- If the model learns parameters from the test set, the evaluation score becomes unrealistically inflated and biased because the model is being tested on data it has already seen. The test set must remain unseen to measure **true generalization performance on real-world data**.

---

## Question 7

### (a) & (b) Step-by-Step Calculation of MAE, MSE, and RMSE
Given:
- Actual: $y = [10, 20, 30, 40, 50]$
- Predicted: $\hat{y} = [12, 18, 29, 42, 48]$
- Number of observations: $n = 5$

| Observation $i$ | $y_i$ | $\hat{y}_i$ | Error $e_i = (y_i - \hat{y}_i)$ | Absolute Error $|e_i|$ | Squared Error $(e_i)^2$ |
|---|---|---|---|---|---|
| 1 | 10 | 12 | $-2$ | 2 | 4 |
| 2 | 20 | 18 | $+2$ | 2 | 4 |
| 3 | 30 | 29 | $+1$ | 1 | 1 |
| 4 | 40 | 42 | $-2$ | 2 | 4 |
| 5 | 50 | 48 | $+2$ | 2 | 4 |
| **Sum** | | | | **$\sum = 9$** | **$\sum = 17$** |

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i| = \frac{9}{5} = \mathbf{1.80}$$
2. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2 = \frac{17}{5} = \mathbf{3.40}$$
3. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{3.40} \approx \mathbf{1.8439}$$

---

### (c) Scikit-Learn Code, Penalty Comparison, and $R^2$ Interpretation
```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 18, 29, 42, 48])

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
```
- **Which penalizes large errors more strongly?**
  - **MSE and RMSE** penalize large errors much more aggressively than MAE because individual residual terms are squared ($e_i^2$). For instance, an error of 10 adds 100 to MSE, whereas an error of 2 adds only 4.
- **Interpretation of $R^2$ close to 1**:
  - $R^2$ represents the proportion of total variance in the dependent variable explained by the model:
    $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$
  - An $R^2$ close to $1$ (e.g., $0.98$) indicates that the model fits the data exceptionally well, and nearly all variations in the target are accurately captured by the input features.

---

## Question 8

### (a) Polynomial Calculation & Linear Regression with Polynomial Features
Given relationship:
$$y = 2 + 3x + 0.5x^2$$

Calculating $y$ for $x \in \{1, 2, 3, 4\}$:
- For $x = 1$: $y = 2 + 3(1) + 0.5(1)^2 = 2 + 3 + 0.5 = \mathbf{5.5}$
- For $x = 2$: $y = 2 + 3(2) + 0.5(4) = 2 + 6 + 2.0 = \mathbf{10.0}$
- For $x = 3$: $y = 2 + 3(3) + 0.5(9) = 2 + 9 + 4.5 = \mathbf{15.5}$
- For $x = 4$: $y = 2 + 3(4) + 0.5(16) = 2 + 12 + 8.0 = \mathbf{22.0}$

**Why Linear Regression can be used with polynomial features:**
- Linearity in Linear Regression refers to **linearity in the parameters/coefficients ($\beta_0, \beta_1, \beta_2$)**, NOT linearity in the features $x$.
- By substituting $z_1 = x$ and $z_2 = x^2$, the model becomes:
  $$y = \beta_0 + \beta_1 z_1 + \beta_2 z_2$$
- This is a standard multiple linear regression equation in the new transformed feature space.

---

### (b) Scikit-Learn Code for Polynomial Regression (Degree 2)
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4]])
y = np.array([5.5, 10.0, 15.5, 22.0])

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

print("Coefficients:", model.coef_)       # Output: [3. , 0.5]
print("Intercept:", model.intercept_)     # Output: 2.0
```

---

### (c) Model Complexity, Fit, and Generalization
- **Low Complexity** (e.g., linear line for non-linear data): Causes **High Bias / Underfitting**. The model fails to capture underlying patterns, resulting in high errors on both training and test data.
- **Optimal Complexity**: Achieves balanced bias and variance, minimizing test generalization error.
- **High Complexity** (e.g., degree 15 polynomial): Causes **High Variance / Overfitting**. The model fits noise and random variations in the training set perfectly ($R^2 \approx 1$), but fluctuates wildly on unseen data, resulting in poor generalization.

---

## Question 9

### (a) & (b) Feature Engineering: Average Feature
```python
import pandas as pd

# Assume df contains 'Temperature' and 'Humidity'
df['Statistical_Mean'] = (df['Temperature'] + df['Humidity']) / 2.0
```

---

### (c) Feature Ranking by Absolute Correlation & Pros/Cons
```python
# Ranking input features by absolute correlation with distance
corr = df[['acc', 'time', 'distance']].corr()['distance'].abs().drop('distance')
ranked_features = corr.sort_values(ascending=False)
print("Ranked Features by Absolute Correlation:\n", ranked_features)
```
- **One Benefit of Feature Engineering**:
  - Exposes non-linear relationships and domain knowledge directly to simple linear models, drastically boosting model accuracy and explanatory power.
- **One Risk of Feature Engineering**:
  - Risk of **Multicollinearity and Curse of Dimensionality / Overfitting**: Creating excessive redundant or noisy interaction features increases model variance and degrades performance on unseen test data.

