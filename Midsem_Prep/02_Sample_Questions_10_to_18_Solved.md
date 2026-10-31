# 📝 Midsem Sample Questions 10 to 18 (Fully Solved)

> **Format**: Question $\rightarrow$ **Simple English 2-Mark Solution** $\rightarrow$ **Calculations / Code** $\rightarrow$ **Hinglish Concept Breakdown**

---

## Question 10

### (a) Cause of Input-Shape Error in `LinearRegression().fit()`
- **Cause**: In Scikit-Learn, all estimators expect the feature matrix `X` to be a **2D array/matrix** of shape `(n_samples, n_features)`.
- When passing `X = df["area"]`, Pandas returns a **1D Series** of shape `(n_samples,)`. Scikit-learn raises:
  `ValueError: Expected 2D array, got 1D array instead`.

---

### (b) Corrected Statement & Expected Shape
```python
# Correction Option 1 (Double brackets to keep 2D DataFrame):
X = df[["area"]]

# Correction Option 2 (NumPy reshape):
X = df["area"].values.reshape(-1, 1)
```
- **Expected Shape**: For a single feature across $N$ samples, the expected shape is `(N, 1)` — meaning $N$ rows and $1$ column.

---

### (c) Two Essential Pre-Fitting Checks
1. **Missing Value Check (`df.isnull().sum()`)**: Scikit-Learn's `LinearRegression` crashes on `NaN`/missing values; data must be imputed or dropped beforehand.
2. **Data Type & Encoding Check (`df.dtypes`)**: Ensure all features are strictly numeric (`float64` or `int64`). Categorical columns must be encoded (One-Hot or Label encoded).

---

## Question 11

### (a) Distinguishing the 4 Sources of Prediction Errors
1. **Sub-optimal Parameter Values**: The model structure is suitable, but training was terminated prematurely, learning rate was unsuitable, or optimization was stuck in a local minimum.
2. **Noisy Data (Irreducible Error $\epsilon$)**: Random measurement errors, sensor inaccuracies, or unpredictable human behaviors inherent in the data that no model can predict.
3. **Omitted Features (Under-specification)**: The dataset lacks critical variables required to explain the target (e.g., predicting house prices without square footage or location).
4. **Model Inadequacy (High Bias / Capacity mismatch)**: The chosen algorithm lacks the capacity to represent the underlying reality (e.g., fitting a straight line to sinusoidal or quadratic data).

---

### (b) Practical Investigation and Remedies
- **Example**: Predicting car prices using only `Color`. The model has high error.
- **Approaches to Reduce Errors**:
  - *For Omitted Features*: Perform feature collection to include `Mileage`, `Engine_CC`, `Year`.
  - *For Model Inadequacy*: Try non-linear models, polynomial features, or tree-based algorithms.
  - *For Sub-optimal Parameters*: Use hyperparameter tuning (`GridSearchCV`) and train with more iterations.
  - *For Noise*: Apply data cleaning, remove extreme bad sensor recordings.

---

### (c) Role of Bias and Variance & Diagnostic Comparison
- **Bias**: Error due to overly simplistic assumptions in the learning algorithm.
- **Variance**: Error due to excessive sensitivity to small fluctuations/noise in the training dataset.

**Diagnostic Table**:
| Observation | Diagnosis |
|---|---|
| **High Training Error + High Test Error** | **Underfitting (High Bias)** |
| **Very Low Training Error + High Test Error** | **Overfitting (High Variance)** |
| **Low Training Error + Low Test Error** (Close to each other) | **Good Generalization (Balanced)** |

---

## Question 12

### (a) Bias-Variance Tradeoff Graph & Complexity Relationship
```
  Error
    ▲
    │ \                          /
    │  \   High Bias            /  High Variance
    │   \ (Underfitting)       /  (Overfitting)
    │    \                    /
    │     \                  /  <── Total Test Error (U-shaped)
    │      \                /
    │       \──────────────/
    │        \
    │         \───────────────── <── Training Error
    └─────────────────────────────► Model Complexity
```
- **Relationship**:
  - As **Model Complexity increases**: **Bias decreases**, but **Variance increases**.
  - **Generalization Error** is U-shaped: it reaches a minimum at the optimal complexity where $\text{Bias}^2 + \text{Variance}$ is minimized.

---

### (b) Model Comparison Table Analysis

| Model | Training Error | Test Error | Gap (Test - Train) |
|---|---|---|---|
| **A** | 25 | 28 | 3 |
| **B** | 5 | 7 | 2 |
| **C** | 1 | 35 | 34 |

- **(i) Model suffering from Underfitting**: **Model A**  
  *Justification*: Both training error (25) and test error (28) are substantially high. The model is too simple to capture the underlying pattern.
- **(ii) Model suffering from Overfitting**: **Model C**  
  *Justification*: Training error is nearly zero (1), but test error is huge (35). The massive divergence proves the model memorized the training data and completely failed to generalize.
- **(iii) Best Generalization Performance**: **Model B**  
  *Justification*: Lowest overall test error (7) with a minimal gap between training (5) and test (7), demonstrating high accuracy and excellent generalization.

---

## Question 13

### (a) Overfitting Evidence: Model A vs Model B
- **Model A**: $\text{Train } R^2 = 0.62$, $\text{Test } R^2 = 0.60$ (Gap = $0.02$)
- **Model B**: $\text{Train } R^2 = 0.99$, $\text{Test } R^2 = 0.51$ (Gap = $0.48$)
- **Answer**: **Model B shows extreme evidence of overfitting**.  
  *Justification*: A training $R^2$ of $0.99$ indicates nearly perfect memorization of the training set, but its test $R^2$ plummets to $0.51$.

---

### (b) Underfitting and Generalization in this Context
- **Model A** suffers from mild **underfitting**: It generalizes stably ($\text{Train} \approx \text{Test}$), but its overall capacity is inadequate ($\text{Score} \approx 0.60$).
- **Generalization** is the ability of a model to maintain consistent prediction accuracy on new, unseen test data.

---

### (c) One Possible Intervention for Model B
- **Intervention**: Apply **Regularization (Ridge $L_2$ or Lasso $L_1$)** or **reduce polynomial degree / features**.
- **Why it helps**: Regularization adds a penalty term ($\lambda \sum w_j^2$) to the loss function, shrinking large regression coefficients towards zero, which prevents the model from fitting high-frequency noise.

---

## Question 14

### (a) Train-Test Sample Count Calculation
- Total Samples $N = 1000$
- Split: `test_size = 0.2` (20%)
- **Test Samples** = $1000 \times 0.2 = \mathbf{200\text{ samples}}$
- **Training Samples** = $1000 - 200 = \mathbf{800\text{ samples}}$

---

### (b) Purpose of `random_state` & Why Training-Only Evaluation Fails
- **Purpose of `random_state`**: Sets a deterministic seed for the pseudo-random shuffling algorithm, ensuring that the exact same data split is produced every time the script runs (100% reproducibility).
- **Why Training-Only Evaluation is Unreliable**: An overfitted model can memorize training data like a lookup table (giving 100% training score), giving a completely false illusion of performance while performing catastrophically on new data.

---

### (c) Limitation of a Single Split & $K$-Fold Cross-Validation
- **Limitation**: A single split is subject to **sampling variance**; critical edge cases or patterns might accidentally end up exclusively in the test set or exclusively in the train set.
- **How $K$-Fold CV helps**: It splits the data into $K$ equal subsets (folds). The model is trained on $K-1$ folds and tested on the remaining fold, rotating $K$ times. Every single sample is used for testing exactly once, yielding an average performance metric with low variance.

---

## Question 15

### (a) Regression vs Classification
- **Regression**: The target variable is continuous and numeric.
  - *Example*: Predicting house price in lakhs (e.g., 45.5, 78.2) or stock price.
- **Classification**: The target variable is categorical/discrete.
  - *Example*: Predicting email as `Spam` vs `Not Spam`, or patient as `Anemic` vs `Non-Anemic`.

---

### (b) Converting Class Probabilities to Class Labels
- Class probabilities are real numbers $P \in [0, 1]$.
- A **decision threshold** $\tau$ (default $\tau = 0.5$) is applied:
  $$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \ge \tau \\ 0 & \text{if } P(y=1|x) < \tau \end{cases}$$
- *Example*: If model predicts probability of disease $P = 0.72$, since $0.72 \ge 0.5$, label is assigned as **Class 1 (Positive)**. If $P = 0.35 < 0.5$, label is **Class 0 (Negative)**.

---

### (c) Purpose & Output Range of Sigmoid Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Purpose**: Maps any arbitrary real-valued number $z = w^T x + b \in (-\infty, +\infty)$ into a valid probability.
- **Output Range**: Strictly bounded in the open interval **$(0, 1)$**.
- At $z = 0$, $\sigma(0) = \frac{1}{1 + 1} = 0.5$.
- As $z \rightarrow +\infty$, $\sigma(z) \rightarrow 1$; as $z \rightarrow -\infty$, $\sigma(z) \rightarrow 0$.

---

## Question 16

### (a) "Classification is Unsupervised Learning" — True or False?
- **FALSE**.
- **Justification**: Classification is a core **Supervised Learning** task because it requires ground-truth labeled training data $(x_i, y_i)$ where the correct class target $y_i$ is known and provided to the model during training.

---

### (b) Anemia Healthcare Problem Formulation
- **Input Features ($X$)**:
  1. `Hemoglobin level` (continuous numeric)
  2. `Red Blood Cell (RBC) count` (continuous numeric)
- **Target Variable ($y$)**: `Anemia Status` ($\{0: \text{Non-Anemic}, 1: \text{Anemic}\}$)
- **Justification as Supervised Binary Classification**:
  - *Supervised*: Ground-truth diagnoses are known for patient training records.
  - *Binary*: The outcome consists of exactly two mutually exclusive classes (Anemic vs Non-Anemic).

---

### (c) Python Code to Generate Synthetic Anemia Dataset
```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_per_class = 200

# 1. Non-Anemic class (Label = 0)
# Hb: 13-17 g/dL, RBC: 4.5-6.0 million cells/uL
hb_non = np.random.uniform(low=13.0, high=17.0, size=n_per_class)
rbc_non = np.random.uniform(low=4.5, high=6.0, size=n_per_class)
labels_non = np.zeros(n_per_class, dtype=int)

# 2. Anemic class (Label = 1)
# Hb: 7-11 g/dL, RBC: 2.5-4.0 million cells/uL
hb_anemic = np.random.uniform(low=7.0, high=11.0, size=n_per_class)
rbc_anemic = np.random.uniform(low=2.5, high=4.0, size=n_per_class)
labels_anemic = np.ones(n_per_class, dtype=int)

# Concatenate and build DataFrame
df_anemia = pd.DataFrame({
    'Hemoglobin': np.concatenate([hb_non, hb_anemic]),
    'RBC_count': np.concatenate([rbc_non, rbc_anemic]),
    'Target': np.concatenate([labels_non, labels_anemic])
})

# Shuffle dataset
df_anemia = df_anemia.sample(frac=1.0, random_state=42).reset_index(drop=True)
print("Synthetic Dataset Created. Shape:", df_anemia.shape)
print(df_anemia.head())
```

---

## Question 17

### (a) Working Principle of Logistic Regression
- Logistic Regression computes a linear combination of input features:
  $$z = w_1 x_1 + w_2 x_2 + \dots + w_p x_p + b$$
- It then passes $z$ through the non-linear **sigmoid activation function** to predict the probability $\hat{p} = \sigma(z) = P(y=1|x)$.
- A decision boundary separates the feature space where $\sigma(z) = 0.5$ (i.e., $z = 0$).

---

### (b) Mathematical Expression & Thresholding
- **Expression**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Range**: $(0, 1)$
- **Threshold Rule**:
  $$\hat{y} = 1 \text{ if } \hat{p} \ge 0.5, \quad \text{else } 0$$

---

### (c) Why Sigmoid is More Appropriate than Linear Regression
1. **Bounded Output**: Probabilities must mathematically lie in $[0, 1]$. Linear regression predicts $(-\infty, +\infty)$, yielding invalid probabilities like $-0.3$ or $1.4$.
2. **Robustness to Extreme Outliers**: In linear regression, distant outlier points pull the regression line heavily, severely displacing the decision boundary. Sigmoid saturates at $0$ and $1$, preventing extreme values from distorting the boundary.

---

## Question 18

### (a) & (b) Converting Probabilities, Comparison, and Accuracy Calculation
Given:
- Actual Class Labels: $y = [1, 0, 1, 1, 0]$
- Predicted Probabilities: $\hat{p} = [0.85, 0.30, 0.65, 0.40, 0.20]$
- Classification Threshold: $\tau = 0.5$

#### Step 1: Convert Probabilities to Class Labels
- For sample 1: $0.85 \ge 0.5 \implies \mathbf{1}$
- For sample 2: $0.30 < 0.5 \implies \mathbf{0}$
- For sample 3: $0.65 \ge 0.5 \implies \mathbf{1}$
- For sample 4: $0.40 < 0.5 \implies \mathbf{0}$
- For sample 5: $0.20 < 0.5 \implies \mathbf{0}$

**Predicted Labels ($\hat{y}$)**: $[1, 0, 1, 0, 0]$

#### Step 2: Comparison & Accuracy Calculation
| Sample $i$ | Actual $y_i$ | Predicted Probability $\hat{p}_i$ | Predicted Label $\hat{y}_i$ | Status |
|---|---|---|---|---|
| 1 | 1 | 0.85 | 1 | **Correct** |
| 2 | 0 | 0.30 | 0 | **Correct** |
| 3 | 1 | 0.65 | 1 | **Correct** |
| 4 | 1 | 0.40 | 0 | **Incorrect** (False Negative) |
| 5 | 0 | 0.20 | 0 | **Correct** |

- Total samples $N = 5$.
- Correctly classified = 4 (Samples 1, 2, 3, 5).
- Incorrectly classified = 1 (Sample 4).
$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Samples}} = \frac{4}{5} = \mathbf{0.80 \ (80\%)}$$

---

### (c) Influence of Changing Classification Threshold
- **Lowering the Threshold (e.g., from 0.5 to 0.3)**:
  - More samples are predicted as Class 1 (Positive).
  - **Increases Recall (Sensitivity)** and decreases False Negatives.
  - *Use-case*: Medical diagnosis (Anemia / Cancer detection), where missing a sick patient (False Negative) is very dangerous.
- **Raising the Threshold (e.g., from 0.5 to 0.7)**:
  - Fewer samples are predicted as Class 1; only high-confidence predictions become 1.
  - **Increases Precision** and decreases False Positives.
  - *Use-case*: Spam detection or fraud alerts where false alarms disrupt legitimate users.

