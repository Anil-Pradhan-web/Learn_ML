# Machine Learning Concepts 1 (CSE3967)
## Assignment-1: Machine Learning Basics (Complete Solutions)

---

# Part 1: Hands-On Exercises

---

### Exercise 1: NumPy 2D Array Operations
**Given:**
$$\text{data} = \begin{bmatrix} 10 & -20 & 30 \\ 40 & -50 & 60 \\ 70 & 80 & -90 \end{bmatrix}$$

#### Python Code:
```python
import numpy as np

data = np.array([
    [10, -20, 30],
    [40, -50, 60],
    [70,  80, -90]
])

print("Mean of each row:", np.mean(data, axis=1))
print("Mean of each column:", np.mean(data, axis=0))
print("Standard deviation of each row:", np.std(data, axis=1))
print("Maximum value of each column:", np.max(data, axis=0))
print("Minimum value of each row:", np.min(data, axis=1))
```

#### Outputs:
1. **Mean of each row (`axis=1`)**:
   - Row 1: $\frac{10 + (-20) + 30}{3} = \mathbf{6.67}$
   - Row 2: $\frac{40 + (-50) + 60}{3} = \mathbf{16.67}$
   - Row 3: $\frac{70 + 80 + (-90)}{3} = \mathbf{20.00}$
   - Vector: `[6.67, 16.67, 20.00]`
2. **Mean of each column (`axis=0`)**:
   - Col 1: $\frac{10 + 40 + 70}{3} = \mathbf{40.00}$
   - Col 2: $\frac{-20 + (-50) + 80}{3} = \mathbf{3.33}$
   - Col 3: $\frac{30 + 60 + (-90)}{3} = \mathbf{0.00}$
   - Vector: `[40.00, 3.33, 0.00]`
3. **Standard deviation of each row (`axis=1`)**:
   - Row 1: $\mathbf{20.55}$
   - Row 2: $\mathbf{47.84}$
   - Row 3: $\mathbf{77.89}$
4. **Maximum value of each column (`axis=0`)**: `[70, 80, 60]`
5. **Minimum value of each row (`axis=1`)**: `[-20, -50, -90]`

---

### Exercise 2: Matrix Operations
**Given:**
$$A = \begin{bmatrix} 2 & -4 \\ -6 & 8 \end{bmatrix}, \quad B = \begin{bmatrix} 1 & -3 \\ -5 & -7 \end{bmatrix}$$

#### Python Code:
```python
A = np.array([[2, -4], [-6, 8]])
B = np.array([[1, -3], [-5, -7]])

print("A + B:\n", A + B)
print("A - B:\n", A - B)
print("A . B:\n", np.dot(A, B))
print("A^T:\n", A.T)
print("det(A):", np.linalg.det(A))
print("A^-1:\n", np.linalg.inv(A))
```

#### Results & Step-by-Step Math:
1. **$A + B$**:
   $$\begin{bmatrix} 2+1 & -4+(-3) \\ -6+(-5) & 8+(-7) \end{bmatrix} = \mathbf{\begin{bmatrix} 3 & -7 \\ -11 & 1 \end{bmatrix}}$$
2. **$A - B$**:
   $$\begin{bmatrix} 2-1 & -4-(-3) \\ -6-(-5) & 8-(-7) \end{bmatrix} = \mathbf{\begin{bmatrix} 1 & -1 \\ -1 & 15 \end{bmatrix}}$$
3. **$A \cdot B$ (Matrix Multiplication)**:
   $$\begin{bmatrix} (2)(1) + (-4)(-5) & (2)(-3) + (-4)(-7) \\ (-6)(1) + (8)(-5) & (-6)(-3) + (8)(-7) \end{bmatrix} = \mathbf{\begin{bmatrix} 22 & 22 \\ -46 & -38 \end{bmatrix}}$$
4. **$A^T$ (Transpose)**:
   $$\mathbf{\begin{bmatrix} 2 & -6 \\ -4 & 8 \end{bmatrix}}$$
5. **$\det(A)$**:
   $$\det(A) = (2)(8) - (-4)(-6) = 16 - 24 = \mathbf{-8.0}$$
6. **$A^{-1}$ (Inverse)**:
   $$A^{-1} = \frac{1}{\det(A)} \begin{bmatrix} 8 & 4 \\ 6 & 2 \end{bmatrix} = -\frac{1}{8} \begin{bmatrix} 8 & 4 \\ 6 & 2 \end{bmatrix} = \mathbf{\begin{bmatrix} -1.0 & -0.5 \\ -0.75 & -0.25 \end{bmatrix}}$$

---

### Exercise 3: Pandas DataFrame Analysis
```python
import pandas as pd

product_dict = {
    'Products': ['Laptop', 'Smartphone', 'Tablet', 'Smartwatch', 'Headphones', 'Monitor'],
    'Price': [60000, 35000, 22000, 8000, 3000, 15000],
    'Sales': [120, 250, 180, 400, 850, 160],
    'Profit': [7200, 5250, 2640, 1600, 1275, 4800]
}
df = pd.DataFrame(product_dict)
```
- **Total Rows & Columns**: $6 \text{ rows} \times 4 \text{ columns}$.
- **Average Sales**: $326.67$, **Max Sales**: $850$ (Headphones), **Min Sales**: $120$ (Laptop).
- **Products with Profit > 4000**: Laptop (7200), Smartphone (5250), Monitor (4800).
- **Sorted by Profit Descending**: Laptop $\rightarrow$ Smartphone $\rightarrow$ Monitor $\rightarrow$ Tablet $\rightarrow$ Smartwatch $\rightarrow$ Headphones.

---

### Exercise 4: Visualizations Generated
All plots generated and saved to `Assignments/Assignment-1/plots/`:
1. `q4_scatter_price_vs_sales.png`: Scatter plot with highest selling product highlighted with red star.
2. `q4_line_sales_profit.png`: Comparative line plot showing Sales & Profit trajectories.
3. `q4_bar_profit.png`: Bar chart of Profit with exact numerical labels above each bar.
4. `q4_bar_profit_margin.png`: Profit Margin ($\frac{\text{Profit}}{\text{Sales}} \times 100$) per product.
5. `q4_pie_sales_distribution.png`: Percentage market share of each product in total units sold.
6. `q4_hist_sales.png`: Frequency distribution of sales volume.

---

### Exercise 5: Synthetic Dataset Generation
- **Sample Count**: $N = 100$
- **Independent Feature ($X$)**: `Study_Hours` (Uniformly generated in $[1.0, 10.0]$)
- **Dependent Target ($y$)**: `Test_score` ($y = 35.0 + 6.5 \cdot X + \mathcal{N}(0, 3.5^2)$)
- **Missing Values**: $0$ (Clean dataset)
- **Scatter Plot**: Saved to `plots/q5_scatter_study_vs_score.png`.

---

### Exercise 6: Simple Linear Regression on Synthetic Data
- **Train/Test Split**: 80 samples for Training (80%), 20 samples for Testing (20%).
- **Learned Parameters**:
  - Slope ($m$): $\mathbf{6.4449}$
  - Intercept ($c$): $\mathbf{35.5757}$
  - **Equation**:
    $$\mathbf{\text{Test\_score} = 6.4449 \times \text{Study\_Hours} + 35.5757}$$
- **Evaluation on Unseen Test Set**:
  - **MAE**: $\mathbf{2.6185}$
  - **MSE**: $\mathbf{10.8176}$
  - **$R^2$ Score**: $\mathbf{0.9423}$ ($94.23\%$ variance explained by study hours)
- **Saved Predictions CSV**: `linear_regression_predictions.csv` created.
- **Plot**: Saved to `plots/q6_linear_regression_line.png`.

---

# Part 2: Home Assignments

---

### Home Assignment 1: Student Marks Analysis
```python
MARKS = np.array([
    [80, 75, 90],  # Student 1
    [65, 70, 72],  # Student 2
    [90, 88, 95],  # Student 3
    [55, 60, 58]   # Student 4
])
```
1. **Each student's average (`axis=1`)**:
   - Student 1: $\frac{80+75+90}{3} = \mathbf{81.67}$
   - Student 2: $\frac{65+70+72}{3} = \mathbf{69.00}$
   - Student 3: $\frac{90+88+95}{3} = \mathbf{91.00}$
   - Student 4: $\frac{55+60+58}{3} = \mathbf{57.67}$
2. **Student with highest average**: **Student 3** ($91.00$).
3. **Average for each subject (`axis=0`)**:
   - Subject 1: $\frac{80+65+90+55}{4} = \mathbf{72.50}$
   - Subject 2: $\frac{75+70+88+60}{4} = \mathbf{73.25}$
   - Subject 3: $\frac{90+72+95+58}{4} = \mathbf{78.75}$
4. **Subject with highest average**: **Subject 3** ($78.75$).
5. **Students whose average > 75**: **Student 1** ($81.67$) and **Student 3** ($91.00$).

---

### Home Assignment 2: Multi-Variable Bubble Visualization
- Scatter plot where **X = Price**, **Y = Sales**, **Marker Color & Size = Profit**, and **Product IDs annotated**.
- Saved to: `plots/ha2_multivariable_bubble_chart.png`.

---

### Home Assignment 3: Real-World Dataset Loading
```python
import pandas as pd

# Load Iris Dataset directly from repository
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)
print(df.head())
```

---

### Home Assignment 4: Mathematical Problem Formulation
Given dataset with $20,000$ customer records and features:
- $x_1$: Age, $x_2$: Income, $x_3$: Number_of_purchases, $x_4$: Membership_duration, $x_5$: Website_visits
- Target: Customer_spending ($y$)

1. **Total number of samples**: $N = \mathbf{20000}$
2. **Number of input variables**: $p = \mathbf{5}$
3. **Input vector for the $i$-th sample**:
   $$\mathbf{x}_i = \begin{bmatrix} x_{i1} \\ x_{i2} \\ x_{i3} \\ x_{i4} \\ x_{i5} \end{bmatrix} \in \mathbb{R}^5$$
4. **Output variable $y_i$**:
   $$y_i \in \mathbb{R}$$
5. **Input matrix $X$ and dimensions**:
   $$X = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_N^T \end{bmatrix} \in \mathbb{R}^{20000 \times 5}$$
   **Dimensions**: $\mathbf{20000 \times 5}$
6. **Output vector $\mathbf{y}$ and dimensions**:
   $$\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix} \in \mathbb{R}^{20000 \times 1}$$
   **Dimensions**: $\mathbf{20000 \times 1}$
7. **Complete dataset representation**:
   $$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{20000} \quad \text{or} \quad (X, \mathbf{y})$$
8. **Machine learning function mapping**:
   $$f: \mathbb{R}^5 \rightarrow \mathbb{R} \quad \text{such that } y_i = f(\mathbf{x}_i) + \epsilon_i, \quad \mathbb{E}[\epsilon_i] = 0$$
9. **Prediction equation using $\hat{y}_i$**:
   $$\hat{y}_i = \hat{f}(\mathbf{x}_i) = w_0 + w_1 x_{i1} + w_2 x_{i2} + w_3 x_{i3} + w_4 x_{i4} + w_5 x_{i5} = \mathbf{w}^T \mathbf{x}_i + w_0$$
10. **Training dataset (16,000 samples)**:
    $$\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{16000}, \quad X_{\text{train}} \in \mathbb{R}^{16000 \times 5}, \quad \mathbf{y}_{\text{train}} \in \mathbb{R}^{16000 \times 1}$$
11. **Testing dataset (4,000 samples)**:
    $$\mathcal{D}_{\text{test}} = \{(\mathbf{x}_i, y_i)\}_{i=16001}^{20000}, \quad X_{\text{test}} \in \mathbb{R}^{4000 \times 5}, \quad \mathbf{y}_{\text{test}} \in \mathbb{R}^{4000 \times 1}$$
12. **Dimensions of $X_{\text{test}}$**:
    $$X_{\text{test}} \in \mathbb{R}^{\mathbf{4000 \times 5}}$$

---

### Home Assignment 5: Simple Linear Regression on `Simple_House_Price.csv`
- **Dataset Size**: $1000$ rows ($N = 1000$).
- **Train/Test Split**:
  - Training: $800$ samples ($80.0\%$)
  - Testing: $200$ samples ($20.0\%$)
- **Dimensions on Training Set**:
  - `y_train.shape`: $(800,)$
  - `y_train_pred.shape`: $(800,)$ (Exact match)
- **Fitted Regression Model**:
  $$\mathbf{\text{Price} = 0.0549 \times \text{Area\_sqft} + 17.9458}$$
- **Residual Definition**:
  $$\text{Residual} = \text{Actual} - \text{Predicted}$$
- **Model Evaluation on Test Set**:
  - **MAE**: $\mathbf{8.0071}$
  - **RMSE**: $\mathbf{9.9123}$
  - **$R^2$ Score**: $\mathbf{0.9375}$
- **Comparison between MAE and RMSE**:
  - **RMSE ($9.9123$) is larger than MAE ($8.0071$)**.
  - **Explanation of Influence of Larger Prediction Errors**:
    RMSE squares the individual residual terms ($e_i^2$) before averaging. Consequently, large prediction errors receive a much heavier penalty in RMSE than in MAE (which treats all errors linearly). The divergence ($\text{RMSE} - \text{MAE} = 1.9052$) indicates that the dataset contains a noticeable variance in prediction errors across different houses.

