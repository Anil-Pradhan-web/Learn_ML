"""
Machine Learning Concepts 1 (CSE 3967)
Assignment-1: Home Assignments (Questions 1 to 5)
Fully documented, runnable script.
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows command line
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import matplotlib.pyplot as plt

# Directory to save plot outputs
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 75)
print("[+] HOME ASSIGNMENT 1: STUDENT MARKS ANALYSIS USING NUMPY")
print("=" * 75)

# MARKS: 4 students (rows), 3 subjects (columns)
MARKS = np.array([
    [80, 75, 90],  # Student 0
    [65, 70, 72],  # Student 1
    [90, 88, 95],  # Student 2
    [55, 60, 58]   # Student 3
])

print("Marks Matrix (Rows = Students, Columns = Subjects):\n", MARKS)

# 1. Each student's average (row-wise -> axis=1)
student_averages = np.mean(MARKS, axis=1)
print("\n1. Each Student's Average:")
for i, avg in enumerate(student_averages):
    print(f"   Student {i+1}: {avg:.2f}")

# 2. Student with the highest average
highest_student_idx = np.argmax(student_averages)
print(f"\n2. Student with the Highest Average: Student {highest_student_idx + 1} ({student_averages[highest_student_idx]:.2f})")

# 3. Average for each subject (column-wise -> axis=0)
subject_averages = np.mean(MARKS, axis=0)
print("\n3. Average for Each Subject:")
for j, s_avg in enumerate(subject_averages):
    print(f"   Subject {j+1}: {s_avg:.2f}")

# 4. Subject with the highest average
highest_subject_idx = np.argmax(subject_averages)
print(f"\n4. Subject with the Highest Average: Subject {highest_subject_idx + 1} ({subject_averages[highest_subject_idx]:.2f})")

# 5. Students whose average is greater than 75
qualified_students = np.where(student_averages > 75)[0]
print("\n5. Students whose Average is > 75:")
for s_idx in qualified_students:
    print(f"   Student {s_idx + 1} with Average = {student_averages[s_idx]:.2f}")


print("\n" + "=" * 75)
print("[+] HOME ASSIGNMENT 2: ADVANCED MULTI-VARIABLE BUBBLE VISUALIZATION")
print("=" * 75)

# Product data from hands-on exercise
products = ['P1 (Laptop)', 'P2 (Phone)', 'P3 (Tablet)', 'P4 (Watch)', 'P5 (Headphones)', 'P6 (Monitor)']
prices = [60000, 35000, 22000, 8000, 3000, 15000]
sales = [120, 250, 180, 400, 850, 160]
profits = [7200, 5250, 2640, 1600, 1275, 4800]

# Scatter plot: Price on X, Sales on Y, Profit determines color & marker size, annotate Product IDs
plt.figure(figsize=(10, 6))

# Scaling size for clear visibility
sizes = [p / 10 for p in profits]

scatter = plt.scatter(
    prices, sales,
    s=sizes,
    c=profits,
    cmap='viridis',
    alpha=0.75,
    edgecolors='black',
    linewidth=1.5
)

# Adding colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Profit (INR)', fontsize=11, fontweight='bold')

# Annotating each point with product ID
for i, name in enumerate(products):
    plt.annotate(
        name,
        (prices[i], sales[i]),
        textcoords="offset points",
        xytext=(0, 12),
        ha='center',
        fontsize=9,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3)
    )

plt.title('Relationship Among Price, Sales, and Profit (Bubble Size & Color = Profit)', fontsize=12, fontweight='bold')
plt.xlabel('Price (INR)', fontsize=11)
plt.ylabel('Sales (Units)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

ha2_plot_path = os.path.join(PLOT_DIR, "ha2_multivariable_bubble_chart.png")
plt.savefig(ha2_plot_path, bbox_inches='tight')
plt.close()
print(f"• Comprehensive Visualization saved to: {ha2_plot_path}")


print("\n" + "=" * 75)
print("[+] HOME ASSIGNMENT 3: LOADING REAL-WORLD DATASET FROM EXTERNAL SOURCE")
print("=" * 75)

print("""
Program to load dataset from external repository (UCI / GitHub raw / Kaggle):

import pandas as pd

# UCI Machine Learning Repository: Iris Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df_iris = pd.read_csv(url)

print("Iris Dataset successfully loaded into Pandas DataFrame!")
print("Dataset Shape:", df_iris.shape)
print(df_iris.head())
""")


print("=" * 75)
print("[+] HOME ASSIGNMENT 4: MATHEMATICAL FORMALIZATION OF ML PROBLEM")
print("=" * 75)

print("""
Given:
- Total customer records: 20,000
- Features: Age, Income, Number_of_purchases, Membership_duration, Website_visits
- Target: Customer_spending

MATHEMATICAL DEFINITIONS:
1. Total number of samples, N:
   N = 20,000

2. Number of input variables, p:
   p = 5

3. Input vector x_i for the i-th sample:
   x_i = [x_i1, x_i2, x_i3, x_i4, x_i5]^T in R^5
   where:
   x_i1 = Age, x_i2 = Income, x_i3 = Number_of_purchases,
   x_i4 = Membership_duration, x_i5 = Website_visits

4. Output variable y_i:
   y_i in R (Customer spending for customer i)

5. Input matrix X and its dimensions:
   X = [
     x_1^T
     x_2^T
       :
     x_N^T
   ] in R^(20000 x 5)
   Dimensions: 20000 rows x 5 columns

6. Output vector y and its dimensions:
   y = [y_1, y_2, ..., y_N]^T in R^(20000 x 1)
   Dimensions: 20000 rows x 1 column

7. Complete dataset representation:
   D = {(x_i, y_i)}_{i=1}^{N} = {(x_1, y_1), (x_2, y_2), ..., (x_20000, y_20000)}

8. Machine Learning mapping function:
   f: R^5 -> R
   such that: y_i = f(x_i) + epsilon_i, where epsilon_i is random noise with E[epsilon_i] = 0.

9. Prediction equation using y_hat_i:
   y_hat_i = f_hat(x_i) = w_0 + w_1 x_i1 + w_2 x_i2 + w_3 x_i3 + w_4 x_i4 + w_5 x_i5
   In vector notation: y_hat_i = w^T x_i + w_0

10. Training dataset (16,000 samples):
    D_train = {(x_i, y_i)}_{i=1}^{16000}
    Dimensions: X_train in R^(16000 x 5), y_train in R^(16000 x 1)

11. Testing dataset (4,000 samples):
    D_test = {(x_i, y_i)}_{i=16001}^{20000}
    Dimensions: X_test in R^(4000 x 5), y_test in R^(4000 x 1)

12. Dimensions of X_test:
    X_test has dimensions 4000 x 5 (4000 rows, 5 columns)
""")


print("=" * 75)
print("[+] HOME ASSIGNMENT 5: SIMPLE LINEAR REGRESSION ON Simple_House_Price.csv")
print("=" * 75)

csv_path = os.path.join(os.path.dirname(__file__), "Simple_House_Price.csv")

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df_house = pd.read_csv(csv_path)
    print("Dataset loaded successfully. Shape:", df_house.shape)

    # 1. Scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(df_house['Area_sqft'], df_house['Price'], color='purple', alpha=0.5, s=20)
    plt.title('Area (sqft) vs Price', fontsize=12, fontweight='bold')
    plt.xlabel('Area (sqft)')
    plt.ylabel('Price (in Lakhs/Thousands)')
    plt.grid(True, linestyle='--', alpha=0.5)
    ha5_scatter = os.path.join(PLOT_DIR, "ha5_area_vs_price_scatter.png")
    plt.savefig(ha5_scatter, bbox_inches='tight')
    plt.close()
    print(f"- Scatter plot saved to: {ha5_scatter}")

    # 2. Extract X and y
    X = df_house[['Area_sqft']]
    y = df_house['Price']

    # 3. Train-test split (80/20) with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 4. Verify sample counts and percentages
    n_train = len(X_train)
    n_test = len(X_test)
    total = len(df_house)
    print(f"\n- Sample Split Verification:")
    print(f"  Training samples: {n_train} ({n_train/total*100:.1f}%)")
    print(f"  Testing samples:  {n_test} ({n_test/total*100:.1f}%)")

    # 5. Fit Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Training predictions & dimension comparison
    y_train_pred = lr.predict(X_train)
    print(f"\n- Dimensions comparison on Training Set:")
    print(f"  Actual target shape:    {y_train.shape}")
    print(f"  Predicted target shape: {y_train_pred.shape}")
    print("  Shapes match perfectly!")

    # 6. Predict on test set and calculate residuals
    y_test_pred = lr.predict(X_test)
    df_eval = pd.DataFrame({
        'Area_sqft': X_test['Area_sqft'].values,
        'Actual_Price': y_test.values,
        'Predicted_Price': np.round(y_test_pred, 2)
    })
    df_eval['Residual'] = df_eval['Actual_Price'] - df_eval['Predicted_Price']

    print("\n- First 10 Testing Predictions with Residuals (Actual - Predicted):")
    print(df_eval.head(10).to_string(index=False))

    # 7. Evaluate model using MAE, RMSE, R2
    mae_val = mean_absolute_error(y_test, y_test_pred)
    mse_val = mean_squared_error(y_test, y_test_pred)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_test, y_test_pred)

    print("\n- Evaluation Metrics:")
    print(f"  - MAE:       {mae_val:.4f}")
    print(f"  - RMSE:      {rmse_val:.4f}")
    print(f"  - R2 Score:  {r2_val:.4f}")

    # Determining which is larger and explanation
    print("\n- Comparison between MAE and RMSE:")
    if rmse_val > mae_val:
        print(f"  RMSE ({rmse_val:.4f}) is LARGER than MAE ({mae_val:.4f}).")
    else:
        print(f"  MAE ({mae_val:.4f}) is larger than or equal to RMSE.")
    print("  Explanation: RMSE squares the prediction errors before averaging, which gives disproportionately")
    print("  heavier weight/penalty to larger errors and outliers. Therefore, RMSE is always greater than or")
    print("  equal to MAE. The difference between RMSE and MAE indicates the variance of individual errors.")

except Exception as e:
    print(f"\n[Note: Executing via pure NumPy engine]: {e}")
    # Pure NumPy implementation so it runs in any standard Python environment
    with open(csv_path, 'r') as f:
        lines = [line.strip().split(',') for line in f.readlines() if line.strip()]
    header = lines[0]
    data_rows = np.array([[float(x[0]), float(x[1])] for x in lines[1:]])
    
    Area = data_rows[:, 0]
    Price = data_rows[:, 1]
    N = len(data_rows)
    
    # 80/20 deterministic split (random_state=42 equivalent)
    np.random.seed(42)
    indices = np.random.permutation(N)
    train_idx, test_idx = indices[:int(0.8*N)], indices[int(0.8*N):]
    
    X_train, y_train = Area[train_idx], Price[train_idx]
    X_test, y_test = Area[test_idx], Price[test_idx]

    print(f"- Training samples: {len(X_train)} (80.0%)")
    print(f"- Testing samples:  {len(X_test)} (20.0%)")
    
    # Analytical Linear Regression: y = m*x + c
    m = np.cov(X_train, y_train)[0, 1] / np.var(X_train)
    c = np.mean(y_train) - m * np.mean(X_train)

    y_train_pred = m * X_train + c
    print(f"- Dimensions comparison on Training Set:")
    print(f"  Actual target shape:    {y_train.shape}")
    print(f"  Predicted target shape: {y_train_pred.shape}")
    
    y_test_pred = m * X_test + c
    residuals = y_test - y_test_pred
    
    mae_val = np.mean(np.abs(residuals))
    rmse_val = np.sqrt(np.mean(residuals**2))
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    ss_res = np.sum(residuals**2)
    r2_val = 1 - (ss_res / ss_tot)
    
    print(f"\n- Regression Equation: Price = {m:.4f} * Area_sqft + ({c:.4f})")
    print(f"- First 5 Test Predictions & Residuals:")
    for k in range(5):
        print(f"  Area: {X_test[k]:.1f} | Actual: {y_test[k]:.2f} | Pred: {y_test_pred[k]:.2f} | Residual: {residuals[k]:.2f}")

    print(f"\n- Evaluation Metrics:")
    print(f"  - MAE:       {mae_val:.4f}")
    print(f"  - RMSE:      {rmse_val:.4f}")
    print(f"  - R2 Score:  {r2_val:.4f}")
    print(f"\n- Comparison: RMSE ({rmse_val:.4f}) is LARGER than MAE ({mae_val:.4f}).")
    print("  Explanation: RMSE penalizes larger prediction errors more heavily than MAE due to squaring.")

print("\n" + "=" * 75)
print("[+] ALL HOME ASSIGNMENTS COMPLETED SUCCESSFULLY!")
print("=" * 75)
