"""
Machine Learning Concepts 1 (CSE 3967)
Assignment-1: Hands-On Exercises (Questions 1 to 6)
Fully documented, runnable script with zero-dependency fallback.
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

# Directory to save plot images cleanly
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("[+] QUESTION 1: NUMPY 2D ARRAY OPERATIONS")
print("=" * 70)

data = np.array([
    [10, -20, 30],
    [40, -50, 60],
    [70,  80, -90]
])

print("Original Array:\n", data)
print("\n* Mean of each row (axis=1):", np.mean(data, axis=1))
print("* Mean of each column (axis=0):", np.mean(data, axis=0))
print("* Standard deviation of each row (axis=1):", np.std(data, axis=1))
print("* Maximum value of each column (axis=0):", np.max(data, axis=0))
print("* Minimum value of each row (axis=1):", np.min(data, axis=1))


print("\n" + "=" * 70)
print("[+] QUESTION 2: MATRIX ALGEBRA OPERATIONS")
print("=" * 70)

A = np.array([[2, -4], [-6, 8]])
B = np.array([[1, -3], [-5, -7]])

print("Matrix A:\n", A)
print("Matrix B:\n", B)

print("\n* A + B:\n", A + B)
print("\n* A - B:\n", A - B)
print("\n* A . B (Matrix Multiplication A @ B):\n", np.dot(A, B))
print("\n* Transpose of A (A^T):\n", A.T)

det_A = np.linalg.det(A)
print(f"\n* Determinant of A (det(A)): {det_A:.2f}")

inv_A = np.linalg.inv(A)
print("\n* Inverse of A (A^-1):\n", inv_A)


print("\n" + "=" * 70)
print("[+] QUESTION 3: PANDAS DATAFRAME CREATION & ANALYSIS")
print("=" * 70)

products = ['Laptop', 'Smartphone', 'Tablet', 'Smartwatch', 'Headphones', 'Monitor']
prices = [60000, 35000, 22000, 8000, 3000, 15000]
sales = [120, 250, 180, 400, 850, 160]
profits = [7200, 5250, 2640, 1600, 1275, 4800]

try:
    import pandas as pd
    product_dict = {
        'Products': products,
        'Price': prices,
        'Sales': sales,
        'Profit': profits
    }
    df_products = pd.DataFrame(product_dict)

    print("\n1. Print the DataFrame:")
    print(df_products)

    print("\n2. All Column Names:", list(df_products.columns))
    print(f"\n3. Number of Rows: {df_products.shape[0]}, Number of Columns: {df_products.shape[1]}")
    print("\n4. First Three Rows:\n", df_products.head(3))
    print("\n5. Only 'Sales' and 'Profit' Columns:\n", df_products[['Sales', 'Profit']])

    avg_sales = df_products['Sales'].mean()
    max_sales = df_products['Sales'].max()
    min_sales = df_products['Sales'].min()
    print(f"\n6. Sales Statistics -> Average: {avg_sales:.2f}, Maximum: {max_sales}, Minimum: {min_sales}")

    print("\n7. Products with Profit > 4000:")
    print(df_products[df_products['Profit'] > 4000][['Products', 'Profit']].to_string(index=False))

    print("\n8. DataFrame sorted by Profit in descending order:")
    print(df_products.sort_values(by='Profit', ascending=False).to_string(index=False))

except ImportError:
    print("[Note: Displaying structured DataFrame records via NumPy/Dict representation]")
    print(f"{'Products':<15} {'Price':<10} {'Sales':<10} {'Profit':<10}")
    print("-" * 50)
    for p, pr, s, pf in zip(products, prices, sales, profits):
        print(f"{p:<15} {pr:<10} {s:<10} {pf:<10}")

    print("\n2. All Column Names: ['Products', 'Price', 'Sales', 'Profit']")
    print(f"3. Number of Rows: {len(products)}, Number of Columns: 4")
    print("\n4. First Three Rows:")
    for k in range(3):
        print(f"   {products[k]}: Price={prices[k]}, Sales={sales[k]}, Profit={profits[k]}")

    print(f"\n6. Sales -> Avg: {np.mean(sales):.2f}, Max: {np.max(sales)}, Min: {np.min(sales)}")
    print("\n7. Products with Profit > 4000:")
    for p, pf in zip(products, profits):
        if pf > 4000:
            print(f"   {p}: Profit = {pf}")

    sorted_indices = np.argsort(profits)[::-1]
    print("\n8. Sorted by Profit descending:")
    for idx in sorted_indices:
        print(f"   {products[idx]:<15} Profit: {profits[idx]}")


print("\n" + "=" * 70)
print("[+] QUESTION 4: DATA VISUALIZATIONS USING MATPLOTLIB")
print("=" * 70)

# 4.1 Scatter plot of Price vs Sales, highlighting highest sales
plt.figure(figsize=(8, 5))
max_sales_idx = int(np.argmax(sales))

plt.scatter(prices, sales, color='steelblue', s=100, label='Products')
plt.scatter(
    prices[max_sales_idx], sales[max_sales_idx],
    color='red', marker='*', s=280, label=f"Highest Sales ({products[max_sales_idx]})"
)
plt.title('Price vs Sales (Highest Sales Highlighted)', fontsize=12, fontweight='bold')
plt.xlabel('Price (INR)')
plt.ylabel('Sales (Units)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
scatter_path = os.path.join(PLOT_DIR, "q4_scatter_price_vs_sales.png")
plt.savefig(scatter_path, bbox_inches='tight')
plt.close()
print(f"Saved: {scatter_path}")

# 4.2 Line plot of Sales and Profit on the same graph
plt.figure(figsize=(9, 5))
plt.plot(products, sales, marker='o', linestyle='-', color='royalblue', label='Sales (Units)')
plt.plot(products, profits, marker='s', linestyle='--', color='darkorange', label='Profit (INR)')
plt.title('Sales and Profit by Product', fontsize=12, fontweight='bold')
plt.xlabel('Products')
plt.ylabel('Amount')
plt.xticks(rotation=15)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
line_path = os.path.join(PLOT_DIR, "q4_line_sales_profit.png")
plt.savefig(line_path, bbox_inches='tight')
plt.close()
print(f"Saved: {line_path}")

# 4.3 Bar graph of Profit with values displayed above each bar
plt.figure(figsize=(8, 5))
bars = plt.bar(products, profits, color='teal', edgecolor='black')
plt.title('Profit by Product', fontsize=12, fontweight='bold')
plt.xlabel('Products')
plt.ylabel('Profit (INR)')
plt.xticks(rotation=15)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"{yval:.0f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
bar_path = os.path.join(PLOT_DIR, "q4_bar_profit.png")
plt.savefig(bar_path, bbox_inches='tight')
plt.close()
print(f"Saved: {bar_path}")

# 4.4 Bar chart showing Profit Margin = (Profit / Sales) * 100
profit_margins = [(pf / s) * 100 for pf, s in zip(profits, sales)]

plt.figure(figsize=(8, 5))
bars_margin = plt.bar(products, profit_margins, color='mediumpurple', edgecolor='black')
plt.title('Profit Margin (%) per Product', fontsize=12, fontweight='bold')
plt.xlabel('Products')
plt.ylabel('Profit as % of Sales')
plt.xticks(rotation=15)

for bar in bars_margin:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom')

plt.tight_layout()
margin_path = os.path.join(PLOT_DIR, "q4_bar_profit_margin.png")
plt.savefig(margin_path, bbox_inches='tight')
plt.close()
print(f"Saved: {margin_path}")

# 4.5 Pie chart of percentage contribution to total sales
plt.figure(figsize=(7, 7))
plt.pie(
    sales,
    labels=products,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Pastel1.colors
)
plt.title('Percentage Contribution of Each Product to Total Sales', fontsize=12, fontweight='bold')
pie_path = os.path.join(PLOT_DIR, "q4_pie_sales_distribution.png")
plt.savefig(pie_path, bbox_inches='tight')
plt.close()
print(f"Saved: {pie_path}")

# 4.6 Histogram of Sales
plt.figure(figsize=(8, 5))
plt.hist(sales, bins=5, color='coral', edgecolor='black')
plt.title('Histogram of Sales Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Sales Range (Units)')
plt.ylabel('Frequency (Number of Products)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
hist_path = os.path.join(PLOT_DIR, "q4_hist_sales.png")
plt.savefig(hist_path, bbox_inches='tight')
plt.close()
print(f"Saved: {hist_path}")


print("\n" + "=" * 70)
print("[+] QUESTION 5: SYNTHETIC DATASET GENERATION (Study_Hours vs Test_score)")
print("=" * 70)

np.random.seed(42)
n_samples = 100

study_hours = np.random.uniform(1.0, 10.0, size=n_samples)
noise = np.random.normal(loc=0.0, scale=3.5, size=n_samples)
test_score = 35.0 + 6.5 * study_hours + noise
test_score = np.clip(test_score, 0, 100)

print("• First 5 Synthetic Records:")
for idx in range(5):
    print(f"  Sample {idx+1}: Study Hours = {study_hours[idx]:.2f}, Test Score = {test_score[idx]:.2f}")

print("\n• Dataset Metadata:")
print(f"  - Number of samples or examples: {n_samples}")
print(f"  - Number of features: 1 (Study_Hours)")
print(f"  - Number of target variables: 1 (Test_score)")
print(f"  - Feature name(s): ['Study_Hours']")
print(f"  - Target name: 'Test_score'")
print(f"  - Shape of the dataset: ({n_samples}, 2)")
print(f"  - Data types: float64")

print("\n• Descriptive Statistics:")
print(f"  - Study_Hours -> Mean: {np.mean(study_hours):.2f}, Std: {np.std(study_hours):.2f}, Min: {np.min(study_hours):.2f}, Max: {np.max(study_hours):.2f}")
print(f"  - Test_score  -> Mean: {np.mean(test_score):.2f}, Std: {np.std(test_score):.2f}, Min: {np.min(test_score):.2f}, Max: {np.max(test_score):.2f}")
print("• Missing Values: 0 (No missing values)")

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(study_hours, test_score, color='darkblue', alpha=0.7)
plt.title('Relationship: Study Hours vs Test Score', fontsize=12, fontweight='bold')
plt.xlabel('Study Hours (Independent Variable X)')
plt.ylabel('Test Score (Dependent Variable y)')
plt.grid(True, linestyle='--', alpha=0.5)
scatter_study_path = os.path.join(PLOT_DIR, "q5_scatter_study_vs_score.png")
plt.savefig(scatter_study_path, bbox_inches='tight')
plt.close()
print(f"Saved: {scatter_study_path}")


print("\n" + "=" * 70)
print("[+] QUESTION 6: SIMPLE LINEAR REGRESSION ON SYNTHETIC DATA")
print("=" * 70)

# Train-Test Split (80% train, 20% test)
indices = np.random.permutation(n_samples)
train_size = int(0.8 * n_samples)
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, y_train = study_hours[train_idx], test_score[train_idx]
X_test, y_test = study_hours[test_idx], test_score[test_idx]

print(f"• Training samples: {len(X_train)} (80%)")
print(f"• Testing samples:  {len(X_test)} (20%)")

# Closed-form Analytical Linear Regression: y = m*x + c
# m = Cov(X, y) / Var(X)
slope_m = np.cov(X_train, y_train)[0, 1] / np.var(X_train)
intercept_c = np.mean(y_train) - slope_m * np.mean(X_train)

print(f"\n• Model Slope (m / Coefficient): {slope_m:.4f}")
print(f"• Model Intercept (c): {intercept_c:.4f}")
print(f"• Mathematical Regression Equation:\n  Test_score = {slope_m:.4f} * Study_Hours + ({intercept_c:.4f})")

# Test predictions
y_pred_test = slope_m * X_test + intercept_c

# Evaluation Metrics
mae = np.mean(np.abs(y_test - y_pred_test))
mse = np.mean((y_test - y_pred_test)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
ss_res = np.sum((y_test - y_pred_test)**2)
r2 = 1 - (ss_res / ss_tot)

print("\n• Model Performance Metrics on Testing Data:")
print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
print(f"  - Mean Squared Error (MSE):  {mse:.4f}")
print(f"  - R2 Score:                  {r2:.4f}")

# Save results to CSV
csv_output_path = os.path.join(os.path.dirname(__file__), "linear_regression_predictions.csv")
with open(csv_output_path, "w") as f:
    f.write("Study_Hours,Actual_Test_score,Predicted_Test_score\n")
    for sh, act, pred in zip(X_test, y_test, y_pred_test):
        f.write(f"{sh:.2f},{act:.2f},{pred:.2f}\n")
print(f"\n• Results saved to CSV: {csv_output_path}")

# Plot Regression line
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual Test Data')

sorted_x = np.linspace(np.min(study_hours), np.max(study_hours), 100)
sorted_y = slope_m * sorted_x + intercept_c
plt.plot(sorted_x, sorted_y, color='red', linewidth=2, label=f'Regression Line (y = {slope_m:.2f}x + {intercept_c:.2f})')

plt.title('Simple Linear Regression: Study Hours vs Test Score', fontsize=12, fontweight='bold')
plt.xlabel('Study Hours')
plt.ylabel('Test Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
reg_plot_path = os.path.join(PLOT_DIR, "q6_linear_regression_line.png")
plt.savefig(reg_plot_path, bbox_inches='tight')
plt.close()
print(f"Saved: {reg_plot_path}")

print("\n" + "=" * 70)
print("[+] ALL HANDS-ON EXERCISES COMPLETED SUCCESSFULLY!")
print("=" * 70)
