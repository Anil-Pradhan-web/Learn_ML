# Chapter 1: Prerequisites (NumPy, Pandas & Basic Statistics)

> **Why this matters for ML**: ML algorithms don't understand raw text or files; they only understand matrices and vectors of numbers. NumPy and Pandas are the foundation tools used to clean and convert real-world data into numerical matrices.

---

## 1. NumPy Essentials (Numerical Python)

### Core Concept (Simple English)
NumPy provides an $N$-dimensional array object (`ndarray`). Unlike standard Python lists, NumPy arrays store data in contiguous memory blocks, enabling vectorized mathematical operations without slow Python `for` loops.

### Key Operations
1. **Creation**:
   ```python
   import numpy as np
   data = np.array([[10, 20, 30], [40, 50, 60]])
   ```
2. **Axes Rule (Very Important for Exams)**:
   - `axis=0`: Column-wise operation (collapses rows vertically).
   - `axis=1`: Row-wise operation (collapses columns horizontally).
   ```python
   col_mean = np.mean(data, axis=0) # [25., 35., 45.]
   row_mean = np.mean(data, axis=1) # [20., 50.]
   ```
3. **Random Generation & Seed**:
   ```python
   np.random.seed(42)  # Ensures reproducibility (same random numbers every run)
   normal_vals = np.random.normal(loc=0.0, scale=1.0, size=50) # Gaussian distribution
   uniform_vals = np.random.uniform(low=0.0, high=1.0, size=50) # Flat distribution
   ```

> 💡 **Hinglish Intuition**:  
> Normal Python list ek aisi bag hai jisme sabzi, kitaab, bottle sab mix ho sakti hain (slow). NumPy array ek organized egg-tray jaisa hai jahan sabhi elements ek hi type aur size ke hote hain, isiliye fast processing hoti hai.

---

## 2. Pandas Essentials (DataFrames)

### Core Concept (Simple English)
Pandas is built on top of NumPy and provides two primary 2D tabular data structures: **Series** (1D column) and **DataFrame** (2D table with labeled rows and columns).

### Daily ML Operations
```python
import pandas as pd

# 1. Load CSV
df = pd.read_csv('dataset.csv')

# 2. Inspect data
print(df.shape)       # (n_rows, n_cols)
print(df.info())        # Column datatypes and memory
print(df.describe())    # Mean, std, min, 25%, 50%, 75%, max

# 3. Handle Missing Values
print(df.isnull().sum())           # Count missing values per column
df['Marks'] = df['Marks'].fillna(df['Marks'].mean()) # Imputation
df_clean = df.dropna()             # Deletion
```

> 💡 **Hinglish Intuition**:  
> `dropna()` ka matlab: Jis student ka marks missing tha, usko poori list se delete kar diya (Loss of data).  
> `fillna(mean)` ka matlab: Missing jagah pe class ka average score daal diya taaki data ka size kam na ho (Imputation).

---

## 3. Basic Statistics & Probability

### 1. Measures of Central Tendency
- **Mean ($\mu$)**: $\frac{\sum x_i}{N}$ — Sum of all values divided by count.
  - *Weakness*: **Extremely sensitive to outliers**.
- **Median**: The middle value when data is sorted.
  - *Strength*: **Robust to outliers**.
- **Mode**: The most frequently repeating value. Useful for categorical data.

> 💡 **The Outlier Trap (Favorite Exam Question)**:  
> Agar class me 5 bacho ke pocket money $[20, 25, 30, 25, 30]$ hain, aur Ambani ka beta aa jaye $[20, 25, 30, 25, 30, 100000]$:
> - **Mean** achanak hazaron me pahunch jayega (distorted).
> - **Median** bas ek do rupaye aage piche hoga. Isiliye skewed data me **Median** best hota hai.

### 2. Dispersion: Standard Deviation ($\sigma$)
$$\sigma = \sqrt{\frac{1}{N}\sum (x_i - \mu)^2}$$
- Tells how tightly or widely the data is scattered around the average.
- Low $\sigma \implies$ All data points are close to average (consistent).
- High $\sigma \implies$ Data is spread out across a large range.

---

## 🎯 Quick B.Tech Viva Questions
1. **Q: Why do we use `np.random.seed(42)`?**  
   *A:* To make random data generation deterministic and reproducible across multiple runs.
2. **Q: In `df[['col']]` vs `df['col']`, what is the difference?**  
   *A:* Double brackets return a 2D DataFrame (required for ML feature matrix $X$), single bracket returns a 1D Series.

