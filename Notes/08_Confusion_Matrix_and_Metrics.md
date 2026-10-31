# Chapter 8: Evaluation of Classifiers & The Confusion Matrix

> **Why this matters for ML**: If a cancer dataset has 99 healthy people and 1 cancer patient, a dumb model that predicts "Healthy" for everyone achieves **99% accuracy**, but fails 100% of the cancer patients. We need the Confusion Matrix to see the true reality.

---

## 1. The Confusion Matrix (2×2 Table)

A Confusion Matrix compares **Actual Labels** against **Predicted Labels**:

```
                    PREDICTED CLASS
                 Positive (1)     Negative (0)
               ┌────────────────┬────────────────┐
  Positive (1) │ True Positive  │ False Negative │
ACTUAL         │     (TP)       │     (FN)       │  <── Type II Error (Missed!)
CLASS          ├────────────────┼────────────────┤
  Negative (0) │ False Positive │ True Negative  │
               │     (FP)       │     (TN)       │  <── Type I Error (False Alarm)
               └────────────────┴────────────────┘
```

> 💡 **Hinglish Intuition**:
> - **TP**: Chor tha, aur police ne chor hi pakda (Good).
> - **TN**: Sharif aadmi tha, police ne chor nahi bola (Good).
> - **FP (Type I Error)**: Sharif aadmi tha, par police ne galti se arrest kar liya (*False Alarm*).
> - **FN (Type II Error)**: Chor tha, par police ne bola "ye chor nahi hai" aur chhod diya (*Dangerously Missed!*).

---

## 2. The 4 Essential Classification Formulas

### 1. Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **When to use**: Only when classes are roughly balanced (e.g., 50% cats, 50% dogs).

### 2. Precision
$$\text{Precision} = \frac{TP}{TP + FP}$$
- **Simple Question**: *"Out of all samples the model predicted as Positive, how many were ACTUALLY Positive?"*
- **Focus**: Minimizing False Positives (FP).
- **Use case**: Spam detection (You don't want an important job offer email sent to spam).

### 3. Recall (Sensitivity / True Positive Rate)
$$\text{Recall} = \frac{TP}{TP + FN}$$
- **Simple Question**: *"Out of all ACTUAL Positive samples in reality, how many did the model successfully catch?"*
- **Focus**: Minimizing False Negatives (FN).
- **Use case**: Disease diagnosis, airport security weapon check (Missing a positive case is fatal).

### 4. F1-Score (Harmonic Mean)
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- Combines Precision and Recall into a single balanced number.
- We use the **Harmonic Mean** (not normal average) because if either Precision or Recall is close to 0, F1-Score crashes to 0.

---

## 3. Step-by-Step Solved Numerical Example

**Given**:
- $TP = 40, \quad FN = 10$
- $FP = 5, \quad TN = 45$
- Total Samples = $40 + 10 + 5 + 45 = 100$

1. $\text{Accuracy} = \frac{40 + 45}{100} = \mathbf{0.85 \ (85\%)}$
2. $\text{Precision} = \frac{40}{40 + 5} = \frac{40}{45} = \mathbf{0.888 \ (88.9\%)}$
3. $\text{Recall} = \frac{40}{40 + 10} = \frac{40}{50} = \mathbf{0.800 \ (80.0\%)}$
4. $\text{F1-Score} = 2 \times \frac{0.888 \times 0.800}{0.888 + 0.800} = \mathbf{0.842}$

---

## 4. Python Scikit-Learn Code

```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Generates Precision, Recall, F1 for both classes
print(classification_report(y_true, y_pred))
```

---

## 🎯 Quick B.Tech Viva Questions
1. **Q: In cancer diagnosis, which metric is more important: Precision or Recall?**  
   *A:* **Recall**. It is critical that no cancer patient is incorrectly diagnosed as healthy ($FN = 0$).
2. **Q: What is Specificity?**  
   *A:* $\frac{TN}{TN + FP}$ — The True Negative Rate (ability to correctly identify negative cases).

