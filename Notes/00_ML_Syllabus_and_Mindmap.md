# CSE 3967: Machine Learning Concepts 1 — Complete Course Roadmap & Index

> **Course Level**: B.Tech Computer Science & Engineering (3rd Year)  
> **Course Format**: 0L - 4P (Hands-On Lab Focus)  
> **Textbook**: *Practical Handbook of Machine Learning* by Bhattacharyya, GK Publications

---

## 🗺️ Master Curriculum Flow ("Kya Kiske Baad Aata Hai?")

```
[Module 0: Prerequisites & Tooling]
  │   └── Ch 01: NumPy Arrays, Pandas DataFrames & Descriptive Statistics
  ▼
[Module 1: Foundations & AIML Evolution]
  │   └── Ch 02: Classical Programming vs Machine Learning Paradigms
  ▼
[Module 2: Regression & Error Analysis]
  │   ├── Ch 03: Simple & Multiple Linear Regression (MSE, OLS, Gradient Descent)
  │   ├── Ch 04: Non-Linear Models & Polynomial Feature Engineering
  │   └── Ch 05: Sources of Model Error & The Bias-Variance Tradeoff (U-Curve)
  ▼
[Module 3: Validation & Classification Fundamentals]
  │   ├── Ch 06: Model Validation (Train-Test Split 80/20 & Shape Error Traps)
  │   ├── Ch 07: Classification & Logistic Regression (Sigmoid Function)
  │   └── Ch 08: Digging Deeper into Classifier Accuracy: The Confusion Matrix
  ▼
[Module 4: Distance-Based ML & Preprocessing]
  │   └── Ch 09: K-Nearest Neighbors (KNN) & Mandatory Feature Scaling (MinMax vs Z-Score)
  ▼
[Module 5: Tree & Ensemble Learning]
  │   ├── Ch 10: Decision Trees (Splitting Math: Entropy, Information Gain & Gini Impurity)
  │   └── Ch 11: Ensemble Models & Random Forest Classifier (Bagging & OOB Error)
  ▼
[Module 6: Unsupervised Learning]
      └── Ch 12: Unsupervised Learning & K-Means Clustering (Elbow Method & WCSS)
```

---

## 📚 Complete Chapter Index & Links

1. [**Chapter 01: Prerequisites (NumPy, Pandas & Basic Statistics)**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/01_Prerequisites_NumPy_Pandas_and_Stats.md)
   - NumPy vectorization, Axis rules (`axis=0` vs `axis=1`), Random seed, Pandas handling missing data (`fillna` vs `dropna`), Outlier impact on Mean vs Median.
2. [**Chapter 02: The AI & Machine Learning Revolution**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/02_AIML_Revolution.md)
   - Traditional code ($Data + Rules \rightarrow Output$) vs ML ($Data + Output \rightarrow Rules$), Historical data significance, Supervised vs Unsupervised vs Reinforcement.
3. [**Chapter 03: Prediction Problems Using Linear Regression**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/03_Linear_Regression.md)
   - $y = mx + c$, MSE Cost function, OLS normal equation vs Gradient Descent, MAE, MSE, RMSE, and $R^2$ Score.
4. [**Chapter 04: Non-Linear Models & Feature Engineering**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/04_NonLinear_Models_and_Feature_Engineering.md)
   - Polynomial regression, Why polynomial models are still linear, Feature engineering interaction terms, Benefits & Overfitting risks.
5. [**Chapter 05: Sources of Model Error & Bias-Variance Tradeoff**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/05_Model_Errors_and_Bias_Variance.md)
   - 4 Error sources, Overfitting vs Underfitting diagnostic table, The U-shaped test error curve, Remedies (Regularization L1/L2, Pruning).
6. [**Chapter 06: Model Validation & Train-Test Split**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/06_Model_Validation_Train_Test_Split.md)
   - 80/20 Split, `random_state` role, Data Leakage prevention, Scikit-learn 2D array input shape error (`df[['col']]` vs `df['col']`), $K$-Fold CV.
7. [**Chapter 07: Classification & Logistic Regression**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/07_Classification_and_Logistic_Regression.md)
   - Why Linear Regression fails for classification, Sigmoid function formula $\sigma(z) = \frac{1}{1 + e^{-z}}$, range $(0, 1)$, Decision threshold $\tau = 0.5$.
8. [**Chapter 08: Evaluation of Classifiers & The Confusion Matrix**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/08_Confusion_Matrix_and_Metrics.md)
   - TP, TN, FP, FN, Accuracy, Precision, Recall, Specificity, F1-Score (Harmonic Mean), Worked numerical problem.
9. [**Chapter 09: Distance-Based ML & Feature Scaling**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/09_Distance_Based_ML_and_Feature_Scaling.md)
   - KNN algorithm, Euclidean vs Manhattan distance, Why distance models strictly fail without scaling, Min-Max Normalization vs Standardization (Z-Score).
10. [**Chapter 10: Decision Trees**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/10_Decision_Trees.md)
    - Tree anatomy, Purity metrics (Entropy, Information Gain, Gini Impurity math), Why deep trees overfit, Hyperparameter pruning (`max_depth`).
11. [**Chapter 11: Ensemble Models & Random Forest Classifier**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/11_Ensemble_Models_and_Random_Forest.md)
    - Wisdom of the crowd, Bagging vs Boosting, Random Forest (Bootstrap sampling, Random feature selection $\sqrt{p}$, Majority voting), Out-of-Bag (OOB) error.
12. [**Chapter 12: Unsupervised Learning & K-Means Clustering**](file:///C:/Users/ANIL/Desktop/Learn_ML/Notes/12_Unsupervised_Learning_and_KMeans.md)
    - Unlabeled data clustering, $K$-Means centroid update steps, Within-Cluster Sum of Squares (WCSS / Inertia), The Elbow Method to find optimal $K$.

---

## 🎯 Exam & Viva Quick Reference Rules
- **Scaling rule**: Distance algorithms (KNN, $K$-Means) strictly require feature scaling. Tree algorithms (Decision Trees, Random Forest) do NOT require feature scaling.
- **Error rule**: High Train + High Test = **Underfitting (High Bias)**. Low Train + High Test = **Overfitting (High Variance)**.
- **Metric rule**: In medical/safety cases, maximize **Recall** (minimize FN). In fraud/spam alerts, maximize **Precision** (minimize FP).

