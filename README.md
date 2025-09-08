❤️ Heart Disease Prediction Project
📌 Overview

This project applies Machine Learning to the UCI Heart Disease Dataset to analyze, predict, and visualize heart disease risk.
We implemented a full ML pipeline: data preprocessing, feature selection, dimensionality reduction (PCA), supervised & unsupervised learning, model optimization, and deployment via Streamlit.

🎯 Objectives

✅ Perform Data Preprocessing & Cleaning

✅ Apply Dimensionality Reduction with PCA

✅ Implement Feature Selection (Random Forest, RFE, Chi-Square)

✅ Train Supervised Models: Logistic Regression, Decision Trees, Random Forest, SVM

✅ Apply Unsupervised Learning: K-Means & Hierarchical Clustering

✅ Optimize Models with GridSearchCV & RandomizedSearchCV

✅ Save the best model as .pkl

✅ Deploy interactive UI using Streamlit (with optional Ngrok)

🛠️ Tools & Libraries

Python

Pandas, NumPy, Scikit-learn

Matplotlib, Seaborn

📊 Results

The best model was Logistic Regression trained on PCA-transformed data.

Achieved:

Accuracy: ~88%

ROC AUC: ~0.93

Feature selection and PCA both improved generalization, but Logistic Regression with PCA gave the best performance.

Streamlit for deployment

Joblib for saving models
