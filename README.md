# AI-basics-course_dicoding-academy

# Bank Transaction Clustering & Classification

Final project for **"Belajar Machine Learning untuk Pemula"** on Dicoding. The project applies an unsupervised → supervised machine learning pipeline on a bank transaction dataset: transactions are first grouped into behavioral segments using clustering, and a classification model is then trained to predict which segment a new transaction belongs to

## 📌 Project Overview

1. **Clustering** – Explore and preprocess raw bank transaction data, then group transactions into clusters using K-Means to uncover behavioral patterns among customers.
2. **Classification** – Use the cluster labels produced in step 1 as the target variable, and train a Decision Tree classifier to predict a transaction's cluster from its features

## 🗂️ Dataset

The dataset contains anonymized bank transaction records, including fields such as transaction amount, transaction type, location, channel, customer age, customer occupation, transaction duration, login attempts, and account balance, along with identifier/date columns (transaction ID, account ID, device ID, IP address, merchant ID, transaction date) that are dropped during preprocessing

## ⚙️ Workflow

**1. Exploratory Data Analysis**
- Inspect the dataset with `head()`, `info()`, and `describe()`

**2. Data Cleaning & Preprocessing**
- Check missing values and duplicates (`isnull().sum()`, `duplicated().sum()`)
- Drop missing values and duplicate rows
- Remove identifier/address/date columns
- Encode categorical features with `LabelEncoder`
- Scale numerical features with `StandardScaler`

**3. Clustering**
- Determine the optimal number of clusters using the Elbow Method (`KElbowVisualizer`)
- Fit a `KMeans` model and assign a `Target` cluster label to each transaction
- Evaluate cluster quality with the Silhouette Score
- Visualize clusters in 2D using PCA
- Interpret each cluster's characteristics based on descriptive statistics (mean, min, max)

**4. Classification**
- Split the clustered dataset with `train_test_split`
- Train a `DecisionTreeClassifier` to predict the `Target` cluster label
- Evaluate performance with accuracy, precision, recall, and F1-score
- (Optional) Compare against Random Forest and Logistic Regression, and tune the Decision Tree with `GridSearchCV`

## 🛠️ Tech Stack

- Python, pandas, NumPy
- scikit-learn (KMeans, DecisionTreeClassifier, preprocessing, model selection, metrics)
- Yellowbrick (`KElbowVisualizer`)
- Matplotlib
- joblib (model persistence)

## 📁 Repository Structure

```
├── [Clustering]_Submission_Akhir_BMLP_Rava_Amesta.ipynb   # Clustering notebook
├── [Klasifikasi]_Submission_Akhir_BMLP_Rava_Amesta.ipynb  # Classification notebook
├── model_clustering.h5           # Trained K-Means model
├── PCA_model_clustering.h5       # Trained PCA model (optional)
├── decision_tree_model.h5        # Trained Decision Tree model
├── explore__classification.h5    # Best model from exploration (optional)
├── tuning_classification.h5      # Tuned model via GridSearchCV (optional)
├── data_clustering.csv           # Preprocessed data + cluster labels
├── data_clustering_inverse.csv   # Clustered data with original category labels (optional)
└── bank_transactions_data_edited.csv  # Raw dataset
```

## 📊 Results

- Optimal number of clusters selected via the Elbow Method
- Cluster separation evaluated with the Silhouette Score
- Decision Tree classifier achieves strong accuracy, precision, recall, and F1-score in predicting cluster membership from transaction features

## 🚀 How to Run

1. Clone this repository.
2. Install dependencies: `pip install pandas numpy scikit-learn yellowbrick matplotlib joblib`
3. Run `[Clustering]_Submission_Akhir_BMLP_Rava_Amesta.ipynb` first to generate `data_clustering.csv` and the clustering model
4. Run `[Klasifikasi]_Submission_Akhir_BMLP_Rava_Amesta.ipynb` to train and evaluate the classification model

## 🎓 Course

This project was submitted as the final assignment for Dicoding's **Belajar Machine Learning untuk Pemula** course
