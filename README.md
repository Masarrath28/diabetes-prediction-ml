# diabetes-prediction-ml
# 🩸 Diabetes Prediction using Machine Learning

This repository features a complete, end-to-end Machine Learning pipeline designed to predict the likelihood of diabetes in patients based on specific medical diagnostic measurements. By leveraging systematic hyperparameter tuning, the final model achieves an exceptional 98% accuracy on unseen test data.


## 📊 Dataset Overview
The dataset was sourced from Kaggl `diabetes.csv`, containing health data for 2,000 patients across 9 diagnostic columns:

| Feature Column | Data Type | Description |
| :--- | :--- | :--- |
| `Pregnancies` | `int64` | Number of times pregnant |
| `Glucose` | `int64` | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| `BloodPressure` | `int64` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | `int64` | Triceps skin fold thickness (mm) |
| `Insulin` | `int64` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | `float64` | Body mass index (weight in kg/(height in m)^2) |
| `DiabetesPedigreeFunction`| `float64` | Diabetes pedigree function (genetic score) |
| `Age` | `int64` | Age in years |
| `Outcome` | `int64` | **Target Variable** (0 = Non-Diabetic, 1 = Diabetic) |


## 🛠️ Data Preprocessing & Engineering
A significant portion of the project focused on data quality to ensure high predictive performance:
Data Cleaning & Missing Values: Handled structural defects and zero-values in columns where zeros physically make no sense (e.g., Glucose, Blood Pressure, BMI) by applying domain-appropriate imputation.
Exploratory Data Analysis (EDA):Utilized `seaborn` and `matplotlib` to plot distributions, identify outliers, and check feature correlations.
Data Splitting:Divided the data using a strict 80/20 train-test split:
  * `X_train` size: `(1600, 8)`
  * `X_test` size: `(400, 8)`
Feature Scaling: Applied StandardScaler from Scikit-Learn to standardize the feature ranges, ensuring algorithm stability.

---

## 🧠 Model Selection & Hyperparameter Tuning
Instead of guessing the best algorithm, `GridSearchCV` combined with a `ShuffleSplit` cross-validation strategy was used to systematically test and tune multiple models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Classifier (SVC)

### The Winning Model: Random Forest
Out of all evaluated pipelines, the **Random Forest Classifier** yielded the strongest cross-validation scores and was selected as the final model.

---

## 📈 Model Performance & Evaluation
To monitor for overfitting, the final Random Forest model was thoroughly validated using a Confusion Matrix on both segments:

Training Set Accuracy: `99%`
Test Set Accuracy (Unseen Data): `98%`



## 💾 Model Deployment Ready
The fully trained, finalized Random Forest model and corresponding preprocessing scalers have been serialized and exported as a Pickle file (`.pkl`). This makes the model production-ready, allowing it to be easily integrated into a Flask/Streamlit web application or backend API for real-time predictions.

---

