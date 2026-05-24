# Insurance Cost Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting medical insurance charges using Machine Learning algorithms. The model analyzes customer information such as age, BMI, smoking habits, gender, region, and number of children to estimate insurance costs accurately.

The project demonstrates the complete Data Science workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and comparison of multiple regression algorithms.

---

# 🎯 Business Problem

Insurance companies need an efficient way to estimate medical insurance costs based on customer health and demographic information. Manual estimation can be time-consuming and inconsistent.

This project helps automate insurance charge prediction using Machine Learning models.

---

# 📂 Dataset Information

The dataset contains the following features:

| Feature  | Description                              |
| -------- | ---------------------------------------- |
| age      | Age of the customer                      |
| sex      | Gender                                   |
| bmi      | Body Mass Index                          |
| children | Number of children                       |
| smoker   | Smoking status                           |
| region   | Residential region                       |
| charges  | Medical insurance cost (Target Variable) |

---

# ⚙️ Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn (sklearn)
* Jupyter Notebook

---

# 🔍 Project Workflow

## 1. Data Collection

* Loaded dataset using Pandas
* Checked dataset shape and structure

## 2. Data Cleaning

* Checked missing values
* Removed duplicates
* Verified data types

## 3. Exploratory Data Analysis (EDA)

* Analyzed feature distributions
* Visualized relationships between variables
* Identified important factors affecting insurance charges

## 4. Feature Engineering

* Applied One-Hot Encoding for categorical variables
* Prepared data for Machine Learning models

## 5. Model Building

Implemented multiple regression models:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

## 6. Model Evaluation

Evaluated models using:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

---

# 📊 Model Performance

| Model                   | R² Score |
| ----------------------- | -------- |
| Linear Regression       | 0.8069   |
| Decision Tree Regressor | 0.8189   |
| Random Forest Regressor | 0.8797   |

✅ **Best Performing Model:** Random Forest Regressor

---

# 📈 Key Insights

* Smoking status has a major impact on insurance charges.
* Higher BMI and age generally increase medical costs.
* Random Forest provided the best prediction accuracy among all models.

---

# 🚀 Future Improvements

* Hyperparameter tuning
* Deploy model using Flask or Streamlit
* Use larger real-world datasets
* Apply advanced algorithms like XGBoost

---

# 🧠 Skills Demonstrated

* Data Cleaning
* Data Visualization
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning
* Regression Modeling
* Model Evaluation
* Python Programming

---

# 📁 Project Structure

```bash
Insurance-Cost-Prediction/
│
├── data/
│   └── insurance.csv
│
├── notebooks/
│   └── Insurance_Cost_Prediction.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# ▶️ How to Run the Project

## Clone the Repository

```bash
git clone <your-github-repository-link>
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Jupyter Notebook

```bash
jupyter notebook
```

---

# 📌 Conclusion

This project successfully predicts medical insurance charges using Machine Learning techniques. Among all models, Random Forest Regressor achieved the highest accuracy and provided the best prediction performance.

---

# 👨‍💻 Author

**Vivek Bhosale**
Aspiring Data Scientist & Data Analyst

---
