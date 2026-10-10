<a id="top"></a>

<div align="center">



# 🛡️ Insurance Cost Prediction

### From customer attributes to estimated medical insurance charges

Explore the data, compare regression models, and try the Streamlit predictor.

[![Open the live app](https://img.shields.io/badge/OPEN_LIVE_APP-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://insurance-cost-prediction-public.streamlit.app/)
[![Explore the notebook](https://img.shields.io/badge/EXPLORE_NOTEBOOK-142D4E?style=for-the-badge&logo=jupyter&logoColor=white)](Insurance_Cost_Prediction.ipynb)

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2.3-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.1.3-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

**[Overview](#overview) · [Live demo](#live-demo) · [Results](#results) · [Insights](#insights) · [Run locally](#run-locally) · [Author](#author)**

</div>

---

| 🗃️ Cleaned records | 🧪 Models compared | 🏆 Best test R² | ⚙️ Selected model |
| :---: | :---: | :---: | :---: |
| **1,337** | **3** | **0.9012** | **Random Forest** |

<a id="overview"></a>

## 📌 Project overview

How much can a customer's age, BMI, smoking status, and other attributes tell us about their medical insurance charges?

This project explores that question through **data cleaning**, **exploratory analysis**, and **regression modeling**. It compares Linear Regression, Decision Tree, and Random Forest, then combines the selected model and preprocessing into a reusable prediction pipeline.

### 🎯 Business problem

Estimating charges consistently from recorded customer attributes can support exploratory cost analysis. The goal is to build a data science prototype that predicts the numerical `charges` target and explains the patterns learned from the dataset.

**The result:** Random Forest performed best among the three tested configurations. Smoking status, BMI, and age led its feature importance ranking.

<a id="live-demo"></a>

## 🚀 Try the live project

### [Launch the Insurance Cost Predictor →](https://insurance-cost-prediction-public.streamlit.app/)

1. Enter **age**, **sex**, **BMI**, and **number of children**.
2. Choose **smoking status** and **region**.
3. Click **Predict Insurance Cost** to view the estimated charges.

> 💡 **Try this profile:** age **30**, male, BMI **25.5**, **2** children, non-smoker, southeast. The notebook's saved pipeline returned **6,579.06** for this profile. This is a recorded example, not a guaranteed live-app result.

<details>
<summary><strong>🔎 What happens after you click Predict?</strong></summary>

The Streamlit interface creates a one-row pandas DataFrame. The saved pipeline encodes categorical inputs, passes through numerical inputs, and applies the Random Forest model.

The companion app accepts age **18–64**, BMI **15.0–55.0**, and children **0–5**. The dataset's observed BMI range is slightly narrower: **15.96–53.13**.

The notebook prints `Rs` for the sample, but the dataset's currency and any currency conversion are not documented. Estimates are described here in **dataset charge units**.

</details>

---

## 📂 Dataset information

| Feature | Description | Role |
| --- | --- | --- |
| `age` | Age; observed range 18–64 | Numerical input |
| `sex` | `female` or `male` | Categorical input |
| `bmi` | Body mass index; observed range 15.96–53.13 | Numerical input |
| `children` | Number of children recorded; range 0–5 | Numerical input |
| `smoker` | `no` or `yes` | Categorical input |
| `region` | northeast, northwest, southeast, southwest | Categorical input |
| `charges` | Recorded medical insurance charges | **Prediction target** |

<details>
<summary><strong>🧹 Expand data quality and split details</strong></summary>

| Check | Finding / action |
| --- | --- |
| Original size | 1,338 rows × 7 columns |
| Missing values | None in any column |
| Duplicate records | One exact duplicate removed |
| Cleaned size | 1,337 rows |
| Duplicate column names | None |
| Training set | 1,069 records |
| Test set | 268 records |
| Split settings | `test_size=0.2`, `random_state=42` |

No imputation, scaling, outlier removal, or target transformation was applied. The original-data mean charge is **13,270.42**, compared with a median of **9,382.03**, consistent with a right-skewed charge distribution.

</details>

## 🔍 Project workflow

| Stage | Work completed |
| --- | --- |
| **01 · Understand** | Loaded the CSV and inspected shape, types, statistics, and example records. |
| **02 · Clean** | Checked nulls, removed one duplicate, and verified column names. |
| **03 · Explore** | Created 16 plots covering distributions, relationships, correlations, predictions, and importance. |
| **04 · Prepare** | Separated `charges` and one-hot encoded the categorical predictors. |
| **05 · Train** | Used the same 80/20 split to fit three regressors. |
| **06 · Evaluate** | Compared MAE, MSE, RMSE, and R² on held-out records. |
| **07 · Package** | Combined encoding and Random Forest in a pipeline and saved it with joblib. |
| **08 · Predict** | Reloaded the pipeline, tested a sample customer, and provided a Streamlit interface. |

<details>
<summary><strong>📊 Explore the correlation heatmap</strong></summary>

![Correlation heatmap from the project notebook](assets/correlation-heatmap.png)

Smoking status has the strongest positive target correlation in the heatmap, followed by age and BMI. This figure uses a separately encoded copy of the cleaned dataset. Region is nominal, so its arbitrary numeric codes do not imply a geographical order. Correlation does not establish causation.

</details>

<a id="results"></a>

## 🏆 Model performance

The following values come from the notebook's saved test outputs.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: |
| Linear Regression | 4,177.05 | 5,956.34 | 0.8069 |
| Decision Tree Regressor | 2,730.63 | 5,769.01 | 0.8189 |
| 🥇 **Random Forest Regressor** | **2,478.18** | **4,260.66** | **0.9012** |

**Selected configuration:** `RandomForestRegressor(max_depth=4, random_state=42)`

Compared with Linear Regression, Random Forest reduced **MAE by 40.67%** and **RMSE by 28.47%** on the recorded test split.

<details>
<summary><strong>📐 Understand the metrics and full comparison</strong></summary>

| Metric | Meaning |
| --- | --- |
| **MAE** | Average absolute error in charge units. Lower is better. |
| **MSE** | Average squared error in squared charge units; emphasizes larger errors. |
| **RMSE** | Square root of MSE, in charge units. Lower is better. |
| **R²** | Fit relative to predicting the test-set mean. Higher is better. |

| Model | MSE |
| --- | ---: |
| Linear Regression | 35,478,020.68 |
| Decision Tree Regressor | 33,281,524.62 |
| Random Forest Regressor | 18,153,245.21 |

R² = **0.9012** means approximately **90.12% of test-set variation** is explained relative to the mean baseline. It is not an individual prediction accuracy percentage.

Linear Regression uses its default constructor. Decision Tree specifies `random_state=42`. Random Forest specifies `max_depth=4` and `random_state=42`; remaining settings use library defaults. No hyperparameter search is recorded.

</details>

<a id="insights"></a>

## 📈 Key insights

- **Smoking status leads**, contributing approximately 69.36% of the fitted forest's reported importance.
- **BMI and age follow**, contributing approximately 17.58% and 11.86%.
- **Charges have a long right tail**, with fewer high-charge records.
- **The nonlinear models improve on the linear baseline** in this recorded comparison.



<details>
<summary><strong>🌲 View the complete feature importance table</strong></summary>

| Feature | Importance | Approximate share |
| --- | ---: | ---: |
| `smoker_yes` | 0.693633 | 69.36% |
| `bmi` | 0.175816 | 17.58% |
| `age` | 0.118560 | 11.86% |
| `children` | 0.011461 | 1.15% |
| `region_southeast` | 0.000227 | 0.0227% |
| `sex_male` | 0.000184 | 0.0184% |
| `region_southwest` | 0.000086 | 0.0086% |
| `region_northwest` | 0.000032 | 0.0032% |

These impurity-based importances describe the fitted model's use of inputs, not causal effects or percentage changes in charges. Smoking status, BMI, and age together account for approximately **98.80%** of the displayed importance.

</details>

## ⚙️ Technologies & libraries

| Area | Tools |
| --- | --- |
| Data preparation | Python, pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | scikit-learn |
| Model persistence | joblib |
| Analysis | Jupyter Notebook |
| Prediction interface | Streamlit |

<details>
<summary><strong>🧩 Inspect the final prediction pipeline</strong></summary>

The saved pipeline accepts the original six named columns and handles encoding internally.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            ["sex", "smoker", "region"],
        )
    ],
    remainder="passthrough",
)

insurance_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_depth=4, random_state=42)),
])
```

The numerical features pass through directly. The transformer and model are fitted together on the training portion of the raw cleaned inputs. Recorded pipeline test R²: **0.9012100890157556**.

The notebook saves the fitted object as `insurance_pipeline.pkl`; it does not refit on all cleaned records before saving.

</details>

## 📁 Project structure

```text
Insurance_Cost_Prediction/
├── README.md
├── Insurance_Cost_Prediction.ipynb   # Analysis and training
├── insurance.csv                   # Input dataset
├── insurance_pipeline.pkl          # Fitted preprocessing + model
├── app.py                          # Streamlit interface
├── requirements.txt                # Application dependencies
└── assets/
    ├── insurance-banner.svg
    ├── correlation-heatmap.png
    └── feature-importance.png
```

<a id="run-locally"></a>

## ▶️ Run the project locally

Run these commands from the folder containing `app.py`, `requirements.txt`, and `insurance_pipeline.pkl`.

### 1. Create an environment

```powershell
python -m venv .venv
```

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Launch the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed by Streamlit. The commands use the environment's Python directly, so activation is optional.

<details>
<summary><strong>📓 Open the notebook or retrain the model</strong></summary>

Install the notebook dependencies, which are additional to the application requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib seaborn notebook
.\.venv\Scripts\python.exe -m notebook Insurance_Cost_Prediction.ipynb
```

Select the environment containing those dependencies and run the cells in order. Keep `insurance.csv` in the working directory. The final pipeline cells create or overwrite `insurance_pipeline.pkl`.

The notebook records Python **3.13.5**, pandas **2.2.3**, NumPy **2.1.3**, scikit-learn **1.6.1**, and joblib **1.4.2**. Streamlit is unpinned in the supplied requirements; Matplotlib and Seaborn versions are not recorded.

</details>

<details>
<summary><strong>🧪 Make a prediction directly in Python</strong></summary>

```python
import joblib
import pandas as pd

pipeline = joblib.load("insurance_pipeline.pkl")

customer = pd.DataFrame({
    "age": [30],
    "sex": ["male"],
    "bmi": [25.5],
    "children": [2],
    "smoker": ["no"],
    "region": ["southeast"],
})

prediction = pipeline.predict(customer)[0]
print(f"Estimated charges: {prediction:,.2f}")
```

Recorded notebook result:

```text
Estimated charges: 6,579.06
```

No actual charge is supplied for this example, so it does not measure prediction accuracy.

</details>

<details>
<summary><strong>🛠️ Troubleshoot common setup issues</strong></summary>

| Issue | Check |
| --- | --- |
| CSV not found | Start Jupyter from the folder containing `insurance.csv`. |
| Pipeline not found | Keep `insurance_pipeline.pkl` beside `app.py` and launch from that folder, or generate it using the notebook. |
| Missing chart libraries | Install Matplotlib and Seaborn with the notebook command above. |
| Saved-model version warning | Use the recorded dependencies or retrain and save in the intended environment. |

</details>

---

## 🚀 Future improvements

- [ ] Tune models using cross-validation on training data.
- [ ] Evaluate the chosen model on a separate untouched holdout.
- [ ] Add Random Forest residual plots and subgroup error comparisons.
- [ ] Check feature rankings with held-out permutation importance.
- [ ] Document dataset origin, currency, and the target's time period.
- [ ] Add deployment input validation, dependency versioning, and monitoring.

<details>
<summary><strong>📝 Read the current evaluation limitations</strong></summary>

Results use one random 80/20 split. The same test set supports model comparison and selection. No cross-validation or hyperparameter search is reported, and detailed prediction diagnostics are shown for Linear Regression rather than the selected forest.

The dataset's representativeness, currency, and charge period are not documented. Model importance does not establish causality. A public app link is provided; reported notebook scores describe the saved local evaluation rather than a separate audit of the live service.

</details>

## 🧠 Skills demonstrated

**Data cleaning · Exploratory analysis · Data visualization · Categorical encoding · Regression modeling · Model evaluation · Pipeline design · Model persistence · Streamlit application development**

## 📌 Conclusion

The project connects data analysis, model comparison, and reusable prediction. Random Forest delivered the strongest recorded test results among the three tested configurations, while smoking status, BMI, and age led its feature importance ranking.

<a id="author"></a>

## 👨‍💻 Author

**Vivek Bhosale**  
Aspiring Data Scientist & Data Analyst

<div align="center">

**[🚀 Try the live app](https://insurance-cost-prediction-public.streamlit.app/) · [📓 Explore the notebook](Insurance_Cost_Prediction.ipynb) · [↑ Back to top](#top)**

<sub>Metrics and charts come from the supplied notebook's saved outputs. The live app link was supplied by the project author.</sub>

</div>

