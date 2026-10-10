# Insurance Cost Prediction

A machine learning project that estimates medical insurance charges from customer attributes. The project covers data inspection, cleaning, exploratory data analysis (EDA), feature preparation, regression model comparison, and a saved prediction pipeline with a Streamlit interface.

The **Random Forest Regressor** achieved the best recorded test performance among the three evaluated models, with an **R² score of 0.9012**.

## Contents

- [Project objective](#project-objective)
- [Dataset](#dataset)
- [Project files](#project-files)
- [Tools and technologies](#tools-and-technologies)
- [Project workflow](#project-workflow)
- [Model performance](#model-performance)
- [Feature importance](#feature-importance)
- [Final prediction pipeline](#final-prediction-pipeline)
- [Setup and usage](#setup-and-usage)
- [Sample prediction](#sample-prediction)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)
- [Conclusion](#conclusion)

## Project objective

Predict the numerical `charges` target using six input attributes:

- Age
- Sex
- Body mass index (BMI)
- Number of children
- Smoking status
- Region

This is a **supervised learning regression problem**. The project also investigates which inputs contribute most to the selected model's predictions.

## Dataset

The notebook reads the local `insurance.csv` file using pandas.

| Property | Value |
| --- | --- |
| Original records | 1,338 |
| Columns | 7 |
| Input variables | 6 |
| Target | `charges` |
| Missing values | 0 in every column |
| Duplicate records removed | 1 |
| Records used for EDA and modeling | 1,337 |

### Data dictionary

| Column | Type | Description |
| --- | --- | --- |
| `age` | Integer | Age; observed range: 18–64. |
| `sex` | Categorical | Recorded values: `female`, `male`. |
| `bmi` | Float | Body mass index; observed range: 15.96–53.13. |
| `children` | Integer | Number of children recorded; range: 0–5. |
| `smoker` | Categorical | Smoking status: `no`, `yes`. |
| `region` | Categorical | `northeast`, `northwest`, `southeast`, `southwest`. |
| `charges` | Float | Continuous target representing recorded medical insurance charges. |

The original dataset has mean charges of **13,270.42** and median charges of **9,382.03**, consistent with the right-skewed distribution shown in the notebook.

**Target units:** The sample notebook output labels its prediction as `Rs`, but the dataset's currency, target period, and currency conversion are not documented. Predictions are therefore described here in **dataset charge units**.

## Project files

Place this README in the same folder as the project's core files:

```text
Insurance_Cost_Prediction/
├── README.md
├── Insurance_Cost_Prediction.ipynb
├── insurance.csv
├── insurance_pipeline.pkl
├── app.py
└── requirements.txt
```

| File | Purpose |
| --- | --- |
| `Insurance_Cost_Prediction.ipynb` | Data analysis, charts, training, evaluation, feature importance, and pipeline creation. |
| `insurance.csv` | Input dataset. |
| `insurance_pipeline.pkl` | Saved fitted preprocessing and Random Forest pipeline. |
| `app.py` | Streamlit application for entering customer details and displaying an estimate. |
| `requirements.txt` | Application dependencies and recorded package pins. |

The project also has supporting report and presentation files. The detailed PDF report documents the notebook's methods, original plots, findings, and limitations.

## Tools and technologies

| Tool | Role |
| --- | --- |
| Python | Programming language. |
| pandas | Data loading, inspection, preparation, and customer input tables. |
| NumPy | Numerical operations and RMSE calculation. |
| Matplotlib and Seaborn | Distributions, scatter plots, boxplots, correlations, and feature importance charts. |
| scikit-learn | Data splitting, preprocessing, regressors, pipelines, and evaluation metrics. |
| joblib | Pipeline serialization and loading. |
| Jupyter Notebook | Interactive project analysis. |
| Streamlit | Customer input and prediction interface. |

The notebook records Python **3.13.5**, scikit-learn **1.6.1**, pandas **2.2.3**, NumPy **2.1.3**, and joblib **1.4.2**. Matplotlib, Seaborn, and Streamlit versions are not recorded in its version output.

## Project workflow

### 1. Data loading and inspection

Loaded `insurance.csv` and inspected the data with:

```python
dataset.head()
dataset.info()
dataset.describe()
dataset.shape
```

These checks established the column types, record count, and numerical distributions.

### 2. Data cleaning

- Checked missing values using `dataset.isnull().sum()`.
- Identified one exact duplicate using `dataset.duplicated().sum()`.
- Removed it using `dataset.drop_duplicates()`.
- Verified that no duplicate rows or duplicate column names remained.

No imputation, feature scaling, outlier removal, or target transformation was applied.

### 3. Exploratory data analysis

The notebook includes 16 charts covering:

- Distributions of charges, age, BMI, children, sex, region, and smoking status.
- Charges grouped by smoking status, sex, and region.
- Age versus charges and BMI versus charges.
- Charges versus number of children.
- An encoded correlation heatmap.
- Actual versus predicted charges for Linear Regression.
- Random Forest feature importance.

Key observations:

- Charges have a long right tail.
- Smokers have a visibly higher charge distribution than non-smokers.
- Age and BMI show associations with charges, with considerable variation at similar input values.
- Charge distributions overlap across the recorded sex and region categories.

The children-versus-charges chart is a **scatter plot**, although its original notebook title calls it a boxplot.

### 4. Feature preparation

Separated the target from the predictors:

```python
x = dataset.drop("charges", axis=1)
y = dataset["charges"]
x = pd.get_dummies(x, drop_first=True)
```

The modeling input contains eight features:

```text
age
bmi
children
sex_male
smoker_yes
region_northwest
region_southeast
region_southwest
```

The omitted reference categories are `female`, `no`, and `northeast`.

A separate integer-encoded copy is used only for the correlation heatmap. Its region codes do not imply an ordered geographical relationship.

### 5. Train/test split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

| Partition | Records | Input features |
| --- | --- | --- |
| Training | 1,069 | 8 |
| Testing | 268 | 8 |

The three comparison models use the same split.

### 6. Model training and evaluation

Trained these regressors:

```python
LinearRegression()
DecisionTreeRegressor(random_state=42)
RandomForestRegressor(max_depth=4, random_state=42)
```

Evaluated their test predictions using MAE, MSE, RMSE, and R². Other model parameters were left at the installed library defaults. The notebook does not implement hyperparameter search or cross-validation.

### 7. Model selection and persistence

Selected Random Forest based on its strongest recorded test metrics, inspected its feature importance, and rebuilt it as a reusable preprocessing-plus-model pipeline. Saved the fitted pipeline with joblib, reloaded it, and tested a sample customer.

## Model performance

The following values are taken from the notebook's saved test outputs and rounded for readability:

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 4,177.05 | 35,478,020.68 | 5,956.34 | 0.8069 |
| Decision Tree Regressor | 2,730.63 | 33,281,524.62 | 5,769.01 | 0.8189 |
| **Random Forest Regressor** | **2,478.18** | **18,153,245.21** | **4,260.66** | **0.9012** |

### Metric interpretation

- **MAE:** Average absolute prediction error in charge units. Lower is better.
- **MSE:** Average squared prediction error in squared charge units. Larger errors receive greater weight.
- **RMSE:** Square root of MSE, expressed in charge units. Lower is better.
- **R²:** Fit relative to predicting the test-set mean. Higher is better.

Random Forest achieved the lowest errors and highest R² among the three tested models. Relative to Linear Regression, it reduced MAE by approximately **40.67%** and RMSE by **28.47%**.

An R² of 0.9012 means the model explains approximately 90.12% of test-set variation relative to the mean baseline. It is **not a 90.12% individual prediction accuracy guarantee**.

## Feature importance

The fitted comparison Random Forest reports these impurity-based feature importances:

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

Smoking status, BMI, and age together account for approximately **98.80%** of the displayed importance. These values describe this model's use of inputs; they do not establish causality or represent percentage changes in charges.

## Final prediction pipeline

The final notebook pipeline accepts the original six named input columns and performs encoding internally:

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

insurance_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(max_depth=4, random_state=42)),
    ]
)
```

- `ColumnTransformer` applies categorical encoding to `sex`, `smoker`, and `region`.
- `remainder="passthrough"` retains `age`, `bmi`, and `children`.
- `OneHotEncoder` creates binary indicators and ignores unknown categories during transformation.
- The Random Forest predicts charges from the transformed inputs.

The pipeline is fitted on the training portion of the raw cleaned data using the same 80/20 split. Its recorded test R² is **0.9012100890157556**.

```python
insurance_pipeline.fit(X_train, y_train)
joblib.dump(insurance_pipeline, "insurance_pipeline.pkl")
```

The notebook saves this training-fitted pipeline; it does not refit the model on all cleaned records before saving.

## Setup and usage

Run the following commands **from the project folder** containing `app.py`, `insurance.csv`, and `insurance_pipeline.pkl`. The existing code uses paths relative to the working directory.

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

The commands below use the virtual environment's Python directly, so activation is optional.

### 2. Install application dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The supplied requirements are:

```text
streamlit
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.6.1
joblib==1.4.2
```

For notebook work, also install its visualization dependencies and Jupyter:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib seaborn notebook
```

These additional packages are not listed in the supplied `requirements.txt`.

### 3. Explore or retrain in the notebook

```powershell
.\.venv\Scripts\python.exe -m notebook Insurance_Cost_Prediction.ipynb
```

Select the environment containing the installed dependencies and run the cells in order. The final pipeline section creates or overwrites `insurance_pipeline.pkl` in the working directory.

### 4. Run the Streamlit app

If the saved pipeline is already available, you can launch the app without retraining:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed by Streamlit, enter the six customer attributes, and select **Predict Insurance Cost**. The app creates a one-row DataFrame, calls the loaded pipeline, and displays the estimated charges.

The app's input ranges are age 18–64, BMI 15.0–55.0, and children 0–5. The BMI input range is slightly wider than the range observed in the dataset.

### Common setup issues

| Issue | Check |
| --- | --- |
| `insurance.csv` not found | Start Jupyter from the folder containing the CSV. |
| `insurance_pipeline.pkl` not found | Run the final pipeline cells or place the saved file beside `app.py`; launch the app from that folder. |
| Missing Matplotlib or Seaborn | Install the notebook dependencies listed above. |
| Saved-model version warning | Use the recorded package versions, especially scikit-learn 1.6.1, or retrain and save the pipeline in the intended environment. |

## Sample prediction

```python
import joblib
import pandas as pd

loaded_pipeline = joblib.load("insurance_pipeline.pkl")

sample_customer = pd.DataFrame({
    "age": [30],
    "sex": ["male"],
    "bmi": [25.5],
    "children": [2],
    "smoker": ["no"],
    "region": ["southeast"],
})

prediction = loaded_pipeline.predict(sample_customer)[0]
print(f"Estimated charges: {prediction:,.2f}")
```

The notebook's saved result for this profile is:

```text
Estimated charges: 6,579.06
```

This is an example inference in dataset charge units. No actual charge is supplied for this customer, so the example does not measure prediction accuracy.

## Limitations

- Model comparison uses one random train/test split; no cross-validation or repeated-split analysis is recorded.
- The same test set is used to compare models and choose the winner, so a separate untouched holdout would strengthen the final assessment.
- No hyperparameter search is performed.
- The notebook shows prediction diagnostics for Linear Regression, but not detailed residual or subgroup analysis for the selected Random Forest.
- Dataset provenance, verified currency, and the target's time period are not documented.
- Feature importance reflects the fitted model and does not establish causal relationships.
- The pipeline and Streamlit source support local inference; hosted deployment and validation on new external records are not demonstrated.

## Future improvements

These are proposed extensions rather than completed notebook steps:

- Tune model settings using cross-validation on training data.
- Evaluate the chosen model on a separate final holdout.
- Add a mean-prediction baseline, residual plots, and Random Forest actual-versus-predicted charts.
- Compare error by smoking status, age group, and region.
- Validate the importance ranking with held-out permutation importance.
- Document dataset origin, currency, and the definition of `charges`.
- Add input validation, dependency versioning, and monitoring if the application is deployed.

## Conclusion

The project completes a practical regression workflow from data exploration to saved-pipeline inference. Random Forest achieved the strongest recorded held-out performance among the three evaluated models, and smoking status, BMI, and age were the leading inputs in its feature importance ranking.

**Evidence basis:** Metrics, feature importances, environment versions, and the sample estimate in this README come from the notebook's saved outputs. The Streamlit description is based on the adjacent `app.py`. The notebook and application were not rerun to prepare this documentation.
