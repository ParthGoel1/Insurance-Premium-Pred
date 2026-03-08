# Insurance Premium Prediction

A machine learning project that predicts health insurance premiums based on customer demographics, health indicators, and lifestyle factors. Built as an end-to-end pipeline from raw data to a deployed Streamlit application.

---

## Project Overview

Insurance premium pricing depends on a complex mix of factors — age, pre-existing conditions, lifestyle choices, and genetic risk. This project builds a regression model that predicts the premium amount a customer should be charged, using a segmented modelling approach to handle performance differences across age groups.

The final application allows a user to input their details and receive a predicted insurance premium in real time.

---

## Key Features

- End-to-end ML pipeline: data cleaning, EDA, feature engineering, model training, and deployment
- Segmented modelling strategy based on age group, improving accuracy for younger customers
- Domain-aware preprocessing including disease-to-risk-score mapping and multicollinearity analysis
- Interactive Streamlit application for real-time premium prediction

---

## Dataset

Source: [Codebasics](https://codebasics.io) — provided as part of a structured ML course.

Two datasets were used:
- **General dataset**: customers across all age groups, with features including number of dependants, income, BMI, smoking status, region, medical history, and insurance plan type
- **Age 18–25 dataset**: a separate dataset for younger customers that includes an additional feature — genetic risk — which significantly improved model performance for this segment

---

## Data Preprocessing

The preprocessing stage involved careful, domain-informed cleaning:

- Handled missing values and removed duplicate records
- Corrected erroneous values based on column context — for example, negative values in the number of dependants column were converted to their absolute values
- Mapped medical history entries to numerical risk scores reflecting the relative severity of different conditions
- Conducted univariate and bivariate analysis to understand feature distributions and relationships with the target variable
- Assessed multicollinearity using correlation matrices and Variance Inflation Factor (VIF), removing or combining features where necessary

---

## Modelling Approach

Initial model training revealed that the **18–25 age group** had significantly higher prediction errors, with some predictions deviating by more than 50% from actual values. This pointed to a structural difference in how premiums are determined for younger customers — confirmed by the availability of genetic risk data for this segment.

The solution was a **segmented modelling strategy**:

| Segment | Model | R² Score |
|---|---|---|
| Age 18–25 | XGBoost Regressor | 99.6% |
| Age 26+ | Linear Regression | 97.3% |

Linear Regression was retained for the older segment deliberately — the model performed well and offered strong explainability, which matters in insurance contexts where predictions need to be interpretable.

Both models were serialised using `joblib` and loaded into the Streamlit application at runtime.

---

## Streamlit Application

The deployed application takes user inputs — age, dependants, income level, BMI category, smoking status, region, medical history, and insurance plan — and returns a predicted annual premium.

The application automatically routes the input to the correct model based on the user's age group.

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

- **Python**
- **Pandas, NumPy** — data manipulation and cleaning
- **Matplotlib, Seaborn** — exploratory data analysis
- **Scikit-learn** — Linear Regression, preprocessing, model evaluation
- **XGBoost** — gradient boosted regression for the 18–25 segment
- **Joblib** — model serialisation
- **Streamlit** — application deployment

---

## Project Structure

```
insurance-premium-prediction/
├── data/
│   ├── premiums.csv
│   └── premiums_young_with_gr.csv
├── notebooks/
│   └── insurance_premium_prediction.ipynb
├── models/
│   ├── model_young.joblib
│   └── model_rest.joblib
├── main.py
├── requirements.txt
└── README.md
```

---

## Acknowledgements

Dataset and project structure sourced from the Codebasics ML course. Feature engineering, domain mapping, segmentation strategy, and deployment implemented as part of independent learning.
