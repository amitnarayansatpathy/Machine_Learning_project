# Machine_Learning_project
3_1 ML course semester project

#  ML for SDGs — Air Quality Prediction (Team 18)

> **BITS F464 – Machine Learning | Semester 1**
> Project under the *Machine Learning for Sustainable Development Goals (SDGs)* initiative.

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [SDG Alignment](#-sdg-alignment)
3. [Repository Structure](#-repository-structure)
4. [Dataset Description](#-dataset-description)
5. [Feature Description](#-feature-description)
6. [Methodology](#-methodology)
7. [Machine Learning Models](#-machine-learning-models)
   - [Model 1: Artificial Neural Network (ANN)](#model-1-artificial-neural-network-ann)
   - [Model 2: Random Forest Regressor](#model-2-random-forest-regressor)
   - [Model 3: Gaussian Naïve Bayes Classifier](#model-3-gaussian-naïve-bayes-classifier)
   - [Model 4: Weighted K-Nearest Neighbours (KNN) — Research Literature](#model-4-weighted-k-nearest-neighbours-knn--research-literature)
   - [Additional: Variable AutoRegression (VAR)](#additional-variable-autoregression-var)
   - [Additional: XGBoost (from scratch)](#additional-xgboost-from-scratch)
8. [Model Performance Summary](#-model-performance-summary)
9. [Key Insights](#-key-insights)
10. [Installation & Usage](#-installation--usage)
11. [Team](#-team)
12. [References](#-references)

---

##  Project Overview

Air pollution is one of the leading environmental health risks globally, contributing to respiratory and cardiovascular diseases, threatening biodiversity, and exacerbating climate change. This project applies machine learning techniques to **predict the Air Quality Index (AQI)** at any given point in time using historical sensor and meteorological data collected at a road-level monitoring station in an Italian city.

The **insight deliverable** is:
> *Identify the air quality at any given point in time based on historical data.*

All models have been **built from scratch** using NumPy and Python — without relying on pre-built sklearn estimators for the core algorithms — fulfilling the academic requirement of implementing models at the algorithmic level.

---

##  SDG Alignment

| SDG | Relevance |
|-----|-----------|
| **SDG 3** — Good Health and Well-Being | Air pollution prediction helps prevent respiratory and cardiovascular diseases |
| **SDG 7** — Affordable and Clean Energy | Encourages a shift from dirty fuels towards cleaner energy sources |
| **SDG 11** — Sustainable Cities and Communities | Supports maintaining safe PM levels in urban environments |
| **SDG 13** — Climate Action | Reducing air pollution directly addresses climate change drivers |
| **SDG 15** — Life on Land | Clean air protects ecosystems and biodiversity |

---

##  Repository Structure

```
ml-sdg-air-quality/
│
├── data/
│   └── Air_Quality.csv                    # Raw dataset (semicolon-delimited, 9471 rows × 15 features)
│
├── docs/
│   └── Air Quality_Feature Description.docx  # Official feature descriptions from project brief
│
├── notebooks/
│   └── Team18_Project.ipynb               # Full Jupyter Notebook with all preprocessing, models & analysis
│
├── Team18_Project.pdf                     # Exported PDF report of the project notebook
│
└── README.md                              # This file
```

---

##  Dataset Description

| Property | Value |
|----------|-------|
| **Source** | UCI Machine Learning Repository – Air Quality Dataset |
| **Period** | March 2004 – April 2005 |
| **Frequency** | Hourly readings |
| **Raw Rows** | 9,471 |
| **Rows after Cleaning** | 6,941 |
| **Features** | 15 (13 numeric + Date + Time) |
| **Missing Value Encoding** | `-200` (replaced with `NaN` during preprocessing) |
| **Delimiter** | Semicolon (`;`) |
| **Decimal Separator** | Comma (`,`) — converted to dot during preprocessing |

The dataset contains **hourly recordings** from a multi-sensor device array co-located alongside a certified reference analyser. Sensor responses are from metal-oxide chemical sensors (tin oxide, titania, tungsten oxide, indium oxide).

---

##  Feature Description

| Feature | Description | Unit |
|---------|-------------|------|
| `Date` | Recording date | DD/MM/YYYY |
| `Time` | Recording time | HH.MM.SS |
| `CO(GT)` | True hourly averaged CO concentration (reference analyser) | mg/m³ |
| `PT08.S1(CO)` | Tin oxide sensor response — nominally CO targeted | — |
| `NMHC(GT)` | True hourly averaged Non-Methanic HydroCarbons concentration (reference analyser) *dropped due to >70% missing values* | µg/m³ |
| `C6H6(GT)` | True hourly averaged Benzene concentration (reference analyser) | µg/m³ |
| `PT08.S2(NMHC)` | Titania sensor response — nominally NMHC targeted | — |
| `NOx(GT)` | True hourly averaged NOx concentration (reference analyser) | ppb |
| `PT08.S3(NOx)` | Tungsten oxide sensor response — nominally NOx targeted | — |
| `NO2(GT)` | True hourly averaged NO₂ concentration (reference analyser) | µg/m³ |
| `PT08.S4(NO2)` | Tungsten oxide sensor response — nominally NO₂ targeted | — |
| `PT08.S5(O3)` | Indium oxide sensor response — nominally O₃ targeted | — |
| `T` | Temperature | °C |
| `RH` | Relative Humidity | % |
| `AH` | Absolute Humidity | — |

> **Engineered Target — `AQI` (Air Quality Index):** Computed from sub-indices of CO, NOx, and NO₂ using standard AQI breakpoint formulas. The AQI is the **maximum sub-index** across all pollutants for each hourly sample.

---

##  Methodology

### 1. Data Preprocessing

- **Type Conversion:** Comma-decimal strings (e.g., `"2,6"`) converted to floats.
- **Missing Value Handling:** `-200` sentinel values replaced with `NaN`; `NMHC(GT)` column dropped entirely due to >70% missingness.
- **Row Dropping:** Remaining rows with any `NaN` retained for clean training data (6,941 rows after cleaning).
- **AQI Engineering:** Three pollutant sub-indices (`CO_SubIndex`, `NOx_SubIndex`, `NO2_SubIndex`) computed using EPA breakpoint tables. The final `AQI = max(sub-indices)`.
- **Feature Scaling:** `StandardScaler` applied on all model inputs.
- **Train/Test Split:** 80/20 split via a custom `train_test_split` using random permutation.
- **Correlation Analysis:** Pairplot and correlation heatmap confirmed high multicollinearity between sensor responses (`PT08.*`) and their ground-truth counterparts.

### 2. Modelling Strategy

| Task | Models Used |
|------|-------------|
| **Regression** (predict continuous AQI) | ANN, Random Forest, XGBoost |
| **Classification** (predict AQI bin/category) | Gaussian Naïve Bayes, Weighted KNN |
| **Multivariate Time-Series** | VAR (Variable AutoRegression) |

For classification, AQI was binned using quantile-based discretisation:
- **Naïve Bayes:** 4 bins → `Good`, `Moderate`, `Unhealthy`, `Hazardous`
- **KNN:** 6 bins → `Good`, `Moderate`, `Unhealthy for Sensitive Groups`, `Unhealthy`, `Very Unhealthy`, `Hazardous`

---

##  Machine Learning Models

### Model 1: Artificial Neural Network (ANN)

Fully implemented **from scratch** using NumPy with a 4-layer architecture:

```
Input (9) → Hidden Layer 1 (30) → Hidden Layer 2 (30) → Hidden Layer 3 (10) → Output (1)
```

| Parameter | Value |
|-----------|-------|
| Activation | Sigmoid |
| Learning Rate | 0.0001 |
| Epochs | 50,000 |
| Loss Function | Mean Squared Error |
| Backpropagation | Manual gradient computation |

**Features used:** `PT08.S1(CO)`, `C6H6(GT)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH`  
*(Raw pollutant concentrations and sub-indices excluded to avoid target leakage)*

---

### Model 2: Random Forest Regressor

A custom **Random Forest** built from scratch on top of a custom `DecisionTreeRegressor`:

| Parameter | Value |
|-----------|-------|
| Number of Trees | 10 |
| Split Criterion | Variance Reduction (Information Gain) |
| Bootstrap Sampling | Yes (with replacement) |
| Prediction Aggregation | Mean of tree predictions |

Each `DecisionTreeRegressor` uses:
- Greedy best-split search over randomly chosen feature subsets
- `max_depth=100`, `min_samples_split=2`

---

### Model 3: Gaussian Naïve Bayes Classifier

Custom implementation of the **Gaussian Naïve Bayes** classifier:

- Assumes Gaussian distribution of features within each class
- Uses log-posterior for numerical stability:  
  `posterior = log(prior) + Σ log(P(x_i | class))`
- AQI binned into **4 quantile classes**: `Good`, `Moderate`, `Unhealthy`, `Hazardous`

---

### Model 4: Weighted K-Nearest Neighbours (KNN) — Research Literature

A **Weighted KNN Classifier** sourced from ML research literature on distance-weighted voting:

| Parameter | Value |
|-----------|-------|
| Distance Metric | Euclidean |
| Optimal k | 7 (validated via accuracy vs. k sweep from 1–30) |
| Weight Scheme | Inverse distance: `w = 1 / (d + ε)` |
| AQI Classes | 6 quantile bins |

Voting is performed by computing inverse-distance weighted votes for each candidate class, selecting the class with the highest total weight.

---

### Additional: Variable AutoRegression (VAR)

A **VAR model** (order `p=10`) was applied to the multivariate time series of all pollutant/sensor features, forecasting each variable simultaneously using OLS:

- Demonstrates that temporal patterns in pollutants can serve as predictors for AQI at future timestamps.
- RMSE computed per feature as a measure of individual predictability.

---

### Additional: XGBoost (from scratch)

A custom **gradient boosting** implementation (XGBoost-style):

| Parameter | Value |
|-----------|-------|
| Number of Estimators | 100 |
| Learning Rate | 0.2 |
| Max Depth | 3 |
| Loss | Square Loss (gradient = y − ŷ) |

Each iteration fits a `DecisionTreeRegressor` to the current gradient residuals, and predictions are accumulated as a weighted sum.

---

##  Model Performance Summary

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| **ANN** | Regression | R² Score | **0.9339** |
| **Random Forest** | Regression | R² Score | **0.9153** |
| **Random Forest** | Regression | MSE | 0.0836 |
| **XGBoost** | Regression | R² Score | **0.9154** |
| **XGBoost** | Regression | MSE | 0.0858 |
| **Gaussian Naïve Bayes** | Classification (4-class) | Accuracy | 0.5745 |
| **Weighted KNN (k=7)** | Classification (6-class) | Accuracy | **~0.68** |
| **Weighted KNN (k=7)** | Classification (6-class) | Macro F1 | 0.68 |
| **VAR (p=10)** | Time-Series Forecasting | Avg RMSE | Per-variable (see notebook) |

>  **Best Performer:** ANN achieved the highest R² of **0.93**, closely followed by XGBoost and Random Forest (both ~0.915). The regression models significantly outperformed classifiers, indicating AQI is better predicted as a continuous value than discretised into bins.

---

##  Key Insights

1. **AQI is highly predictable** from sensor responses alone (R² > 0.91 for all regression models), validating the utility of low-cost metal-oxide sensors as AQI proxies.
2. **Sensor responses** (`PT08.*`) are strongly correlated with their reference-analyser counterparts and serve as effective features when ground-truth concentrations are unavailable.
3. **NMHC(GT)** was dropped due to over 70% missing data without impacting model performance, suggesting sensor-based features can substitute for reference analysers.
4. **ANN (R² = 0.934)** slightly outperforms ensemble methods due to its capacity to model non-linear interactions between pollutants.
5. **Naïve Bayes (57%)** underperforms classification because the conditional independence assumption is violated — pollutants are highly correlated.
6. **Weighted KNN (68%)** outperforms Naïve Bayes on the 6-class problem, benefiting from distance-based locality for non-linear decision boundaries.
7. **VAR model** confirms that pollutant concentrations exhibit temporal autocorrelation, enabling short-term forecasting of air quality from historical data.
8. **Confusion matrix analysis** for all classifiers shows best performance on extreme classes (`Good` / `Hazardous`) and poorest on intermediate bins, reflecting the ambiguity of mid-range AQI.

---

##  Installation & Usage

### Prerequisites

```bash
Python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn       # Only for preprocessing utilities (StandardScaler, LabelEncoder, metrics)
```

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/ml-sdg-air-quality-team18.git
cd ml-sdg-air-quality-team18

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the Notebook

```bash
jupyter notebook notebooks/Team18_Project.ipynb
```

> **Note:** The notebook originally loaded the dataset from a Google Drive URL. With this repo, it uses the local `data/Air_Quality.csv` file. Update the data loading cell to:
> ```python
> df = pd.read_csv('../data/Air_Quality.csv', sep=';')
> ```



##  References

1. Wang S, Ren Y, Xia B. *Estimation of urban AQI based on interpretable machine learning.* Environ Sci Pollut Res Int. 2023 Sep;30(42):96562–96574. doi: 10.1007/s11356-023-29336-5.

2. Méndez M, Merayo MG & Núñez M. *Machine learning algorithms to forecast air quality: a survey.* Artif Intell Rev 56, 10031–10066 (2023).

3. Lei TMT, Ng SCW, and Siu SWI. *Application of ANN, XGBoost, and Other ML Methods to Forecast Air Quality in Macau.* Sustainability 15, no. 6: 5341 (2023).

4. Breiman L. *Random Forests.* Machine Learning, 45, 5–32 (2001).

5. Zhang H. *The optimality of Naive Bayes.* FLAIRS Conference, 2004.

6. Chen T, Guestrin C. *XGBoost: A Scalable Tree Boosting System.* KDD '16 (2016).

7. Cover TM, Hart PE. *Nearest neighbor pattern classification.* IEEE Trans. Inf. Theory 13(1):21–27 (1967).

---

