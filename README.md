# Freight Rate Prediction

Machine learning pipeline for predicting spot freight rates across US corridors. The model predicts rates for 12,000 loads in `data/validation.csv` and a 31-day test scenario in `data/december-chart-inputs.csv`.

---

## Setup and How to Run

### Requirements
- Python 3.10+ (tested on Python 3.14)
- Dependencies listed in `requirements.txt`

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MohammedTaharwah/freight-rate-prediction.git
   cd freight-rate-prediction
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Alternatively, if using `uv`:*
   ```bash
   uv sync
   ```

4. **Run the scoring script:**
   To validate the generated predictions and produce the December chart:
   ```bash
   python score.py --predictions validation_predictions.csv --december-predictions data/december_predictions.csv
   ```
   Or via `uv`:
   ```bash
   uv run python score.py --predictions validation_predictions.csv --december-predictions data/december_predictions.csv
   ```

   **Expected Output:**
   ```text
   Validated 12,000 final predictions.
   Validated 31 fixed December predictions.
   Created chart: scorer_results/candidate_december.png
   Final validation metrics are calculated by Spotter after submission.
   ```

---

## Repository Structure

```text
freight-rate-prediction/
├── Notebooks/
│   └── freight-rate-prediction.ipynb    # Full exploratory analysis, cleaning, training, and inference
├── data/
│   ├── train-test.csv                   # Historical training data (Jan 1 - Oct 31, 2025)
│   ├── validation.csv                   # Target evaluation set (12,000 loads, Nov 2025)
│   ├── validation-predictions-template.csv
│   ├── december-chart-inputs.csv        # 31-day fixed input template for Dec 2025
│   └── december_predictions.csv         # Generated predictions for December 2025
├── scorer_results/
│   └── candidate_december.png           # Scorer-generated December plot
├── src/                                 # Project source package
├── requirements.txt                     # Pinned dependencies
├── pyproject.toml                       # Build & project metadata
├── score.py                             # Evaluation and chart generation script
├── validation_predictions.csv           # 12,000 predictions for validation.csv
└── README.md                            # Project documentation
```

---

## Exploratory Data Analysis & Data Quality

### Key Observations
1. **Distance**: Strongest linear predictor with a Pearson correlation of +0.9085 with `posted_rate`.
2. **Equipment Type**: Median rates per mile follow realistic equipment operating costs:
   - Dry Van: ~$2.05 / mile
   - Flatbed: ~$2.22 / mile
   - Reefer: ~$2.31 / mile (additional fuel for refrigeration units)
3. **Network Coverage**: 64 origin cities and 64 destination cities covering 4,014 unique lanes.

### Missing Data Handling (No Leakage)
- `weight` (300 missing values): Imputed using the median weight of the corresponding equipment type calculated strictly from the training set.
- `market_index` (374 missing values): Imputed using the median of the training set.

### Outlier Detection & Data Cleaning
Residual analysis on the initial models revealed severe pricing anomalies in the training set:
- **669 rows (1.39% of the dataset)** had rates per mile below $1.00 or above $4.00 (peaking at $14.12/mile) despite standard equipment and normal `market_index` / `quote_signal` values (~1.0 and ~2.1).
- Training directly on these corrupt records distorted squared-error loss gradients and degraded predictions on normal loads.
- Removing these 669 records from the training set reduced overall validation MAE from $147.39 to $110.93.

---

## Validation Strategy

Spot freight rates are time-dependent. Standard random K-fold cross-validation causes lookahead bias because future market trends leak into past predictions.

To properly simulate predicting unseen future periods (November `validation.csv` and December `december-chart-inputs.csv`), a temporal split was used:
- **Training Set (Months 1–9)**: January 1 to September 30, 2025 (43,147 loads, 89.9%).
- **Validation Set (Month 10)**: October 1 to October 31, 2025 (4,853 loads, 10.1%).

All imputations, feature encodings, and scalers were fit on the training portion and applied to the validation portion.

---

## Model Progression & Benchmark Results

Models were evaluated on the Month 10 temporal holdout set:

| Model | Training Subset | Overall Val RMSE | Overall Val MAE | Normal Market MAE | Normal Market R² |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (Baseline)** | Raw Jan–Sep | $666.08 | $198.32 | $120.40 | 0.8101 |
| **LightGBM (Raw)** | Raw Jan–Sep | $657.55 | $147.39 | $79.12 | 0.8149 |
| **LightGBM (Clean Data)** | Clean Jan–Sep | $646.40 | $110.93 | **$50.54** | **0.9968** |
| **Final Production Model** | Clean Full (Jan–Oct) | -- | -- | *Retrained on all 47,331 clean records* |

### Feature Engineering
- **Temporal**: `month`, `day`, `dayofweek`, `is_weekend`, `dayofyear`, `quarter`.
- **Spatial / Geographic**: `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`.
- **Domain Interactors**: `weight_per_mile = weight / (distance + 1)`.
- **Categoricals**: `equipment`, `pickup`, `delivery` handled natively via LightGBM categorical features.

---

## December 2025 Trajectory Analysis

The December test inputs evaluate model behavior when all parameters are held constant and only the date changes:
- Route: Lexington, KY to Fort Wayne, IN (360 miles)
- Equipment: Dry Van, 32,000 lbs
- Dates: 2025-12-01 to 2025-12-31

![December 2025 Predicted Load Rate](scorer_results/candidate_december.png)

### Observations:
- **Early December**: Base rate starts at $824.77 ($2.29/mile), consistent with standard Midwest dry van rates.
- **Mid-December**: Rises to $843 - $847 as retail volume peaks.
- **Late December**: Peaks at $849.40 during December 24–31, reflecting holiday carrier capacity tightening and winter weather adjustments.

---

## Reproducing the Notebook

To re-run the full pipeline from raw data to final prediction files:
1. Open `Notebooks/freight-rate-prediction.ipynb` in VS Code or JupyterLab.
2. Select the virtual environment kernel (`.venv`).
3. Run all cells sequentially. The notebook regenerates `validation_predictions.csv` and `data/december_predictions.csv`.
