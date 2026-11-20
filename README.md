# AI4I 2020 Predictive Maintenance Dashboard

Interactive Streamlit dashboard for exploring the **AI4I 2020 Predictive Maintenance Dataset** (synthetic, UCI ML Repository, DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)). It also has a **Failure Risk Prediction** page backed by a trained gradient-boosting model.

## Pages

| Page | What it shows |
|---|---|
| **Overview** | KPI cards (records, failures, failure rate), failure rate by product type (L/M/H), records by type and failure status, dataset description |
| **Sensor distributions** | Pick a sensor (or an engineered feature) and see its histogram and box plot split by failure status, plus medians |
| **Sensor relationships** | Scatter plot (presets: speed vs torque, air vs process temperature, speed vs temperature difference, or custom X/Y) and a correlation heatmap |
| **Failure modes** | Counts of TWF / HDF / PWF / OSF / RNF, overall and by product type (EDA only, never model inputs) |
| **Failure Risk Prediction** | Enter machine settings (physical units) or load a demo scenario: failure probability, risk level, generation rules met, a what-if chart for one setting, and test-set performance |

The **sidebar filters** (product type, failure status, a range slider for each sensor) apply to every exploratory page. Every KPI and chart updates when they change. The selection is kept in `st.session_state` when you switch pages, and **Reset filters** restores the defaults. If no records match, the page shows a "No data" message.

Dashed reference lines are **dataset generation rules**: the thresholds the AI4I authors used to create the synthetic failures. They are not real safety limits.

## Project structure

```
app.py                  entry point: page config, shared sidebar filters, st.navigation
app_pages/              one file per page (named app_pages/, not pages/; see note)
src/
  config.py             paths, column names, unit labels, colours, generation rules
  data.py               load_data() (@st.cache_data) + summary helpers
  filters.py            sidebar filters, apply_filters(), "no data" handling
  charts.py             Plotly figure builders
  prediction.py         render_prediction(): the prediction page UI
  model.py              validation, scaling, prediction, what-if sweep, rule checks (no Streamlit)
  features.py           engineered features used inside the saved model pipeline
  prediction_charts.py  gauge, what-if and confusion-matrix figures
data/ai4i_cleaned.csv   10,000 rows, physical units (the only data the dashboard reads)
models/                 ai4i_sensor_scaler.joblib, failure_model.joblib, model_card.json
training/               train_model.py + Kiat's train/test (and SMOTE) CSVs
reports/                model_evaluation.md, model_comparison.csv, screenshots/ (for the report)
tests/                  sanity checks + end-to-end app tests
.streamlit/config.toml  theme
```

> **Why `app_pages/` and not `pages/`?** Streamlit treats a folder named `pages/` as legacy auto-discovered pages. Combined with `st.navigation` this made Streamlit skip `app.py` on reruns, so the shared sidebar filters disappeared after the first click. Please keep the folder name.

## Setup and run

Requires **Python 3.11+** (tested with 3.14.5). Run these commands from the `ai4i-dashboard` folder.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.\.venv\Scripts\python.exe -m streamlit run app.py` without activating.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

### Tests

```bash
python -m pytest -q tests
```

- `tests/test_sanity.py` checks that the unfiltered data matches the EDA: 339 failures / 10,000 = 3.39%; L 235/6,000, M 83/2,997, H 21/1,003; median torque 53.7 vs 39.9 Nm; median tool wear 165 vs 107 min. It also checks the scaler's input order.
- `tests/test_prediction.py` checks the model contract, that the saved metrics reproduce on the test set, input validation, extreme inputs and the prediction page (scenarios, warnings, missing-model error).
- `tests/test_app.py` runs the app headless. It checks that every page renders, that each filter changes every chart and KPI on every exploratory page, the "No data" case, that filters persist across pages, and the reset button.

## The prediction model

Retrain (about 30 seconds) from the project folder:

```bash
python training/train_model.py
```

- **Data:** Kiat's `train_scaled.csv` (8,000 rows, 271 failures) for training; `test_scaled.csv` (2,000 rows, 68 failures) used once for the final numbers.
- **Inputs:** `type` (L=0, M=1, H=2, not scaled) + the five sensors scaled with `ai4i_sensor_scaler.joblib`, in that order. The failure-mode columns are never inputs.
- **Model:** histogram gradient boosting. The pipeline adds temperature difference and power internally (`src/features.py`), so training, testing and the app compute them identically and the saved 5-sensor scaler is still the only scaler.
- **Selection:** six candidates compared with 5-fold cross-validation on the training set; the decision threshold maximises failure-class F1 on out-of-fold predictions.
- **Test set:** recall 82.4%, precision 84.8%, F1 83.6% (56 of 68 failures caught, 10 false alarms). Full write-up: `reports/model_evaluation.md`.

`models/model_card.json` records the input order, threshold and metrics. `tests/test_prediction.py` checks the card's metrics reproduce, the input contract, validation, extreme values and the page itself.

## Data notes

- AI4I is **synthetic**. Patterns are associations within generated data, not causal findings.
- The dataset has no repair costs or downtime, so real-world savings cannot be measured from it.
- Only `data/ai4i_cleaned.csv` is used. The scaled and SMOTE files are for model training and are not used by the dashboard.
