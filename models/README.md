# models/

| File | Owner | Notes |
|---|---|---|
| `ai4i_sensor_scaler.joblib` | Ilhan | `StandardScaler` fitted on the 5 training sensors (stratified 80/20 split, `random_state=42`). |
| `failure_model.joblib` | Kelvin | sklearn Pipeline: engineered features (`src/features.py`) + histogram gradient boosting. Created by `python training/train_model.py`. |
| `model_card.json` | Kelvin | Input order, decision threshold, CV and test-set metrics. Read by the prediction page. |

**Model input order (exact):**
`type, air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min`

`type` (L=0, M=1, H=2) is **not** scaled; the five sensors are scaled with `ai4i_sensor_scaler.joblib`. Failure-mode columns (`twf, hdf, pwf, osf, rnf`) are never model inputs. The model file needs `src.features` importable and scikit-learn 1.9.x (pinned in `requirements.txt`).
