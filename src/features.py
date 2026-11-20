"""Engineered features used inside the saved model pipeline.

The model receives exactly what `src.prediction.build_input_frame()` produces:
`type` plus the five sensors standardised with models/ai4i_sensor_scaler.joblib
(the same format as Kiat's train_scaled.csv / test_scaled.csv).

`EngineeredFeatures` undoes that scaling internally (with the scaler's own mean
and scale, stored on the transformer) to compute two physical quantities:

    temperature_difference_k = process temperature - air temperature      (K)
    calculated_power_w       = torque * rotational speed * 2*pi / 60       (W)

and appends them, standardised with statistics learned on the training set.
Because this lives inside the pipeline, training, test evaluation and the app
compute the features identically, and the saved 5-sensor scaler is still the
only scaler the app needs.

This module must stay importable as `src.features`: joblib stores a reference
to this class inside models/failure_model.joblib.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import SENSOR_ORDER

ENGINEERED = ["temperature_difference_k", "calculated_power_w"]


def physical_engineered(physical: pd.DataFrame) -> pd.DataFrame:
    """The two engineered features from sensors in physical units."""
    return pd.DataFrame(
        {
            "temperature_difference_k": physical["process_temperature_k"] - physical["air_temperature_k"],
            "calculated_power_w": physical["torque_nm"] * physical["rotational_speed_rpm"] * 2 * np.pi / 60,
        },
        index=physical.index,
    )


class EngineeredFeatures(BaseEstimator, TransformerMixin):
    """Append standardised temperature difference and power to the model inputs.

    sensor_mean / sensor_scale: the saved StandardScaler's `mean_` and `scale_`
    (in SENSOR_ORDER), used to recover physical units from the scaled inputs.
    """

    def __init__(self, sensor_mean=None, sensor_scale=None):
        self.sensor_mean = sensor_mean
        self.sensor_scale = sensor_scale

    def _engineered(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = X[SENSOR_ORDER].to_numpy(dtype=float)
        physical = pd.DataFrame(
            scaled * np.asarray(self.sensor_scale) + np.asarray(self.sensor_mean),
            columns=SENSOR_ORDER,
            index=X.index,
        )
        return physical_engineered(physical)

    def fit(self, X, y=None):
        eng = self._engineered(X)
        self.eng_mean_ = eng.mean().to_numpy()
        self.eng_std_ = eng.std(ddof=0).to_numpy()
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        eng = (self._engineered(X) - self.eng_mean_) / self.eng_std_
        return pd.concat([X.reset_index(drop=True), eng.reset_index(drop=True)], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*self.feature_names_in_, *ENGINEERED], dtype=object)
