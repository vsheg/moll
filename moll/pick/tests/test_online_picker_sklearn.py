from copy import deepcopy

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from moll.pick._online_picker import OnlineVectorPicker


@pytest.fixture
def X():
    return np.random.rand(10, 3)


@pytest.fixture
def y():
    return np.arange(10)


def test_pipeline(X, y):
    # Fit the picker through a pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("picker", OnlineVectorPicker(capacity=10)),
            # to test it can handle extra steps, nothing to impute here
            ("imp", SimpleImputer(strategy="mean")),
        ]
    )
    pipeline.fit(X, y)
    X_transformed_pipeline = pipeline.transform(X)

    # Fit the picker directly
    picker2 = OnlineVectorPicker(capacity=10)
    picker2.fit(X, y)
    X_transformed2 = picker2.transform(X)

    # Labels are the same
    assert picker2.labels == pipeline.named_steps["picker"].labels

    # Vectors are the same after scaling
    X_transformed2_scaled = (X_transformed2 - np.mean(X, axis=0)) / np.std(X, axis=0)

    assert np.allclose(X_transformed_pipeline, X_transformed2_scaled)
