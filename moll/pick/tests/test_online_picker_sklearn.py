from copy import deepcopy

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from moll.pick._online_picker import OnlineVectorPicker


@pytest.fixture
def picker():
    return OnlineVectorPicker(capacity=10)


@pytest.fixture
def X():
    return np.random.rand(10, 3)


@pytest.fixture
def y():
    return np.arange(10)


def test_pipeline(picker, X, y):
    # Fit the picker directly
    picker_copied = deepcopy(picker)
    picker_copied.fit(X, y)
    X_transformed_picker_copied = picker_copied.transform(X)

    # Fit the picker through a pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("picker", picker),
            # to test it can handle extra steps, nothing to impute here
            ("imp", SimpleImputer(strategy="mean")),
        ]
    )
    pipeline.fit(X, y)
    X_transformed_pipeline = pipeline.transform(X)

    # Labels are the same
    assert picker_copied.labels == pipeline.named_steps["picker"].labels

    # Vectors are the same after scaling
    X_transformed_picker_copied_scaled = (
        X_transformed_picker_copied - np.mean(X, axis=0)
    ) / np.std(X, axis=0)

    assert np.allclose(X_transformed_pipeline, X_transformed_picker_copied_scaled)
