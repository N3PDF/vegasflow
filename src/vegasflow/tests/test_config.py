"""
    Test that the configuration is consistent
"""

import importlib
import os

import numpy as np

import vegasflow.configflow
from vegasflow.configflow import DTYPE, DTYPEINT, float_me, int_me


def test_int_me():
    res = int_me(4)
    assert res.dtype == DTYPEINT


def test_float_me():
    res = float_me(4.0)
    assert res.dtype == DTYPE


def test_float_env():
    os.environ["VEGASFLOW_FLOAT"] = "32"
    importlib.reload(vegasflow.configflow)
    from vegasflow.configflow import DTYPE

    assert DTYPE.as_numpy_dtype == np.float32
    os.environ["VEGASFLOW_FLOAT"] = "64"
    importlib.reload(vegasflow.configflow)
    from vegasflow.configflow import DTYPE

    assert DTYPE.as_numpy_dtype == np.float64
    # Reset to default
    os.environ["VEGASFLOW_FLOAT"] = "64"
    importlib.reload(vegasflow.configflow)


def test_int_env():
    os.environ["VEGASFLOW_INT"] = "32"
    importlib.reload(vegasflow.configflow)
    from vegasflow.configflow import DTYPEINT

    assert DTYPEINT.as_numpy_dtype == np.int32
    os.environ["VEGASFLOW_INT"] = "64"
    importlib.reload(vegasflow.configflow)
    from vegasflow.configflow import DTYPEINT

    assert DTYPEINT.as_numpy_dtype == np.int64
    # Reset to default
    os.environ["VEGASFLOW_INT"] = "32"
    importlib.reload(vegasflow.configflow)
