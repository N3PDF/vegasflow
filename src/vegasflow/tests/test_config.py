"""
    Test that the configuration is consistent
"""
from vegasflow.configflow import DTYPEINT, DTYPE, int_me, float_me

def test_int_me():
    res = int_me(4)
    assert res.dtype == DTYPEINT

def test_float_me():
    res = float_me(4.)
    assert res.dtype == DTYPE
