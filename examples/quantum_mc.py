"""
Run integration with qibo as the backend
and compares it to other algorithms
"""

from vegasflow.quantum import quantum_wrapper, quantumflow_wrapper
from vegasflow import float_me, vegas_wrapper, plain_wrapper, run_eager
import time
import numpy as np
import tensorflow as tf

run_eager(True)
# MC integration setup
dim = 2
ncalls = int(1e5)
n_iter = 5


def symgauss(xarr):
    """symgauss test function"""
    n_dim = xarr.shape[-1]
    a = float_me(0.1)
    n100 = float_me(100 * n_dim)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


def test_me(wrapper, nev):
    nev = int(nev)
    algo = wrapper.__name__.split("_")[0]
    print(f"> Running {algo} for {nev} events")
    start = time.time()
    result = wrapper(symgauss, dim, n_iter, nev)
    end = time.time()
    print(f"This run took {end-start}\n")
    return result


if __name__ == "__main__":
    test_me(plain_wrapper, ncalls)
    test_me(vegas_wrapper, ncalls)
    test_me(quantum_wrapper, ncalls)
    test_me(quantumflow_wrapper, ncalls)
