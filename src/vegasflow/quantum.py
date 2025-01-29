"""
A Monte Carlo integrator built upon Qibo for quantum integration
"""

from .monte_carlo import wrapper, sampler
from .plain import PlainFlow  # start building upon a naive idiotic integrator
from .configflow import run_eager, DTYPE
import tensorflow as tf


class QuantumIntegrator(PlainFlow):
    """
    Simple Monte Carlo integrator.
    """

    _CAN_RUN_VECTORIAL = False

    def __init__(self, *args, **kwargs):
        # This integrator can only run for now in eager mode and needs qibolab to be installed
        run_eager(True)

        try:
            from qibolab.instruments.qrng import QRNG
            from serial.serialutil import SerialException
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("You can do pip install vegasflow[quantum]") from e

        try:
            qrng = QRNG(address="/dev/ttyACM0")
            qrng.connect()
        except SerialException as e:
            raise SerialException("No quantum device found") from e

        self._quantum_sampler = qrng
        super().__init__(*args, **kwargs)

    def run_integration(self, *args, **kwargs):
        ret = super().run_integration(*args, **kwargs)
        self._quantum_sampler.disconnect()
        return ret

    def _generate_random_array(self, n_events, *args):
        """
        Returns
            -------
                `rnds`: array of (n_events, n_dim) random points
                `idx` : index associated to each random point
                `wgt` : wgt associated to the random point
        """
        quantum_rnds_raw = self._quantum_sampler.random((n_events, self.n_dim))
        rnds_raw = tf.cast(quantum_rnds_raw, dtype=DTYPE)

        rnds, wgts_raw, *extra = self._digest_random_generation(rnds_raw, *args)

        wgts = wgts_raw * self.xjac
        if self._xdelta is not None:
            # Now apply integration limits
            rnds = self._xmin + rnds * self._xdelta
            wgts *= self._xdeltajac
        return rnds, wgts, *extra


def quantum_wrapper(*args, **kwargs):
    """Wrapper around QuantumIntegrator"""
    return wrapper(QuantumIntegrator, *args, **kwargs)


def quantum_sampler(*args, **kwargs):
    """Wrapper sampler around QuantumIntegrator"""
    return sampler(QuantumIntegrator, *args, **kwargs)
