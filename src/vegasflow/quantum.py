"""
A Monte Carlo integrator built upon Qibo for quantum integration
"""

import tensorflow as tf

from .configflow import DTYPE, run_eager
from .monte_carlo import MonteCarloFlow, sampler, wrapper
from .plain import PlainFlow
from .vflow import VegasFlow


class QuantumBase(MonteCarloFlow):
    """
    This class serves as a basis for the quantum monte carlo integrator.
    At initialization it tries to import qibolab and connect to the quantum device,
    if successful, saves the reference to _quantum_sampler.

    This class is compatible with all ``MonteCarloFlow`` classes, it overrides
    the uniform sampling and uses the quantum device instead.
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

        qaddress = "/dev/ttyACM0"
        try:
            # Check whether the quantum device is available and we can connect
            qrng = QRNG(address=qaddress)
            qrng.connect()
            qrng.disconnect()
        except SerialException as e:
            raise SerialException(f"No quantum device found at {qaddress}") from e

        print(f"Sucessfuly connected to quantum device in {qaddress}")

        self._quantum_sampler = qrng
        super().__init__(*args, **kwargs)

    def _internal_sampler(self, n_events):
        """Sample ``n_events x n_dim`` numbers from the quantum device
        and cast them to a TF DTYPE to pass down to the MC algorithm"""
        self._quantum_sampler.connect()
        quantum_rnds_raw = self._quantum_sampler.random((n_events, self.n_dim))
        self._quantum_sampler.disconnect()
        return tf.cast(quantum_rnds_raw, dtype=DTYPE)


class QuantumIntegrator(PlainFlow, QuantumBase):
    pass


class QuantumFlow(VegasFlow, QuantumBase):
    pass


def quantum_wrapper(*args, **kwargs):
    """Wrapper around QuantumIntegrator"""
    return wrapper(QuantumIntegrator, *args, **kwargs)


def quantumflow_wrapper(*args, **kwargs):
    return wrapper(QuantumFlow, *args, **kwargs)


def quantum_sampler(*args, **kwargs):
    """Wrapper sampler around QuantumIntegrator"""
    return sampler(QuantumIntegrator, *args, **kwargs)
