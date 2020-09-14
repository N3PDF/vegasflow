"""Monte Carlo integration with Tensorflow"""

from vegasflow.configflow import int_me, float_me, run_eager
# Expose the main interfaces
from vegasflow.vflow import VegasFlow, vegas_wrapper
from vegasflow.monte_carlo import MonteCarloFlow

__version__ = "1.2.0"
