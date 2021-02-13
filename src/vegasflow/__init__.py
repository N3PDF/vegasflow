"""Monte Carlo integration with Tensorflow"""

from vegasflow.configflow import int_me, float_me, run_eager
# Expose the main interfaces
from vegasflow.vflow import VegasFlow, vegas_wrapper
from vegasflow.plain import PlainFlow, plain_wrapper
from vegasflow.stratified import StratifiedFlow, stratified_wrapper

__version__ = "1.2.0"
