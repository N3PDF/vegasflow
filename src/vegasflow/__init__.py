"""Monte Carlo integration with Tensorflow"""

from vegasflow.configflow import DTYPE, DTYPEINT, float_me, int_me, run_eager
from vegasflow.plain import PlainFlow, plain_sampler, plain_wrapper

# Expose the main interfaces
from vegasflow.vflow import VegasFlow, vegas_sampler, vegas_wrapper
from vegasflow.vflowplus import VegasFlowPlus, vegasflowplus_sampler, vegasflowplus_wrapper

__version__ = "1.4.0"
