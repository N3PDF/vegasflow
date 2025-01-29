"""Monte Carlo integration with Tensorflow"""

from vegasflow.configflow import int_me, float_me, run_eager, DTYPE, DTYPEINT

# Expose the main interfaces
from vegasflow.vflow import VegasFlow, vegas_wrapper, vegas_sampler
from vegasflow.plain import PlainFlow, plain_wrapper, plain_sampler
from vegasflow.vflowplus import VegasFlowPlus, vegasflowplus_wrapper, vegasflowplus_sampler

__version__ = "1.4.0"
