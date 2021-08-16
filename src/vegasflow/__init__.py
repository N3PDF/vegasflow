"""Monte Carlo integration with Tensorflow"""

from vegasflow.configflow import int_me, float_me, run_eager
# Expose the main interfaces
from vegasflow.vflow import VegasFlow, vegas_wrapper, vegas_sampler
from vegasflow.plain import PlainFlow, plain_wrapper, plain_sampler
from vegasflow.vflowplus import VegasFlowPlus, vegasflowplus_wrapper
# TODO: create a sampler for vegasflowplus

__version__ = "1.2.2"
