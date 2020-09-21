"""
    Example: cluster usage

    Basic example of running a VegasFlow job on a distributed system
    using dask and the SLURMCluster backend
"""

from dask_jobqueue import SLURMCluster  # pylint: disable=import-error
from vegasflow.vflow import VegasFlow
import tensorflow as tf


def integrand(xarr, **kwargs):
    return tf.reduce_sum(xarr, axis=1)


if __name__ == "__main__":
    cluster = SLURMCluster(
        memory="2g",
        processes=1,
        cores=4,
        queue="<partition_name>",
        project="<accout_name>",
        job_extra=["--get-user-env", "--nodes=1"],
    )

    mc_instance = VegasFlow(4, int(1e7), events_limit=int(1e6))
    mc_instance.set_distribute(cluster)
    mc_instance.compile(integrand)
    mc_instance.run_integration(5)
