# Place your function here
import time
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from vegasflow.configflow import DTYPE, DTYPEINT
from vegasflow.vflow import VegasFlow


def parse_setup():
    DIM = 4
    NCALLS = np.int32(1e5)
    N_ITER = 3
    args = ArgumentParser()
    args.add_argument("-d", "--dimensions", default=DIM, type=int)
    args.add_argument("-n", "--ncalls", default=NCALLS, type=int)
    args.add_argument("-i", "--iter", default=N_ITER, type=int)
    args.add_argument(
        "-l",
        "--limit",
        help="Max number of events per runstep (1e6 is usually a good limit)",
        default=int(1e6),
        type=int,
    )
    args.add_argument("-q", "--quiet", action = "store_true", help = "Printout only results and times")
    return args.parse_args()


@tf.function
def lepage(xarr, n_dim=None, **kwargs):
    """Lepage test function"""
    if n_dim is None:
        n_dim = xarr.shape[-1]
    a = tf.constant(0.1, dtype=DTYPE)
    n100 = tf.cast(100 * n_dim, dtype=DTYPE)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


if __name__ == "__main__":
    # Load the setup
    args = parse_setup()
    ncalls = args.ncalls
    n_iter = args.iter
    dim = args.dimensions
    quiet = args.quiet
    limit = args.limit

    if not quiet:
        print("Testing a basic integration")
        print(f"VEGAS MC, ncalls={ncalls}, dim={dim}, niter={n_iter}, limit={limit}")
    start = time.time()

    # Create the instance of Vegasflow
    mc_instance = VegasFlow(dim, ncalls, events_limit=limit)
    mc_instance.compile(lepage)
    # Train the grid for {n_iter} iterations
    result_1 = mc_instance.run_integration(n_iter)
    print(f"Result after the training: {result_1[0]} +/- {result_1[1]}")

    # Now freeze the grid and get a new result
    mc_instance.freeze_grid()
    mc_instance.compile(lepage)
    result_2 = mc_instance.run_integration(n_iter)
    print(f"Final result: {result_1[0]} +/- {result_1[1]}")
    end = time.time()
    print(f"time (s): {end-start}")
