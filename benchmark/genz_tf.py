# Place your function here
import time
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from vegasflow.configflow import DTYPE, DTYPEINT, fone, fzero
from vegasflow.vflow import VegasFlow


def parse_setup():
    DIM = 4
    NCALLS = np.int32(1e5)
    N_ITER = 3
    args = ArgumentParser()
    args.add_argument("-d", "--dimensions", default=DIM, type=int)
    args.add_argument("-n", "--ncalls", default=NCALLS, type=int)
    args.add_argument("-D", "--difficulty", default=10.0, type=float)
    args.add_argument("-f", "--input_file")
    args.add_argument("-i", "--iter", default=N_ITER, type=int)
    args.add_argument("-g", "--genz", default="product_peak")
    args.add_argument(
        "-l",
        "--limit",
        help="Max number of events per runstep (1e6 is usually a good limit)",
        default=int(1e6),
        type=int,
    )
    args.add_argument("-q", "--quiet", action = "store_true", help = "Printout only results and times")
    return args.parse_args()

def get_num(input_file):
    a = np.loadtxt(input_file)
    w = a[1]
    c = a[0]
    return w, c

if __name__ == "__main__":
    # Load the setup
    args = parse_setup()
    ncalls = args.ncalls
    n_iter = args.iter
    dim = args.dimensions
    quiet = args.quiet
    limit = args.limit


    if args.input_file:
        print(f"Using numbers from {args.input_file}")
        npwvec, npcvec = get_num(args.input_file)
    else:
        diff = args.difficulty
        npcvec=np.random.rand(dim)*diff
        npwvec = np.random.rand(dim)

    cvec = tf.constant(npcvec, dtype=DTYPE)
    wvec = tf.constant(npwvec, dtype=DTYPE)

    if args.genz == "product_peak":
        tan1 = np.arctan( (1-npwvec)*npcvec )
        tan2 = np.arctan( npwvec*npcvec )
        res = np.prod((tan1+tan2)*npcvec)
        norm = tf.constant(res, dtype=DTYPE)

        @tf.function
        def genz(xarr, n_dim=None, **kwargs):
            ci = tf.pow(cvec, -2)
            den = ci + tf.pow(xarr-wvec, 2)
            result = tf.pow(den, -1)
            return tf.reduce_prod(result, axis=1)/norm
    elif args.genz == "oscillatory":
        npu1 = np.pi*2*wvec[0]
        u1 = tf.constant(npu1, dtype=DTYPE)
        # normalize
        num = pow(2,dim)*np.cos(npu1+np.sum(npcvec)/2)*np.prod(np.sin(npcvec/2))
        den = np.prod(cvec)
        result = num/den
        norm = tf.constant(result, dtype=DTYPE)

        @tf.function
        def genz(xarr, n_dim=None, **kwargs):
            internal = tf.reduce_sum(cvec*xarr, axis=1)
            res = tf.cos(u1 + internal)
            return res/norm
    else:
        raise NotImplementedError("This genz function is not implemented")

    if args.input_file is None:
        savevals = np.concatenate([npcvec.reshape(1,-1), npwvec.reshape(1,-1)], axis=0)
        np.savetxt("numbers.txt", savevals)

    if not quiet:
        print("Testing a basic integration")
        print(f"VEGAS MC, ncalls={ncalls}, dim={dim}, niter={n_iter}, limit={limit}")
    start = time.time()

    # Create the instance of Vegasflow
    # For training use 1/10 as many events per run
    training_limit = int(limit/10)
    mc_instance = VegasFlow(dim, ncalls, events_limit=training_limit)
    mc_instance.compile(genz)
    # Train the grid for {n_iter} iterations
    result_1 = mc_instance.run_integration(n_iter)
    print(f"Result after the training: {result_1[0]} +/- {result_1[1]}")

    # Now freeze the grid and get a new result
    mc_instance.freeze_grid()
    # After freezing the grid change the number of events per run
    mc_instance.events_per_run = limit
    result_2 = mc_instance.run_integration(n_iter)
    print(f"Final result: {result_2[0]} +/- {result_2[1]}")
    end = time.time()
    print(f"time (s): {end-start}")
    print(f"Normalization factor: {norm}")
