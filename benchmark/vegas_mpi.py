#!/usr/bin/env python

""" Numerical comparison of the integrals with
Lepage's Vegas.
In order to make things as fair as possible the
integrand will be a TF compiled function """

import time
import numpy as np
from argparse import ArgumentParser
from vegas import Integrator as LepageIntegrator

from mpi4py import MPI
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()

BINS_MAX = 50
ALPHA = 1.5

def parse_setup():
    DIM = 4
    NCALLS = np.int32(1e5)
    N_ITER = 3
    args = ArgumentParser()
    args.add_argument("-d", "--dimensions", default=DIM, type=int)
    args.add_argument("-n", "--ncalls", default=NCALLS, type=int)
    args.add_argument("-i", "--iter", default=N_ITER, type=int)
    args.add_argument("-g", "--genz", default="lepage")
    args.add_argument("-f", "--input_file", default="numbers.txt")
    args.add_argument(
        "-l",
        "--limit",
        help="Max number of events per runstep (1e6 is usually a good limit)",
        default=int(1e6),
        type=int,
    )
    args.add_argument("-q", "--quiet", action = "store_true", help = "Printout only results and times")
    return args.parse_args()

def lepage_vegas_integrate(fun, dim, ncalls, n_iter):
    if RANK == 0:
        print("Results with VEGAS from Lepage module")
    limits = dim*[[0.0, 1.0]]
    bins = BINS_MAX
    damping_alpha = ALPHA
    integrator = LepageIntegrator(limits) #,maxinc_axis = bins )
    start = time.time()
    _ = integrator(fun, neval = ncalls, nitn = n_iter, alpha = damping_alpha)
    if RANK == 0:
        print(_.summary())
    result = integrator(fun, neval = ncalls, nitn = n_iter, adapt = False)
    end = time.time()
    if RANK == 0:
        print(result.summary())
        print(f"Final result: {result}")
        print(f"time (s): {end-start}")
        print(f"Per iteration (s): {(end-start)/n_iter/2}")
    return result

a = 0.1
def symgauss(xarr):
    n_dim = len(xarr)
    pref = pow(1.0/a/np.sqrt(np.pi), n_dim)
    coef = np.sum( pow(xarr - 1.0/2.0, 2) / pow(a, 2) )
    return pref*np.exp(-coef)

def reminder():
    if RANK == 0:
        print("Remember to always run first genz_tf to generate the numbers.txt file")

def get_num(input_file):
    a = np.loadtxt(input_file)
    w = a[1]
    c = a[0]
    return w, c
    

if __name__ == "__main__":
    args = parse_setup()
    dim = args.dimensions
    ncalls = args.ncalls
    niter = args.iter
    if RANK == 0:
        print(f" > > > Running for ncalls:{ncalls}, dims:{dim}, niter:{niter}")
        print(f" > Testing {args.genz} ")

    if args.genz == "lepage":
        integrand = symgauss
        normfac = 1.0
    elif args.genz == "product_peak":
        reminder()
        npwvec, npcvec = get_num(args.input_file)
        tan1 = np.arctan( (1-npwvec)*npcvec )
        tan2 = np.arctan( npwvec*npcvec )
        normfac = np.prod((tan1+tan2)*npcvec)
        def integrand(xarr):
            ci = pow(npcvec, -2)
            den = ci + pow(xarr - npwvec, 2)
            result = pow(den, -1)
            return np.product(result)/normfac
    elif args.genz == "discontinuous":
        reminder()
        npwvec, npcvec = get_num(args.input_file)
        ar = (np.exp(npcvec*npwvec) - 1.0)/npcvec
        normfac = np.prod(ar)

        def integrand(xarr):
            flag = np.any(xarr <= npwvec, axis=1)
            res = np.exp(np.sum(xarr*npcvec))
            return res*flag/normfac

    else:
        raise NotImplementedError("This genz function is not implemented")

    lepage_vegas_integrate(symgauss, dim, ncalls, niter)
    if RANK == 0:
        print(f"Normalization factor: {normfac}")
