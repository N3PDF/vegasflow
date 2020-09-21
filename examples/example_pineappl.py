#!/usr/bin/env python
import random as rn
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from pdfflow.pflow import mkPDF
import pineappl
from vegasflow.utils import generate_condition_function
from vegasflow.vflow import VegasFlow
from vegasflow.configflow import DTYPE, MAX_EVENTS_LIMIT
import time
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)


# Seed everything seedable
seed = 7
np.random.seed(seed)
rn.seed(seed+1)
tf.random.set_seed(seed+2)


# configuration
dim = 3
ncalls = np.int32(10000000)
n_iter = 3
events_limit = MAX_EVENTS_LIMIT

# Constants in GeV^2 pbarn
hbarc2 = tf.constant(389379372.1, dtype=DTYPE)
alpha0 = tf.constant(1.0 / 137.03599911, dtype=DTYPE)
cuts = generate_condition_function(6, condition='and')


@tf.function
def int_photo(s, t, u):
    return alpha0 * alpha0 / 2.0 / s * (t / u + u / t)


@tf.function
def hadronic_pspgen(xarr, mmin, mmax):
    smin = mmin * mmin
    smax = mmax * mmax

    r1 = xarr[:, 0]
    r2 = xarr[:, 1]
    r3 = xarr[:, 2]

    tau0 = smin / smax
    tau = tf.pow(tau0, r1)
    y = tf.pow(tau, 1.0 - r2)
    x1 = y
    x2 = tau / y
    s = tau * smax

    jacobian = np.math.log(tau0) * np.math.log(tau0) * tau * r1

    # theta integration (in the CMS)
    cos_theta = 2.0 * r3 - 1.0
    jacobian *= 2.0

    t = -0.5 * s * (1.0 - cos_theta)
    u = -0.5 * s * (1.0 + cos_theta)

    # phi integration
    jacobian *= 2.0 * np.math.acos(-1.0)

    return s, t, u, x1, x2, jacobian


def fill(grid, x1, x2, q2, yll, weight):
    zeros = np.zeros(len(weight), dtype=np.uintp)
    grid.fill_array(x1, x2, q2, zeros, yll, zeros, weight)


def fill_grid(xarr, n_dim=None, **kwargs):
    s, t, u, x1, x2, jacobian = hadronic_pspgen(xarr, 10.0, 7000.0)

    ptl = tf.sqrt((t * u / s))
    mll = tf.sqrt(s)
    yll = 0.5 * tf.math.log(x1 / x2)
    ylp = tf.abs(yll + tf.math.acosh(0.5 * mll / ptl))
    ylm = tf.abs(yll - tf.math.acosh(0.5 * mll / ptl))

    jacobian *= hbarc2

    # apply cuts
    t_1 = ptl >= 14.0
    t_2 = tf.abs(yll) <= 2.4
    t_3 = ylp <= 2.4
    t_4 = ylm <= 2.4
    t_5 = mll >= 60.0
    t_6 = mll <= 120.0
    full_mask, indices = cuts(t_1, t_2, t_3, t_4, t_5, t_6)

    weight = tf.boolean_mask(jacobian * int_photo(s, u, t), full_mask, axis=0)
    x1 = tf.boolean_mask(x1, full_mask, axis=0)
    x2 = tf.boolean_mask(x2, full_mask, axis=0)
    yll = tf.boolean_mask(yll, full_mask, axis=0)
    vweight = weight * tf.boolean_mask(kwargs.get('weight'), full_mask, axis=0)

    if kwargs.get('fill_pineappl'):
        q2 = 90.0 * 90.0 * tf.ones(weight.shape, dtype=tf.float64)
        kwargs.get('pool').apply_async(fill, [kwargs.get('grid'), x1.numpy(), x2.numpy(),
                                              q2.numpy(), tf.abs(yll).numpy(), vweight.numpy()])

    return tf.scatter_nd(indices, weight, shape=xarr.shape[0:1])


if __name__ == "__main__":
    start = time.time()

    lumi = pineappl.lumi()
    pdg_ids = [22, 22]
    ckm_factors = [1.0]
    lumi.add(pdg_ids, ckm_factors)

    # only LO $\alpha_\mathrm{s}^0 \alpha^2 \log^0(\xi_\mathrm{R}) \log^0(\xi_\mathrm{F})$
    orders = [0, 2, 0, 0]
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
    grid = pineappl.grid(lumi, orders, bins)
    pool = Pool(processes=1)

    # fill the grid with phase-space points
    print('Generating events, please wait...')

    print(f"VEGAS MC, ncalls={ncalls}:")
    mc_instance = VegasFlow(dim, ncalls, events_limit=events_limit)
    mc_instance.compile(
        partial(fill_grid, fill_pineappl=False, grid=grid, pool=pool))
    mc_instance.run_integration(n_iter)
    mc_instance.compile(
        partial(fill_grid, fill_pineappl=True, grid=grid, pool=pool))
    mc_instance.freeze_grid()
    mc_instance.run_integration(1)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")

    # wait until pineappl has filled the grids properly
    pool.close()
    pool.join()

    # write the grid to disk
    filename = 'DY-LO-AA.pineappl'
    print(f'Writing PineAPPL grid to disk: {filename}')
    grid.write(filename)

    # check convolution
    # load pdf for testing
    pdf = mkPDF('NNPDF31_nlo_as_0118_luxqed/0')

    def xfx(id, x, q2, p):
        return pdf.py_xfxQ2([id], [x], [q2])

    def alphas(q2, p):
        return pdf.py_alphasQ2([q2])

    # perform convolution
    dxsec = grid.convolute(xfx, xfx, alphas, None, None, 1.0, 1.0)
    for i in range(len(dxsec)):
        print(f'{bins[i]:.1f} {bins[i + 1]:.1f} {dxsec[i]:.3e}')
