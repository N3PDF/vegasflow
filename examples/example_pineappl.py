#!/usr/bin/env python
from vegasflow.configflow import DTYPE, DTYPEINT
import time
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from vegasflow.vflow import vegas_wrapper, VegasFlow
import pineappl
from pdfflow.pflow import mkPDF
from functools import partial
from multiprocessing.pool import ThreadPool as Pool

# configuration
dim = 3
ncalls = np.int32(10000)
n_iter = 3

# Constants in GeV^2 pbarn
hbarc2 = tf.constant(389379372.1, dtype=DTYPE)
alpha0 = tf.constant(1.0 / 137.03599911, dtype=DTYPE)


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
    for ix1, ix2, iyll, iw in zip(x1, x2, yll, weight):
        grid.fill(ix1, ix2, q2, 0, np.abs(iyll), 0, iw)
pool = Pool(processes=1)


def fill_grid(xarr, n_dim=None, **kwargs):
    s, t, u, x1, x2, jacobian = hadronic_pspgen(xarr, 10.0, 7000.0)

    ptl = tf.sqrt((t * u / s))
    mll = tf.sqrt(s)
    yll = 0.5 * tf.math.log(x1 / x2)
    ylp = tf.abs(yll + tf.math.acosh(0.5 * mll / ptl))
    ylm = tf.abs(yll - tf.math.acosh(0.5 * mll / ptl))

    jacobian *= hbarc2 / ncalls

    # cuts for LO for the invariant-mass slice containing the
    # Z-peak from CMSDY2D11
    #if ptl < 14.0 or np.abs(yll) > 2.4 or ylp > 2.4 \
    #    or ylm > 2.4 or mll < 60.0 or mll > 120.0:
    #    continue

    weight = jacobian * int_photo(s, u, t)
    q2 = 90.0 * 90.0

    #fill(kwargs['grid'], x1, x2, q2, yll, weight)
    pool.apply_async(fill, [kwargs['grid'], x1, x2, q2, yll, weight]).get()
    return weight


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

    # fill the grid with phase-space points
    print('Generating events, please wait...')

    print(f"VEGAS MC, ncalls={ncalls}:")
    mc_instance = VegasFlow(dim, ncalls)
    mc_instance.compile(partial(fill_grid, grid=grid))
    mc_instance.run_integration(n_iter)

    end = time.time()
    print(f"Vegas took: time (s): {end-start}")

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
