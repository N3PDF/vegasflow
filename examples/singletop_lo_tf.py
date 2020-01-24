#!/usr/bin/env python3
import sys
import time
import numpy as np
import tensorflow as tf
from vegasflow.configflow import DTYPE
from vegasflow.vflow import vegas_wrapper

# MC integration setup
dim = 3
ncalls = np.int32(1e5)
n_iter = 5

# Physics setup
# top mass
mt = tf.constant(173.2, dtype=DTYPE)
# center of mass energy
sqrts = tf.constant(8000, dtype=DTYPE)
# minimum allowed center of mass energy
sqrtsmin = tf.constant(173.2, dtype=DTYPE)
# W-boson mass
mw = tf.constant(80.419, dtype=DTYPE)
# gaw
gaw = tf.constant(2.1054, dtype=DTYPE)
# GF
gf = tf.constant(1.16639e-5, dtype=DTYPE)


# auxiliary variables
colf_bt = tf.constant(9, dtype=DTYPE)
mt2 = tf.square(mt)
s = tf.square(sqrts)
s2 = tf.square(s)
smin = tf.square(sqrtsmin)
bmax = tf.sqrt(1 - smin / s)
conv = tf.constant(0.3893793e9, dtype=DTYPE)  # GeV to pb conversion
gaw2 = tf.square(gaw)
mw2 = tf.square(mw)
gw4 = tf.square(4 * np.sqrt(2) * mw2 * gf)


@tf.function
def get_x1x2(xarr):
    """Remapping [0,1] to tau-y"""
    # building shat
    b = bmax * xarr[:, 0]
    onemb2 = 1 - tf.square(b)
    shat = smin / onemb2
    tau = shat / s

    # building rapidity
    ymax = -0.5 * tf.math.log(tau)
    y = ymax * (2 * xarr[:, 1] - 1)

    # building jacobian
    jac = 2 * tau * b * bmax / onemb2  # tau
    jac *= 2 * ymax  # y

    # building x1 and x2
    sqrttau = tf.sqrt(tau)
    expy = tf.exp(y)
    x1 = sqrttau * expy
    x2 = sqrttau / expy

    return shat, jac, x1, x2


@tf.function
def make_event(xarr):
    """Generate event kinematics"""
    shat, jac, x1, x2 = get_x1x2(xarr)

    ecmo2 = tf.sqrt(shat) / 2
    cc = ecmo2 * (1 - mt2 / shat)
    cos = 1 - 2 * xarr[:, 2]
    sinxi = cc * tf.sqrt(1 - cos * cos)
    cosxi = cc * cos
    zeros = tf.zeros(ecmo2.shape, dtype=DTYPE)

    p0 = tf.stack([ecmo2, zeros, zeros, ecmo2])
    p1 = tf.stack([ecmo2, zeros, zeros, -ecmo2])
    p2 = tf.stack([cc, sinxi, zeros, cosxi])
    p3 = tf.stack([tf.sqrt(cc * cc + mt2), -sinxi, zeros, -cosxi])

    psw = (1 - mt2 / shat) / (8 * np.pi)  # psw
    psw *= jac  # jac for tau, y
    flux = 1 / (2 * shat)  # flux

    return psw, flux, p0, p1, p2, p3, x1, x2


@tf.function
def dot(p1, p2):
    """Dot product 4-momenta"""
    e = p1[0] * p2[0]
    px = p1[1] * p2[1]
    py = p1[2] * p2[2]
    pz = p1[3] * p2[3]
    return e - px - py - pz


@tf.function
def u0(p, i):
    """Compute the dirac spinor u0"""

    zeros = tf.zeros(p[0].shape, dtype=DTYPE)
    czeros = tf.complex(zeros, zeros)
    ones = tf.ones(p[0].shape, dtype=DTYPE)

    # case 1) py == 0
    rz = p[3] / p[0]
    theta1 = tf.where(rz > 0, zeros, rz)
    theta1 = tf.where(rz < 0, np.pi * ones, theta1)
    phi1 = zeros

    # case 2) py != 0
    rrr = rz
    rrr = tf.where(rz < -1, -ones, rz)
    rrr = tf.where(rz > 1, ones, rrr)
    theta2 = tf.acos(rrr)
    rx = p[1] / p[0]
    phi2 = zeros
    phi2 = tf.where(rx < 0, np.pi * ones, phi2)

    # combine
    theta = tf.where(p[1] == 0, theta1, theta2)
    phi = tf.where(p[1] == 0, phi1, phi2)

    prefact = tf.complex(np.sqrt(2), zeros) * tf.sqrt(tf.complex(p[0], zeros))
    if i == 1:
        a = tf.complex(tf.cos(theta / 2), zeros)
        b = tf.complex(tf.sin(theta / 2), zeros)
        u0_0 = prefact * a
        u0_1 = prefact * b * tf.complex(tf.cos(phi), tf.sin(phi))
        u0_2 = czeros
        u0_3 = czeros
    else:
        a = tf.complex(tf.sin(theta / 2), zeros)
        b = tf.complex(tf.cos(theta / 2), zeros)
        u0_0 = czeros
        u0_1 = czeros
        u0_2 = prefact * a * tf.complex(tf.cos(phi), -tf.sin(phi))
        u0_3 = -prefact * b

    return tf.stack([u0_0, u0_1, u0_2, u0_3])


@tf.function
def ubar0(p, i):
    """Compute the dirac spinor ubar0"""

    zeros = tf.zeros(p[0].shape, dtype=DTYPE)
    czeros = tf.complex(zeros, zeros)
    ones = tf.ones(p[0].shape, dtype=DTYPE)

    # case 1) py == 0
    rz = p[3] / p[0]
    theta1 = tf.where(rz > 0, zeros, rz)
    theta1 = tf.where(rz < 0, np.pi * ones, theta1)
    phi1 = zeros

    # case 2) py != 0
    rrr = rz
    rrr = tf.where(rz < -1, -ones, rrr)
    rrr = tf.where(rz > 1, ones, rrr)
    theta2 = tf.acos(rrr)
    rrr = p[1] / p[0] / tf.sin(theta2)
    rrr = tf.where(rrr < -1, -ones, rrr)
    rrr = tf.where(rrr > 1, ones, rrr)
    phi2 = tf.acos(rrr)
    ry = p[2] / p[0]
    phi2 = tf.where(ry < 0, -phi2, phi2)

    # combine
    theta = tf.where(p[1] == 0, theta1, theta2)
    phi = tf.where(p[1] == 0, phi1, phi2)

    prefact = tf.complex(np.sqrt(2), zeros) * tf.sqrt(tf.complex(p[0], zeros))
    if i == -1:
        a = tf.complex(tf.sin(theta / 2), zeros)
        b = tf.complex(tf.abs(tf.cos(theta / 2)), zeros)
        ubar0_0 = prefact * a * tf.complex(tf.cos(phi), tf.sin(phi))
        ubar0_1 = -prefact * b
        ubar0_2 = czeros
        ubar0_3 = czeros
    else:
        a = tf.complex(tf.cos(theta / 2), zeros)
        b = tf.complex(tf.sin(theta / 2), zeros)
        ubar0_0 = czeros
        ubar0_1 = czeros
        ubar0_2 = prefact * a
        ubar0_3 = prefact * b * tf.complex(tf.cos(phi), -tf.sin(phi))

    return tf.stack([ubar0_0, ubar0_1, ubar0_2, ubar0_3])


@tf.function
def za(p1, p2):
    ket = u0(p2, 1)
    bra = ubar0(p1, -1)
    return tf.reduce_sum(bra * ket, axis=0)


@tf.function
def zb(p1, p2):
    ket = u0(p2, -1)
    bra = ubar0(p1, 1)
    return tf.reduce_sum(bra * ket, axis=0)


@tf.function
def sprod(p1, p2):
    a = za(p1, p2)
    b = zb(p2, p1)
    return tf.math.real(a * b)


@tf.function
def qqxtbx(p0, p1, p2, p3):
    """Evaluate 0 -> qqbarttbar"""
    pw2 = sprod(p0, p1)
    wprop = tf.square(pw2 - mw2) + mw2 * gaw2
    a = sprod(p0, p2)
    b = sprod(p0, p3)
    c = sprod(p2, p3)
    d = sprod(p3, p1)
    return tf.abs((a + mt2 * b / c) * d) * colf_bt / wprop * gw4 / 36


@tf.function
def evaluate_matrix_element_square(p0, p1, p2, p3):
    """Evaluate Matrix Element square"""

    # massless projection
    k = mt2 / dot(p3, p0) / 2
    p3 -= p0 * k

    # channels evaluation
    c1 = qqxtbx(p2, -p1, p3, -p0)  # BBARQBARQT +2 -1 +3 -0
    c2 = qqxtbx(-p1, p2, p3, -p0)  # BBARQQBART -1 +2 +3 -0

    return tf.stack([c1, c2])


@tf.function
def pdf(fl1, fl2, x1, x2):
    """Dummy toy PDF"""
    return x1 * x2


@tf.function
def build_luminosity(x1, x2):
    """Single-top t-channel luminosity"""
    lumi1 = pdf(5, 2, x1, x2) + pdf(5, 4, x1, x2)
    lumi2 = pdf(5, -1, x1, x2) + pdf(5, -3, x1, x2)
    lumis = tf.stack([lumi1, lumi2]) / x1 / x2
    return lumis


@tf.function
def singletop(xarr, n_dim=None):
    """Single-top (t-channel) at LO"""
    psw, flux, p0, p1, p2, p3, x1, x2 = make_event(xarr)
    wgts = evaluate_matrix_element_square(p0, p1, p2, p3)
    lumis = build_luminosity(x1, x2)
    lumi_me2 = tf.reduce_sum(2 * lumis * wgts, axis=0)
    return lumi_me2 * psw * flux * conv


if __name__ == "__main__":
    """Testing a basic integration"""
    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    r = vegas_wrapper(singletop, dim, n_iter, ncalls)
    end = time.time()
    print(f"time (s): {end-start}")

    try:
        from vegas import Integrator
    except ModuleNotFoundError:
        sys.exit(0)

    if len(sys.argv) > 1:
        print(" > Doing also the comparison with original Vegas ")

        def fun(xarr):
            x = xarr.reshape(1, -1)
            return singletop(x)

        print("Comparing with Lepage's Vegas")
        limits = dim * [[0.0, 1.0]]
        integrator = Integrator(limits)
        start = time.time()
        vr = integrator(fun, neval=ncalls, nitn=n_iter)
        end = time.time()
        print(vr.summary())
        print(f"time (s): {end-start}")
        print(f"Per iteration (s): {(end-start)/n_iter}")
