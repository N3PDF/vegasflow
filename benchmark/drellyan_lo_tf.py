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

# Physics setup
# center of mass energy
sqrts = 14000

# auxiliary variables
s = tf.constant(pow(sqrts, 2), dtype=DTYPE)
conv = tf.constant(0.3893793e9, dtype=DTYPE) # GeV to pb conversion


@tf.function
def get_x1x2(xarr):
    """Remapping [0,1] to kappa-y"""
    kappa = xarr[:, 0]
    y = xarr[:, 1]
    logkappa = tf.math.log(kappa)
    sqrtkappa = tf.sqrt(kappa)
    Ycm = tf.exp(logkappa*(y - 0.5))

    shat = s
    x1 = sqrtkappa * Ycm
    x2 = sqrtkappa / Ycm
    jac = tf.abs(logkappa)

    return shat, jac, x1, x2


@tf.function
def make_event(xarr):
    """Generate event kinematics"""
    shat, jac, x1, x2 = get_x1x2(xarr)

    mV = tf.sqrt(shat * x1 * x2)
    mV2 = mV*mV
    ecmo2 = mV/2
    zeros = tf.zeros(ecmo2.shape, dtype=DTYPE)

    p0 = tf.stack([ecmo2, zeros, zeros, ecmo2])
    p1 = tf.stack([ecmo2, zeros, zeros,-ecmo2])

    pV = p0 + p1
    YV = 0.5 * tf.math.log(tf.abs((pV[0] + pV[3])/(pV[0] - pV[3])))
    pVt2 = tf.square(pV[1]) + tf.square(pV[2])
    phi = 2 * np.pi * xarr[:, 3]
    ptmax = 0.5 * mV2 / (tf.sqrt(mV2 + pVt2) - (pV[1]*tf.cos(phi) + pV[2]*tf.sin(phi)))
    pta = ptmax * xarr[:, 2]
    pt = tf.stack([zeros, pta*tf.cos(phi), pta*tf.sin(phi), zeros])
    Delta = (mV2 + 2 * (pV[1]*pt[1] + pV[2]*pt[2]))/2.0/pta/tf.sqrt(mV2 + pVt2)
    y = YV - tf.acosh(Delta)
    kallenF = 2.0 * ptmax/tf.sqrt(mV2 + pVt2)/tf.abs(tf.sinh(YV-y))

    p2 = tf.stack([pta*tf.cosh(y), pta*tf.cos(phi), pta*tf.sin(phi),pta*tf.sinh(y)])
    p3 = pV - p2

    psw = 1 / (8*np.pi)* kallenF # psw
    psw *= jac # jac for tau, y
    psw /= conv
    psw *= 36*np.pi
    flux = 1 / (2 * mV2) # flux

    return psw, flux, p0, p1, p2, p3, x1, x2


@tf.function
def dot(p1, p2):
    """Dot product 4-momenta"""
    e  = p1[0]*p2[0]
    px = p1[1]*p2[1]
    py = p1[2]*p2[2]
    pz = p1[3]*p2[3]
    return e - px - py - pz


@tf.function
def u0(p, i):
    """Compute the dirac spinor u0"""

    zeros = tf.zeros(p[0].shape, dtype=DTYPE)
    czeros = tf.complex(zeros, zeros)
    ones = tf.ones(p[0].shape, dtype=DTYPE)

    # case 1) py == 0
    rz = p[3]/p[0]
    theta1 = tf.where(rz > 0, zeros, rz)
    theta1 = tf.where(rz < 0, np.pi*ones, theta1)
    phi1 = zeros

    # case 2) py != 0
    rrr = rz
    rrr = tf.where(rz < -1, -ones, rz)
    rrr = tf.where(rz > 1, ones, rrr)
    theta2 = tf.acos(rrr)
    rx = p[1]/p[0]/tf.sin(theta2)
    rrr = tf.where(rx < -1, -ones, rx)
    rrr = tf.where(rx > 1, ones, rrr)
    phi2 = tf.acos(rrr)
    ry = p[2]/p[0]
    phi2 = tf.where(ry < 0, -phi2, phi2)

    # combine
    theta = tf.where(p[1] == 0, theta1, theta2)
    phi = tf.where(p[1] == 0, phi1, phi2)

    prefact = tf.complex(np.sqrt(2), zeros)*tf.sqrt(tf.complex(p[0], zeros))
    if i == 1:
        a = tf.complex(tf.cos(theta/2), zeros)
        b = tf.complex(tf.sin(theta/2), zeros)
        u0_0 = prefact*a
        u0_1 = prefact*b*tf.complex(tf.cos(phi), tf.sin(phi))
        u0_2 = czeros
        u0_3 = czeros
    else:
        a = tf.complex(tf.sin(theta/2), zeros)
        b = tf.complex(tf.cos(theta/2), zeros)
        u0_0 = czeros
        u0_1 = czeros
        u0_2 = prefact*a*tf.complex(tf.cos(phi), -tf.sin(phi))
        u0_3 = -prefact*b

    return tf.stack([u0_0, u0_1, u0_2, u0_3])


@tf.function
def ubar0(p, i):
    """Compute the dirac spinor ubar0"""

    zeros = tf.zeros(p[0].shape, dtype=DTYPE)
    czeros = tf.complex(zeros, zeros)
    ones = tf.ones(p[0].shape, dtype=DTYPE)

    # case 1) py == 0
    rz = p[3]/p[0]
    theta1 = tf.where(rz > 0, zeros, rz)
    theta1 = tf.where(rz < 0, np.pi*ones, theta1)
    phi1 = zeros

    # case 2) py != 0
    rrr = rz
    rrr = tf.where(rz < -1, -ones, rrr)
    rrr = tf.where(rz > 1, ones, rrr)
    theta2 = tf.acos(rrr)
    rrr = p[1]/p[0]/tf.sin(theta2)
    rrr = tf.where(rrr < -1, -ones, rrr)
    rrr = tf.where(rrr > 1, ones, rrr)
    phi2 = tf.acos(rrr)
    ry = p[2]/p[0]
    phi2 = tf.where(ry < 0, -phi2, phi2)

    # combine
    theta = tf.where(p[1] == 0, theta1, theta2)
    phi = tf.where(p[1] == 0, phi1, phi2)

    prefact = tf.complex(np.sqrt(2), zeros)*tf.sqrt(tf.complex(p[0], zeros))
    if i == -1:
        a = tf.complex(tf.sin(theta/2), zeros)
        b = tf.complex(tf.abs(tf.cos(theta/2)), zeros)
        ubar0_0 = prefact*a*tf.complex(tf.cos(phi), tf.sin(phi))
        ubar0_1 = -prefact*b
        ubar0_2 = czeros
        ubar0_3 = czeros
    else:
        a = tf.complex(tf.cos(theta/2), zeros)
        b = tf.complex(tf.sin(theta/2), zeros)
        ubar0_0 = czeros
        ubar0_1 = czeros
        ubar0_2 = prefact*a
        ubar0_3 = prefact*b*tf.complex(tf.cos(phi), -tf.sin(phi))

    return tf.stack([ubar0_0, ubar0_1, ubar0_2, ubar0_3])


@tf.function
def za(p1, p2):
    ket = u0(p2, 1)
    bra = ubar0(p1, -1)
    return tf.reduce_sum(bra*ket, axis=0)


@tf.function
def zb(p1, p2):
    ket = u0(p2, -1)
    bra = ubar0(p1, 1)
    return tf.reduce_sum(bra*ket, axis=0)


@tf.function
def sprod(p1, p2):
    a = za(p1, p2)
    b = zb(p2, p1)
    return tf.math.real(a*b)


@tf.function
def qqxllx(p0, p1, p2, p3):
    """Evaluate 0 -> qqbarttbar"""
    lsprod = sprod(p0, p1)
    a = 2 * tf.abs(za(p0, p2) * zb(p3, p1)) / lsprod
    b = 2 * tf.abs(za(p0, p3) * zb(p2, p1)) / lsprod
    return 6.0 * (tf.square(a)+tf.square(b)) / 36.0


@tf.function
def evaluate_matrix_element_square(p0, p1, p2, p3):
    """Evaluate Matrix Element square"""

    # channels evaluation
    c1 = qqxllx(-p1,-p0, p2, p3) # QQBARLLBAR

    return c1


@tf.function
def pdf(fl1, fl2, x1, x2):
    """Dummy toy PDF"""
    return x1*x2


@tf.function
def build_luminosity(x1, x2):
    """Single-top t-channel luminosity"""
    lumi = (
        pdf(1, -1, x1, x2) + pdf(2, -2, x1, x2) +
        pdf(3, -3, x1, x2) + pdf(4, -4, x1, x2)
        ) / x1 / x2
    return lumi


@tf.function
def drellyan(xarr, n_dim=None, **kwargs):
    """Single-top (t-channel) at LO"""
    psw, flux, p0, p1, p2, p3, x1, x2 = make_event(xarr)
    wgts = evaluate_matrix_element_square(p0, p1, p2, p3)
    lumis = build_luminosity(x1, x2)
    lumi_me2 = 2*lumis*wgts
    return lumi_me2*psw*flux*conv


if __name__ == "__main__":
    # Load the setup
    args = parse_setup()
    ncalls = args.ncalls
    n_iter = args.iter
    dim = args.dimensions
    quiet = args.quiet
    limit = args.limit

    if not quiet:
        print("Testing a basic Drell Yan integration")
        print(f"VEGAS MC, {ncalls=}, {dim=}, {n_iter=}, {limit=}")
    start = time.time()

    # Create the instance of Vegasflow
    mc_instance = VegasFlow(dim, ncalls, events_limit=limit)
    mc_instance.compile(drellyan)
    # Train the grid for {n_iter} iterations
    result_1 = mc_instance.run_integration(n_iter)
    print(f"Result after the training: {result_1[0]} +/- {result_1[1]}")

    # Now freeze the grid and get a new result
    mc_instance.freeze_grid()
    result_2 = mc_instance.run_integration(n_iter)
    print(f"Final result: {result_1[0]} +/- {result_1[1]}")
    end = time.time()
    print(f"time (s): {end-start}")
