"""
    Implementation of vegas+ algorithm:

    adaptive importance sampling + adaptive stratified sampling
    from https://arxiv.org/abs/2009.05112

    The main interface is the `VegasFlowPlus` class.
"""

from functools import partial
from itertools import product
import logging

import numpy as np
import tensorflow as tf

from vegasflow.configflow import (
    BETA,
    BINS_MAX,
    DTYPE,
    DTYPEINT,
    MAX_NEVAL_HCUBE,
    float_me,
    fone,
    fzero,
    int_me,
)
from vegasflow.monte_carlo import MonteCarloFlow, sampler, wrapper
from vegasflow.utils import consume_array_into_indices
from vegasflow.vflow import VegasFlow, importance_sampling_digest

logger = logging.getLogger(__name__)

FBINS = float_me(BINS_MAX)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPEINT),
        tf.TensorSpec(shape=[None], dtype=DTYPEINT),
        tf.TensorSpec(shape=[None, None], dtype=DTYPEINT),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
    ]
)
def generate_samples_in_hypercubes(rnds, n_strat, n_ev, hypercubes, divisions):
    """Receives an array of random numbers 0 and 1 and
    distribute them in each hypercube according to the
    number of samples in each hypercube specified by n_ev

    Parameters
    ----------
        `rnds`: tensor of random number between 0 and 1
        `n_strat`: tensor with number of stratifications in each dimension
        `n_ev`: tensor containing number of samples per hypercube
        `hypercubes`: tensor containing all different hypercube
        `divisions`: vegas grid

    Returns
    -------
        `x` : random numbers collocated in hypercubes
        `w` : weight of each event
        `ind`: division index in which each (n_dim) set of random numbers fall
        `segm` : segmentation for later computations
    """
    # Use the event-per-hypercube information to fix each random event to a hypercube
    indices = tf.repeat(tf.range(tf.shape(hypercubes, out_type=DTYPEINT)[0]), n_ev)
    points = float_me(tf.gather(hypercubes, indices))
    n_evs = float_me(tf.gather(n_ev, indices))

    # Compute in which division of the importance_sampling grid the points fall
    xn = tf.transpose(points + rnds) * FBINS / float_me(n_strat)

    ind_xn, x, weights = importance_sampling_digest(xn, divisions)

    # Reweighs taking into account the number of events per hypercube
    final_weights = weights / n_evs

    segm = indices
    return x, final_weights, ind_xn, segm


class VegasFlowPlus(VegasFlow):
    """
    Implementation of the VEGAS+ algorithm
    """

    def __init__(self, n_dim, n_events, train=True, adaptive=False, events_limit=None, **kwargs):
        # https://github.com/N3PDF/vegasflow/issues/78
        if events_limit is None:
            logger.info("Events per device limit set to %d", n_events)
            events_limit = n_events
        elif events_limit < n_events:
            logger.warning(
                "VegasFlowPlus needs to hold all events in memory at once, "
                "setting the `events_limit` to be equal to `n_events=%d`",
                n_events,
            )
            events_limit = n_events
        super().__init__(n_dim, n_events, train, events_limit=events_limit, **kwargs)

        # Save the initial number of events
        self._init_calls = n_events

        # Don't use adaptive if the number of dimension is too big
        if n_dim > 13 and adaptive:
            self._adaptive = False
            logger.warning("Disabling adaptive mode from VegasFlowPlus, too many dimensions!")
        else:
            self._adaptive = adaptive

        # Initialize stratifications
        if self._adaptive:
            neval_eff = int(self.n_events / 2)
            self._n_strat = tf.math.floor(tf.math.pow(neval_eff / 2, 1 / n_dim))
        else:
            neval_eff = self.n_events
            self._n_strat = tf.math.floor(tf.math.pow(neval_eff / 2, 1 / n_dim))

        if tf.math.pow(self._n_strat, n_dim) > MAX_NEVAL_HCUBE:
            self._n_strat = tf.math.floor(tf.math.pow(1e4, 1 / n_dim))

        self._n_strat = int_me(self._n_strat)

        # Initialize hypercubes
        hypercubes_one_dim = np.arange(0, int(self._n_strat))
        hypercubes = [list(p) for p in product(hypercubes_one_dim, repeat=int(n_dim))]
        self._hypercubes = tf.convert_to_tensor(hypercubes, dtype=DTYPEINT)

        if len(hypercubes) != int(tf.math.pow(self._n_strat, n_dim)):
            raise ValueError("Hypercubes are not equal to n_strat^n_dim")

        self.min_neval_hcube = int(neval_eff // len(hypercubes))
        self.min_neval_hcube = max(self.min_neval_hcube, 2)

        self.n_ev = tf.fill([1, len(hypercubes)], self.min_neval_hcube)
        self.n_ev = int_me(tf.reshape(self.n_ev, [-1]))
        self._n_events = int(tf.reduce_sum(self.n_ev))
        self._modified_jac = float_me(1 / len(hypercubes))

        if self._adaptive:
            logger.warning("Variable number of events requires function signatures all across")

    @property
    def xjac(self):
        return self._modified_jac

    def make_differentiable(self):
        """Overrides make_differentiable to make sure the runner has a reference to n_ev"""
        runner = super().make_differentiable()
        return partial(runner, n_ev=self.n_ev)

    def redistribute_samples(self, arr_var):
        """Receives an array with the variance of the integrand in each
        hypercube and recalculate the samples per hypercube according
        to VEGAS+ algorithm"""
        damped_arr_sdev = tf.pow(arr_var, BETA / 2)
        new_n_ev = tf.maximum(
            self.min_neval_hcube,
            damped_arr_sdev * self._init_calls / 2 / tf.reduce_sum(damped_arr_sdev),
        )
        self.n_ev = int_me(new_n_ev)
        self.n_events = int(tf.reduce_sum(self.n_ev))

    def _digest_random_generation(self, rnds, n_ev):
        """Generate a random array for a given number of events divided in hypercubes"""
        # Get random numbers from hypercubes
        x, w, ind, segm = generate_samples_in_hypercubes(
            rnds, self._n_strat, n_ev, self._hypercubes, self.divisions
        )
        return x, w, ind, segm

    def generate_random_array(self, n_events, *args):
        """Override the behaviour of ``generate_random_array``
        to accomodate for the peculiarities of VegasFlowPlus
        """
        rnds = []
        wgts = []
        for _ in range(n_events // self.n_events + 1):
            r, w = super().generate_random_array(self.n_events, self.n_ev)
            rnds.append(r)
            wgts.append(w)
        final_r = tf.concat(rnds, axis=0)[:n_events]
        final_w = tf.concat(wgts, axis=0)[:n_events] * self.n_events / n_events
        return final_r, final_w

    def _run_event(self, integrand, ncalls=None, n_ev=None):
        """Run one step of VegasFlowPlus
        Similar to the event step for importance sampling VegasFlow
        adding the n_ev argument for the segmentation into hypercubes
        n_ev is a tensor containing the number of samples per hypercube

        Parameters
        ----------
            `integrand`: function to integrate
            `ncalls`: how many events to run in this step
            `n_ev`: number of samples per hypercube

        Returns
        -------
            `res`: sum of the result of the integrand for all events per segment
            `res2`: sum of the result squared of the integrand for all events per segment
            `arr_res2`: result of the integrand squared per dimension and grid bin
        """
        # NOTE: needs to receive both ncalls and n_ev
        x, xjac, ind, segm = self._generate_random_array(ncalls, n_ev)

        # compute integrand
        tmp = xjac * integrand(x, weight=xjac)
        tmp2 = tf.square(tmp)

        # tensor containing resummed component for each hypercubes
        ress = tf.math.segment_sum(tmp, segm)
        ress2 = tf.math.segment_sum(tmp2, segm)

        fn_ev = float_me(n_ev)
        arr_var = ress2 * fn_ev - tf.square(ress)
        arr_res2 = self._importance_sampling_array_filling(tmp2, ind)

        return ress, arr_var, arr_res2

    def _iteration_content(self):
        """Steps to follow per iteration
        Differently from importance-sampling Vegas, the result of the integration
        is a result _per segment_ and thus the total result needs to be computed at this point
        """
        ress, arr_var, arr_res2 = self.run_event(n_ev=self.n_ev)

        # Compute the rror
        sigmas2 = tf.maximum(arr_var, fzero)
        res = tf.reduce_sum(ress)
        sigma2 = tf.reduce_sum(sigmas2 / (float_me(self.n_ev) - fone))
        sigma = tf.sqrt(sigma2)

        # If adaptive is active redistribute the samples
        if self._adaptive:
            self.redistribute_samples(arr_var)

        if self.train:
            self.refine_grid(arr_res2)

        return res, sigma

    def run_event(self, tensorize_events=None, **kwargs):
        """Tensorizes the number of events
        so they are not python or numpy primitives if self._adaptive=True"""
        return super().run_event(tensorize_events=self._adaptive, **kwargs)


def vegasflowplus_wrapper(integrand, n_dim, n_iter, total_n_events, **kwargs):
    """Convenience wrapper

    Parameters
    ----------
        `integrand`: tf.function
        `n_dim`: number of dimensions
        `n_iter`: number of iterations
        `n_events`: number of events per iteration

    Returns
    -------
        `final_result`: integral value
        `sigma`: monte carlo error
    """
    return wrapper(VegasFlowPlus, integrand, n_dim, n_iter, total_n_events, **kwargs)


def vegasflowplus_sampler(*args, **kwargs):
    """Convenience wrapper for sampling random numbers

    Parameters
    ----------
        `integrand`: tf.function
        `n_dim`: number of dimensions
        `n_events`: number of events per iteration
        `training_steps`: number of training_iterations

    Returns
    -------
        `sampler`: a reference to the generate_random_array method of the integrator class
    """
    return sampler(VegasFlowPlus, *args, **kwargs)
