"""
    Plain implementation of the plainest possible MonteCarlo
"""
import copy
import numpy as np
from vegasflow.configflow import DTYPE, fone, fzero, float_me
from vegasflow.monte_carlo import MonteCarloFlow, wrapper
import tensorflow as tf

from theta.rtbm import RTBM
from theta import costfunctions
from cma import CMAEvolutionStrategy

import logging

logger = logging.getLogger(__name__)


class RTBMFlow(MonteCarloFlow):
    """
    RTBM based Monte Carlo integrator
    """

    def __init__(self, n_hidden=2, rtbm=None, train=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train = train
        self._first_run = True
        if rtbm is None:
            logger.info(
                "Generating a RTBM with %d visible nodes and %d hidden" % (self.n_dim, n_hidden)
            )
            self._rtbm = RTBM(
                self.n_dim,
                n_hidden,
                minimization_bound=50,
                gaussian_init=True,
                diagonal_T=False,
                positive_T=True,
                positive_Q=True,
            )
        else:
            # Check whether it is a valid rtbm model
            if not hasattr(rtbm, "make_sample"):
                raise TypeError(f"{rtbm} is not a valid boltzman machine")
            self._rtbm = rtbm

    def freeze(self):
        self.train = False

    def unfreeze(self):
        self.train = True

    def compile(self, integrand, compilable=False, **kwargs):
        if compilable:
            logger.warning("RTBMFlow is still WIP and not compilable")
        super().compile(integrand, compilable=False, **kwargs)

    def generate_random_array(self, n_events):
        """
        Returns (xrand, original_r, xjac)
        where xrand is the integration variable between 0 and 1
        and xjac the correspondent jacobian factor.
        original_r is the original random values sampled by the RTBM to be used at training
        """
        xrand, px, original_r = self._rtbm.make_sample_rho(n_events)
        xjac = float_me(1.0/px/n_events)
        return float_me(xrand), original_r, xjac

    def _train_machine(self, x, yraw):
        """ Takes as input the random sample from the previous step
        and the values of the calculation"""
        # Get a reference to the initial solution of the CMA
        x0 = copy.deepcopy(self._rtbm.get_parameters())
        bounds = self._rtbm.get_bounds()

        options = {
                "bounds": bounds,
                "maxiter": 250,
                }

        xnp = x.numpy().T
        ynp = yraw.numpy()

        sol_found = False
        def optimization(n=1):
            sigma = np.min(bounds[1])/(4.0*n)
            es = CMAEvolutionStrategy(x0, sigma, options)

            def target(params):
                if not self._rtbm.set_parameters(params):
                    return np.NaN
                prob = self._rtbm(xnp)
                return costfunctions.kullbackLeibler.cost(prob, ynp)

            es.optimize(target)
            return es.result

        n = 1
        while not sol_found:
            res = optimization(n)
            sol_found = self._rtbm.set_parameters(res.xbest)
            if not sol_found:
                logger.warning("Optimization failed, trying again!")
            n+=1

        self._first_run = False

    @staticmethod
    def _accumulate(accumulators):
        """For the RTBM accumulation strategy we need to keep track
        of all results and who produced it"""
        # In the accumulators we should receive a number of items with
        # (res, xrands) which have shape ( (n_events,), (n_events, n_dim) )
        all_res = []
        all_rnd = []
        for (
            res,
            rnds,
        ) in accumulators:
            all_res.append(res)
            all_rnd.append(rnds)
        return tf.concat(all_res, axis=0), tf.concat(all_rnd, axis=0)

    def _run_event(self, integrand, ncalls=None):
        if ncalls is None:
            n_events = self.n_events
        else:
            n_events = ncalls

        # Generate all random number for this iteration
        rnds, original_r, xjac = self.generate_random_array(n_events)
        # Compute the integrand
        res = integrand(rnds, n_dim=self.n_dim, weight=xjac) * xjac

        # Clean up the array from numbers outside the 0-1 range
        if True: # They should always be between 0 and 1 anyway
            # Accept for now only random number between 0 and 1
            condition = tf.reduce_all(rnds >= 0.0, axis=1) & tf.reduce_all(rnds <= 1.0, axis=1)
            res = tf.where(condition, res, fzero)
            if np.count_nonzero(res) != n_events:
                logger.warning(f"Found only {np.count_nonzero(res)} of {n_events} valid values\n")

        return res, original_r

    def _run_iteration(self):
        all_res, rnds = self.run_event()
        if self.train:
            self._train_machine(rnds, all_res)

        res = tf.reduce_sum(all_res)
        all_res2 = all_res ** 2
        res2 = tf.reduce_sum(all_res2) * self.n_events

        # Compute the error
        err_tmp2 = (res2 - tf.square(res)) / (self.n_events - fone)
        sigma = tf.sqrt(tf.maximum(err_tmp2, fzero))
        return res, sigma


def rtbm_wrapper(integrand, n_dim, n_iter, total_n_events, **kwargs):
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
    return wrapper(RTBMFlow, integrand, n_dim, n_iter, total_n_events, **kwargs)
