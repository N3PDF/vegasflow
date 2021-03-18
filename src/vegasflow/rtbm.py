"""
    Plain implementation of the plainest possible MonteCarlo
"""
import copy
import numpy as np
from vegasflow.configflow import DTYPE, fone, fzero, float_me
from vegasflow.monte_carlo import MonteCarloFlow, wrapper
import tensorflow as tf

from theta.rtbm import RTBM # pylint:disable=import-error
from theta import costfunctions # pylint:disable=import-error
from cma import CMAEvolutionStrategy # pylint:disable=import-error

import logging
from joblib import Parallel, delayed
from time import time

logger = logging.getLogger(__name__)

# Cost functions for training
def _kl(x, ytarget):
    return ytarget*np.log(ytarget/x)

_loss = _kl

def _train_machine(rtbm, target_tf, original_r_tf):

    logger.info("Training RTBM")

    target = target_tf.numpy()
    original_r = original_r_tf.numpy()

    n_jobs = 8
    max_iterations = 250


    def target_loss(params):
        if not rtbm.set_parameters(params):
            return np.NaN
        _, prob = rtbm.get_transformation(original_r)
        return np.sum(_loss(prob, target))

    best_parameters = copy.deepcopy(rtbm.get_parameters())
    min_bound, max_bound = rtbm.get_bounds()

    with Parallel(n_jobs=n_jobs) as parallel:
        prev = time()
        n_parameters = len(best_parameters)

        # random hyperparameters
        pop_per_rate = 32
        mutation_rates = np.array([0.2, 0.4, 0.6, 0.8])
        rates = np.concatenate([np.ones(pop_per_rate)*mr for mr in mutation_rates])
        original_sigma = 0.25
        sigma = original_sigma
        repeats = 3

        for it in range(max_iterations):

            # Get the best parameters from the previous iteration
            p0 = copy.deepcopy(best_parameters)
            loss_val = target_loss(p0)

            def compute_mutant(mutation_rate):
                number_of_mut = int(mutation_rate*n_parameters)
                mut_idx = np.random.choice(n_parameters, number_of_mut, replace=False)
                r1, r2 = np.random.rand(2, number_of_mut)*sigma

                mutant = copy.deepcopy(p0)
                var_plus = max_bound - p0
                var_minus = min_bound - p0
                mutant[mut_idx] += var_plus[mut_idx]*r1 + var_minus[mut_idx]*r2

                return target_loss(mutant), mutant

            parallel_runs = [delayed(compute_mutant)(rate) for rate in rates]
            result = parallel(parallel_runs)
            losses, mutants = zip(*result)

            best_loss = np.nanmin(losses)
            if best_loss < loss_val:
                loss_val = best_loss
                best_parameters = mutants[losses.index(best_loss)]
            else:
                sigma /= 2

            if it % 10 == 0:
                current = time()
                print(f"Iteration {it}, best loss: {loss_val:.2f}, time; {current-prev:.2f}s")
                prev = current

            if sigma < 1e-3:
                sigma = original_sigma
                print(f"Resetting sigma with loss: {loss_val:.2f}")
                repeats -= 1
                break

            if not repeats:
                print(f"No more repeats allowed, iteration: {it}, loss: {loss_val:2.f}")

        rtbm.set_parameters(best_parameters)
        return rtbm

class RTBMFlow(MonteCarloFlow):
    """
    RTBM based Monte Carlo integrator
    """

    def __init__(self, n_hidden=3, rtbm=None, train=True, *args, **kwargs):
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
                positive_T=True,
                positive_Q=True,
                gaussian_parameters={"mean":0.0, "std": 0.55},
                sampling_activation="tanh"
            )
        else:
            # Check whether it is a valid rtbm model
            if not hasattr(rtbm, "make_sample"):
                raise TypeError(f"{rtbm} is not a valid boltzman machine")
            self._rtbm = rtbm
        self._p0 = self._rtbm.get_parameters()

    def freeze(self):
        """ Stop the training """
        self.train = False

    def unfreeze(self, reset_p0 = False):
        """ Restart the training """
        self.train = True
        if reset_p0:
            self._p0 = self._rtbm.get_parameters()

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

        The return dimension of the random variables is (n_events,n_dim) and the jacobian (n_events)
        """
        xrand, px, original_r = self._rtbm.make_sample_rho(n_events)
        # Since we are using the tanh function, the integration limits are (-1,1), move:
        xjac = 1.0/px/n_events
        return float_me(xrand), original_r, float_me(xjac)

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
        tmp = integrand(rnds, n_dim=self.n_dim, weight=xjac)
        res = tmp*xjac

        return res, original_r

    def _run_iteration(self):
        all_res, original_r = self.run_event()

        if self.train:
            _train_machine(self._rtbm, all_res, original_r)

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
