"""
    Plain implementation of the plainest possible MonteCarlo
"""
import copy
import itertools
import numpy as np
from vegasflow.configflow import DTYPE, fone, fzero, float_me, run_eager
from vegasflow.monte_carlo import MonteCarloFlow, wrapper
import tensorflow as tf

run_eager(True)

try:
    from theta.rtbm import RTBM  # pylint:disable=import-error
    from theta import costfunctions  # pylint:disable=import-error
except ImportError as e:
    raise ValueError(
        "Cannot use the RTBM based integrator without installing the appropiate libraries"
    ) from e

import logging
from joblib import Parallel, delayed
from time import time

logger = logging.getLogger(__name__)
TOL = 1e-6

# Cost functions for training
def _kl(x, ytarget_raw):
    ytarget = ytarget_raw + TOL

    ytarget /= np.sum(ytarget)
    x /= np.sum(x)

    return ytarget * np.log(ytarget / x)


def _mse(x, y):
    integral = np.sum(y)
    x /= np.sum(x)
    y /= integral
    return integral * pow(x - y, 2)


_loss = _kl


def _generate_target_loss(rtbm, original_r, target):
    def target_loss(params):
        if not rtbm.set_parameters(params):
            return np.NaN
        _, prob = rtbm.get_transformation(original_r)
        if (prob <= 0.0).any():
            return np.NaN
        return np.sum(_loss(prob, target))

    return target_loss


def _train_machine(
    rtbm,
    target_tf,
    original_r_tf,
    n_jobs=32,
    max_iterations=3000,
    pop_per_rate=512,
    verbose=True,
    resets=True,
    timeout=5 * 60,  # dont wait more than 5 minutes per iteration
    timeout_repeat=3,
    fail_ontimeout=False,
    skip_on_nan=True,
):

    # note that if between 2 timeout there are severaliterations
    # but they do not produce a better esult it doesn't count
    timeout_original = timeout_repeat

    if verbose:
        logger.info("Training RTBM")

    if isinstance(target_tf, np.ndarray):
        target = target_tf
    else:
        target = target_tf.numpy()

    if isinstance(original_r_tf, np.ndarray):
        original_r = original_r_tf
    else:
        original_r = original_r_tf.numpy()

    target_loss = _generate_target_loss(rtbm, original_r, target)

    best_parameters = copy.deepcopy(rtbm.get_parameters())
    min_bound, max_bound = rtbm.get_bounds()
    loss_val = target_loss(best_parameters)

    with Parallel(n_jobs=n_jobs, timeout=timeout) as parallel:
        prev = time()
        n_parameters = len(best_parameters)

        # training hyperparameters
        mutation_rates = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.96])
        rates = np.concatenate([np.ones(pop_per_rate) * mr for mr in mutation_rates])
        original_sigma = 0.55
        sigma = original_sigma
        repeats = 3
        #######

        for it in range(max_iterations):

            # Get the best parameters from the previous iteration
            p0 = copy.deepcopy(best_parameters)

            def compute_mutant(mutation_rate):
                number_of_mut = int(mutation_rate * n_parameters)
                mut_idx = np.random.choice(n_parameters, number_of_mut, replace=False)
                r1, r2 = np.random.rand(2, number_of_mut) * sigma

                mutant = copy.deepcopy(p0)
                var_plus = max_bound - p0
                var_minus = min_bound - p0
                mutant[mut_idx] += var_plus[mut_idx] * r1 + var_minus[mut_idx] * r2

                return target_loss(mutant), mutant

            parallel_runs = [delayed(compute_mutant)(rate) for rate in rates]
            try:
                result = parallel(parallel_runs)
            except:  # TODO control better
                logger.debug("Time'd out, skip me")
                timeout_repeat -= 1
                if fail_ontimeout:
                    raise ValueError
                if timeout_repeat == 0:
                    logger.debug("Full timeout, out")
                    break
                result = [(np.nan, None)]
            losses, mutants = zip(*result)

            # Insert the last best to avoid runtime errors
            best_loss = np.nanmin(list(losses) + [loss_val + 1.0])
            if best_loss < loss_val:
                timeout_repeat = timeout_original  # reset the timeouts
                loss_val = best_loss
                best_parameters = mutants[losses.index(best_loss)]
            else:
                sigma *= 0.9

            if it % 50 == 0 and verbose:
                current = time()
                logger.debug(
                    "Iteration %d, best_loss: %.4f, (%2.fs)",
                    it,
                    loss_val,
                    current - prev,
                )
                logger.debug(
                    "NaNs in last iteration: %d/%d, sigma=%.4f",
                    np.count_nonzero(np.isnan(losses)),
                    len(losses),
                    sigma,
                )
                prev = current

            if sigma < 1e-2:
                sigma = original_sigma
                logger.debug("Resetting sigma with loss: %.4f after %d iterations", loss_val, it)
                if resets:
                    repeats -= 1
                else:
                    repeats = 0

            if not repeats:
                print(f"No more repeats allowed, iteration: {it}, loss: {loss_val:.4f}")
                break

        rtbm.set_parameters(best_parameters)
        return loss_val


class RTBMFlow(MonteCarloFlow):
    """
    RTBM based Monte Carlo integrator
    """

    def __init__(self, n_hidden=3, rtbm=None, train=True, generations=3000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train = train
        self._first_run = False
        self._n_hidden = n_hidden
        self._ga_generations = generations
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
                gaussian_parameters={"mean": 0.0, "std": 0.75},
                sampling_activations="tanh",
            )
        else:
            # Check whether it is a valid rtbm model
            if not hasattr(rtbm, "make_sample"):
                raise TypeError(f"{rtbm} is not a valid boltzman machine")
            self._rtbm = rtbm
            self._first_run = False

    def freeze(self):
        """Stop the training"""
        self.train = False

    def unfreeze(self):
        """Restart the training"""
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

        The return dimension of the random variables is (n_events,n_dim) and the jacobian (n_events)
        """
        if self._first_run:
            rnds, _, xjac = super().generate_random_array(n_events)
            return rnds, rnds, xjac

        xrand, px, original_r = self._rtbm.make_sample_rho(n_events)
        # Since we are using the tanh function, the integration limits are (-1,1), move:
        xjac = 1.0 / px / n_events
        return float_me(xrand), original_r, float_me(xjac)

    @staticmethod
    def _accumulate(accumulators):
        """For the RTBM accumulation strategy we need to keep track
        of all results and who produced it"""
        # In the accumulators we should receive a number of items with
        # (res, xrands) which have shape ( (n_events,), (n_events, n_dim) )
        all_res = []
        all_unw = []
        all_rnd = []
        for (res, unw, rnds) in accumulators:
            all_res.append(res)
            all_unw.append(unw)
            all_rnd.append(rnds)
        return (
            tf.concat(all_res, axis=0),
            tf.concat(all_unw, axis=0),
            tf.concat(all_rnd, axis=0),
        )

    def _run_event(self, integrand, ncalls=None):
        if ncalls is None:
            n_events = self.n_events
        else:
            n_events = ncalls

        # Generate all random number for this iteration
        rnds, original_r, xjac = self.generate_random_array(n_events)
        # Compute the integrand
        unw = integrand(rnds, n_dim=self.n_dim, weight=xjac)
        res = unw * xjac

        return res, unw, original_r

    def _run_iteration(self):
        all_res, unw, original_r = self.run_event()
        generations = self._ga_generations

        if self.train and self._first_run:
            original_r = self._run_first_run(unw, original_r)
            self._first_run = False
            generations = 250

        if self.train:
            _train_machine(self._rtbm, unw, original_r, max_iterations=generations)

        res = tf.reduce_sum(all_res)
        all_res2 = all_res ** 2
        res2 = tf.reduce_sum(all_res2) * self.n_events

        # Compute the error
        err_tmp2 = (res2 - tf.square(res)) / (self.n_events - fone)
        sigma = tf.sqrt(tf.maximum(err_tmp2, fzero))
        return res, sigma

    def _run_first_run(self, unweighted_events, tf_rnds):
        """
        Run the first iteration of the RTBM
        """
        rnds = tf_rnds.numpy()
        configurations_raw = [
            ("tanh", 0.0, 0.75),
            ("sigmoid", 0.0, 1.5),
            #             ("softmax", 0.0, 0.75),
        ]
        names = ("name", "mean", "std")

        configuration_per_d = []
        for rnd in rnds.T:
            vals, lims = np.histogram(rnd, bins=3, weights=unweighted_events, density=True)
            mean = None
            if vals[0] > 2 * vals[-1]:
                mean = -0.5
            elif vals[-1] > 2 * vals[0]:
                mean = 0.5
            tmp_list = copy.copy(configurations_raw)
            if mean is not None:
                tmp_list.append(("tanh", mean, 0.75))
            #                 tmp_list.append(("sigmoid", mean, 1.5))
            configuration_per_d.append([dict(zip(names, tup)) for tup in tmp_list])

        # TODO:
        # The number of combinations will intractably grow with the number of dimensions
        # put some limits to this
        all_configs = list(itertools.product(*configuration_per_d))
        logger.info(f"Generating {len(all_configs)} configurations")

        best_loss = 1e9
        best_params = None
        winner_config = None

        for configuration in all_configs:
            logger.info(f"Testing {configuration}")
            acts, means, stds = zip(*[i.values() for i in configuration])
            rtbm = RTBM(
                self.n_dim,
                self._n_hidden,
                minimization_bound=50,
                gaussian_init=True,
                positive_T=True,
                positive_Q=True,
                gaussian_parameters={"mean": means, "std": stds},
                sampling_activations=acts,
            )
            guessed_original_r = rtbm.undo_transformation(rnds)
            try:
                loss = _train_machine(
                    rtbm,
                    unweighted_events,
                    guessed_original_r,
                    max_iterations=32,
                    pop_per_rate=32,
                    resets=False,
                    verbose=False,
                    fail_ontimeout=True,
                    timeout=10,  # TODO if any initial iteration takes more than 10 seconds, get out
                )
            except ValueError:
                loss = 1e9
            if loss < best_loss:
                best_loss = loss
                best_params = rtbm.get_parameters()
                winner_config = configuration
        self._rtbm = rtbm
        rtbm.set_parameters(best_params)
        logger.info(f"And the winner is: {configuration}")
        return rtbm.undo_transformation(rnds)


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
