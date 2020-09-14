#!/usr/env python
"""
    This module contains the VegasFlow class and all its auxuliary functions

    The main interfaces of this class are the class `VegasFlow` and the
    `vegas_wrapper`
"""
from vegasflow.configflow import DTYPE, DTYPEINT, fone, fzero, float_me, ione, int_me
import json
import numpy as np
import tensorflow as tf

from vegasflow.configflow import DTYPE, DTYPEINT, fone, fzero, float_me, ione, int_me
from vegasflow.configflow import BINS_MAX, ALPHA
from vegasflow.monte_carlo import MonteCarloFlow, wrapper
from vegasflow.utils import consume_array_into_indices

import logging

logger = logging.getLogger(__name__)

FBINS = float_me(BINS_MAX)

# Auxiliary functions for Vegas
@tf.function
def generate_random_array(rnds, divisions):
    """
        Generates the Vegas random array for any number of events

        Parameters
        ----------
            rnds: array shaped (None, n_dim)
                Random numbers used as an input for Vegas
            divisions: array shaped (n_dim, BINS_MAX)
                vegas grid

        Returns
        -------
            x: array (None, n_dim)
                Vegas random output
            div_index: array (None, n_dim)
                division index in which each (n_dim) set of random numbers fall
            w: array (None,)
                Weight of each set of (n_dim) random numbers
    """
    # Get the boundaries of the random numbers
    #     reg_i = fzero
    #     reg_f = fone
    # Get the index of the division we are interested in
    xn = FBINS * (fone - rnds)

    @tf.function
    def digest(xn):
        ind_i = tf.cast(xn, DTYPEINT)
        # Get the value of the left and right sides of the bins
        ind_f = ind_i + ione
        x_ini = tf.gather(divisions, ind_i, batch_dims=1)
        x_fin = tf.gather(divisions, ind_f, batch_dims=1)
        # Compute the width of the bins
        xdelta = x_fin - x_ini
        return ind_i, x_ini, xdelta

    ind_i, x_ini, xdelta = digest(xn)
    # Compute the random number between 0 and 1
    # This is the heavy part of the calc
    @tf.function
    def compute_x(x_ini, xn, xdelta):
        aux_rand = xn - tf.math.floor(xn)
        return x_ini + xdelta * aux_rand

    x = compute_x(x_ini, xn, xdelta)
    # Compute the random number between the limits
    #     x = reg_i + rand_x * (reg_f - reg_i)
    # and the weight
    weights = tf.reduce_prod(xdelta * FBINS, axis=0)
    x_t = tf.transpose(x)
    int_xn = tf.transpose(ind_i)
    return x_t, int_xn, weights


@tf.function
def refine_grid_per_dimension(t_res_sq, subdivisions):
    """
        Modifies the boundaries for the vegas grid for a given dimension

        Parameters
        ----------
            `t_res_sq`: tensor
                array of results squared per bin
            `subdivision`: tensor
                current boundaries for the grid

        Returns
        -------
            `new_divisions`: tensor
                array with the new boundaries of the grid
    """
    # Define some constants
    paddings = tf.constant([[1, 1],])
    tmp_meaner = tf.fill([BINS_MAX - 2,], float_me(3.0))
    meaner = tf.pad(tmp_meaner, paddings, constant_values=2.0)
    # Pad the vector of results
    res_padded = tf.pad(t_res_sq, paddings)
    # First we need to smear out the array of results squared
    smeared_tensor_tmp = res_padded[1:-1] + res_padded[2:] + res_padded[:-2]
    smeared_tensor = tf.maximum(smeared_tensor_tmp / meaner, float_me(1e-30))
    # Now we refine the grid according to
    # journal of comp phys, 27, 192-203 (1978) G.P. Lepage
    sum_t = tf.reduce_sum(smeared_tensor)
    log_t = tf.math.log(smeared_tensor)
    aux_t = (1.0 - smeared_tensor / sum_t) / (tf.math.log(sum_t) - log_t)
    wei_t = tf.pow(aux_t, ALPHA)
    ave_t = tf.reduce_sum(wei_t) / BINS_MAX

    ###### Auxiliary functions for the while loop
    @tf.function
    def while_check(bin_weight, *args):
        """ Checks whether the bin has enough weight
        to beat the average """
        return bin_weight < ave_t

    @tf.function
    def while_body(bin_weight, n_bin, cur, prev):
        """ Fills the bin weight until it surpassed the avg
        once it's done, returns the limits of the last bin """
        n_bin += 1
        bin_weight += wei_t[n_bin]
        prev = cur
        cur = subdivisions[n_bin + 1]
        return bin_weight, n_bin, cur, prev

    ###########################

    # And now resize all bins
    new_bins = [fzero]
    # Auxiliary variables
    bin_weight = fzero
    n_bin = -1
    cur = fzero
    prev = fzero
    for _ in range(BINS_MAX - 1):
        bin_weight, n_bin, cur, prev = tf.while_loop(
            while_check, while_body, (bin_weight, n_bin, cur, prev), parallel_iterations=1,
        )
        bin_weight -= ave_t
        delta = (cur - prev) * bin_weight / wei_t[n_bin]
        new_bins.append(cur - delta)
    new_bins.append(fone)

    new_divisions = tf.stack(new_bins)
    return new_divisions


####### VegasFlow
class VegasFlow(MonteCarloFlow):
    """
    Implementation of the important sampling algorithm from Vegas
    """

    def __init__(self, n_dim, n_events, train=True, **kwargs):
        super().__init__(n_dim, n_events, **kwargs)

        # If training is True, the grid will be changed after every iteration
        # otherwise it will be frozen
        self.train = train
        self.iteration_content = None
        self.compile_args = None

        # Initialize grid
        self.grid_bins = BINS_MAX + 1
        subdivision_np = np.linspace(0, 1, self.grid_bins)
        divisions_np = subdivision_np.repeat(n_dim).reshape(-1, n_dim).T
        self.divisions = tf.Variable(divisions_np, dtype=DTYPE)

    def freeze_grid(self):
        """ Stops the grid from refining any more """
        self.train = False
        self.recompile()

    def unfreeze_grid(self):
        """ Enable the refining of the grid """
        self.train = True
        self.recompile()

    def save_grid(self, file_name):
        """ Save the `divisions` array in a json file

        Parameters
        ----------
            `file_name`: str
            Filename in which to save the checkpoint
        """
        div_np = self.divisions.numpy()
        if self.integrand:
            int_name = self.integrand.__name__
        else:
            int_name = ""
        json_dict = {
            "dimensions": self.n_dim,
            "ALPHA": ALPHA,
            "BINS": self.grid_bins,
            "integrand": int_name,
            "grid": div_np.tolist(),
        }
        with open(file_name, "w") as f:
            json.dump(json_dict, f, indent=True)

    def load_grid(self, file_name=None, numpy_grid=None):
        """ Load the `divisions` array from a json file
        or from a numpy_array

        Parameters
        ----------
            `file_name`: str
            Filename in which the grid json is stored
            `numpy_grid`: np.array
            Numpy array to substitute divisions with
        """
        if file_name is not None and numpy_grid is not None:
            raise ValueError(
                "Received both a numpy grid and a file_name to load the grid from."
                "Ambiguous call to `load_grid`"
            )

        # If it received a file, loads up the grid
        if file_name:
            with open(file_name, "r") as f:
                json_dict = json.load(f)
            # First check the parameters of the grid are unchanged
            grid_dim = json_dict.get("dimensions")
            grid_bins = json_dict.get("BINS")
            # Check that the integrand is the same one
            if self.integrand:
                integrand_name = self.integrand.__name__
                integrand_grid = json_dict.get("integrand")
                if integrand_name != integrand_grid:
                    logger.warning(
                        f"The grid was written for the integrand: {integrand_grid}"
                        f"which is different from {integrand_name}"
                    )
            # Now that everything is clear, let's load up the grid
            numpy_grid = np.array(json_dict["grid"])
        elif numpy_grid is not None:
            grid_dim = numpy_grid.shape[0]
            grid_bins = numpy_grid.shape[1]
        else:
            raise ValueError("load_grid was called but no grid was provided!")
        # Check that the grid has the right dimensions
        if grid_dim is not None and self.n_dim != grid_dim:
            raise ValueError(
                f"Received a {grid_dim}-dimensional grid while VegasFlow"
                f"was instantiated with {self.n_dim} dimensions"
            )
        if grid_bins is not None and self.grid_bins != grid_bins:
            raise ValueError(
                f"The received grid contains {grid_bins} bins while the"
                f"current settings is of {self.grid_bins} bins"
            )
        if file_name:
            logger.info(f" > SUCCESS: Loaded grid from {file_name}")
        self.divisions.assign(numpy_grid)

    def refine_grid(self, arr_res2):
        """ Receives an array with the values of the integral squared per
        bin per dimension (`arr_res2.shape = (n_dim, self.grid_bins)`)
        and reshapes the `divisions` attribute accordingly

        Parameters
        ----------
            `arr_res2`: result the integrand sq per dimension and grid bin

        Function not compiled
        """
        for j in range(self.n_dim):
            new_divisions = refine_grid_per_dimension(arr_res2[j, :], self.divisions[j, :])
            self.divisions[j, :].assign(new_divisions)

    def _run_event(self, integrand, ncalls=None):
        """ Runs one step of Vegas.

        Parameters
        ----------
            `integrand`: function to integrate
            `ncalls`: how many events to run in this step

        Returns
        -------
            `res`: sum of the result of the integrand for all events
            `res2`: sum of the result squared of the integrand for all events
            `arr_res2`: result of the integrand squared per dimension and grid bin
        """
        if ncalls is None:
            n_events = self.n_events
        else:
            n_events = ncalls

        tech_cut = 1e-8
        # Generate all random number for this iteration
        rnds = tf.random.uniform(
            (self.n_dim, n_events), minval=tech_cut, maxval=1.0 - tech_cut, dtype=DTYPE
        )

        # Pass them through the Vegas digestion
        x, ind, w = generate_random_array(rnds, self.divisions)

        # Now compute the integrand
        xjac = self.xjac * w
        if self.simplify_signature:
            tmp = xjac * integrand(x)
        else:
            tmp = xjac * integrand(x, n_dim=self.n_dim, weight=xjac)
        tmp2 = tf.square(tmp)

        # Compute the final result for this step
        res = tf.reduce_sum(tmp)
        res2 = tf.reduce_sum(tmp2)

        arr_res2 = []
        if self.train:
            # If the training is active, save the result of the integral sq
            for j in range(self.n_dim):
                arr_res2.append(
                    consume_array_into_indices(tmp2, ind[:, j : j + 1], int_me(self.grid_bins - 1))
                )
            arr_res2 = tf.reshape(arr_res2, (self.n_dim, -1))

        return res, res2, arr_res2

    def _iteration_content(self):
        # Compute the result
        res, res2, arr_res2 = self.run_event()
        # Compute the error
        err_tmp2 = (self.n_events * res2 - tf.square(res)) / (self.n_events - fone)
        sigma = tf.sqrt(tf.maximum(err_tmp2, fzero))
        # If training is active, act post integration
        if self.train:
            self.refine_grid(arr_res2)
        return res, sigma

    def compile(self, integrand, compilable=True, **kwargs):
        self.compile_args = (integrand, compilable, kwargs)
        super().compile(integrand, compilable=compilable, **kwargs)
        self.iteration_content = self._iteration_content

    def recompile(self):
        """ Forces recompilation with the same arguments that have
        previously been used for compilation"""
        if self.compile_args is None:
            raise RuntimeError("recompile was called without ever having called compile")
        a = self.compile_args
        self.compile(a[0], a[1], **a[2])

    def _run_iteration(self):
        """ Runs one iteration of the Vegas integrator """
        res, sigma = self.iteration_content()
        return res, sigma


def vegas_wrapper(integrand, n_dim, n_iter, total_n_events, **kwargs):
    """ Convenience wrapper

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
    return wrapper(VegasFlow, integrand, n_dim, n_iter, total_n_events, **kwargs)
