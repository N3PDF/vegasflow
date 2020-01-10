"""
    Abstract class for Monte Carlo integrators
"""

from abc import abstractmethod, ABC
import numpy as np

class MonteCarloFlow(ABC):
    """
    Parameters
    ----------
        `n_dim`: number of dimensions of the integrand
        `n_events`: number of events per iteration
    """

    def __init__(self, n_dim, n_events):
        # Save some parameters
        self.n_dim = n_dim
        self.n_events = n_events
        self.xjac = 1.0 / n_events
        self.integrand = None
        self.compiled = False
        self.all_results = []

    def compile(self, integrand):
        """ Receives an integrand, prepares it for integration
        and forces the compilation

        Parameters
        ----------
            `integrand`: the output of a tf.function, must receive as input a tensor
        """
        self.integrand = integrand
        self.compiled = True

    @abstractmethod
    def _run_iteration(self):
        pass

    def run_integration(self, n_iter):
        """ Runs the integrator for the chosen number of iterations
        Parameters
        ---------
            `n_iter`: number of iterations
        Returns
        -------
            `final_result`: integral value
            `sigma`: monte carlo error
        """
        for _ in range(n_iter):
            self._run_iteration()
        aux_res = 0.0
        weight_sum = 0.0
        for result in self.all_results:
            res = result[0]
            sigma = result[1]
            wgt_tmp = 1.0 / pow(sigma, 2)
            aux_res += res * wgt_tmp
            weight_sum += wgt_tmp

        final_result = aux_res / weight_sum
        sigma = np.sqrt(1.0 / weight_sum)
        print(f" > Final results: {final_result.numpy()} +/- {sigma}")
        return final_result, sigma
