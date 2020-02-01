"""
    Abstract class for Monte Carlo integrators
    implements a distribution of events across multiple devices and tensorflow graph technology

    Usage:
        In order to implement a new MonteCarloFlow integrator
        it is necessary to implement (at least) two methods:
        - `_run_event`: integrand
            This function defines what to do in order to run one event
            of the Monte Carlo. It is used only for compilation, as the
            actual integration is done by the `run_event` method.
            In order to use the full capabilities of this library, `_run_event`
            can take a number of events as its input so it can run more than one
            event at the same time.
            All results from `_run_event` will be accumulated before being passed
            to `_run_iteration`.
        - `_run_iteration`:
            This function defines what to do in a full iteration of the
            MonteCarlo (i.e., what to do in order to run for n_events)

    Device distribution:
        The default behaviour is defined in the `configflow.py` file.
    This class will go through the devices given in the `list_devices` argument and consider
    them all active and enabled. Then the integration will be broken in batches of `events_limit`
    which will be given to the first idle device found.
    This means if device A is two times faster than device B, it will be expected to get two times
    as many events.
    Equally so, if `events_limit` is greater than `n_events`, all events will be given to device A
    as it is the first one found idle.
"""

import time
import threading
from abc import abstractmethod, ABC
import joblib
import numpy as np
import tensorflow as tf
from vegasflow.configflow import MAX_EVENTS_LIMIT, DEFAULT_ACTIVE_DEVICES


def print_iteration(it, res, error, extra="", threshold=0.1):
    """ Checks the size of the result to select between
    scientific notation and floating point notation """
    # note: actually, the flag 'g' does this automatically
    # but I prefer to choose the precision myself...
    if res < threshold:
        print(f"Result for iteration {it}: {res:.3e} +/- {error:.3e}" + extra)
    else:
        print(f"Result for iteration {it}: {res:.4f} +/- {error:.4f}" + extra)


class MonteCarloFlow(ABC):
    """
    Parent class of all Monte Carlo integrators using tensorflow

    Parameters
    ----------
        `n_dim`: number of dimensions of the integrand
        `n_events`: number of events per iteration
        `events_limit`: maximum number of events per step
            if `events_limit` is below `n_events` each iteration of the MC
            will be broken down into several steps. Do it in order to limit memory.
            Note: for a better performance, when n_events is greater than the event limit,
            `n_events` should be exactly divisible by `events_limit`
        `list_devices`: list of device type to use (use `None` to do the tensorflow default)
    """

    def __init__(
        self,
        n_dim,
        n_events,
        events_limit=MAX_EVENTS_LIMIT,
        list_devices=DEFAULT_ACTIVE_DEVICES,
    ):
        # Save some parameters
        self.n_dim = n_dim
        self.xjac = 1.0 / n_events
        self.integrand = None
        self.event = None
        self.all_results = []
        self.n_events = n_events
        self.events_per_run = min(events_limit, n_events)
        self.lock = threading.Lock()
        if list_devices:
            # List all devices from the list that can be found by tensorflow
            devices = []
            for device_type in list_devices:
                devices += tf.config.experimental.list_logical_devices(device_type)
            # For the moment we assume they are ordered by preference
            # Make all devices available
            self.devices = {}
            for dev in devices:
                self.devices[dev.name] = True
            # Generate the pool of workers
            self.pool = joblib.Parallel(n_jobs=len(devices), prefer="threads")
        else:
            self.devices = None

    #### Abstract methods
    @abstractmethod
    def _run_iteration(self):
        """ Run one iteration (i.e., `self.n_events`) of the
        Monte Carlo integration """

    @abstractmethod
    def _run_event(self, integrand, ncalls=None):
        """ Run one single event of the Monte Carlo integration
        the output must be a tuple """
        result = self.event()
        return result, pow(result, 2)

    #### Device management methods
    def get_device(self):
        """ Looks in the list of devices until it finds a device available, once found
        makes the device unavailable and returns it """
        use_dev = None
        self.lock.acquire()
        try:
            for device, available in self.devices.items():
                # Get the first available device (which will be the fastest one hopefully)
                if available:
                    use_dev = device
                    self.devices[device] = False
                    break
        finally:
            self.lock.release()
        return use_dev

    def release_device(self, device):
        """ Makes `device` available again """
        self.lock.acquire()
        try:
            self.devices[device] = True
        finally:
            self.lock.release()

    def accumulate(self, accumulators):
        """ Accumulate all the quantities in accumulators
        The default accumulation is implemented for tensorflow tensors
        as a sum of all partial results.

        Parameters
        ----------
            `accumulators`: list of tensorflow tensors

        Returns
        -------
            `results`: `sum` for each element of the accumulators

        Function not compiled
        """
        results = []
        len_acc = len(accumulators[0])
        for i in range(len_acc):
            total = tf.reduce_sum([acc[i] for acc in accumulators], axis=0)
            results.append(total)
        return results

    def device_run(self, ncalls, **kwargs):
        """ Wrapper function to select a specific device when running the event
        If the devices were not set, tensorflow default will be used

        Parameters
        ----------
            `ncalls`: number of calls to pass to the integrand

        Returns
        -------
            `result`: raw result from the integrator
        """
        if not self.event:
            raise RuntimeError("Compile must be ran before running any iterations")
        if self.devices:
            device = self.get_device()
            with tf.device(device):
                result = self.event(ncalls=ncalls, **kwargs)
            self.release_device(device)
        else:
            result = self.event(ncalls=ncalls, **kwargs)
        return result

    def run_event(self, **kwargs):
        """
        Runs the Monte Carlo event. This corresponds to a number of calls
        decided by the `events_per_run` variable. The variable `acc` is exposed
        in order to pass the tensor output back to the integrator in case it needs
        to accumulate.

        The main driver of this function is the `event` attribute which corresponds
        to the `tensorflor` compilation of the `_run_event` method together with the
        `integrand`.

        Returns
        -------
            The accumulated result of running all steps
        """
        if not self.event:
            raise RuntimeError("Compile must be ran before running any iterations")
        # Run until there are no events left to do
        events_left = self.n_events
        events_to_do = []
        # Fill the array of event distribution
        # If using multiple devices, decide the policy for job sharing
        while events_left > 0:
            ncalls = min(events_left, self.events_per_run)
            events_to_do.append(ncalls)
            events_left -= self.events_per_run

        if self.devices:
            accumulators = self.pool(
                joblib.delayed(self.device_run)(ncalls, **kwargs)
                for ncalls in events_to_do
            )
        else:
            accumulators = [
                self.device_run(ncalls, **kwargs) for ncalls in events_to_do
            ]
        return self.accumulate(accumulators)

    def compile(self, integrand, compilable=True):
        """ Receives an integrand, prepares it for integration
        and tries to compile unless told otherwise.

        Parameters
        ----------
            `integrand`: the function to integrate
            `compilable`: (default True) if False, the integration
                is not passed through `tf.function`
        """
        if compilable:
            tf_integrand = tf.function(integrand)

            def run_event(**kwargs):
                return self._run_event(tf_integrand, **kwargs)

            self.event = tf.function(run_event)
        else:

            def run_event(**kwargs):
                return self._run_event(integrand, **kwargs)

            self.event = run_event

    def run_integration(self, n_iter, log_time=True):
        """ Runs the integrator for the chosen number of iterations
        Parameters
        ---------
            `n_iter`: number of iterations
        Returns
        -------
            `final_result`: integral value
            `sigma`: monte carlo error
        """
        for i in range(n_iter):
            # Save start time
            if log_time:
                start = time.time()

            # Run one single iteration and append results
            res, error = self._run_iteration()
            self.all_results.append((res, error))

            # Logs result and end time
            if log_time:
                end = time.time()
                time_str = f"(took {end-start:.5f} s)"
            else:
                time_str = ""
            print_iteration(i, res, error, extra=time_str)

        # Once all iterations are finished, print out
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
        print(f" > Final results: {final_result.numpy():g} +/- {sigma:g}")
        return final_result, sigma


def wrapper(integrator_class, integrand, n_dim, n_iter, total_n_events):
    """ Convenience wrapper

    Parameters
    ----------
        `integrator_class`: MonteCarloFlow inherited class
        `integrand`: tf.function
        `n_dim`: number of dimensions
        `n_iter`: number of iterations
        `n_events`: number of events per iteration

    Returns
    -------
        `final_result`: integral value
        `sigma`: monte carlo error
    """
    mc_instance = integrator_class(n_dim, total_n_events)
    mc_instance.compile(integrand)
    return mc_instance.run_integration(n_iter)
