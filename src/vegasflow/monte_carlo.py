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

import inspect
import time
import copy
import threading
import logging
from abc import abstractmethod, ABC
import joblib
import numpy as np
import tensorflow as tf
from vegasflow.configflow import (
    MAX_EVENTS_LIMIT,
    DEFAULT_ACTIVE_DEVICES,
    DTYPE,
    DTYPEINT,
    TECH_CUT,
    float_me,
    int_me,
)


logger = logging.getLogger(__name__)


def print_iteration(it, res, error, extra="", threshold=0.1):
    """Checks the size of the result to select between
    scientific notation and floating point notation"""
    # note: actually, the flag 'g' does this automatically
    # but I prefer to choose the precision myself...
    if res < threshold:
        logger.info(f"Result for iteration {it}: {res:.3e} +/- {error:.3e}" + extra)
    else:
        logger.info(f"Result for iteration {it}: {res:.4f} +/- {error:.4f}" + extra)


def _accumulate(accumulators):
    """Accumulate all the quantities in accumulators
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
        list_devices=DEFAULT_ACTIVE_DEVICES,  # pylint: disable=dangerous-default-value
        verbose=True,
        **kwargs,
    ):
        if "simplify_signature" in kwargs:
            logger.warning("simplify_signature is deprecated and will be removed")
        # Save some parameters
        self.n_dim = n_dim
        self._integrand = None
        self.event = None
        self._verbose = verbose
        self._history = []
        self._n_events = n_events
        self._events_limit = events_limit
        self._events_per_run = min(events_limit, n_events)
        self._compilation_arguments = None
        self.distribute = False
        # If any of the pass variables below is set to true
        # the integrand will be expecting them so the integrator
        # should pass them through
        self._pass_weight = False
        if list_devices:
            self.lock = threading.Lock()
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

    # Note:
    # The number of events to run in a single iteration is `n_events`
    # while the total number of events to be run per step (so, for instance, per GPU call)
    # is `events_per_run`

    @property
    def n_events(self):
        """Number of events to run in a single iteration"""
        return self._n_events

    @n_events.setter
    def n_events(self, val):
        """Number of events to run in a single iteration, triggers a recompile"""
        self._n_events = val
        # Reset `events_per_run` if needed
        self.events_per_run = self._events_limit
        self._recompile()

    @property
    def events_per_run(self):
        """Number of events to run in a single step.
        Use this variable to control how much the memory will be loaded"""
        return self._events_per_run

    @events_per_run.setter
    def events_per_run(self, val):
        """Set the number of events per single step"""
        self._events_per_run = min(val, self.n_events)
        if self.n_events % self._events_per_run != 0:
            logger.warning(
                f"The number of events per run step {self._events_per_run} doesn't perfectly"
                f"divide the number of events {self.n_events}, which can harm performance"
            )

    @property
    def history(self):
        """Returns a list with a tuple of results per iteration
        This tuple contains:

        - `result`: result of each iteration

        - `error`: error of the corresponding iteration

        - `histograms`: list of histograms for the corresponding iteration
        """
        return self._history

    @property
    def xjac(self):
        """The default jacobian is 1 / total number of events"""
        return float_me([1.0 / self.n_events])

    def generate_random_array(self, n_events):
        """External interface for the generation of random
        points as a 2D array of (n_events, n_dim).
        It calls the internal version of ``_generate_random_array``

        Parameters
        ----------
            `n_events`: number of events to generate


        Returns
        -------
            `rnds`: array of (n_events, n_dim) random points
            `idx` : index associated to each random point
            `p(x)` : p(x) associated to the random points
        """
        rnds, idx, xjac_raw = self._generate_random_array(n_events)
        # returns a p(x) corresponding to the number of events
        # the algorithm was trained with, reweight
        xjac = xjac_raw / self.xjac / n_events
        return rnds, idx, xjac

    def _generate_random_array(self, n_events):
        """Generate a 2D array of (n_events, n_dim) points
        For the weight of the given point, this function is considered
        as part of an integration with ``self.n_events`` calls.

        Parameters
        ----------
            `n_events`: number of events to generate

        Returns
        -------
            `rnds`: array of (n_events, n_dim) random points
            `idx` : index associated to each random point
            `wgt` : wgt associated to the random point
        """
        rnds = tf.random.uniform(
            (n_events, self.n_dim), minval=TECH_CUT, maxval=1.0 - TECH_CUT, dtype=DTYPE
        )
        idx = 0
        return rnds, idx, self.xjac

    #### Abstract methods
    @abstractmethod
    def _run_iteration(self):
        """Run one iteration (i.e., `self.n_events`) of the
        Monte Carlo integration"""

    @abstractmethod
    def _run_event(self, integrand, ncalls=None):
        """Run one single event (batch of calls) of the Monte Carlo integration
        the output must be a tuple"""
        result = self.event()
        return result, pow(result, 2)

    #### Integration management
    def set_seed(self, seed):
        """Sets the interation seed"""
        tf.random.set_seed(seed)

    #### Device management methods
    def get_device(self):
        """Looks in the list of devices until it finds a device available, once found
        makes the device unavailable and returns it"""
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
        """Makes `device` available again"""
        self.lock.acquire()
        try:
            self.devices[device] = True
        finally:
            self.lock.release()

    def device_run(self, ncalls, sent_pc=100.0, **kwargs):
        """Wrapper function to select a specific device when running the event
        If the devices were not set, tensorflow default will be used

        Parameters
        ----------
            `ncalls`: number of calls to pass to the integrand

        Returns
        -------
            `result`: raw result from the integrator
        """
        if self._verbose:
            print(f"Events sent to the computing device: {sent_pc:.1f} %", end="\r")
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

    def set_distribute(self, queue_object):
        """Uses dask to distribute the vegasflow run onto a cluster
        Takes as input a queue_object defining the jobs to be sent

        Parameters
        ----------
            `queue_object`: dask_jobqueue object
        """
        try:
            import dask.distributed  # pylint: disable=import-error
        except ImportError as e:
            raise ImportError("Install dask and distributed to use `set_distribute`") from e
        if self.devices is not None:
            logger.warning("`set_distribute` overrides any previous device configuration")
        self.list_devices = None
        self.lock = None
        self.pool = None
        self.devices = None
        self.cluster = queue_object
        self.distribute = True

    def make_differentiable(self):
        """Modifies the attributes of the integration so that it can be compiled inside
        Tensorflow functions (and, therefore, gradients calculated)
        Returns a reference to `run_event`, a method that upon calling it with no arguments
        will produce results and unceranties for a ncalls number of events
        """
        if self.distribute:
            raise ValueError("Differentiation is not compatible with distribution")
        if self.devices and len(self.devices) > 1:
            main_device = None
            self.devices = {}

            # Look for devices
            for device, active in self.devices.items():
                if active:
                    main_device = device
                    logger.warning(
                        "Limiting the integration to device #0: %s to make the run differentiable",
                        main_device[0],
                    )
                    break

            if main_device is not None:
                self.devices = {main_device: True}

        if self.event:
            logger.warning("Recompiling...")
            self._recompile()

        return self.run_event

    def run_event(self, tensorize_events=False, **kwargs):
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
        percentages = []
        # Fill the array of event distribution
        # If using multiple devices, decide the policy for job sharing
        pc = 0.0
        while events_left > 0:
            ncalls = min(events_left, self.events_per_run)
            pc += ncalls / self.n_events * 100
            percentages.append(pc)
            if tensorize_events:
                events_to_do.append(int_me(ncalls))
            else:
                events_to_do.append(ncalls)
            events_left -= self.events_per_run

        if self.devices and len(self.devices) > 1:
            running_pool = []
            for ncalls, pc in zip(events_to_do, percentages):
                delay_job = joblib.delayed(self.device_run)(ncalls, sent_pc=pc, **kwargs)
                running_pool.append(delay_job)
            accumulators = self.pool(running_pool)
        elif self.distribute:
            from dask.distributed import Client  # pylint: disable=import-error

            cluster = self.cluster
            self.cluster = None  # the cluster might not be pickable # TODO
            # Generate the client to control the distribution using the cluster variable
            client = Client(cluster)
            accumulators_future = client.map(self.device_run, events_to_do, percentages)
            result_future = client.submit(_accumulate, accumulators_future)
            result = result_future.result()
            # Liberate the client
            client.close()
            self.cluster = cluster
            return result
        else:
            accumulators = []
            for ncalls, pc in zip(events_to_do, percentages):
                res = self.device_run(ncalls, sent_pc=pc, **kwargs)
                accumulators.append(res)
        return _accumulate(accumulators)

    def trace(self, n_events=50):
        """Run a mock integration with just 50 events to trigger compilation"""
        true_events = self.n_events
        true_verbosity = self._verbose
        # Change the integration options
        self.n_events = n_events
        self._verbose = False
        self.run_integration(1, log_time=False)
        # Recover the actual values
        self.n_events = true_events
        self._verbose = true_verbosity

    def compile(self, integrand, compilable=True, signature=None, trace=False):
        """Receives an integrand, prepares it for integration
        and tries to compile unless told otherwise.

        The input integrand must receive, as an input, an array of random numbers.
        There are also one optional arguments that will be passed to the function:

        - `weight`: weight of each event,

        so that the most general signature for the integrand is:

        - `integrand(array_random, weight = None)`,

        the minimal working signature fo the integrand will be

        - `integrand(array_random)`.

        In other words, the integrand must take at least one argument
        and the integrator will always pass the array of random numbers.
        For legacy compatibility, the keyword argument `n_dim` will be accepted
        but it will fixed to be equal to `self.n_dim`

        This function will try to understand the signature of the function and compile
        it accordingly, this means:
            <1> array_random: DTYPE of shape [None, n_dim]
            <2> weight: DTYPE of shape [None]

        if the function posses any kewyword arguments not included in this list,
        it will be compiled with a generic `tf.function` call.
        This will work most of the time but could trigger retracing on shape-shifting
        calculations.

        If the signature is not to be used, it can be set to false


        Parameters
        ----------
            `integrand`: the function to integrate
            `compilable`: (default True) if False, the integration
                is not passed through `tf.function`
            `signature`: (default: True)
                whether to autodiscover the signature of the integrand

        """
        kwargs = {"compilable": compilable, "signature": signature, "trace": trace}
        self._compilation_arguments = (integrand, kwargs)
        tf_integrand = None
        self._integrand = integrand

        # Check whether the function is already compiled
        if hasattr(integrand, "function_spec"):
            tf_integrand = integrand
            integrand = tf_integrand.python_function
            signature = tf_integrand.input_signature

        args = inspect.getfullargspec(integrand).args[1:]
        # Loop over the argument to see whether it can be compiled
        if tf_integrand is None and compilable:
            autodiscover_signature = [tf.TensorSpec(shape=[None, self.n_dim], dtype=DTYPE)]

            for arg in args:
                if arg == "n_dim":
                    autodiscover_signature.append(tf.TensorSpec(shape=[], dtype=DTYPEINT))
                elif arg == "weight":
                    autodiscover_signature.append(tf.TensorSpec(shape=[None], dtype=DTYPE))
                elif signature is None and compilable:
                    logger.warning(
                        "The signature could not be autodiscovered due to argument %s", arg
                    )
                    logger.warning(
                        "Provide a signature argument to compile with a non-std signature"
                    )
                    signature = False

            compile_options = {
                "experimental_autograph_options": tf.autograph.experimental.Feature.ALL,
            }
            if signature is None:
                signature = autodiscover_signature
            if signature:
                compile_options["input_signature"] = signature
                logger.debug("Compiling with signature: %s", str(signature))
            tf_integrand = tf.function(integrand, **compile_options)
        else:
            tf_integrand = integrand

        # The algorithms will always call the function with
        # (xrand, weight=)
        # therefore create a wrapper withi whatever integrand to become this
        # n_dim was an option there during development and needs to be left because of legacy code

        def new_integrand(xarr, weight=None, **kwargs):
            """Wrapper to account for n_dim/weights"""
            if "weight" in args and "n_dim" in args:
                return tf_integrand(xarr, n_dim=self.n_dim, weight=weight, **kwargs)
            if "weight" in args:
                return tf_integrand(xarr, weight=weight, **kwargs)
            if "n_dim" in args:
                return tf_integrand(xarr, n_dim=self.n_dim, **kwargs)
            return tf_integrand(xarr, **kwargs)

        def batch_events(**kwargs):
            """Pass any arguments to the underlying integrand"""
            return self._run_event(new_integrand, **kwargs)

        if compilable:
            self.event = tf.function(batch_events)
        else:
            self.event = batch_events

        if trace:
            self.trace()

    def _recompile(self):
        """Forces recompilation with the same arguments that have
        previously been used for compilation"""
        if self._compilation_arguments is None:
            raise RuntimeError("recompile was called without ever having called compile")
        self.compile(self._compilation_arguments[0], **self._compilation_arguments[1])

    def run_integration(self, n_iter, log_time=True, histograms=None):
        """Runs the integrator for the chosen number of iterations.

        `histograms` must be a tuple of tf.Variables.
        At the end of all iterations the histograms per iteration will
        be output.
        The variable `histograms` instead will contain the weighted
        accumulation of all histograms

        Parameters
        ---------
            `n_iter`: int
                number of iterations
            `log_time`: bool
                flag to decide whether to log the time each iteration takes
            `histograms`: tuple of tf.Variable
                tuple containing the histogram variables so they can be emptied
                each each iteration

        Returns
        -------
            `final_result`: float
                integral value
            `sigma`: float
                monte carlo error

        Note: it is possible not to pass any histogram variable and still fill
        some histogram variable at integration time, but then it is the responsability
        of the integrand to empty the histograms each iteration and accumulate them.

        """
        all_results = []
        histo_results = []

        for i in range(n_iter):
            # Save start time
            if log_time:
                start = time.time()

            # Run one single iteration and append results
            res, error = self._run_iteration()
            all_results.append((res, error))

            # If there is a histogram variable, store it and empty it
            hist_copy = copy.deepcopy(histograms)
            if histograms:
                histo_results.append(hist_copy)
                for histo_tensor in histograms:
                    histo_tensor.assign(tf.zeros_like(histo_tensor, dtype=DTYPE))
            self._history.append((res, error, hist_copy))

            # Logs result and end time
            if log_time:
                end = time.time()
                time_str = f"(took {end-start:.5f} s)"
            else:
                time_str = ""
            if self._verbose:
                print_iteration(i, res, error, extra=time_str)

        # Once all iterations are finished, print out
        aux_res = 0.0
        weight_sum = 0.0
        for i, result in enumerate(all_results):
            res = result[0]
            sigma = result[1]
            wgt_tmp = 1.0 / pow(sigma, 2)
            aux_res += res * wgt_tmp
            weight_sum += wgt_tmp
            # Accumulate the histograms
            if histograms:
                current = histo_results[i]
                for aux_h, curr_h in zip(histograms, current):
                    aux_h.assign(aux_h + curr_h * wgt_tmp)

        if histograms:
            for histogram in histograms:
                histogram.assign(histogram / weight_sum)

        final_result = aux_res / weight_sum
        sigma = np.sqrt(1.0 / weight_sum)
        if self._verbose:
            logger.info(f" > Final results: {final_result.numpy():g} +/- {sigma:g}")
        return final_result.numpy(), sigma


def wrapper(integrator_class, integrand, n_dim, n_iter, total_n_events, compilable=True):
    """Convenience wrapper for integration

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
    mc_instance.compile(integrand, compilable=compilable)
    return mc_instance.run_integration(n_iter)


def sampler(
    integrator_class,
    integrand,
    n_dim,
    total_n_events,
    training_steps=5,
    compilable=True,
    return_class=False,
):
    """Convenience wrapper for sampling random numbers

    Parameters
    ----------
        `integrator_class`: MonteCarloFlow inherited class
        `integrand`: tf.function
        `n_dim`: number of dimensions
        `n_events`: number of events per iteration
        `training_steps`: number of training_iterations
        `return_class`: whether to return the full instance of the class or only the random method

    Returns
    -------
        `sampler`: a reference to the generate_random_array method of the integrator class
    """
    mc_instance = integrator_class(n_dim, total_n_events, verbose=False)
    mc_instance.compile(integrand, compilable=compilable)
    _ = mc_instance.run_integration(training_steps, log_time=False)
    if return_class:
        return mc_instance
    return mc_instance.generate_random_array
