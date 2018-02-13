"""
Copyright (c) 2017 Manuel Peuster
ALL RIGHTS RESERVED.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
bess or implied.
See the License for the specific language governing permissions and
limitations under the License.

Manuel Peuster, Paderborn University, manuel@peuster.de
"""
import numpy as np
import simpy
import logging
import os
from nfvppsim.helper import flatten_conf

LOG = logging.getLogger(os.path.basename(__file__))


# global cache vars
CACHE_REF_RESULTS = dict()


class Profiler(object):
    """
    This component simulates a profiling system
    that performs performance measurements for different
    service configurations.
    """

    def __init__(self,
                 pmodel,
                 selector,
                 predictor,
                 error_lst,
                 repetition_id):
        """
        Initialize profiler for one experiment configuration.
        """
        self.pm = pmodel
        self.pm_conf_space = pmodel.get_conf_space()
        self.pm_conf_space_flat = flatten_conf(self.pm_conf_space)
        self.s = selector
        self.s.set_inputs(self.pm_conf_space, pmodel)
        self.p = predictor
        self.e_lst = error_lst
        self.repetition_id = repetition_id
        self._tmp_train_c = list()
        self._tmp_train_r = list()
        self._sim_t_total = 0
        self._sim_t_mean = list()
        # re-initialize models for repetition
        self.pm.reinitialize(self.repetition_id)
        self.s.reinitialize(self.repetition_id)
        self.p.reinitialize(self.repetition_id)
        # initialize simulation environment
        self.env = simpy.Environment()
        self.profile_proc = self.env.process(self.simulate_measurement())

    def simulate_measurement(self):
        """
        Method to simulate the performance measurement
        for a list of configurations. The actual configurations to
        be tested are dynamically fetched from the specified 'selector'.
        Method implements a discrete event simulator and can work
        with arbitrary timing modes.
        """
        while self.s.has_next():
            c = self.s.next()
            _start_t = self.env.now
            LOG.debug("t={} measuring config: {}".format(_start_t, c))
            r = self.pm.evaluate(c)
            self._tmp_train_c.append(c)  # store configs ...
            self._tmp_train_r.append(r)  # ... and results of profiling run
            self.s.feedback(c, r)  # inform selector about result
            # TODO: Allow to use timing model: boot, config, shutdown etc.
            yield self.env.timeout(60)  # Fix: assumes 60s per measurement
            # sim time bookkeeping (needed for more complex timing models)
            _end_t = self.env.now
            self._sim_t_total = _end_t
            self._sim_t_mean.append(_end_t - _start_t)
            LOG.debug("t={} result: {}".format(self._sim_t_total, r))
        LOG.debug("No configurations left. Stopping simulation.")

    def calculate_reference_result(self):
        """
        Calculate the reference result for the complete configuration space.
        This might be a bottleneck when the space is huge.
        We try to cache the results to speed up the simulation.
        """
        global CACHE_REF_RESULTS
        if id(self.pm_conf_space) in CACHE_REF_RESULTS:
            LOG.debug("Using reference result from cache from cache.")
            return CACHE_REF_RESULTS[id(self.pm_conf_space)]
        LOG.info("Calculating reference results for {} configurations."
                 .format(len(self.pm_conf_space)))
        r = list()
        last_p = None
        for c in self.pm_conf_space:
            p = int(len(r) / len(self.pm_conf_space) * 100)
            if last_p != p:
                LOG.info("Calculating ... {}.0 %".format(p))
                last_p = p
            r.append(self.pm.evaluate(c))
        CACHE_REF_RESULTS[id(self.pm_conf_space)] = r
        return r

    def run(self, until=None):
        """
        Run profiling measurement simulation using the configurations
        (pmodel, selector, predictor, error, timing) specified during
        object initialization.
        :param until: max. time for measurements (simulated seconds)
        :return: result dict (used as row of a Pandas DF)
        """
        # reset tmp. results
        self._tmp_train_c = list()
        self._tmp_train_r = list()
        # simulate profiling process
        self.env.run(until=until)  # time limit in seconds
        # predict full result using training sets
        self.p.train(flatten_conf(self._tmp_train_c), self._tmp_train_r)
        r_hat = self.p.predict(self.pm_conf_space_flat)
        # calculate reference result (evaluate pmodel for all configs)
        r = self.calculate_reference_result()
        # calculate error between prediction (r_hat) and reference results (r)
        # self.e_lst can contain multiple error metrics to be calculated
        r_lst = list()
        for e in self.e_lst:
            err_val = e.calculate(r, r_hat)
            #  build/return result dict (used as row of a Pandas DF)
            result = dict()
            result.update(self.pm.get_results())
            result.update(self.s.get_results())
            result.update(self.p.get_results())
            result.update(e.get_results())
            result.update({"sim_t_total": self._sim_t_total,
                           "sim_t_mean": np.mean(self._sim_t_mean),
                           "sim_t_max": until,
                           "error_value": err_val})
            LOG.debug("Done. Resulting error({0})={1:.4g}, sim_t_total={2}s"
                      .format(e.short_name, err_val, self._sim_t_total))
            r_lst.append(result)
        return r_lst

        
def run(sim_t_max, pmodel, selector, predictor, error, repetition_id):
    p = Profiler(pmodel, selector, predictor, error, repetition_id)
    return p.run(until=sim_t_max)
