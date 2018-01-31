"""
Copyright (c) 2017 Manuel Peuster
ALL RIGHTS RESERVED.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Manuel Peuster, Paderborn University, manuel@peuster.de
"""
import numpy as np
import logging
import os
import re
from nfvppsim.config import expand_parameters
from nfvppsim.helper import dict_to_short_str

LOG = logging.getLogger(os.path.basename(__file__))


def get_by_name(name):
    if name == "UniformRandomSelector":
        return UniformRandomSelector
    if name == "UniformGridSelector":
        return UniformGridSelector
    if name == "UniformGridSelectorRandomOffset":
        return UniformGridSelectorRandomOffset
    if name == "UniformGridSelectorIncrementalOffset":
        return UniformGridSelectorIncrementalOffset
    if name == "PanicGreedyAdaptiveSelector":
        return PanicGreedyAdaptiveSelector
    raise NotImplementedError("'{}' not implemented".format(name))


class Selector(object):

    @classmethod
    def generate(cls, conf):
        """
        Generate list of model objects. One for each conf. to be tested.
        """
        r = list()
        # extract expansion parameter but keep the others
        conf_max_samples = conf.get("max_samples")
        del conf["max_samples"]
        del conf["name"]  # no name in params (implicitly given by class)
        # generate one object for each expanded parameter
        for max_samples in expand_parameters(conf_max_samples):
            r.append(cls(max_samples=max_samples, **conf))
        return r

    def __init__(self, **kwargs):
        # apply default params
        p = {"max_samples": -1}  # -1 infinite samples
        p.update(kwargs)
        # members
        self.pm_inputs = list()
        self.pm_parameter = dict()
        self.params = p
        self.k_samples = 0
        LOG.debug("Initialized selector: {}".format(self))

    def reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        """
        pass

    def set_inputs(self, pm_inputs, pm_parameter):
        self.pm_inputs = pm_inputs
        self.pm_parameter = pm_parameter

    def __repr__(self):
        return "{}({})".format(self.name, self.params)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def short_name(self):
        return re.sub('[^A-Z]', '', self.name)

    @property
    def short_config(self):
        # sort out config parameters that change in each simulation
        sparams = self.params.copy()
        del sparams["max_samples"]
        return "{}_{}".format(
            self.short_name, dict_to_short_str(sparams))

    def next(self):
        self.k_samples += 1
        LOG.error("Not implemented.")

    def has_next(self):
        if self.params.get("max_samples") < 0:
            return True  # -1 infinite samples
        return (self.k_samples < self.params.get("max_samples", 0))

    def feedback(self, c, r):
        """
        Inform selector about result for single configuration.
        """
        pass  # TODO store as internal state if needed

    def get_results(self):
        """
        Getter for global result collection.
        :return: dict for result row
        """
        r = {"selector": self.short_name,
             "selector_conf": self.short_config,
             "k_samples": self.k_samples}
        r.update(self.params)
        # LOG.debug("Get results from {}: {}".format(self, r))
        return r


class UniformRandomSelector(Selector):

    def __init__(self, **kwargs):
        # apply default params
        p = {"max_samples": -1}  # -1 infinite samples
        p.update(kwargs)
        # members
        self.pm_inputs = list()
        self.params = p
        self.k_samples = 0
        LOG.debug("Initialized selector: {}".format(self))

    def next(self):
        idx = np.random.randint(0, len(self.pm_inputs))
        self.k_samples += 1
        return self.pm_inputs[idx]


class UniformGridSelector(Selector):

    def __init__(self, **kwargs):
        # apply default params
        p = {"max_samples": -1,  # -1 infinite samples
             "random_offset": False,
             "incremental_offset": False}
        p.update(kwargs)
        # members
        self.pm_inputs = list()
        self.params = p
        self.k_samples = 0
        self.offset = 0
        LOG.debug("Initialized selector: {}".format(self))

    def reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        We re-initialize the random grid offset here (if enabled)
        """
        if self.params.get("random_offset"):
            # calculate step size of grind based on size and max_samples
            step_size = int(
                len(self.pm_inputs) / self.params.get("max_samples"))
            # pick random offset (0, step_size]
            self.offset = np.random.randint(0, step_size)
            LOG.debug("Re-initialized random grid offset: {}"
                      .format(self.offset))
        if self.params.get("incremental_offset"):
            # later applied with modulo to fit into step size
            self.offset = (float(len(self.pm_inputs))
                           / self.params.get("max_samples"))
            LOG.debug("Re-initialized incremental grid offset: {}"
                      .format(self.offset))

    def next(self):
        if self.params.get("max_samples") < 0:
            LOG.error("{} will not work without positive max_samples setting."
                      .format(self))
            LOG.error("Exit!")
            exit(1)
        # calculate step size of grid based on size and max_samples
        step_size = int(len(self.pm_inputs) / self.params.get("max_samples"))
        if step_size < 1:
            LOG.warning("Bad config: max_samples larger than config. space!")
        # calculate value to be used in this iteration
        idx = int(self.offset / 2.0 + (self.k_samples * step_size))
        self.k_samples += 1
        return self.pm_inputs[idx % len(self.pm_inputs)]


class UniformGridSelectorRandomOffset(UniformGridSelector):
    """
    Same as UniformGridSelector but with random grid offset enabled.
    """
    def __init__(self, **kwargs):
        # change config of base selector
        kwargs["random_offset"] = True
        super().__init__(**kwargs)


class UniformGridSelectorIncrementalOffset(UniformGridSelector):
    """
    Same as UniformGridSelector but with incremental grid offset enabled.
    """
    def __init__(self, **kwargs):
        # change config of base selector
        kwargs["incremental_offset"] = True
        super().__init__(**kwargs)


class PanicGreedyAdaptiveSelector(Selector):
    """
    Greedy adaptive selection algorithm presented in PANIC paper.
    """

    def __init__(self, **kwargs):
        # apply default params
        p = {"max_samples": -1,
             "max_border_points": 4}  # -1 infinite samples
        p.update(kwargs)
        # members
        self.pm_inputs = list()
        self.pm_parameter = dict()
        self.params = p
        self.k_samples = 0
        self._border_points = None
        self._previous_samples = list()
        LOG.debug("Initialized selector: {}".format(self))

    def reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        """
        pass

    def _conf_geq(self, d1, d2):
        """
        True if all components of d1 >= d2 else False
        """
        for k, v in d1.items():
            if v < d2[k]:
                return False
        return True

    def _conf_one_or_more_components_equal(self, d1, d2):
        """
        Returns true if there is a k with d1[k] == d2[k]
        """
        for k, v in d1.items():
            if d1[k] == d2[k]:
                return True
        return False

    def _calc_border_points_global_min_max(self):
        """
        DEPRECATED!
        Calculate the border points to be used for initial selection.
        This method calculates the 2 global border points:
        [min, min, ..., min] and [max, max, ..., max]
        """
        # find min/max configs for single VNFs
        min_conf = self.pm_inputs[0][0].copy()
        max_conf = self.pm_inputs[0][0].copy()
        for c in self.pm_inputs:
            for vnf_c in c:
                if self._conf_geq(min_conf, vnf_c):
                    min_conf = vnf_c.copy()
                if self._conf_geq(vnf_c, max_conf):
                    max_conf = vnf_c.copy()
        LOG.debug("min_conf={}".format(min_conf))
        LOG.debug("max_conf={}".format(max_conf))

        # return all configurations in which at least one VNF
        # has either a min or a max configuration
        r = list()
        for c in self.pm_inputs:
            for vnf_c in c:
                if vnf_c == min_conf:
                    r.append(c)
                    break
                if vnf_c == max_conf:
                    r.append(c)
                    break
        LOG.debug("Found {} border points.".format(len(r)))
        return r

    def _calc_border_points(self):
        """
        Calculate the border points to be used for initial selection.
        Every configuration point in which at least one
        component (config parameter) is min/max is considered as a
        border point.
        [min, x, ..., x] ... [x, max, ..., x] ... [x, x, ..., max]
        """
        # get min/max for every parameter
        min_parameter = dict()
        max_parameter = dict()
        for k, v in self.pm_parameter.items():
            min_parameter[k] = min(v)
            max_parameter[k] = max(v)
        LOG.debug("min_paramter={}".format(min_parameter))
        LOG.debug("max_paramter={}".format(max_parameter))

        # find configurations that are border points (single VNF)
        border_points = list()
        for c in self.pm_inputs:
            for vnf_c in c:
                # add point if at least one component is min/max
                if (self._conf_one_or_more_components_equal(
                        vnf_c, min_parameter)
                        or self._conf_one_or_more_components_equal(
                        vnf_c, min_parameter)):
                    # always add complete configuration to result
                    # if it has at least one border point
                    border_points.append(c)
                    # but do not duplicate (break inner loop)
                    break
        # LOG.debug("vnf_c_border_points={}".format(border_points))
        LOG.debug("Identified {}/{} VNF border points".format(
            len(border_points),
            len(self.pm_inputs)
        ))
        return border_points

    def _find_midpoint(self, t1, t2):
        """
        ti = (ci, ri)
        midpoint(c1, c2) is defined as the point matching the avg. among
        each dimension of (c1, c2)
        """

        def find_closest_parameter(k, v):
            """
            We have a discrete parameter space
            Find the possible paramter value that is closest
            to v.
            k = parameter name
            """
            min_dist = float("inf")
            r = self.pm_parameter.get(k)[0]
            for p in self.pm_parameter.get(k):
                dist = abs(v - p)
                if dist < min_dist:
                    min_dist = dist
                    r = p
            return r

        def calc_avg_conf(c1, c2):
            """
            calculate the parameter-wise avg. of both configs
            """
            avg_conf_result = list()
            for i in range(0, len(c1)):  # iterate over VNFs
                ac = dict()
                for k in c1[i].keys():  # iterate over conf. parameter
                    tmp = (c1[i][k] + c2[i][k]) / 2  # calculate avarage
                    ac[k] = find_closest_parameter(k, tmp)
                avg_conf_result.append(ac)
            return tuple(avg_conf_result)

        avg_conf = calc_avg_conf(t1[0], t2[0])
        # LOG.debug("Found midpoint: {}/{} -> {}"
        #           .format(t1[0], t2[0], avg_conf))
        return avg_conf

    def _distance(self, t1, t2):
        """
        Distance between two configs based on results.
        ti = (ci, ri)
        distance according to PANIC paper: |r1 - r2|
        """
        return abs(t1[1] - t2[1])

    def _conf_not_used(self, c):
        """
        True if c is not in previous samples.
        """
        for ps in self._previous_samples:
            if ps[0] == c:
                return False
        return True

    def next(self):
        result = None
        # initially select border points if not yet done
        if self._border_points is None:
            self._border_points = self._calc_border_points()
        # PANIC algorithm (see paper)
        if self.k_samples < self.params.get("max_border_points"):
            # select (randomly) border points until "max_border_points"
            idx = np.random.randint(0, len(self._border_points))
            result = self._border_points[idx]
            LOG.debug("Return border point: {}"
                      .format(result))
        else:
            # adaptively select next point based on previous measurements
            assert(len(self._previous_samples)
                   >= self.params.get("max_border_points"))
            max_distance = -1
            for t1 in self._previous_samples:
                for t2 in self._previous_samples:
                    a = self._find_midpoint(t1, t2)
                    if (self._distance(t1, t2) > max_distance
                            # attention PANIC BUG: can get stuck local min/max
                            and self._conf_not_used(a)):
                        max_distance = self._distance(t1, t2)
                        result = a
            LOG.debug("Return mid point: {}"
                      .format(result))
        # workaround for PANIC BUG (randomly re-return result)
        # TODO find a better solution for this (e.g. neighbor points of a)
        if result is None:
            idx = np.random.randint(0, len(self._previous_samples))
            result = self._previous_samples[idx][0]
            LOG.warning("PANIC selector got stuck."
                        + " Re-using configurations after {} samples.".format(
                            len(self._previous_samples)
                        ))
        self.k_samples += 1
        return result

    def feedback(self, c, r):
        """
        Inform selector about result for single configuration.
        """
        self._previous_samples.append((c, r))
