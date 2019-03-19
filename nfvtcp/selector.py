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
import random
import logging
import os
import re
import statistics
import time
import collections
from nfvtcp.config import expand_parameters
from nfvtcp.helper import dict_to_short_str, compress_keys, flatten_conf
from nfvtcp.decisiontree import DecisionTree

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
    if name == "UniformGridSelectorRandomStepBias":
        return UniformGridSelectorRandomStepBias
    if name == "HyperGridSelector":
        return HyperGridSelector
    if name == "PanicGreedyAdaptiveSelector":
        return PanicGreedyAdaptiveSelector
    if name == "WeightedVnfSelector":
        return WeightedVnfSelector
    if name == "WeightedRandomizedVnfSelector":
        return WeightedRandomizedVnfSelector
    raise NotImplementedError("'{}' not implemented".format(name))


class Selector(object):

    @classmethod
    def generate(cls, conf):
        """
        Generate list of model objects. One for each conf. to be tested.
        """
        r = list()
        # extract max_samples parameter because it is the X axis
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
        self.selector_time_next_sum = 0
        self.selector_time_reinit_sum = 0
        LOG.debug("Initialized selector: {}".format(self))

    def reinitialize(self, repetition_id):
        t_start = time.time()
        r = self._reinitialize(repetition_id)
        self.selector_time_reinit_sum += (time.time() - t_start)
        return r
        
    def _reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        """
        pass

    def set_inputs(self, pm_inputs, pm):
        self.pm_inputs = pm_inputs
        self.pm_parameter = pm.parameter
        self.pm = pm

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
        cparams = compress_keys(self.params)
        sparams = collections.OrderedDict(
            sorted(cparams.items(), key=lambda t: t[0]))
        del sparams["max_samples"]
        return "{}_{}".format(
            self.short_name, dict_to_short_str(sparams)).strip("_- ")

    def next(self):
        t_start = time.time()
        r = self._next()
        self.selector_time_next_sum += (time.time() - t_start)
        return r

    def _next(self):
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
             "k_samples": self.k_samples,
             "selector_time_next_sum": self.selector_time_next_sum,
             "selector_time_reinit_sum": self.selector_time_reinit_sum}
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
        self.selector_time_next_sum = 0
        self.selector_time_reinit_sum = 0
        LOG.debug("Initialized selector: {}".format(self))

    def _next(self):
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
        self.selector_time_next_sum = 0
        self.selector_time_reinit_sum = 0
        LOG.debug("Initialized selector: {}".format(self))

    def _reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        We re-initialize the random grid offset here (if enabled)
        """
        # calculate step size of grind based on size and max_samples
        self.step_size = (float(len(self.pm_inputs))
                          / float(self.params.get("max_samples")))
        if self.step_size < 1:
            LOG.warning("Bad config: max_samples larger than config. space!")
        if self.params.get("random_offset"):
            # pick random offset (0, step_size]
            self.offset = random.uniform(0.0, 1.0) * self.step_size
            LOG.debug("Re-initialized random grid offset: {}"
                      .format(self.offset))
        if self.params.get("incremental_offset"):
            # applied with modulo to fit into step size
            self.offset = ((0.1 * self.step_size)
                           * repetition_id
                           % int(self.step_size))
            LOG.debug("Re-initialized incremental grid offset: {}"
                      .format(self.offset))

    def _next(self):
        if self.params.get("max_samples") < 0:
            LOG.error("{} will not work without positive max_samples setting."
                      .format(self))
            LOG.error("Exit!")
            exit(1)
        # apply step bias if enabled
        step_bias = 0.0
        if self.params.get("step_bias"):
            step_bias = self.get_step_bias(self.params.get("step_bias_e"))
        # calculate value to be used in this iteration
        idx = (int(round(
            self.offset + (self.k_samples * self.step_size) + step_bias))
               % len(self.pm_inputs))
        # LOG.warning("{}: ss:{}, offset {}, bias:{: 01.2f}, idx:{:04d}, v:{}"
        #            .format(self.short_name, self.step_size,
        #                    self.offset, step_bias, idx, self.pm_inputs[idx]))
        # increment number of already seen samples
        self.k_samples += 1
        return self.pm_inputs[idx]


class UniformGridSelectorRandomOffset(UniformGridSelector):
    """
    Same as UniformGridSelector but with random grid offset enabled.
    """
    def __init__(self, **kwargs):
        # change config of base selector
        kwargs["random_offset"] = True
        super().__init__(**kwargs)


class UniformGridSelectorRandomStepBias(UniformGridSelector):
    """
    Same as UniformGridSelector but with random bias that
    influences step size.
    """
    def __init__(self, **kwargs):
        # change config of base selector
        kwargs["step_bias"] = True
        kwargs["step_bias_e"] = 0.5
        super().__init__(**kwargs)

    def get_step_bias(self, e=0.1):
        """
        Offset added/substracted from next step.
        In this case rnd * e * step_size: Move point
        for e*100% of step_size / left, right.
        """
        return np.random.normal() * e * self.step_size


class UniformGridSelectorIncrementalOffset(UniformGridSelector):
    """
    Same as UniformGridSelector but with incremental grid offset enabled.
    """
    def __init__(self, **kwargs):
        # change config of base selector
        kwargs["incremental_offset"] = True
        super().__init__(**kwargs)


class HyperGridSelector(Selector):

    def __init__(self, **kwargs):
        # apply default params
        p = {"max_samples": -1}
        p.update(kwargs)
        # members
        self.pm_inputs = list()
        self.params = p
        self.k_samples = 0
        self.selector_time_next_sum = 0
        self.selector_time_reinit_sum = 0
        LOG.debug("Initialized selector: {}".format(self))

    def _get_n_samples_from_list(self, lst, n):
        """
        Equally spaced samples from list.
        If n==1: midpoint
        If n > 1: Start, End and n-2 midpoints
        """
        if n > len(lst):
            n = len(lst)
        if n == 1:
            return [lst[int(len(lst)/2)]]
        elif n > 1:  # first last and mid elements
            return [lst[int(idx)] for idx in np.linspace(0, len(lst)-1, n)]
        return []  # n==0

    def _calculate_grid(self):
        m_parameters = len(self.pm_parameter)
        n_vnfs = len(self.pm_inputs[0])
        # samples per feature
        spf = self.params.get("max_samples")**(1/float(n_vnfs * m_parameters))
        # compute reduced parameter set
        reduced_parameters = dict()
        for pk, pv in self.pm_parameter.items():
            reduced_parameters[pk] = self._get_n_samples_from_list(
                pv, int(spf + 1))
        # expand reduced parameter set using the pmodel object
        cs = self.pm.get_conf_space(
            modified_parameter=reduced_parameters, no_cache=True)
        # adapt grid to max_samples
        self.csr = [cs[int(idx)]
                    for idx in np.linspace(0, len(cs)-1,
                                           self.params.get("max_samples"))]
        return self.csr

    def set_inputs(self, pm_inputs, pm_parameter):
        super().set_inputs(pm_inputs, pm_parameter)
        assert(len(self.pm_inputs) > 0)
        self._calculate_grid()

    def _next(self):
        r = self.csr[self.k_samples % len(self.csr)]
        self.k_samples += 1
        return r


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
        self.selector_time_next_sum = 0
        self.selector_time_reinit_sum = 0
        self._border_points = None
        self._previous_samples = list()
        LOG.debug("Initialized selector: {}".format(self))

    def _conf_geq(self, d1, d2):
        """
        True if all components of d1 >= d2 else False
        """
        for k, v in d1.items():
            if v < d2[k]:
                return False
        return True

    @staticmethod
    def _conf_one_or_more_components_equal(d1, d2):
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

    @staticmethod
    def _calc_border_points(pm_parameter, pm_inputs):
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
        for k, v in pm_parameter.items():
            min_parameter[k] = min(v)
            max_parameter[k] = max(v)
        LOG.debug("min_paramter={}".format(min_parameter))
        LOG.debug("max_paramter={}".format(max_parameter))

        # find configurations that are border points (single VNF)
        border_points = list()
        for c in pm_inputs:
            for vnf_c in c:
                # add point if at least one component is min/max
                if (PanicGreedyAdaptiveSelector.
                    _conf_one_or_more_components_equal(
                        vnf_c, min_parameter)
                        or PanicGreedyAdaptiveSelector.
                    _conf_one_or_more_components_equal(
                        vnf_c, min_parameter)):
                    # always add complete configuration to result
                    # if it has at least one border point
                    border_points.append(c)
                    # but do not duplicate (break inner loop)
                    break
        # LOG.debug("vnf_c_border_points={}".format(border_points))
        LOG.debug("Identified {}/{} VNF border points".format(
            len(border_points),
            len(pm_inputs)
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

    def _next(self):
        result = None
        # initially select border points if not yet done
        if self._border_points is None:
            self._border_points = PanicGreedyAdaptiveSelector. \
                                  _calc_border_points(self.pm_parameter,
                                                      self.pm_inputs)
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
        # workaround for PANIC BUG:
        # option A:  randomly re-return result
        # if result is None:
        #    idx = np.random.randint(0, len(self._previous_samples))
        #    result = self._previous_samples[idx][0]
        #    LOG.warning("PANIC selector got stuck."
        #                + " Re-using configurations after {} samples".format(
        #                    len(self._previous_samples)
        #                ))

        # option B: re-use maxdistance point (don't check for re-use)
        if result is None:
            max_distance = -1
            for t1 in self._previous_samples:
                for t2 in self._previous_samples:
                    a = self._find_midpoint(t1, t2)
                    if (self._distance(t1, t2) > max_distance):
                        max_distance = self._distance(t1, t2)
                        result = a
            LOG.warning("PANIC selector got stuck."
                        + " Re-using configurations after {} samples.".format(
                            len(self._previous_samples)
                        ))
            LOG.debug("Return mid point: {}"
                      .format(result))
            
        self.k_samples += 1
        return result

    def feedback(self, c, r):
        """
        Inform selector about result for single configuration.
        """
        self._previous_samples.append((c, r))


class WeightedVnfSelector(Selector):
    """
    UPB's weighted VNF selector.
    """

    def __init__(self, **kwargs):
        # apply default params
        p = {"max_samples": -1,
             "border_point_mode": 0,
             "sampling_mode_maxmin": 0,
             "p_samples_per_vnf": -1}
        p.update(kwargs)
        # members
        self.pm_inputs = list()
        self.pm_parameter = dict()
        self.params = p
        self.k_samples = 0
        self.p_samples = 0
        self._border_points = None
        self._weights = None
        self._prioritized_vnf_idxs = None
        self._previous_samples = list()
        self.selector_time_next_sum = 0
        self.selector_time_reinit_sum = 0
        LOG.debug("Initialized selector: {}".format(self))

    def _reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        """
        self.k_samples = 0
        self.p_samples = 0
        self._border_points = None
        self._weights = None
        self._prioritized_vnf_idxs = None
        self._previous_samples = list()

    def _get_vnf_bp_minmax(self, p1, p2):
        """
        Call (p_min, p_max) to get max BPs
        Call (p_max, p_min) to get min BPs
        """
        points = list()
        vnf_config_list = [p2 for vnf in self.pm.vnfs]
        # add 0 point (all VNFs with all parameters p2)
        points.append(tuple(vnf_config_list.copy()))
        # add n further points with one VNF set to p1
        for n in range(0, len(self.pm.vnfs)):
            tmp = vnf_config_list.copy()
            tmp[n] = p1  # set one VNF to p1
            points.append(tuple(tmp))
        return points

    def _get_min_max_parameter(self):
        p = self.pm_parameter
        p_min = dict()
        p_max = dict()
        for k, v in p.items():
            p_min[k] = min(v)
            p_max[k] = max(v)
        return p_min, p_max

    def _get_median_parameter(self):
        p = self.pm_parameter
        p_median = dict()
        for k, v in p.items():
            p_median[k] = statistics.median(v)
        return p_median

    def _calc_border_points(self, mode=0, border_point_mode_panic=False):
        """
        Border points across VNFs.
        Modes:
        - 0: return n + 1 BPs (fix. to max)
        - 1: return n + 1 BPs (fix. to min)
        - 2: return 2*(n + 1) BPs (combine min/max from 1/0)
        - 3: return 2^n BPs (cross product over min/max points of VNFs)

        border_point_mode_panic:
        Special case in which the border point calculation of PANIC
        is used and the corresponding number of points (depending on mode)
        is selected randomly from the PANIC result.
        """
        LOG.debug("WVS calculating border points with mode={}".format(mode))
        if not border_point_mode_panic:
            # Default WVS border point calculation.
            # preparations
            p_min, p_max = self._get_min_max_parameter()
            # calculate result depending on mode
            if mode == 0:
                return self._get_vnf_bp_minmax(p_min, p_max)
            if mode == 1:
                return self._get_vnf_bp_minmax(p_max, p_min)
            if mode == 2:
                return (self._get_vnf_bp_minmax(p_min, p_max)
                        + self._get_vnf_bp_minmax(p_max, p_min))
            if mode == 3:
                pass  # TODO implement mode 3
        else:
            # Use PANIC's border point mode
            LOG.debug("WVS Using PANIC's border point calculation!")
            # re-use PANIC bp calculation
            panic_bp_lst = PanicGreedyAdaptiveSelector. \
                _calc_border_points(self.pm_parameter, self.pm_inputs)
            # randomly pick sample with right size from PANIC points:
            n_points = [len(self.pm.vnfs) + 1,
                        len(self.pm.vnfs) + 1,
                        2 * len(self.pm.vnfs) + 2,
                        0]  # 2 ** len(self.pm.vnfs)
            if len(panic_bp_lst) < n_points[mode]:
                # duplicate bpoints found by PANIC
                LOG.warning("Expanding PANIC's border point list from {} to {}"
                            .format(len(panic_bp_lst), n_points[mode]))
                panic_bp_lst = panic_bp_lst * (
                    1 + int(n_points[mode] / len(panic_bp_lst)))
            # LOG.warning(panic_bp_lst)
            assert(len(panic_bp_lst) >= n_points[mode])
            return random.sample(panic_bp_lst, n_points[mode])
        return list()

    def _distance(self, r1, r2):
        """
        Use Euclidean distance as default.
        """
        return float(abs(r1 - r2))

    def _get_vnf_idx_ordered_by_weight(self, weights):
        # ordered indexes of weight array
        w_idx = np.argsort(weights)[::-1]
        # normalize values to vnf indexes (mode > 1)
        r = [i % len(self.pm.vnfs) for i in w_idx]
        LOG.debug("WVS: Prioritized VNF idxs: {}".format(r))
        return r

    def _calc_weights(self, mode=0):
        """
        Returns a list of weights.
        There could be more than one weight per VNF in mode > 1.
        The idx of the list value is x*vnf_id.
        Weights are normalized
        """
        n_vnfs = len(self.pm.vnfs)
        samples = self._previous_samples
        # input validation
        if mode == 0 or mode == 1:
            assert(len(samples) == n_vnfs + 1)
        elif mode == 2:
            assert(len(samples) == 2 * n_vnfs + 2)
        else:  # TODO implement mode 3
            return []
        # calculate distances
        dists = list()
        base_index = 0
        for i in range(0, len(samples)):
            # always compare against 0 element in list
            # this element is a multiple of (n_vnfs + 1)
            if i % (n_vnfs + 1) == 0:
                base_index = i
            else:
                # actual distance calculation
                dists.append(
                    self._distance(
                        samples[base_index][1],
                        samples[i][1]))
        # normalize weigths to [0.0, 1.0]
        if sum(dists) == 0:  # should not happen
            LOG.debug("WVS: All distances have been 0!")
            return dists
        norm = [d/sum(dists) for d in dists]
        LOG.debug("WVS: Calculated weights: {}".format(norm))
        return norm

    def _get_point_with_given_vnf_parameter(self, vnf_idx, p_base, p_vnf):
        """
        vnf_idx: Index of the VNF to which another config is applied
        p_base: Config applied too all VNFs except vnf_idx
        p_vnf: Config applied to vnf_idx
        """
        tmp_lst = [p_base for vnf in self.pm.vnfs]
        tmp_lst[vnf_idx] = p_vnf
        return tuple(tmp_lst)

    def _sample_points_of_vnf_random(self, vnf_idx, mode=0):
        """
        Returns configuration points that only differ.
        in parameters of the given VNF.
        Random sampling.
        Modes:
        - 0: fix other VNF configs to max
        - 1: fix other VNF configs to min
        - 2: fix other VNF configs to median
        """
        # preparations
        p_min, p_max = self._get_min_max_parameter()
        p_median = self._get_median_parameter()
        p_base = p_max
        if mode == 1:
            p_base = p_min
        if mode == 2:
            p_base = p_median
        # get all VNF configs to select from
        vnf_conf_space_lst = self.pm.get_conf_space_vnf()
        # select single VNF config to be applied to vnf_idx
        # random but without p_max/p_min
        p_vnf = p_min
        while p_vnf == p_min or p_vnf == p_max:
            # pick random until we have something which is not p_min / p_max
            r_idx = np.random.randint(0, len(vnf_conf_space_lst))
            p_vnf = vnf_conf_space_lst[r_idx]
            if len(vnf_conf_space_lst) < 3:
                break  # ensure to stop in small conf spaces
        return self._get_point_with_given_vnf_parameter(
            vnf_idx, p_base, p_vnf)

    def _pre_calculate_prioritized_samples_for_vnfs(
            self,
            prioritized_vnf_idxs,
            p_samples_per_vnf,
            mode=0):
        return list()

    def _next(self):
        result = None
        # initially select border points if not yet done
        if self._border_points is None:
            self._border_points = self._calc_border_points(
                mode=self.params.get("border_point_mode"),
                border_point_mode_panic=self.params.get(
                    "border_point_mode_panic"))
        # first return all our border points to get some initial results
        if len(self._border_points) > 0:
            result = self._border_points.pop(0)
            # print(result)
        else:  # then return points from VNFs with high weights
            if self._weights is None:
                # compute weights once we have returned all border points
                self._weights = self._calc_weights(
                    mode=self.params.get("border_point_mode"))
                # select next points using the weigths
                self._prioritized_vnf_idxs = \
                    self._get_vnf_idx_ordered_by_weight(self._weights)
            #  return p samples, then switch to next VNF
            if self.p_samples == 0 and len(self._prioritized_vnf_idxs) > 0:
                # store current active VNF index
                self._active_vnf_idx = self._prioritized_vnf_idxs.pop(0)
                LOG.debug("WVS New VNF idx selected: {} (at k={})".format(
                    self._active_vnf_idx, self.k_samples))
            # get configuration to return
            result = self._sample_points_of_vnf_random(
                self._active_vnf_idx,
                mode=self.params.get("sampling_mode_maxmin"))
            self.p_samples += 1
            # if p_samples_per_vnf is reached: trigger VNF idx switch
            if (self.p_samples >= self.params.get("p_samples_per_vnf")
                    and self.params.get("p_samples_per_vnf") > 0):
                self.p_samples = 0
        # return and increase sample count
        self.k_samples += 1
        return result

    def feedback(self, c, r):
        """
        Inform selector about result for single configuration.
        """
        self._previous_samples.append((c, r))


class WeightedRandomizedVnfSelector(WeightedVnfSelector):
    """
    UPB's weighted randomized VNF selector.
    Select VNF to be sampled randomly according to its
    weight.
    Slightly modifies original WVS
    Parameter p_samples_per_vnf is obsolete here.
    """

    def _compute_cdf(self, weights):
        cdf = list()
        s = 0.0
        for w in weights:
            s += w
            cdf.append(s)
        return cdf

    def _random_weighted_vnf_selection(self, weights, mode=0):
        """
        Return VNF index randomly considering the weights of the VNF.
        Needs to know mode to return right idxs if len(weights) is
        bigger than len(vnf) e.g. in mode == 2.
        """            
        # compute CDF
        weight_cdf = self._compute_cdf(weights)
        # randomly pick [0, sum(weights)]
        rnd = random.uniform(0.0, sum(weights))
        LOG.debug("WRVS:\n\t Weights {}\n\t CDF: {}\n\t RND: {}"
                  .format(weights, weight_cdf, rnd))
        # find index
        idx = -1
        for i in range(0, len(weight_cdf)):
            if rnd < weight_cdf[i]:
                idx = i
                break
        # if weights == 0 pick any random vnf
        if sum(weights) == 0.0:
            LOG.warning(
                "WRVS: Weights are 0.0. Picking VNF uniformly at random.")
            idx = random.randint(0, len(weights) - 1)
        assert(idx >= 0)
        # fit index for special case (mode 2)
        if mode == 2:
            idx = idx % int(len(weights) / 2)
        # return VNF index
        return idx

    def _next(self):
        result = None
        # initially select border points if not yet done
        if self._border_points is None:
            self._border_points = self._calc_border_points(
                mode=self.params.get("border_point_mode"),
                border_point_mode_panic=self.params.get(
                    "border_point_mode_panic"))
        # first return all our border points to get some initial results
        if len(self._border_points) > 0:
            result = self._border_points.pop(0)
            # print(result)
        else:  # then return points from random weighted VNF
            if self._weights is None:
                # compute weights once we have returned all border points
                self._weights = self._calc_weights(
                    mode=self.params.get("border_point_mode"))
            #  return point form randomly (weighted) VNF
            vnf_idx = self._random_weighted_vnf_selection(
                self._weights, mode=self.params.get("border_point_mode"))
            result = self._sample_points_of_vnf_random(
                vnf_idx,
                mode=self.params.get("sampling_mode_maxmin"))
            LOG.debug("WRVS return point from VNF {} (at k={})".format(
                vnf_idx, self.k_samples))
        # return and increase sample count
        self.k_samples += 1
        return result


class DecisionTreeSelector(Selector):
    """
    Adaptive DT-based Selector.
    """

    def __init__(self, **kwargs):
        # apply default params
        # should contain number of initial samples for DT construction
        p = {"max_samples": -1, "intial_samples": 10}
        p.update(kwargs)

        # members
        self.pm_inputs = list()  # = config space

        # parameter dict set through Selector set_inputs, e.g. {'c1': [0.01, 0.2575, 0.505, 0.7525, 1.0], 'c2': ...}
        self.pm_parameter = dict()
        self.params = p
        self.k_samples = 0
        self.selector_time_next_sum = 0
        self.selector_time_reinit_sum = 0
        self._previous_samples = list()
        self._sampled_configs_as_feature = None
        self._sampled_results = None
        self._tree = None
        LOG.debug("Initialized selector: {}".format(self))

    def _initialize_tree(self):
        # TODO: get samples in right format
        # needs (configs (flat), self.pm_parameter, features, target, regression='default', homog_metric='mse',
        #                  min_homogeneity_gain=0.05, max_depth=10, min_samples_split=2)
        self.tree = DecisionTree()
        self.tree.build_tree()

    def _next(self):
        """
        Called in sim.py simulate measurement (it loops).
        Base class 'has_next' checks if max_samples were reached.

        Until number of initial samples is reached, configs are sampled at random.
        Afterwards the DT is constructed and used for selection.

        :return: selected configuration
        """
        if self.k_samples == self.params.get("initial_samples"):
            self._initialize_tree()
        if self._tree is None:
            result = self._select_random_config()
        else:
            result = DecisionTree.select_next()
        self.k_samples += 1
        return result

    def _select_random_config(self):
        # Todo: check if config sampled already?
        idx = np.random.randint(0, len(self.pm_inputs))
        self.k_samples += 1
        return self.pm_inputs[idx]

    def feedback(self, c, r):
        """
        Inform selector about result for single configuration.
        Adapt tree to newest profiling result.
        """
        self._previous_samples.append((c, r))
        if self._tree is not None:
            feature, target = None  # Todo: flat np values from self._previous_samples[-1]
            self._tree.adapt_tree(feature, target)


