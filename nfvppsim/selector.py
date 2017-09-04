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

LOG = logging.getLogger(os.path.basename(__file__))


class UniformRandomSelector(object):

    def __init__(self, pmodel_inputs, **kwargs):
        # apply default params
        p = {"max_samples": -1}  # -1 infinite samples
        p.update(kwargs)
        # members
        self.pm_inputs = pmodel_inputs
        self.params = p
        self.k_samples = 0
        LOG.info("Initialized selector: {}".format(self))

    def __repr__(self):
        return "UniformRandomSelector({})".format(self.params)

    def next(self):
        idx = np.random.randint(0, len(self.pm_inputs))
        self.k_samples += 1
        return self.pm_inputs[idx]

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
        r = {"k_samples": self.k_samples}
        r.update(self.params)
        # LOG.debug("Get results from {}: {}".format(self, r))
        return r
