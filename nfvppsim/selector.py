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

    def __init__(self, pmodel_inputs, params={}):
        # apply default params
        p = {}
        p.update(params)
        # members
        self.pm_inputs = pmodel_inputs
        self.params = p
        LOG.info("Initialized {} selector".format(self))

    def __repr__(self):
        return "UniformRandomSelector({})".format(self.params)

    def next(self):
        idx = np.random.randint(0, len(self.pm_inputs))
        return self.pm_inputs[idx]

    def has_next(self):
        return True

    def feedback(self, c, r):
        """
        Inform selector about result for single configuration.
        """
        pass  # TODO store as internal state if needed
