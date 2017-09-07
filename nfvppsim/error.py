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
import logging
import os
import re
from sklearn.metrics import mean_squared_error

LOG = logging.getLogger(os.path.basename(__file__))


def get_by_name(name):
    if name == "MSE":
        return MSE
    raise NotImplementedError("'{}' not implemented".format(name))


class MSE(object):

    @classmethod
    def generate(cls, conf):
        """
        Generate list of model objects. One for each conf. to be tested.
        """
        r = list()
        r.append(cls())
        return r
    
    def __init__(self, **kwargs):
        LOG.debug("Initialized {} error metric".format(self))

    def __repr__(self):
        return "{}".format(self.name)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def short_name(self):
        return re.sub('[^A-Z]', '', self.name)

    def calculate(self, r_hat, r):
        return mean_squared_error(r_hat, r)

    def get_results(self):
        """
        Getter for global result collection.
        :return: dict for result row
        """
        r = {"error": self.short_name}
        # r.update(self.params)
        # LOG.debug("Get results from {}: {}".format(self, r))
        return r
