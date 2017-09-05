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
from sklearn.metrics import mean_squared_error

LOG = logging.getLogger(os.path.basename(__file__))


class MSE(object):

    def __init__(self, **kwargs):
        LOG.info("Initialized {} error metric".format(self))

    def __repr__(self):
        return "mean-squared-error (MSE)"

    def calculate(self, r_hat, r):
        return mean_squared_error(r_hat, r)
