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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score, explained_variance_score

LOG = logging.getLogger(os.path.basename(__file__))


def get_by_name(name):
    if name == "MSE":
        return MSE
    if name == "MAE":
        return MAE
    if name == "R2":
        return R2
    if name == "EVS":
        return EVS
    if name == "MEDAE":
        return MEDAE
    raise NotImplementedError("'{}' not implemented".format(name))


class BaseError(object):
    """
    Mean squared error regression loss
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    """

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
        LOG.error("Error calculation not implemented.")
        return 0

    def get_results(self):
        """
        Getter for global result collection.
        :return: dict for result row
        """
        r = {"error": self.short_name}
        # r.update(self.params)
        # LOG.debug("Get results from {}: {}".format(self, r))
        return r


class MSE(BaseError):
    """
    Mean squared error regression loss
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    """
    def calculate(self, r, r_hat):
        return mean_squared_error(r, r_hat)


class MAE(BaseError):
    """
    Mean absolute error regression loss
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    """
    def calculate(self, r, r_hat):
        return mean_absolute_error(r, r_hat)


class MEDAE(BaseError):
    """
    MEDIAN absolute error regression loss
  http://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error
    """
    def calculate(self, r, r_hat):
        return median_absolute_error(r, r_hat)


class R2(BaseError):
    """
    R^2 (coefficient of determination) regression score function.
    Best possible score is 1.0 and it can be negative
    (because the model can be arbitrarily worse). A constant model
    that always predicts the expected value of y, disregarding
    the input features, would get a R^2 score of 0.0.
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    """
    def calculate(self, r, r_hat):
        return r2_score(r, r_hat)


class EVS(BaseError):
    """
    Explained variance regression score function
    Best possible score is 1.0, lower values are worse.
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html
    """
    def calculate(self, r, r_hat):
        return explained_variance_score(r, r_hat)
