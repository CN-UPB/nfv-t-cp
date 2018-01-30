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
import warnings
import re
from nfvppsim.config import expand_parameters
from nfvppsim.helper import dict_to_short_str
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

LOG = logging.getLogger(os.path.basename(__file__))


def get_by_name(name):
    if name == "PolynomialRegressionPredictor":
        return PolynomialRegressionPredictor
    if name == "SupportVectorRegressionPredictor":
        return SupportVectorRegressionPredictor
    raise NotImplementedError("'{}' not implemented".format(name))


class PolynomialRegressionPredictor(object):
    """
    Polynomial interpolation with given degree based on sklearn.
    http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
    """

    @classmethod
    def generate(cls, conf):
        """
        Generate list of model objects. One for each conf. to be tested.
        """
        r = list()
        for degree in expand_parameters(conf.get("degree")):
            r.append(cls(degree=degree))
        return r

    def __init__(self, **kwargs):
        # apply default params
        p = {"degree": 2,
             "epsilon": 0}
        p.update(kwargs)
        # members
        self.m = None
        self.poly = None
        self.params = p
        # disable scipy warning: https://github.com/scipy/scipy/issues/5998
        warnings.filterwarnings(
            action="ignore", module="scipy", message="^internal gelsd")
        LOG.debug("Initialized predictor: {}".format(self))

    def reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        """
        pass

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
        return "{}_{}".format(
            self.short_name, dict_to_short_str(self.params))

    def train(self, c_tilde, r_tilde):
        self.poly = PolynomialFeatures(degree=self.params.get("degree"))
        c_tilde = self.poly.fit_transform(c_tilde)
        self.m = LinearRegression()
        self.m.fit(c_tilde, r_tilde)
        LOG.debug("Trained {}: coef={} intercept={}".format(
            self, self.m.coef_, self.m.intercept_))

    def predict(self, c_hat):
        if self.m is None or self.poly is None:
            LOG.error("Model not trained!")
        c_hat = self.poly.fit_transform(c_hat)
        return self.m.predict(c_hat)

    def get_results(self):
        """
        Getter for global result collection.
        :return: dict for result row
        """
        r = {"predictor": self.short_name,
             "predictor_conf": self.short_config}
        r.update(self.params)
        # LOG.debug("Get results from {}: {}".format(self, r))
        return r


class SupportVectorRegressionPredictor(object):
    """
    Support Vector Regression
    http://scikit-learn.org/stable/modules/svm.html#svm-regression
    """
    # TODO see notes in docu: we may want to normalize the input data (?!)

    @classmethod
    def generate(cls, conf):
        """
        Generate list of model objects. One for each conf. to be tested.
        """
        r = list()
        for e in expand_parameters(conf.get("epsilon")):
            r.append(cls(epsilon=e))
        return r

    def __init__(self, **kwargs):
        # apply default params
        p = {"degree": 0,
             "epsilon": 0.1}
        p.update(kwargs)
        # members
        self.m = None
        self.params = p
        # disable scipy warning: https://github.com/scipy/scipy/issues/5998
        warnings.filterwarnings(
            action="ignore", module="scipy", message="^internal gelsd")
        LOG.debug("Initialized predictor: {}".format(self))

    def reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        """
        pass

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
        return "{}_{}".format(
            self.short_name, dict_to_short_str(self.params))

    def train(self, c_tilde, r_tilde):
        self.m = SVR(C=1.0, epsilon=self.params.get("epsilon"))
        self.m.fit(c_tilde, r_tilde)
        LOG.debug("Trained: {}".format(self.m))

    def predict(self, c_hat):
        if self.m is None:
            LOG.error("Model not trained!")
        return self.m.predict(c_hat)

    def get_results(self):
        """
        Getter for global result collection.
        :return: dict for result row
        """
        r = {"predictor": self.short_name,
             "predictor_conf": self.short_config}
        r.update(self.params)
        # LOG.debug("Get results from {}: {}".format(self, r))
        return r
