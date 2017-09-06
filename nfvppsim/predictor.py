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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

LOG = logging.getLogger(os.path.basename(__file__))


def get_by_name(name):
    if name == "PolynomialRegressionPredictor":
        return PolynomialRegressionPredictor
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
        for degree in range(2, 5):  # TODO do config expansion
            r.append(cls(degree=degree))
        return r

    def __init__(self, **kwargs):
        # apply default params
        p = {"degree": 2}
        p.update(kwargs)
        # members
        self.m = None
        self.poly = None
        self.params = p
        # disable scipy warning: https://github.com/scipy/scipy/issues/5998
        warnings.filterwarnings(
            action="ignore", module="scipy", message="^internal gelsd")
        LOG.debug("Initialized predictor: {}".format(self))

    def __repr__(self):
        return "{}({})".format(self.name, self.params)

    @property
    def name(self):
        return self.__class__.__name__

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
        r = {"predictor": self.name}
        r.update(self.params)
        # LOG.debug("Get results from {}: {}".format(self, r))
        return r
