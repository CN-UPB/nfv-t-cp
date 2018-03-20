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
import collections
from nfvppsim.config import expand_parameters
from nfvppsim.helper import dict_to_short_str
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import preprocessing

LOG = logging.getLogger(os.path.basename(__file__))


def get_by_name(name):
    if name == "PolynomialRegressionPredictor":
        return PolynomialRegressionPredictor
    if name == "SupportVectorRegressionPredictor":
        return SupportVectorRegressionPredictor
    if name == "SVRPredictorRbfKernel":
        return SVRPredictorRbfKernel
    if name == "SVRPredictorLinearKernel":
        return SVRPredictorLinearKernel
    if name == "SVRPredictorPolyKernel":
        return SVRPredictorPolyKernel
    raise NotImplementedError("'{}' not implemented".format(name))


class Predictor(object):

    @classmethod
    def generate(cls, conf):
        """
        Generate list of model objects. One for each conf. to be tested.
        """
        r = list()
        degrees = conf.get("degree", 2)
        epsilons = conf.get("epsilon", 0.1)
        if "degree" in conf:
            del conf["degree"]
        if "epsilon" in conf:
            del conf["epsilon"]
        for degree in expand_parameters(degrees):
            for e in expand_parameters(epsilons):
                r.append(cls(degree=degree,
                             epsilon=e,
                             **conf))
        return r

    def __init__(self, **kwargs):
        # disable scipy warning: https://github.com/scipy/scipy/issues/5998
        warnings.filterwarnings(
            action="ignore", module="scipy", message="^internal gelsd")
        # apply default params
        p = {"degree": 2,
             "epsilon": 0.1,
             "scale_x": True}  # normalize inputs to [0, 1]
        p.update(kwargs)
        self.params = p

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
        sparams = collections.OrderedDict(self.params.copy())
        return "{}_{}".format(
            self.short_name, dict_to_short_str(sparams))

    def reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        """
        pass

    def train(self, c_tilde, r_tilde):
        if self.params.get("scale_x"):
            min_max_scaler = preprocessing.MinMaxScaler()
            c_tilde_scaled = min_max_scaler.fit_transform(c_tilde.copy())
        else:
            c_tilde_scaled = c_tilde.copy()
        return self._train(c_tilde_scaled, r_tilde)

    def predict(self, c_hat):
        if self.params.get("scale_x"):
            min_max_scaler = preprocessing.MinMaxScaler()
            c_hat_scaled = min_max_scaler.fit_transform(c_hat.copy())
        else:
            c_hat_scaled = c_hat.copy()
        return self._predict(c_hat_scaled)

    def _train(self, c_tilde, r_tilde):
        LOG.error("Not implemented.")

    def _predict(self, c_hat):
        LOG.error("Not implemented.")

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
   

class PolynomialRegressionPredictor(Predictor):
    """
    Polynomial interpolation with given degree based on sklearn.
    http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # members
        self.m = None
        self.poly = None
        LOG.debug("Initialized predictor: {}".format(self))

    def _train(self, c_tilde, r_tilde):
        self.poly = PolynomialFeatures(degree=self.params.get("degree"))
        c_tilde = self.poly.fit_transform(c_tilde)
        self.m = LinearRegression()
        self.m.fit(c_tilde, r_tilde)
        LOG.debug("Trained {}: coef={} intercept={}".format(
            self, self.m.coef_, self.m.intercept_))

    def _predict(self, c_hat):
        if self.m is None or self.poly is None:
            LOG.error("Model not trained!")
        c_hat = self.poly.fit_transform(c_hat)
        return self.m.predict(c_hat)


class SupportVectorRegressionPredictor(Predictor):
    """
    Support Vector Regression
    http://scikit-learn.org/stable/modules/svm.html#svm-regression
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # members
        self.m = None
        LOG.debug("Initialized predictor: {}".format(self))
        self.m = SVR(C=1.0, epsilon=self.params.get("epsilon"))
        self.trained = False

    def _train(self, c_tilde, r_tilde):
        self.m.fit(c_tilde, r_tilde)
        self.trained = True
        LOG.debug("Trained: {}".format(self.m))

    def _predict(self, c_hat):
        if self.m is None or not self.trained:
            LOG.error("Model not trained!")
        return self.m.predict(c_hat)


class SVRPredictorRbfKernel(SupportVectorRegressionPredictor):
    """
    SVR with RBF kernel (default == SupportVectorRegressionPredictor)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m = SVR(
            kernel="rbf",
            C=1.0,
            epsilon=self.params.get("epsilon"))


class SVRPredictorLinearKernel(SupportVectorRegressionPredictor):
    """
    SVR with linear kernel
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m = SVR(
            kernel="linear",
            C=1.0,
            epsilon=self.params.get("epsilon"))


class SVRPredictorPolyKernel(SupportVectorRegressionPredictor):
    """
    SVR with polynomial kernel
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m = SVR(
            kernel="poly",
            C=1.0,
            epsilon=self.params.get("epsilon"),
            degree=self.params.get("degree"))
