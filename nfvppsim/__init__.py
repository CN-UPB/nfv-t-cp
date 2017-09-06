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
import coloredlogs
import os

from nfvppsim import sim
from nfvppsim.config import read_config
import nfvppsim.pmodel
import nfvppsim.selector
import nfvppsim.predictor
import nfvppsim.error

LOG = logging.getLogger(os.path.basename(__file__))


def logging_setup():
    os.environ["COLOREDLOGS_LOG_FORMAT"] \
        = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"


class Experiment(object):
    # TODO Refactor: move to own module experiment.py?

    def __init__(self, conf):
        """
        Load modules and configure experiment.
        """
        # TODO logging
        self.conf = conf
        # get classes of modules to be use based on config
        self._pmodel_cls = nfvppsim.pmodel.get_by_name(
            conf.get("pmodel").get("name"))
        self._selector_cls = nfvppsim.selector.get_by_name(
            conf.get("selector").get("name"))
        self._predictor_cls = nfvppsim.predictor.get_by_name(
            conf.get("predictor").get("name"))
        self._error_cls = nfvppsim.error.get_by_name(
            conf.get("error").get("name"))
        
    def prepare(self):
        """
        Prepare experiment: Generate configurations to be simulated.
        """
        # TODO logging
        self._lst_pmodel = self._pmodel_cls.generate(
            self.conf.get("pmodel"))
        self._lst_selector = self._selector_cls.generate(
            self.conf.get("selector"))
        self._lst_predictor = self._predictor_cls.generate(
            self.conf.get("predictor"))
        self._lst_error = self._error_cls.generate(
            self.conf.get("error"))
        LOG.info("Prepared {} configurations to be simulated.".format(
            self._get_number_of_configurations()))

    def _get_number_of_configurations(self):
        return (len(self._lst_pmodel) *
                len(self._lst_selector) *
                len(self._lst_predictor) *
                len(self._lst_error))


def main():
    # TODO CLI interface
    logging_setup()
    
    coloredlogs.install(level="DEBUG")
    # TODO replace this with configuration runner module
    # initialize and configure involved modules
    conf = read_config("example_experiment.yaml")
    e = Experiment(conf)
    e.prepare()
    #print(conf)
    # TODO dynamically import classes for models specified in config
    # TODO call model_cls.generate(conf) ...
    # ... to expand configs to a list of model objects
    # TODO loop over all returned lists (nested) an collect results in DF
    # TODO pickle DF to disk if path in config
    return
    # network service performance model
    #pmodel = SNSTM("test_ns_model",
    #               vnfs=[lambda x: x**2 + (x * 2) + 0.1,
    #                     lambda x: x**4 + (.5 * x)],
    #               alphas=None)
    # all potential possible service configurations
    #pmodel_inputs = [[c1, c2] for c2 in np.linspace(0.01, 1.0, num=20)
    #                 for c1 in np.linspace(0.01, 1.0, num=20)]
    #selector = UniformRandomSelector(pmodel_inputs, max_samples=3)
    #predictor = PolynomialRegressionPredictor(degree=3)
    #error = MSE()
    # TODO initialize profiler object with model etc.
    # TODO use configuration list as run input? or time limit only?
    #row = sim.run(pmodel, pmodel_inputs, selector, predictor, error)
    #print(row)
