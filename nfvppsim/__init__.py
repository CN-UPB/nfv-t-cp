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

from nfvppsim.core import sim
from nfvppsim.pmodel import SimpleNetworkServiceThroughputModel as SNSTM
from nfvppsim.selector import UniformRandomSelector
from nfvppsim.predictor import PolynomialRegressionPredictor
from nfvppsim.error import MSE

LOG = logging.getLogger(os.path.basename(__file__))


def logging_setup():
    os.environ["COLOREDLOGS_LOG_FORMAT"] \
        = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"


def main():
    # TODO CLI interface
    logging_setup()
    
    coloredlogs.install(level="DEBUG")
    # TODO replace this with configuration runner module
    # initialize and configure involved modules

    # network service performance model
    pmodel = SNSTM("test_ns_model",
                   [lambda x: x**2 + (x * 2) + 0.1,
                    lambda x: x**4 + (.5 * x)])
    # all potential possible service configurations
    pmodel_inputs = [[c1, c2] for c2 in np.linspace(0.01, 1.0, num=20)
                     for c1 in np.linspace(0.01, 1.0, num=20)]
    selector = UniformRandomSelector(pmodel_inputs, params={"max_samples": 3})
    predictor = PolynomialRegressionPredictor(params={"degree": 3})
    error = MSE()
    # TODO initialize profiler object with model etc.
    # TODO use configuration list as run input? or time limit only?
    row = sim.run(pmodel, pmodel_inputs, selector, predictor, error)
    print(row)
