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
import coloredlogs
import os
import copy
import pandas as pd

from nfvppsim import sim
from nfvppsim.config import read_config, expand_parameters
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
        self.conf = conf
        # Pandas DF to hold result after run()
        self.result_df = None
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
        self._lst_sim_t_max = expand_parameters(
            self.conf.get("sim_t_max"))
        self._lst_pmodel = self._pmodel_cls.generate(
            self.conf.get("pmodel"))
        self._lst_selector = self._selector_cls.generate(
            self.conf.get("selector"))
        self._lst_predictor = self._predictor_cls.generate(
            self.conf.get("predictor"))
        self._lst_error = self._error_cls.generate(
            self.conf.get("error"))
        LOG.info("Prepared {}x{} configurations to be simulated.".format(
            self._get_number_of_configurations(),
            self.conf.get("repetitions", 1)))

    def _get_number_of_configurations(self):
        return (len(self._lst_sim_t_max) *
                len(self._lst_pmodel) *
                len(self._lst_selector) *
                len(self._lst_predictor) *
                len(self._lst_error))

    def run(self):
        """
        Executes an experiment by iterating over all prepared
        configurations that should be tested.
        Uses deepcopy do ensure fresh internal states of all
        algorithm objects passed to the simulator module.
        """
        # list to hold results before moved to Pandas DF
        tmp_results = list()
        conf_id = 0
        # iterate over all sim. configurations and run simulation
        for sim_t_max in self._lst_sim_t_max:
            for pm_obj in self._lst_pmodel:
                for s_obj in self._lst_selector:
                    for p_obj in self._lst_predictor:
                        for e_obj in self._lst_error:
                            conf_id += 1
                            for r_id in range(0, self.conf.get(
                                    "repetitions", 1)):
                                # Attention: We need to copy the models objects
                                # to have fresh states for each run!
                                # TODO Can we optimize?
                                row = sim.run(sim_t_max,
                                              copy.deepcopy(pm_obj),
                                              copy.deepcopy(s_obj),
                                              copy.deepcopy(p_obj),
                                              copy.deepcopy(e_obj))
                                # extend result
                                row.update({"conf_id": conf_id,
                                            "repetition_id": r_id})
                                tmp_results.append(row)
        self.result_df = pd.DataFrame(tmp_results)

    def store_result(self, path):
        """
        Stores result DF in pickle file if path
        is not None.
        """
        assert(self.result_df is not None)
        if path is None:
            print(self.result_df)
            LOG.warning("'result_path' not specified. No results stored.")
            return
        with open(path, "wb") as f:
            self.result_df.to_pickle(f)
            

def main():
    # TODO CLI interface
    logging_setup()
    print("")
    print("*" * 64)
    print("nfv-pp-sim by Manuel Peuster <manuel@peuster.de>")
    print("*" * 64)
    coloredlogs.install(level="INFO")
    # TODO replace this with configuration runner module
    # initialize and configure involved modules
    conf = read_config("example_experiment.yaml")
    e = Experiment(conf)
    e.prepare()
    e.run()
    e.store_result(conf.get("result_path"))
    return

    # TODO dynamically import classes for models specified in config
    # TODO call model_cls.generate(conf) ...
    # ... to expand configs to a list of model objects
    # TODO loop over all returned lists (nested) an collect results in DF
    # TODO pickle DF to disk if path in config

    # single instance run (export for test?)
    # network service performance model
    # pmodel = e._pmodel_cls(vnfs=[lambda x: x**2 + (x * 2) + 0.1,
    #                             lambda x: x**4 + (.5 * x)],
    #                       alphas=None)
    # all potential possible service configurations
    # pmodel_inputs = [[c1, c2] for c2 in np.linspace(0.01, 1.0, num=20)
    #                 for c1 in np.linspace(0.01, 1.0, num=20)]
    # selector = e._selector_cls(max_samples=3)
    # selector.set_inputs(pmodel_inputs)
    # predictor = e._predictor_cls(degree=3)
    # error = e._error_cls()
    # TODO initialize profiler object with model etc.
    # TODO use configuration list as run input? or time limit only?
    # row = sim.run(pmodel, pmodel_inputs, selector, predictor, error)
    # print(row)
