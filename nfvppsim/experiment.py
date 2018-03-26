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
import copy
import datetime
import pandas as pd

from nfvppsim import sim
from nfvppsim.config import expand_parameters
import nfvppsim.pmodel
import nfvppsim.selector
import nfvppsim.predictor
import nfvppsim.error
import nfvppsim.plot

LOG = logging.getLogger(os.path.basename(__file__))


class Experiment(object):
    """
    An experiment is a single simulation run.
    Its behavior is defined by the config file.
    """

    def __init__(self, conf):
        """
        Load modules and configure experiment.
        """
        self.conf = conf
        # Pandas DF to hold result after run()
        self.result_df = None
        # get classes of modules to be use based on config
        # pmodel
        self._pmodel_cls_lst = list()
        for p in conf.get("pmodels"):
            pcls = nfvppsim.pmodel.get_by_name(p.get("name"))
            self._pmodel_cls_lst.append((pcls, p))
        # selector(s)
        self._selector_cls_lst = list()
        for s in conf.get("selector"):
            scls = nfvppsim.selector.get_by_name(s.get("name"))
            self._selector_cls_lst.append((scls, s))
        # predictor
        self._predictor_cls_lst = list()
        for p in conf.get("predictor"):
            pcls = nfvppsim.predictor.get_by_name(p.get("name"))
            self._predictor_cls_lst.append((pcls, p))
        # error metrics
        self._error_cls_lst = list()
        for em in conf.get("error_metrics"):
            ecls = nfvppsim.error.get_by_name(em.get("name"))
            self._error_cls_lst.append((ecls, em))
        # plots
        self._plot_cls_lst = list()
        if conf.get("plot") is not None:
            for p in conf.get("plot"):
                pcls = nfvppsim.plot.get_by_name(p.get("name"))
                self._plot_cls_lst.append((pcls, p))
        
    def prepare(self):
        """
        Prepare experiment: Generate configurations to be simulated.
        """
        self._lst_sim_t_max = expand_parameters(
            self.conf.get("sim_t_max"))
        # pmodel
        self._lst_pmodel = list()
        for pmcls, pmconf in self._pmodel_cls_lst:
            self._lst_pmodel += pmcls.generate(pmconf)
        # selector(s)
        self._lst_selector = list()
        for scls, sconf in self._selector_cls_lst:
            self._lst_selector += scls.generate(sconf)
        # predictor
        self._lst_predictor = list()
        for pcls, pconf in self._predictor_cls_lst:
            self._lst_predictor += pcls.generate(pconf)
        # error metrics
        self._lst_error = list()
        for ecls, econf in self._error_cls_lst:
            self._lst_error += ecls.generate(econf)
        # plots
        self._lst_plot = list()
        for pcls, pconf in self._plot_cls_lst:
            self._lst_plot += pcls.generate(pconf)

        LOG.info("*** Prepared {}x{} configurations to be simulated.".format(
            self.n_configs,
            self.conf.get("repetitions", 1)))

    @property
    def n_configs(self):
        """
        Attention: Does not consider number of repetitions.
        Keep in sync with prepare method.
        """
        return (len(self._lst_sim_t_max) *
                len(self._lst_pmodel) *
                len(self._lst_selector) *
                len(self._lst_predictor))

    def generate(self):
        """
        Generate all configurations to be simulated to have
        a flat list that can be split among several simulation
        jobs/processes.
        """
        configs = list()
        conf_id = 0
        # iterate over all sim. configurations and run simulation
        for sim_t_max in self._lst_sim_t_max:
            for pm_obj in self._lst_pmodel:
                for s_obj in self._lst_selector:
                    for p_obj in self._lst_predictor:
                        conf_id += 1
                        if conf_id % int(self.n_configs / 100) == 0:
                            LOG.info("Generating configuration {}/{}"
                                     .format(conf_id,
                                             self.n_configs))
                        for r_id in range(0, self.conf.get(
                                "repetitions", 1)):
                            # add arguments for run call
                            configs.append((sim_t_max,
                                            pm_obj,
                                            s_obj,
                                            p_obj,
                                            self._lst_error,
                                            r_id,
                                            conf_id))
        LOG.info("*** Generated {}x{}={} configurations to be simulated."
                 .format(self.n_configs,
                         self.conf.get("repetitions", 1),
                         len(configs)))
        return configs

    def run(self, configs):
        """
        Executes an experiment by iterating all configurations
        that have been generated for simulation.
        Uses deepcopy do ensure fresh internal states of all
        algorithm objects passed to the simulator module.
        """
        job_id = self.conf.get("job_id", 0)
        job_no = self.conf.get("job_no", 1)
        if job_id >= job_no:
            LOG.error("job_id {} >= job_no {}! Abort run!".format(
                job_id, job_no))
            return
        job_size, job_size_rest = divmod(len(configs), job_no)
        start_idx = job_id * job_size
        end_idx = (job_id + 1) * job_size
        if (job_id + 1) == job_no:
            end_idx += job_size_rest
        LOG.info("*** Running job: {} start/end: {}/{} of {} configs".format(
            job_id, start_idx, end_idx, len(configs)))
        self.result_df = self._run_process(configs, start_idx, end_idx)

    def _run_process(self, configs, start_idx, end_idx):
        """
        Simulate only given subset of configurations.
        """
        # list to hold results before moved to Pandas DF
        tmp_results = list()
        # iterate over all sim. configurations and run simulation
        for i in range(start_idx, end_idx):
            c = configs[i]
            if i % int((end_idx - start_idx) / 100) == 0:  # progress
                LOG.info("*** Simulating conf. {} from [{}, {}] ({:.1f}%)"
                         .format(
                             i, start_idx, end_idx - 1,
                             ((i - start_idx) / (end_idx - start_idx) * 100)))
            # Attention: We need to copy the models objects
            # to have fresh states for each run!
            # TODO Can we optimize?
            row_lst = sim.run(c[0],  # sim_t_max
                              copy.deepcopy(c[1]),  # pm_obj
                              copy.deepcopy(c[2]),  # s_obj
                              copy.deepcopy(c[3]),  # p_obj
                              copy.deepcopy(c[4]),  # _lst_error
                              c[5])  # r_id
            for row in row_lst:
                # extend result
                row.update({"conf_id": c[6],  # TODO from gen!
                            "repetition_id": c[5]})
                tmp_results.append(row)
        return pd.DataFrame(tmp_results)

    def plot(self, data_path):
        """
        Plot results using each initialized plotter.
        data_path: Path to pickle file to be plotted.
        """
        df = None
        try:
            df = pd.read_pickle(data_path, compression="bz2")
            # l = list(set(list(df["selector_conf"])))
            # for x in l:
            #    print(x)
            # exit(1)
        except BaseException as e:
            LOG.exception("Could not find '{}'. Abort plotting."
                          .format(data_path))
            exit(1)
        assert(df is not None)
        print(df)
        for p in self._lst_plot:
            p.plot(df)

    def store_result(self, path):
        """
        Stores result DF in pickle file if path
        is not None.
        """
        assert(self.result_df is not None)
        if path is None:
            LOG.warning("'result_path' not specified. No results stored.")
            return
        if self.conf.get("result_path_add_timestamp", False):
            # add timestamp to path
            fname = os.path.basename(path)
            nfname = "{}_{}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
                fname)
            path = path.replace(fname, nfname)
        if self.conf.get("job_id") is not None:
            path = path.replace(".pkl", ".job{}.pkl".format(
                self.conf.get("job_id")))
        with open(path, "wb") as f:
            self.result_df.to_pickle(f, compression="bz2")
        LOG.info("Wrote result with {} rows to '{}'".format(
            len(self.result_df.index), path))

    def print_results(self):
        """
        Print result DF to screen.
        """
        LOG.info("Printing result DF to 'stdout'")
        print(self.result_df)
        self.result_df.info()

    @property
    def result_number(self):
        if self.result_df is None:
            return 0
        return len(self.result_df.index)

