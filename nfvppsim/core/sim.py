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
import simpy
import logging
import os

LOG = logging.getLogger(os.path.basename(__file__))


class Profiler(object):

    def __init__(self, pmodel, pmodel_inputs, selector, predictor, result):
        self.pm = pmodel
        self.pm_inputs = pmodel_inputs
        self.s = selector
        self.p = predictor
        self.r = result
        self._tmp_train_c = list()
        self._tmp_train_r = list()
        # initialize simulation environment
        self.env = simpy.Environment()
        self.profile_proc = self.env.process(self.do_measurement())

    def do_measurement(self):
        while self.s.has_next():
            c = self.s.next()
            LOG.debug("t={} measuring config: {}".format(self.env.now, c))
            r = self.pm.evaluate(c)
            self._tmp_train_c.append(c)  # store configs ...
            self._tmp_train_r.append(r)  # ... and results of profiling run
            self.s.feedback(c, r)  # inform selector about result
            # Note: Timing could be randomized, or a more complex function:
            yield self.env.timeout(60)  # Fix: assumes 60s per measurement
            LOG.debug("t={} result: {}".format(self.env.now, r))
        LOG.debug("No configurations left. Stopping simulation.")

    def run(self, until=None):
        # TODO refactor: initialize, postprocess
        # reset tmp. results
        self._tmp_train_c = list()
        self._tmp_train_r = list()
        # simulate profiling process
        self.env.run(until=until)  # time limit in seconds
        # predict full result using training sets
        # TODO p.train( self._tmp_train_c, self._tmp_train_r)
        # TODO p.predict(self.pm_inputs) -> final result
        # TODO calculate error
        # TODO add to global results (or return?)
        LOG.debug("Done. Resulting MSE={}".format(0.0))

        
def run(pmodel, pmodel_inputs, selector, predictor, result):
    p = Profiler(pmodel, pmodel_inputs, selector, predictor, result)
    p.run(until=400)
