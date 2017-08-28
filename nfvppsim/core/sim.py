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


class Profiler(object):

    def __init__(self, pmodel, selector, predictor, result):
        self.pm = pmodel
        self.s = selector
        self.p = predictor
        self.r = result
        # initialize simulation environment
        self.env = simpy.Environment()
        self.profile_proc = self.env.process(self.do_measurement())

    def do_measurement(self):
        while True:
            print("Select config")
            print("Start measurement at {} ...".format(self.env.now))
            print("Evaluate performance model")
            # Note: Timing could be randomized, or a more complex function:
            yield self.env.timeout(60)  # Fix: assumes 60s per measurement
            print("... done at {}".format(self.env.now))
            print("Store single result.")

    def run(self, until=None):
        self.env.run(until=until)  # time limit in seconds
        print("Predict to have full result from selected.")

        
def run(pmodel, selector, predictor, result):
    p = Profiler(pmodel, selector, predictor, result)
    p.run(until=400)
