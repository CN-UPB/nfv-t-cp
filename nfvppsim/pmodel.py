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
import sys

LOG = logging.getLogger(os.path.basename(__file__))


def get_by_name(name):
    if name == "SimpleNetworkServiceThroughputModel":
        return SimpleNetworkServiceThroughputModel
    raise NotImplementedError("'{}' not implemented".format(name))


class SimpleNetworkServiceThroughputModel(object):
    """
    A network service based on a linear SFC: f1 -> f2 -> ... -> fN
    """
    @classmethod
    def generate(cls, conf):
        """
        Generate list of model objects. One for each conf. to be tested.
        """
        # TODO change to generate real models
        pm_obj = cls(vnfs=[lambda x: x**2 + (x * 2) + 0.1,
                           lambda x: x**4 + (.5 * x)],
                     alphas=None)
        return [pm_obj]
    
    def __init__(self, **kwargs):
        """
        vnfs: vector of functions representing the
              VNF's CPU-time -> throughput mapping
        alphas: vector of floats to scale the performance of a VNF
                at the corresponding position (set to [1.0,...,1.0] if None)
        """
        self.vnfs = kwargs.get("vnfs", [])
        self.alphas = kwargs.get("alphas", None)
        if self.alphas is None:
            self.alphas = [1.0 for _ in self.vnfs]
        if len(self.vnfs) < 1:
            LOG.error("{} with 0 VNFs not supported. Stopping.".format(self))
            sys.exit(1)
        LOG.debug("Initialized performance model: '{}' with {} VNFs".format(
            self, len(self.vnfs)))

    def __repr__(self):
        return "{}({})".format(
            self.name, self.alphas)

    @property
    def name(self):
        return self.__class__.__name__
        
    def _calc_vnf_tp(self, cpu_times):
        """
        calculate TP for each function in self.vnfs
        cpu_times: CPU time available for each VNF
        """
        assert len(cpu_times) == len(self.vnfs) == len(self.alphas)
        # calculate result for each vnf and multiply by corresponding alpha
        return [f(r) * a for f, r, a in zip(self.vnfs, cpu_times, self.alphas)]
           
    def evaluate(self, cpu_times):
        """
        calculate TP of SFC
        cpu_times: CPU time available for each VNF
        """
        # uses "naive" minimum-TP model from NFV-SDN'17 paper for now
        return min(self._calc_vnf_tp(cpu_times))

    def get_results(self):
        """
        Getter for global result collection.
        :return: dict for result row
        """
        r = {"pmodel": self.name}
        # r.update(self.params)
        # LOG.debug("Get results from {}: {}".format(self, r))
        return r
