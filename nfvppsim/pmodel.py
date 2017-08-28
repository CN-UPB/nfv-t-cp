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


class SimpleNetworkServiceThroughputModel(object):
    """
    A network service based on a linear SFC: f1 -> f2 -> ... -> fN
    """
    def __init__(self, name, vnfs, alphas=None):
        """
        name: name of service (string)
        vnfs: vector of functions representing the
              VNF's CPU-time -> throughput mapping
        alphas: vector of floats to scale the performance of a VNF
                at the corresponding position (set to [1.0,...,1.0] if None)
        """
        self.name = name
        self.vnfs = vnfs
        self.alphas = alphas if alphas else [1.0 for _ in self.vnfs]
        print("Initialized '{}' with alphas={} and {} VNFs".format(
            self.name, self.alphas, len(self.vnfs)))
        
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
