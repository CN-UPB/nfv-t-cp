"""
Copyright (c) 2018 Manuel Peuster
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
import unittest
import networkx as nx
import numpy as np
from nfvppsim.selector import WeightedVnfSelector
from nfvppsim.pmodel import SfcPerformanceModel, VnfPerformanceModel


class PerformanceModel_4VNF(SfcPerformanceModel):
    """
    (s) - (v0) - (v1) - (v2) - (v3) - (t)
    """

    @classmethod
    def generate_vnfs(cls, conf):
        # define parameters
        # dict of lists defining possible configuration parameters
        # use normalized inputs for now
        p = {"p1": list(np.linspace(0.0, 1.0, num=10))}

        # create vnfs
        # function: config_list -> performance
        # REQUIREMENT: vnf_ids of objects == idx in list
        vnf0 = VnfPerformanceModel(0, "vnf0", p,
                                   lambda c: (c["p1"] * 0.2))
        vnf1 = VnfPerformanceModel(0, "vnf1", p,
                                   lambda c: (c["p1"] * 0.1))
        vnf2 = VnfPerformanceModel(0, "vnf2", p,
                                   lambda c: (c["p1"] * 0.1))
        vnf3 = VnfPerformanceModel(0, "vnf3", p,
                                   lambda c: (c["p1"] * 0.4))
        
        # return parameters, list of vnfs
        return p, [vnf0, vnf1, vnf2, vnf3]

    @classmethod
    def generate_sfc_graph(cls, conf, vnfs):
        # create a directed graph
        G = nx.DiGraph()
        # add nodes and assign VNF objects
        G.add_node(0, vnf=vnfs[0])
        G.add_node(1, vnf=vnfs[1])
        G.add_node(2, vnf=vnfs[2])
        G.add_node(3, vnf=vnfs[3])
        G.add_node("s", vnf=None)
        G.add_node("t", vnf=None)
        # s -> 0 -> 1,2 -> 3 -> 4 -> t
        G.add_edges_from([("s", 0),
                          (0, 1),
                          (1, 2),
                          (2, 3),
                          (3, "t")])
        return G


class TestWeightedVnfSelector(unittest.TestCase):
    def setUp(self):
        # instantiate a performance model for the tests
        [pm] = PerformanceModel_4VNF.generate(None)
        # definition of possible PM parameters
        self.DEFAULT_PM = pm
        # cross product of possible PM parameters
        self.DEFAULT_PM_INPUTS = pm.get_conf_space()

    def tearDown(self):
        pass

    def _new_WVS(self, max_samples=60, conf={}):
        s = WeightedVnfSelector(max_samples=max_samples, **conf)
        s.set_inputs(self.DEFAULT_PM_INPUTS, self.DEFAULT_PM)
        return s

    def test_initialize(self):
        s = self._new_WVS()
        del s

    def test_calc_border_points(self):
        pass

    def test_calc_weights(self):
        pass

    def test_next_until_max_border_points(self):
        pass

    def test_next_after_max_border_points(self):
        pass

    
if __name__ == '__main__':
    unittest.main()