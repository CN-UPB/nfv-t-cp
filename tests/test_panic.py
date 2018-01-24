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
from nfvppsim.selector import PanicGreedyAdaptiveSelector
from nfvppsim.pmodel import SfcPerformanceModel, VnfPerformanceModel


class PerformanceModel_1VNF(SfcPerformanceModel):
    """
    Test PMODEL with one VNF.
    """
    @classmethod
    def generate_vnfs(cls, conf):
        # define possible parameters
        p = {"c1": [1, 2, 3],
             "c2": [4, 5],
             "c3": [6, 7, 8]}
        # create vnfs
        vnf0 = VnfPerformanceModel(0, "vnf_0", p,
                                   lambda c: (((1 * c["c1"] + 0)
                                               + (1 * c["c2"] + 0)
                                               + (1 * c["c3"] + 0))))
        # return parameters, list of vnfs
        return p, [vnf0]

    @classmethod
    def generate_sfc_graph(cls, conf, vnfs):
        G = nx.DiGraph()
        G.add_node(0, vnf=vnfs[0])
        G.add_node("s", vnf=None)
        G.add_node("t", vnf=None)
        # simple linear: s -> 0 -> t
        G.add_edges_from([("s", 0), (0, "t")])
        return G


class TestPanicSelector(unittest.TestCase):
    def setUp(self):
        # instantiate a performance model for the tests
        [pm] = PerformanceModel_1VNF.generate(None)
        # definition of possible PM parameters
        self.DEFAULT_PM_PARAMETERS = pm.parameter
        # cross product of possible PM parameters
        self.DEFAULT_PM_INPUTS = pm.get_conf_space()

    def tearDown(self):
        pass

    def _new_PGAS(self, max_samples=12, max_border_points=4, conf={}):
        s = PanicGreedyAdaptiveSelector(max_samples=max_samples,
                                        max_border_points=4,
                                        **conf)
        s.set_inputs(self.DEFAULT_PM_INPUTS, self.DEFAULT_PM_PARAMETERS)
        return s

    def test_initialize(self):
        s = self._new_PGAS()
        del s

    def test_calc_border_points(self):
        s = self._new_PGAS()
        s._calc_border_points()


if __name__ == '__main__':
    unittest.main()
