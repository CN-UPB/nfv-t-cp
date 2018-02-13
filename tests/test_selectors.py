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
from nfvppsim.selector import HyperGridSelector
from nfvppsim.pmodel import SfcPerformanceModel, VnfPerformanceModel


class PerformanceModel_2VNF(SfcPerformanceModel):
    """
    Test PMODEL with one VNF.
    """
    @classmethod
    def generate_vnfs(cls, conf):
        # define possible parameters
        p = {"c1": [1, 2, 3, 4, 5],
             "c2": [1, 2, 3],
             "c3": [1]}
        # create vnfs
        vnf0 = VnfPerformanceModel(0, "vnf_0", p,
                                   lambda c: (((1 * c["c1"] + 0)
                                               + (1 * c["c2"] + 0)
                                               + (1 * c["c3"] + 0))))
        vnf1 = VnfPerformanceModel(0, "vnf_1", p,
                                   lambda c: (((1 * c["c1"] + 0)
                                               + (1 * c["c2"] + 0)
                                               + (1 * c["c3"] + 0))))
        # return parameters, list of vnfs
        return p, [vnf0, vnf1]

    @classmethod
    def generate_sfc_graph(cls, conf, vnfs):
        G = nx.DiGraph()
        G.add_node(0, vnf=vnfs[0])
        G.add_node(1, vnf=vnfs[1])
        G.add_node("s", vnf=None)
        G.add_node("t", vnf=None)
        # simple linear: s -> 0 -> 1 -> t
        G.add_edges_from([("s", 0), (0, 1), (1, "t")])
        return G


class TestHyperGridSelector(unittest.TestCase):
    def setUp(self):
        # instantiate a performance model for the tests
        [pm] = PerformanceModel_2VNF.generate(None)
        # definition of possible PM parameters
        self.DEFAULT_PM = pm
        # cross product of possible PM parameters
        self.DEFAULT_PM_INPUTS = pm.get_conf_space()

    def tearDown(self):
        pass

    def _new_HGS(self, max_samples=12, conf={}):
        s = HyperGridSelector(max_samples=max_samples, **conf)
        s.set_inputs(self.DEFAULT_PM_INPUTS, self.DEFAULT_PM)
        return s

    def test_initialize(self):
        s = self._new_HGS()
        del s

    def test_get_n_samples_from_list(self):
        s = self._new_HGS()
        lst = [1, 2, 3, 4, 5, 6, 7]
        # zero elements
        r = s._get_n_samples_from_list(lst, 0)
        self.assertEqual(r, [])
        # one element
        r = s._get_n_samples_from_list(lst, 1)
        self.assertEqual(r, [4])
        # two elements
        r = s._get_n_samples_from_list(lst, 2)
        self.assertEqual(r, [1, 7])
        # three elements
        r = s._get_n_samples_from_list(lst, 3)
        self.assertEqual(r, [1, 4, 7])
        # four elements
        r = s._get_n_samples_from_list(lst, 4)
        self.assertEqual(r, [1, 3, 5, 7])

        lst = [1, 2, 3, 4]
        # zero elements
        r = s._get_n_samples_from_list(lst, 0)
        self.assertEqual(r, [])
        # one element
        r = s._get_n_samples_from_list(lst, 1)
        self.assertEqual(r, [3])
        # two elements
        r = s._get_n_samples_from_list(lst, 2)
        self.assertEqual(r, [1, 4])
        # three elements
        r = s._get_n_samples_from_list(lst, 3)
        self.assertEqual(r, [1, 2, 4])
        # four elements
        r = s._get_n_samples_from_list(lst, 4)
        self.assertEqual(r, [1, 2, 3, 4])

        lst = [2]
        # zero elements
        r = s._get_n_samples_from_list(lst, 0)
        self.assertEqual(r, [])
        # one element
        r = s._get_n_samples_from_list(lst, 1)
        self.assertEqual(r, [2])
        # two elements
        r = s._get_n_samples_from_list(lst, 2)
        self.assertEqual(r, [2])
        # three elements
        r = s._get_n_samples_from_list(lst, 3)
        self.assertEqual(r, [2])
        # four elements
        r = s._get_n_samples_from_list(lst, 4)
        self.assertEqual(r, [2])

    def test_calculate_grid(self):
        for i in range(1, len(self.DEFAULT_PM_INPUTS)):
            s = self._new_HGS(max_samples=i)
            r = s._calculate_grid()
            self.assertEqual(len(r), i)

    def test_next(self):
        s = self._new_HGS(max_samples=16)
        for i in range(0, 16):
            r = s.next()
            self.assertEqual(len(r), 2)


if __name__ == '__main__':
    unittest.main()
