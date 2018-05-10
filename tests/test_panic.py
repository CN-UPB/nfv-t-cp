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
from nfvtcp.selector import PanicGreedyAdaptiveSelector
from nfvtcp.pmodel import SfcPerformanceModel, VnfPerformanceModel


class PerformanceModel_1VNF(SfcPerformanceModel):
    """
    Test PMODEL with one VNF.
    """
    @classmethod
    def generate_vnfs(cls, conf):
        # define possible parameters
        p = {"c1": [11, 21, 31, 41, 51],
             "c2": [12, 22, 32, 42],
             "c3": [13, 23, 33, 43, 53]}
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
        self.DEFAULT_PM = pm
        # cross product of possible PM parameters
        self.DEFAULT_PM_INPUTS = pm.get_conf_space()

    def tearDown(self):
        pass

    def _new_PGAS(self, max_samples=12, max_border_points=4, conf={}):
        s = PanicGreedyAdaptiveSelector(max_samples=max_samples,
                                        max_border_points=max_border_points,
                                        **conf)
        s.set_inputs(self.DEFAULT_PM_INPUTS, self.DEFAULT_PM)
        return s

    def test_initialize(self):
        s = self._new_PGAS()
        del s

    def test_calc_border_points(self):
        s = self._new_PGAS()
        bp = s._calc_border_points(s.pm_parameter, s.pm_inputs)
        self.assertEqual(len(bp), 52)

    def test_next_until_max_border_points(self):
        MS = 48
        MBP = 12
        s = self._new_PGAS(max_samples=MS, max_border_points=MBP)
        bp = s._calc_border_points(s.pm_parameter, s.pm_inputs)
        # pick MS samples
        for i in range(0, MS):
            # get next point to check
            p = s.next()
            self.assertIsNotNone(p)
            # inform selector about result of previous sample
            s.feedback(p, 0)
            if i < MBP:
                # ensure that MPB are border points not midpoints
                self.assertIn(p, bp)
        self.assertEqual(len(s._previous_samples), MS)

    def test_distance(self):
        s = self._new_PGAS()
        d = s._distance((None, 1), (None, 6))
        self.assertEqual(d, 5)
        d = s._distance((None, -1), (None, 6))
        self.assertEqual(d, 7)
        d = s._distance((None, 1), (None, -3))
        self.assertEqual(d, 4)
        d = s._distance((None, 2), (None, 2))
        self.assertEqual(d, 0)

    def test_find_midpoint(self):
        s = self._new_PGAS()
        # Define discrete conf. space, since the
        # method should only return points that exist
        # in this discrete space. Irrespective
        # of the input parameter of the method (cf. test3).

        class FakePM(object):
            pass
            
        fpm = FakePM()
        fpm.parameter = {"c1": [1, 4, 5, 6, 10],
                         "c2": [2, 3, 4, 5, 6, 7],
                         "c3": [20, 10, 30, 40]}
        
        s.set_inputs(None, fpm)
        # test 1
        p1 = [{"c1": 1, "c2": 2, "c3": 10}]
        p2 = [{"c1": 10, "c2": 4, "c3": 30}]
        mp = s._find_midpoint((p1, 0), (p2, 0))
        self.assertEqual(mp, ({"c1": 5, "c2": 3, "c3": 20},))
        # test 2
        p1 = [{"c1": 5, "c2": 4, "c3": 30}]
        p2 = [{"c1": 1, "c2": 7, "c3": 30}]
        mp = s._find_midpoint((p1, 0), (p2, 0))
        self.assertEqual(mp, ({"c1": 4, "c2": 5, "c3": 30},))
        # test 3
        p1 = [{"c1": -10, "c2": 7, "c3": 40}]
        p2 = [{"c1": 10, "c2": 6, "c3": 10}]
        mp = s._find_midpoint((p1, 0), (p2, 0))
        self.assertEqual(mp, ({"c1": 1, "c2": 6, "c3": 20},))


if __name__ == '__main__':
    unittest.main()
