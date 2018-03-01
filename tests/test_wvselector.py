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
import random
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
        p = {"p1": list(np.linspace(0.0, 1.0, num=5)),
             "p2": list(np.linspace(0.5, 1.5, num=4))}

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

    def _new_WVS(self, max_samples=60,
                 border_point_mode=0,
                 p_samples_per_vnf=-1,
                 conf={}):
        s = WeightedVnfSelector(max_samples=max_samples,
                                border_point_mode=border_point_mode,
                                p_samples_per_vnf=p_samples_per_vnf,
                                **conf)
        s.set_inputs(self.DEFAULT_PM_INPUTS, self.DEFAULT_PM)
        return s

    def test_wvs_initialize(self):
        s = self._new_WVS()
        del s

    def test_wvs_calc_border_points(self):
        n_vnfs = 4
        s = self._new_WVS()
        r = s._calc_border_points(mode=0)
        self.assertEqual(len(r), n_vnfs + 1,
                         msg="wrong number of border points returned")
        r = s._calc_border_points(mode=1)
        self.assertEqual(len(r), n_vnfs + 1,
                         msg="wrong number of border points returned")
        r = s._calc_border_points(mode=2)
        self.assertEqual(len(r), 2 * n_vnfs + 2,
                         msg="wrong number of border points returned")
        r = s._calc_border_points(mode=3)
        self.assertEqual(len(r), 0,  # TODO not implemented yet
                         msg="wrong number of border points returned")

    def test_wvs_distance(self):
        s = self._new_WVS()
        self.assertEqual(s._distance(0, 2), 2.0)
        self.assertEqual(s._distance(1, 1), 0.0)
        self.assertEqual(s._distance(-1, 2.0), 3.0)
        self.assertEqual(s._distance(3, -1), 4.0)
        self.assertEqual(s._distance(-3, -2), 1.0)

    def test_wvs_calc_weights(self):
        for mode in [0, 1]:
            s = self._new_WVS(border_point_mode=mode)
            # add fake feedback results for weight calculation
            s.feedback(0, 0)
            s.feedback(1, 1)
            s.feedback(2, 5)
            s.feedback(3, 3)
            s.feedback(4, -2)
            # calculate weights
            w = s._calc_weights(mode=mode)
            self.assertEqual(len(w), len(s.pm.vnfs))
        mode = 2
        s = self._new_WVS(border_point_mode=mode)
        # add fake feedback results for weight calculation
        # max
        s.feedback(0, 0)
        s.feedback(1, 1)
        s.feedback(2, 5)
        s.feedback(3, 3)
        s.feedback(4, -2)
        # min
        s.feedback(0, 0)
        s.feedback(1, -1)
        s.feedback(2, 2)
        s.feedback(3, -7)
        s.feedback(4, 2)
        # calculate weights
        w = s._calc_weights(mode=mode)
        self.assertEqual(len(w), len(s.pm.vnfs) * 2)
        # TODO mode 3

    def test_wvs_get_vnf_idx_ordered_by_weight(self):
        w1 = [0.0909, 0.45, 0.27, 0.18]
        w2 = [0.0434, 0.217, 0.13, 0.086, 0.04, 0.086, 0.304, 0.086]
        s = self._new_WVS()
        r1 = s._get_vnf_idx_ordered_by_weight(w1)
        self.assertEqual(r1, [1, 2, 3, 0])
        r2 = s._get_vnf_idx_ordered_by_weight(w2)
        self.assertEqual(r2, [2, 1, 2, 3, 1, 3, 0, 0])

    def test_wvs_sample_points_of_vnf_random(self):
        mode = 0
        n_vnfs = 4
        s = self._new_WVS(border_point_mode=mode)
        p_min, p_max = s._get_min_max_parameter()
        for repetition in range(0, 100):  # try it 100 times
            for vnf_idx in range(0, n_vnfs):
                r = s._sample_points_of_vnf_random(vnf_idx, mode=mode)
                # print(r)
                self.assertEqual(len(r), n_vnfs)
                for tst_idx in range(0, n_vnfs):
                    if tst_idx == vnf_idx:
                        self.assertNotEqual(r[tst_idx], p_max)
                        self.assertNotEqual(r[tst_idx], p_min)
                        self.assertEqual(len(r[tst_idx]), len(s.pm.parameter))
                    else:  # fixed VNF config
                        if mode == 0:
                            self.assertEqual(r[tst_idx], p_max)
                        else:
                            self.assertEqual(r[tst_idx], p_min)

    def test_wvs_next(self):
        n_vnfs = 4
        n_bps = [n_vnfs + 1, n_vnfs + 1, 2 * n_vnfs + 2]
        p_samples_per_vnf_lst = [-1, 1, 10, 100]
        for pspv in p_samples_per_vnf_lst:
            for mode in range(0, 3):  # TODO mode 3 not implemented yet
                s = self._new_WVS(
                    border_point_mode=mode,
                    p_samples_per_vnf=pspv
                )
                # reference border point list (the internal one is modified)
                bps = s._calc_border_points(mode=mode)
                for i in range(0, (n_bps[mode] + 1) * 10):  # 10x tests
                    c = s.next()
                    # give random feedback
                    s.feedback(c, random.uniform(1, 10))
                    # validate point
                    self.assertTrue(c is not None)
                    self.assertEqual(len(c), n_vnfs)
                    # check if weight calculation is triggered
                    if i < n_bps[mode]:
                        self.assertTrue(s._weights is None)
                        self.assertIn(c, bps)
                    else:
                        self.assertTrue(s._weights is not None)
                        self.assertNotIn(c, bps)


if __name__ == '__main__':
    unittest.main()
