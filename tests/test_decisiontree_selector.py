"""
Copyright (c) 2019 Heidi Neuhäuser
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
"""
import unittest
import networkx as nx
import numpy as np
import random
from nfvtcp.selector import DecisionTreeSelector
from nfvtcp.pmodel import SfcPerformanceModel, VnfPerformanceModel, ExampleModel


# Test Oblique Tree

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


class TestDecisionTreeSelector(unittest.TestCase):
    def setUp(self):
        # instantiate a performance model for the tests
        [pm] = PerformanceModel_4VNF.generate(None)
        [pm1] = ExampleModel.generate(None)
        # definition of possible PM parameters
        self.DEFAULT_PM = pm
        self.DEFAULT_PM_EXAMPLE = pm1
        # cross product of possible PM parameters
        self.DEFAULT_PM_INPUTS = pm.get_conf_space()
        self.DEFAULT_PM_INPUTS_EXAMPLE = pm1.get_conf_space()

    def _new_DTS(self, max_samples=60,
                 initial_samples=10,
                 max_depth=100,
                 regression="default",
                 min_samples_split=2,
                 example=False,
                 conf={}):

        s = DecisionTreeSelector(
            max_samples=max_samples,
            initial_samples=initial_samples,
            max_depth=max_depth,
            regression=regression,
            min_samples_split=min_samples_split,
            **conf)

        if not example:
            s.set_inputs(self.DEFAULT_PM_INPUTS, self.DEFAULT_PM)
        else:
            s.set_inputs(self.DEFAULT_PM_INPUTS_EXAMPLE, self.DEFAULT_PM_EXAMPLE)
        return s

    def test_initialize(self):
        s = self._new_DTS()
        del s

        s = self._new_DTS(regression="oblique")
        del s

    def test_select_random_config(self):
        s = self._new_DTS()
        vnf_count = 4
        c = s._select_random_config()
        self.assertTrue(c is not None)
        self.assertEqual(len(c), vnf_count)
        del s

    def test_feedback(self):
        s = self._new_DTS()
        c = s._select_random_config()
        s.feedback(c, random.uniform(1, 10))
        self.assertEqual(len(s._sampled_configs), 1)
        self.assertEqual(len(s._sample_results), 1)
        del s

    def test_initialize_tree(self):
        s = self._new_DTS()
        for i in range(10):
            c = s._select_random_config()
            s.feedback(c, random.uniform(1, 10))

        s._initialize_tree()
        self.assertTrue(s._tree is not None)
        self.assertEqual(len(s._sampled_configs), 10)
        self.assertEqual(len(s._sample_results), 10)

        del s

    def test_initialize_tree_oblique(self):
        # test with ExampleModel
        s = self._new_DTS(regression="oblique", min_samples_split=4, example=True)
        for i in range(10):
            c = s._select_random_config()
            s.feedback(c, random.uniform(1, 10))

        s._initialize_tree()
        self.assertTrue(s._tree is not None)
        self.assertEqual(len(s._sampled_configs), 10)
        self.assertEqual(len(s._sample_results), 10)

        del s

    def test_next(self):
        s = self._new_DTS()
        c = s._next()
        self.assertTrue(s._tree is None)
        self.assertEqual(len(s._sampled_configs), 0)
        self.assertEqual(len(s._sample_results), 0)
        self.assertEqual(s.k_samples, 1)
        s.feedback(c, random.uniform(1, 10))
        for i in range(10):
            c = s._next()
            s.feedback(c, random.uniform(1, 10))

        self.assertTrue(s._tree is not None)
        self.assertEqual(len(s._sampled_configs), 10)
        self.assertEqual(len(s._sample_results), 10)
        self.assertEqual(s.k_samples, 11)

        del s

    """
    def test_next_oblique(self):
        s = self._new_DTS(regression="oblique", min_samples_split=4, example=True)
        c = s._next()
        self.assertTrue(s._tree is None)
        self.assertEqual(len(s._sampled_configs), 0)
        self.assertEqual(len(s._sample_results), 0)
        self.assertEqual(s.k_samples, 1)
        s.feedback(c, random.uniform(1, 10))
        for i in range(10):
            c = s._next()
            s.feedback(c, random.uniform(1, 10))

        self.assertTrue(s._tree is not None)
        self.assertEqual(len(s._sampled_configs), 10)
        self.assertEqual(len(s._sample_results), 10)
        self.assertEqual(s.k_samples, 11)

        del s
    """


if __name__ == '__main__':
    unittest.main()
