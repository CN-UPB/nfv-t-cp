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
from nfvppsim.pmodel import RandomSyntheticModel


class TestRandomSyntheticModel(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_initialize(self):
        conf = {
            "a1_range": [0.1, 2.0],
            "func_set": [1, 2, 3, 4, 5, 6, 7, 8],
            "topologies": ["d4"]
        }
        m_lst = RandomSyntheticModel.generate(conf)
        self.assertEqual(len(m_lst), 1)

    def test_vnf_eval(self):
        conf = {
            "a1_range": [0.1, 2.0],
            "func_set": [1, 2, 3, 4, 5, 6, 7, 8],
            "topologies": ["d4"]
        }
        m_lst = RandomSyntheticModel.generate(conf)
        for m in m_lst:
            for v in m.vnfs:
                v.evaluate({"p1": random.random()})

    def test_vnf_func_set(self):
        conf = {
            "a1_range": [1.0, 1.0],
            "func_set": [2],
            "topologies": ["d4"]
        }
        m_lst = RandomSyntheticModel.generate(conf)
        for m in m_lst:
            for v in m.vnfs:
                r = v.evaluate({"p1": 2})
                self.assertEqual(r, 4.0)

    def test_multi_modelgeneration(self):
        conf = {
            "n_model_instances": 10,
            "a1_range": [0.1, 2.0],
            "func_set": [1, 2, 3, 4, 5, 6, 7, 8],
            "topologies": ["d4"]
        }
        m_lst = RandomSyntheticModel.generate(conf)
        self.assertEqual(len(m_lst), 10)

    def test_multi_modelgeneration_multi_topologies(self):
        conf = {
            "n_model_instances": 2,
            "a1_range": [0.1, 2.0],
            "func_set": [1, 2, 3, 4, 5, 6, 7, 8],
            "topologies": ["l1", "l2", "l3", "l4", "l5",
                           "d2", "d3", "d4", "d5"]
        }
        m_lst = RandomSyntheticModel.generate(conf.copy())
        self.assertEqual(len(m_lst), 18)
        for m in m_lst:
            _, n_vnfs = m.parse_topology_name(m.conf.get("topology"))
            self.assertEqual(n_vnfs, len(m.vnfs), msg=m.sfc_graph.nodes())
            self.assertEqual(n_vnfs + 2, len(m.sfc_graph), msg="{}/{}".format(
                m.sfc_graph.nodes(), m.conf.get("topology")))

    def test_parse_topology_name(self):
        conf = {
            "a1_range": [1.0, 1.0],
            "func_set": [2],
            "topologies": ["d4"]
        }
        m_lst = RandomSyntheticModel.generate(conf)
        self.assertEqual(m_lst[0].parse_topology_name("d1"), ("d", 1))
        self.assertEqual(m_lst[0].parse_topology_name("l7"), ("l", 7))
        self.assertEqual(m_lst[0].parse_topology_name("d1a"), ("d", 1))
        self.assertEqual(m_lst[0].parse_topology_name("d1b"), ("d", 1))
