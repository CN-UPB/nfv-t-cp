"""
Copyright (c) 2019 Heidi Neuhäuser
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
import numpy as np
from nfvtcp.decisiontree import *
from nfvtcp.helper import *


class TestNode(unittest.TestCase):

    def test_initialize(self):
        params = [{"a": [1, 2, 3], "b": [32, 64, 256]}, {"a": [1, 2], "b": [8, 16, 32, 64, 256]}]
        features = np.array([[1, 32, 1, 16], [1, 32, 1, 64], [2, 64, 2, 64], [3, 32, 1, 8]])
        target = np.array([0.61, 0.55, 0.32, 0.91])
        d = 4

        node = Node(params, features, target, d, 0, 0)

        self.assertEqual(node.parameters, params)
        self.assertTrue((node.features == features).all())
        self.assertTrue((node.target == target).all())
        self.assertEqual(node.depth, d)

        del node

    def test_calculate_partition_size(self):
        params = [{"a": [1, 2, 3], "b": [32]}, {"a": [1, 2], "b": [8]}]

        node = Node(params, None, None, 0, 0, 0)
        self.assertEqual(node.partition_size, 6)

        node.parameters = [{"a": [1, 2], "b": [32]}, {"a": [1, 2], "b": [8]}]
        node.partition_size = node._get_partition_size()
        self.assertEqual(node.partition_size, 4)
        del node

    def test_calculate_score(self):
        params = [{"a": [1, 2, 3], "b": [32]}, {"a": [1, 2], "b": [8]}]

        node = Node(params, None, None, 0, 0, 0)
        node.error = 0.25
        node.calculate_score(0.5, 4, 1234, 0.1, 0.8)

        score = 0.5 * (0.25 - 0.1) / (0.8 - 0.1) + 0.5 * (6 - 4) / (1234 - 4)

        self.assertEqual(node.score, score)
        del node

    def test_normalize_val(self):
        res = Node._normalize_val(1, 1, 3)
        self.assertEqual(res, 0)

        res = Node._normalize_val(2, 1, 3)
        self.assertEqual(res, 0.5)

        res = Node._normalize_val(3, 3, 3)
        self.assertEqual(res, 0)

    def test_is_leaf_node(self):
        params = [{"a": [1, 2, 3], "b": [32, 64, 256]}, {"a": [1, 2], "b": [8, 16, 32, 64, 256]}]
        features = np.array([[1, 32, 1, 16], [1, 32, 1, 64], [2, 64, 2, 64], [3, 32, 1, 8]])
        target = np.array([0.61, 0.55, 0.32, 0.91])
        d = 4

        node = Node(params, features, target, d, 0, 0)
        self.assertEqual(node.is_leaf_node(), True)

        node.split_feature_index = 3
        self.assertEqual(node.is_leaf_node(), False)
        del node


class TestDecisionTree(unittest.TestCase):

    def test_initialize(self):
        params = {"a": [1, 2, 3], "b": [32, 64, 256]}
        features = np.array([[1, 32, 1, 16], [1, 32, 1, 64], [2, 64, 2, 64], [3, 32, 1, 8]])
        target = np.array([0.61, 0.55, 0.32, 0.91])

        dtree = DecisionTree(params, [1, 32, 1, 16], 0.61)
        root = dtree.get_tree()
        root.features = features
        root.target = target

        self.assertEqual(root.parameters, [dict(params), dict(params)])
        self.assertTrue((root.features == features).all())
        self.assertTrue((root.target == target).all())
        self.assertEqual(dtree.vnf_count, 2)
        self.assertEqual(dtree._depth, 1)
        del dtree

    def test_calculate_new_params(self):
        params = {"a": [1, 2, 3], "b": [32, 64, 256]}
        features = np.array([[1, 32, 1, 16], [1, 32, 1, 64], [2, 64, 2, 64], [3, 32, 1, 8]])
        target = np.array([0.61, 0.55, 0.32, 0.91])

        dtree = DecisionTree(params, [1, 32, 1, 16], 0.61)
        root = dtree.get_tree()
        root.features = features
        root.target = target
        p_left, p_right = dtree._calculate_new_parameters(root.parameters, 1, (32 + 64) / 2)

        self.assertEqual(len(dtree.feature_idx_to_name), 4)
        self.assertEqual(dtree.vnf_count, 2)
        self.assertEqual(len(p_left), 2)
        self.assertEqual(len(p_right), 2)

        for i in range(len(dtree.params_per_vnf)):
            if i != 0:
                self.assertEqual(p_left[i], {"a": [1, 2, 3], "b": [32, 64, 256]})
                self.assertEqual(p_right[i], {"a": [1, 2, 3], "b": [32, 64, 256]})
            else:
                self.assertEqual(p_left[i], {"a": [1, 2, 3], "b": [32]})
                self.assertEqual(p_right[i], {"a": [1, 2, 3], "b": [64, 256]})

        del dtree

    def test_split_node(self):
        params = {"a": [1, 2, 3], "b": [32, 64, 256]}
        features = np.array([[1, 32, 1, 16], [1, 32, 1, 64], [2, 64, 2, 64], [3, 32, 1, 8]])
        target = np.array([0.61, 0.55, 0.32, 0.91])

        dtree = DecisionTree(params, [1, 32, 1, 16], 0.61)
        root = dtree.get_tree()
        root.features = features
        root.target = target

        root.split_feature_index = 3
        root.split_feature_cut_val = 50

        self.assertEqual(dtree._depth, 1)
        self.assertEqual(len(dtree.leaf_nodes), 1)
        print("root:\n{}".format(str(root)))

        dtree._split_node(root)

        self.assertEqual(dtree._depth, 2)
        self.assertEqual(len(dtree.leaf_nodes), 2)
        print("left child:\n{}".format(str(root.left)))
        print("right child:\n{}".format(str(root.right)))

        del dtree

    def test_split_samples(self):
        params = {"a": [1, 2, 3], "b": [32, 64, 256]}
        features = [1, 32, 1, 16]
        target = 0.61

        dtree = DecisionTree(params, features, target)

        features = np.array([[1, 32, 1, 16], [1, 32, 1, 64], [2, 64, 2, 64], [3, 32, 1, 8]])
        target = np.array([0.61, 0.55, 0.32, 0.91])
        left_f, left_t, right_f, right_t = dtree._split_samples(features, target, feature_idx=1, cut_val=48)

        left_features = np.array([[1, 32, 1, 16], [1, 32, 1, 64], [3, 32, 1, 8]])
        right_features = np.array([[2, 64, 2, 64]])
        left_target = np.array([0.61, 0.55, 0.91])
        right_target = np.array([0.32])

        self.assertTrue((left_f == left_features).all())
        self.assertTrue((right_f == right_features).all())
        self.assertTrue((left_t == left_target).all())
        self.assertTrue((right_t == right_target).all())

        self.assertEqual(len(left_f), 3)
        self.assertEqual(len(left_t), 3)
        self.assertEqual(len(right_f), 1)
        self.assertEqual(len(right_t), 1)

        del dtree

    def test_grow_tree_at_node(self):
        params = {"a": [1, 2, 3], "b": [32, 64, 256]}
        features = [1, 32, 1, 16]
        target = 0.61
        dtree = DecisionTree(params, features, target)
        root = dtree.get_tree()

        root.features = np.array([[1, 32, 1, 16], [1, 32, 1, 64], [2, 64, 2, 64], [3, 32, 1, 8]])
        root.target = np.array([0.61, 0.55, 0.32, 0.91])
        root.split_feature_index = 1
        root.split_feature_cut_val = 48

        dtree._grow_tree_at_node(root)
        dtree.print_tree(root)
        del dtree

    def test_get_config_from_partition(self):
        params = {"a": [1, 2, 3], "b": [32, 64, 256]}
        features = [1, 32, 1, 16]
        target = 0.61
        dtree = DecisionTree(params, features, target)
        root = dtree.get_tree()

        c = dtree._get_config_from_partition(root)
        print(c)
        self.assertEqual(len(c), 2)
        self.assertEqual(len(c[0]), 2)

        del dtree

    def test_adapt_tree(self):
        params = {"a": [1, 2, 3], "b": [32, 64, 256]}
        features = [1, 32, 1, 16]
        target = 0.61
        dtree = DecisionTree(params, features, target)
        root = dtree.get_tree()
        dtree.last_sampled_node = root

        self.assertEqual(len(root.features), 1)
        self.assertEqual(len(root.target), 1)

        dtree.adapt_tree(([1, 2, 3, 4], 0.54))
        dtree.adapt_tree(([2, 2, 3, 1], 0.73))

        self.assertEqual(len(root.features), 3)
        self.assertEqual(len(root.target), 3)
        del dtree

    def test_get_possible_splits(self):
        params = {"a": [1, 2, 3], "b": [32, 64, 256]}
        features = [1, 32, 1, 16]
        target = 0.61
        dtree = DecisionTree(params, features, target)

        f = np.array([[1, 32, 1, 16], [1, 32, 1, 64], [2, 64, 2, 64], [3, 32, 3, 8]])

        poss_splits = dtree._get_possible_splits(f, 1)
        splits = np.array([(32 + 64) / 2])
        self.assertTrue(len(poss_splits), 1)
        self.assertTrue((poss_splits == splits).all())

        poss_splits = dtree._get_possible_splits(f, 2)
        splits = np.array([1.5, 2.5])
        self.assertTrue(len(poss_splits), 2)
        self.assertTrue((poss_splits == splits).all())

        del dtree


class TestObliqueDecisionTree(unittest.TestCase):
    # Todo
    pass


if __name__ == '__main__':
    unittest.main()
