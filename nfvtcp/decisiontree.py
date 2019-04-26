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
import logging
import os
import numpy as np
from random import randint, random, choice
import heapq

LOG = logging.getLogger(os.path.basename(__file__))


class Node:
    """
    Base Class for Decision Tree Nodes.
    """

    _config_size = 0

    def __init__(self, params, features, target, depth, idx):
        """
        :param params: list of dictionaries with possible parameter values for each vnf
        :param features: sampled configurations as flat 2D numpy array
        :param target: 1D numpy array of performance values of sampled configs
        :param depth: depth of node in whole tree
        """
        # Todo: Delete feature/target/params if node is no leaf no save memory? (Can be recalculated for pruning)

        self.parameters = params
        self.features = features
        self.target = target
        self.left = None
        self.right = None
        self.depth = depth
        self.idx = idx
        self.split_feature_index = None
        self.split_feature_cut_val = None
        self.split_improvement = 0
        self.pred_value = None
        self.error = None  # deviation from prediction. Smaller = better
        self.partition_size = None  # number of configs in partition
        self.score = None

    def __str__(self):
        return "params:\t{}\ndepth:\t{}\npartition size:\t{}\nerror:\t{}\nscore:\t{}\n".format(self.parameters,
                                                                                               self.depth,
                                                                                               self.partition_size,
                                                                                               self.error, self.score)

    def set_config_size(self, s):
        Node._config_size = s

    def calculate_partition_size(self):
        p = self.parameters
        res = 1
        for dict in p:
            for key in dict.keys():
                res *= len(dict.get(key))

        self.partition_size = res

    def calculate_pred_value(self):
        self.pred_value = np.mean(self.target)

    def calculate_score(self, weight_size):
        """
        Calculate the node's score.
        Partition size needs to be (re)calculated, in case the node was sampled.
        :param weight_size: determines the weight of the size of each partition compared to the error.
        """
        self.calculate_partition_size()
        weight_error = 1 - weight_size
        t = weight_error * self.error + weight_size * (self.partition_size / Node._config_size)

        # negate because of min heap and we want biggest score
        self.score = (-1) * t


class ONode(Node):
    """
    Node in Oblique Decision Tree.
    """

    def __init__(self, params, features, target, depth, idx):
        """
        :param params: list of dictionaries with possible parameter values for each vnf
        :param features: sampled configurations as flat 2D numpy array
        :param target: 1D numpy array of performance values of sampled configs
        :param depth: depth of node in whole tree
        """

        self.parameters = params
        self.features = features
        self.target = target
        self.left = None  # data below split line
        self.right = None  # data above split line
        self.depth = depth
        self.idx = idx
        self.split_vector = None
        self.split_improvement = 0
        self.pred_value = None
        self.error = None  # deviation from prediction. Smaller = better
        self.partition_size = None  # number of configs in partition
        self.score = None


class DecisionTree:
    """
    Decision Tree Base Class.
    """

    def __init__(self, parameters, sampled_configs, sample_results, error_metric='mse',
                 min_error_gain=0.001, max_depth=None, weight_size=0.2, min_samples_split=2, max_features_split=1.0):

        self._root = None
        self._depth = 1
        self.leaf_nodes = []  # needed for selection of node to sample, heapq heap of node scores
        self.max_depth = ((2 ** 31) - 1 if max_depth is None else max_depth)
        self.error_metric = error_metric
        self.min_samples_split = min_samples_split  # minimum number of samples a node needs to have for split
        self.min_error_gain = min_error_gain  # minimum improvement to do a split
        self.min_samples_leaf = 1  # minimum required number of samples within one leaf
        self.max_features_split = max_features_split  # consider only 30-40% of features for split search?
        self.weight_size = weight_size
        self.vnf_count = None
        self.feature_idx_to_name = {}  # maps indices of features rows to corresponding vnf and parameter
        self.last_sampled_node = None
        self.node_count = 1

        self._prepare_tree(parameters, sampled_configs, sample_results)

    def _prepare_tree(self, parameters, sampled_cfgs, sample_res):
        """
        Set root node, VNF count and Feature-index-to-name dictionary.
        """
        features = np.array(sampled_cfgs)
        target = np.array(sample_res)
        self.vnf_count = features.shape[1] // len(parameters)

        params = [dict(parameters)]
        if self.vnf_count != len(params):
            # if vnf_count is bigger than 1, append parameter dictionary for each vnf
            for vnf in range(1, self.vnf_count):
                params.append(dict(parameters))

        index = 0
        for vnf in range(len(params)):
            for key in params[vnf].keys():
                self.feature_idx_to_name[index] = (vnf, key)
                index += 1

        self._root = Node(params, features, target, 1, 0)

        # determine overall config space size for calculating score
        self._root.calculate_partition_size()
        self._root.set_config_size(self._root.partition_size)
        LOG.info("Decision Tree Model initialized.")

    def _determine_node_to_sample(self):
        """
        Determine which leaf node (and thus config partition) needs to be explored further.
        Done by returning leaf node with the lowest score value.
        Assumes that no node is sampled twice, since the node is split after sampling from it.
        """
        if not self.leaf_nodes:
            LOG.error("Decision Tree model has no leaf nodes to sample.")
            LOG.error("Exit programme!")
            exit(1)

        # remove node with lowest score from heap, will be split upon call of "adapt_tree"
        next_node = heapq.heappop(self.leaf_nodes)
        next_node = next_node[2]
        while self.leaf_nodes and (next_node.split_feature_index or next_node.depth == self.max_depth):
            next_node = heapq.heappop(self.leaf_nodes)
            next_node = next_node[2]

        if next_node.split_feature_index or next_node.depth == self.max_depth:
            LOG.debug("Decision Tree has reached its maximum depth.")

        self.last_sampled_node = next_node
        return next_node

    def _grow_tree_at_node(self, node):
        """
        Grow (sub)tree until defined termination criterion is reached. Initially called for root node.
        """
        if node.depth == self.max_depth or len(node.target) < self.min_samples_split:
            return  # stop growing

        # set node's split improvement, split feature and split value
        self._determine_best_split_of_node(node)

        if node.split_improvement < self.min_error_gain:
            return  # stop growing

        self._split_node(node)

        # depth first approach, does it matter?
        self._grow_tree_at_node(node.left)
        self._grow_tree_at_node(node.right)

    def _determine_best_split_of_node(self, node):
        """
        Given a node, determine the best feature and the best feature value to split the node.
        Error improvement, best feature and split value are set in the node object.
        """
        if node.error is None:
            node.error = self._calculate_partition_error(node.target)

        # Todo: only evaluate 40% (max_features_split) of features?
        feature_count = node.features.shape[1]
        for col in range(feature_count):
            cut, split_error = self._get_best_split_of_feature(node, col)
            if cut == split_error == -1:
                # Todo: better solution if feature not splittable?
                continue

            error_improvement = node.error - split_error

            if error_improvement > node.split_improvement:
                node.split_improvement = error_improvement
                node.split_feature_index = col
                node.split_feature_cut_val = cut

    def _get_best_split_of_feature(self, node, feature_idx):
        """
        Get a tuple of (cut value, cut error value) where new error value is minimal
        """
        split_vals = self._get_possible_splits(node.features, feature_idx)

        if len(split_vals) == 0:
            # no split possible
            return -1, -1

        sample_count = node.features.shape[0]
        split_error = {}
        for cut in split_vals:
            target_left_partition = node.target[node.features[:, feature_idx] <= cut]
            target_right_partition = node.target[node.features[:, feature_idx] > cut]

            error_split = self._get_after_split_error(target_left_partition, target_right_partition, sample_count)

            split_error[cut] = error_split
        # return cut value that belongs to biggest error improvement
        return min(split_error.items(), key=lambda x: x[1])

    def _get_possible_splits(self, features, feature_idx):
        """
        Returns a numpy array of the mean (midpoints) between values for a given array.
        """
        feature_vals = np.unique(features[:, feature_idx])
        return (feature_vals[:-1] + feature_vals[1:]) / 2.0

    def _get_after_split_error(self, left_target, right_target, sample_count):
        """
        Get the error value of the partitions that would be created by a split.
        """
        error_left_partition = self._calculate_partition_error(left_target)
        error_right_partition = self._calculate_partition_error(right_target)

        left_percentage = float(left_target.shape[0]) / sample_count
        right_percentage = 1 - left_percentage

        error_split = left_percentage * error_left_partition + right_percentage * error_right_partition
        return error_split

    def _split_node(self, node):
        """
        Split tree at given (leaf) node according to its defined split-feature und split-threshold value.
        Create two new leaf nodes with adjusted parameter, feature and target values.
        """
        # get all rows where the split value is less/equal or greater than cut value
        left_features = node.features[node.features[:, node.split_feature_index] <= node.split_feature_cut_val]
        left_target = node.target[node.features[:, node.split_feature_index] <= node.split_feature_cut_val]
        right_features = node.features[node.features[:, node.split_feature_index] > node.split_feature_cut_val]
        right_target = node.target[node.features[:, node.split_feature_index] > node.split_feature_cut_val]

        # adjust parameter values for childnodes
        params_left, params_right = self._calculate_new_parameters(node.parameters, node.split_feature_index,
                                                                   node.split_feature_cut_val)
        node.left = Node(params_left, left_features, left_target, node.depth + 1, self.node_count)
        node.right = Node(params_right, right_features, right_target, node.depth + 1, self.node_count + 1)
        self.node_count += 2

        if node.depth + 1 > self._depth:
            self._depth = node.depth + 1

        # calculate error for child nodes
        node.left.error = self._calculate_partition_error(node.left.target)
        node.right.error = self._calculate_partition_error(node.right.target)

        # calculate score for child nodes
        node.left.calculate_score(self.weight_size)
        node.right.calculate_score(self.weight_size)

        # add child nodes to leaf-node heap
        heapq.heappush(self.leaf_nodes, (node.left.score, node.left.idx, node.left))
        heapq.heappush(self.leaf_nodes, (node.right.score, node.right.idx, node.right))
        # Todo Does node not need to be removed from heap?

    def _calculate_new_parameters(self, params, param_index, cut_value):
        """
        Return two adjusted parameter arrays that remove parameter values below/above cut_value.
        """
        params_left = [dict(d) for d in params]
        params_right = [dict(d) for d in params]

        vnf_idx, param = self.feature_idx_to_name.get(param_index)
        values = params[vnf_idx].get(param)
        params_left[vnf_idx][param] = [val for val in values if val <= cut_value]
        params_right[vnf_idx][param] = [val for val in values if val > cut_value]

        return params_left, params_right

    def _calculate_partition_error(self, target):
        """
        Calculate the error value of a given node according to homogeneity metric (self.homog_metric)
        """
        # Todo: more? Std deviation?
        if self.error_metric == 'mse':  # same as mse? Lowest value = best
            # for each target in node, calculate error value from predicted node
            return np.mean((target - np.mean(target)) ** 2.0)
        LOG.error("Error metric {} not implemented.".format(self.error_metric))

    def _get_config_from_partition(self, node):
        """
        Given the node to sample from, randomly select a configuration from the node's partition space.
        Done by randomly choosing parameter values within the node's parameter thresholds.

        Config format should be: ({'c1': 1, 'c2': 1, 'c3': 1}, {'c1': 1, 'c2': 1, 'c3': 1})
        """
        c = []
        for dict in node.parameters:
            vnf = {}
            for param in dict.keys():
                vnf[param] = choice(dict.get(param))
            c.append(vnf)

        return tuple(c)

    def select_next(self):
        """
        Return next configuration to be profiled.
        """
        next_node = self._determine_node_to_sample()
        config = self._get_config_from_partition(next_node)
        return config

    def build_tree(self):
        """
        Build tree initially until termination criterion is reached.
        """
        self._grow_tree_at_node(self._root)

    def adapt_tree(self, sample):
        """
        Add new sample values (config and performance) to feature/target of node that has last been sampled and
        grow at that node. Re-Calculate the nodes' error value.

        :param sample: A tuple of a flat config (np.array) and a target value.
        """
        curr_node = self.last_sampled_node
        c, t = sample[0], sample[1]

        curr_node.features = np.append(curr_node.features, [c], axis=0)
        curr_node.target = np.append(curr_node.target, t)

        curr_node.error = self._calculate_partition_error(curr_node.target)
        self._grow_tree_at_node(curr_node)

    def prune_tree(self):
        """
        Optional. Prune tree after selection process is complete to see if the Decision Tree model yields
        better selections during tree construction compared to afterwards.
        """
        # Todo
        pass

    def get_tree(self):
        """
        Return Decision Tree model.
        """
        return self._root

    def print_tree(self, node, condition=""):
        """
        Print tree to STDOUT.
        """
        base = "   " * node.depth + condition
        if node.split_feature_index:
            print("%s if X[%s] <= %s" % (base, node.split_feature_index, node.split_feature_cut_val))
            self.print_tree(node.left, "then")
            self.print_tree(node.right, "else")

        else:
            node.calculate_pred_value()
            print("%s <value: %s, samples in partition: %s>" % (base, node.pred_value, node.partition_size))


class ObliqueDecisionTree(DecisionTree):

    # Todo: Function split_samples, welche samples nach cut-value splitted (both feature and target!)

    def _prepare_tree(self, parameters, sampled_cfgs, sample_res):
        """
        Set root node, VNF count and Feature-index-to-name dictionary.
        """
        features = np.array(sampled_cfgs)
        target = np.array(sample_res)
        self.vnf_count = features.shape[1] // len(parameters)

        params = [dict(parameters)]
        if self.vnf_count != len(params):
            # if vnf_count is bigger than 1, append parameter dictionary for each vnf
            for vnf in range(1, self.vnf_count):
                params.append(dict(parameters))

        index = 0
        for vnf in range(len(params)):
            for key in params[vnf].keys():
                self.feature_idx_to_name[index] = (vnf, key)
                index += 1

        self._root = ONode(params, features, target, 1, 0)

        # determine overall config space size for calculating score
        self._root.calculate_partition_size()
        self._root.set_config_size(self._root.partition_size)
        LOG.info("Decision Tree Model initialized.")

    def _determine_best_split_of_node(self, node):
        """
        Determines oblique split line for a given node and sets it in the node object
        as well as the resulting improvement in homogeneity.

        :param node: The node to be split.
        """
        feature_count = node.features.shape[1]
        splits = self._get_best_splits_for_all_features(node)
        # get feature index and resulting error of split with lowest error
        index, split = min(enumerate(splits), key=lambda x: x[1][1])

        # initiate split_vector to enable the linear combination split
        node.split_vector = np.zeros((node.features.shape[1],))

        # set last elem in split_vector to negated cut value
        node.split_vector[-1] = -split[0]
        node.split_vector[index] = 1

        # TODO really ugly, change! Maybe add last feature col that saves position? Keep features and target in one Array!
        # get feature partitions
        features_above = np.zeros((1, node.features.shape[1]))
        features_below = np.zeros((1, node.features.shape[1]))
        target_right_partition = np.zeros((1, 1))
        target_left_partition = np.zeros((1, 1))

        for row in node.features:
            config_pos = self._check_config_position(row, node.split_vector)
            if config_pos > 0:
                features_above = np.vstack((features_above, row))
                target_right_partition = np.vstack((node.target[row]))
            else:
                features_below = np.vstack((features_below, row))
                target_left_partition = np.vstack((node.target[row]))

        features_above = np.delete(features_above, 0, axis=0)
        features_below = np.delete(features_below, 0, axis=0)
        target_right_partition = np.delete(target_right_partition, 0, axis=0)
        target_left_partition = np.delete(target_left_partition, 0, axis=0)

        sample_count = node.features.shape[0]
        error_split = self._get_after_split_error(target_left_partition, target_right_partition, sample_count)

        # perturb a random feature in split vector 10 times
        for c in range(10):
            feature_idx = randint(0, feature_count)
            error_split, split_vector = self._perturb_hyperplane_coefficients(node, feature_idx, error_split)

        # Todo: set node values (error, split vector...)

    def _get_best_splits_for_all_features(self, node):
        feature_count = node.features.shape[1] # -1 one if we add target
        result = [self._get_best_split_of_feature(node, i) for i in range(feature_count)]
        return np.array(result)

    def _calculate_U_value(self, row, split_vector, feature_idx):
        """
        Calculate Uj as in p.10 of Murthy.
        Checks if row is below or above split vector? (returns either pos or neg?)
        """
        top = split_vector[feature_idx] * row[feature_idx] - self._check_config_position(row, split_vector)
        return top / row[feature_idx]

    def _perturb_hyperplane_coefficients(self, node, feature_idx, prev_error_val):
        # Todo: besser error val in node aendern und nicht in vars hier? also kein return value fuer error? Same for split vect
        # calculate all values of U with the current value for feature_idx in split_vector
        u_values = np.array(sorted([[self._calculate_U_value(row, node.split_vector, feature_idx)] for row in node.features]))

        # possible splits are the midpoints between Uj values
        possible_cuts = self._get_possible_splits(u_values, 0)

        # Find best split in possible splits
        coefficients = {}
        for cut in possible_cuts:
            new_split_vector = np.array(node.split_vector)
            new_split_vector[feature_idx] = cut
            # todo: siehe oben, schreibe function zum splitten von feature/target arrays. Ggf beides in einem?
            low, high = self._split_samples(node.features, new_split_vector)
            error_after_split = self._get_after_split_error(low, high, node.features.shape[1])
            coefficients[cut] = (error_after_split, new_split_vector)

        best_error_val, best_split_vector = min(coefficients.values(), key=lambda x: x[0])
        if best_error_val > prev_error_val:
            return best_error_val, best_split_vector
        elif best_error_val == prev_error_val:
            # 0.3 is P_stag, the stagnation probability. Determines if hyperplane should be perturbed
            # even if the impurity measure of new H is the same as before
            if random() < 0.3:
                return best_error_val, best_split_vector
        return prev_error_val, node.split_vector

    def _check_config_position(self, row, split_vector):
        """
        Calculates Vj as in p.10 of Murthy. All configs in the same partition have the same sign?

        Used to check if row is above or below vector in _split_data()
        """
        temp = np.multiply(row, split_vector[:-1])
        # add negated cut value to sum
        return np.sum(temp) + split_vector[-1]

    def _split_samples(self, samples, split_vector):
        # Todo: split feature and target according to split vector
        below = above = None
        return below, above

    def _split_node(self, node):
        # Todo split node  by creating corresponding feature and target arrs,
        # calculate child nodes errors and scores, push children to heap
        # creates the lower and upper partitions (two row-partitions)? according to split vector
        pass

    def _calculate_new_parameters(self, params, param_index, cut_value, split_vector=None):
        # Todo determine where split line cuts the feature values? --> split vec has cut value for each attr?
        # Todo: override method nur bei Erhalt der Signature --> arbeite mit optionalen args oder **kawrgs?
        # Or call for every param? nicht so schoen
        """
        Return two adjusted parameter arrays that remove parameter values below/above split_vector.
        """
        params_left = [dict(d) for d in params]
        params_right = [dict(d) for d in params]

        for i in range(len(split_vector) - 1):
            vnf_idx, param = self.feature_idx_to_name.get(i)
            values = params[vnf_idx].get(param)
            cut_value = split_vector[i]
            params_left[vnf_idx][param] = [val for val in values if val <= cut_value]
            params_right[vnf_idx][param] = [val for val in values if val > cut_value]

        return params_left, params_right
