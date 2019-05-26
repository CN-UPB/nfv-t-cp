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
from random import randint, random, choice, sample

LOG = logging.getLogger(os.path.basename(__file__))


def log_error(reason):
    LOG.error(reason)
    LOG.error("Exit programme!")
    exit(1)


class Node:
    """
    Base Class for Decision Tree Nodes.
    """

    def __init__(self, params, features, target, depth, idx, error):
        """
        :param params: list of dictionaries with possible parameter values for each vnf
        :param features: sampled configurations as flat 2D numpy array
        :param target: 1D numpy array of performance values of sampled configs
        :param depth: depth of node in whole tree
        """

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
        self.error = error  # deviation from prediction. Smaller = better
        self.partition_size = self._get_partition_size()  # number of configs in partition
        self.score = None

    def __str__(self):
        return "params:\t{}\ndepth:\t{}\npartition size:\t{}\nerror:\t{}\nscore:\t{}\n".format(self.parameters,
                                                                                               self.depth,
                                                                                               self.partition_size,
                                                                                               self.error, self.score)

    def _normalize_val(self, val, min_val, max_val):
        """
        Used to normalize the partition size and error values of each node
        """
        # TODO: case max_val == min_val? Nur ein leaf node, partition sizes are the same, error vals are the same
        upper, lower = (val - min_val), (max_val - min_val)
        if upper == 0:
            return 0
        if lower == 0:
            return upper
        return upper / lower

    def _get_partition_size(self):
        res = 1
        for vnf in self.parameters:
            for key in vnf.keys():
                res *= len(vnf.get(key))

        return res

    def calculate_score(self, weight_size, min_p, max_p, min_e, max_e):
        """
        Calculate the node's score.
        Partition size needs to be (re)calculated, in case the node was sampled.
        :param weight_size: determines the weight of the size of each partition compared to the error.
        """
        weight_error = 1 - weight_size
        normalized_error = self._normalize_val(self.error, min_e, max_e)
        normalized_size = self._normalize_val(self.partition_size, min_p, max_p)
        self.score = weight_error * normalized_error + weight_size * normalized_size

    def is_leaf_node(self):
        """
        Check if node is a leaf node.
        """
        return self.split_feature_index is None


class ONode(Node):
    """
    Node in Oblique Decision Tree.
    """

    def __init__(self, config_part, features, target, depth, idx, error):
        """
        :param params: list of dictionaries with possible parameter values for each vnf
        :param features: sampled configurations as flat 2D numpy array
        :param target: 1D numpy array of performance values of sampled configs
        :param depth: depth of node in whole tree
        """

        self.config_partition = config_part
        self.features = features
        self.target = target
        self.left = None  # data below split line
        self.right = None  # data above split line
        self.depth = depth
        self.idx = idx
        self.split_vector = None
        self.split_improvement = 0
        self.error = error  # deviation from prediction. Smaller = better
        self.partition_size = len(self.config_partition)  # number of configs in partition
        self.score = None

    def __str__(self):
        return "depth:\t{}\nfeatures:\t{}\ntarget:\t{}\npartition size:\t{}\nerror:\t{}\nscore:\t{}\nvector:\t{}\npartitionNone:\t{}\n".format(
            self.depth,
            self.features,
            self.target,
            self.partition_size,
            self.error,
            self.score,
            self.split_vector,
            self.config_partition is None)

    def is_leaf_node(self):
        return self.split_vector is None


class DecisionTree:
    """
    Decision Tree Base Class.
    """

    # Todo: asserts statt if then error?

    def __init__(self, parameters, feature, target, **kwargs):
        self.p = {"max_depth": ((2 ** 31) - 1),
                  "weight_size": 0.6,  # weight of the partition size
                  "min_samples_split": 2,  # minimum number of samples a node needs to have for split
                  "max_features_split": 1.0,  # percentage of features to be considered for split
                  "error_metric": "mse"}
        self.p.update(kwargs)

        self.parameters_unique = parameters
        self.vnf_count = len(feature) // len(self.parameters_unique)
        self._root = None
        self._depth = 1
        self.leaf_nodes = dict()  # hashmap of node.idx --> (node)
        self.feature_idx_to_name = {}  # maps indices of features to corresponding vnf and parameter
        self.params_per_vnf = self._duplicate_parameters_for_vnfs()
        self.last_sampled_node = None
        self.node_count = 0

        self._prepare_tree(feature, target)

        LOG.info("Decision Tree Model initialized.")

    def _prepare_tree(self, feature, target):
        """
        Set root node, VNF count and Feature-index-to-name dictionary.
        """
        features = np.array([feature])
        target = np.array([target])

        self._prepare_index_to_vnf_mapping(self.params_per_vnf)

        err = self._calculate_prediction_error(target)
        self._root = Node(self.params_per_vnf, features, target, 1, self.node_count, err)
        self.last_sampled_node = self._root
        self.node_count += 1
        self.leaf_nodes[self._root.idx] = self._root

    def _duplicate_parameters_for_vnfs(self):
        params = [dict(self.parameters_unique)]
        if self.vnf_count != len(params):
            # if vnf_count is bigger than 1, append parameter dictionary for each vnf
            for vnf in range(1, self.vnf_count):
                params.append(dict(self.parameters_unique))
        return params

    def _prepare_index_to_vnf_mapping(self, params):
        index = 0
        for vnf in range(len(params)):
            for key in params[vnf].keys():
                self.feature_idx_to_name[index] = (vnf, key)
                index += 1

    def _get_normalization_boundaries(self):
        """
        Get the minimum/maximum parition size and error values.
        """
        min_partition_size = max_partition_size = min_error = max_error = -1
        for node_id in self.leaf_nodes:
            curr_node = self.leaf_nodes[node_id]
            if min_partition_size == -1:
                # initialize boundaries
                min_partition_size = max_partition_size = curr_node.partition_size
                min_error = max_error = curr_node.error
            else:
                # update boundaries
                min_partition_size = min(min_partition_size, curr_node.partition_size)
                max_partition_size = max(max_partition_size, curr_node.partition_size)
                min_error = min(min_error, curr_node.error)
                max_error = max(max_error, curr_node.error)

        return min_partition_size, max_partition_size, min_error, max_error

    def _determine_node_to_sample(self):
        """
        Determine which leaf node (and thus config partition) needs to be explored further.
        Done by returning leaf node with the highest score value.
        """
        next_node = None
        max_score = 0
        min_partition, max_partition, min_error, max_error = self._get_normalization_boundaries()

        for node_id in self.leaf_nodes:
            curr_node = self.leaf_nodes[node_id]
            curr_node.calculate_score(self.p.get("weight_size"), min_partition, max_partition, min_error, max_error)
            if curr_node.score > max_score or next_node is None:
                next_node = curr_node
                max_score = curr_node.score

        self.last_sampled_node = next_node
        return next_node

    def _grow_tree_at_node(self, node):
        """
        Grow (sub)tree until defined termination criterion is reached. Initially called for root node.
        """
        # check termination criterion and if node needs to be pushed to leaf nodes
        if node.depth == self.p.get("max_depth") or len(node.target) < self.p.get("min_samples_split"):
            return  # stop growing

        # set node's split improvement and split values
        self._determine_best_split_of_node(node)

        if node.is_leaf_node() is True:
            LOG.debug("It's not possible to split this node further.")
            return

        self._split_node(node)

        # depth first approach
        self._grow_tree_at_node(node.left)
        self._grow_tree_at_node(node.right)

    def _determine_best_split_of_node(self, node):
        """
        Given a node, determine the best feature and the best feature value to split the node.
        Error improvement, best feature and split value are set in the node object.
        """
        feature_count = node.features.shape[1]
        included_features = list(range(feature_count))

        if self.p.get("max_features_split") < 1.0:
            # only look at a percentage of the features
            reduced_count = int(feature_count * self.p.get("max_features_split"))
            included_features = sample(range(0, feature_count - 1), reduced_count)

        for col in included_features:
            cut, split_error = self._get_best_split_of_feature(node.features, node.target, col)
            if cut != -1 and split_error != -1:
                error_improvement = node.error - split_error

                if error_improvement > node.split_improvement:
                    node.split_improvement = error_improvement
                    node.split_feature_index = col
                    node.split_feature_cut_val = cut

    def _get_best_split_of_feature(self, features, target, feature_idx):
        """
        Get a tuple of (cut value, cut error value) where new error value is minimal
        """
        split_vals = self._get_possible_splits(features, feature_idx)

        sample_count = features.shape[0]
        split_error = {}
        for cut in split_vals:
            target_left_partition = target[features[:, feature_idx] <= cut]
            target_right_partition = target[features[:, feature_idx] > cut]

            if len(target_left_partition) != 0 and len(target_left_partition) != len(target):
                error_split = self._get_after_split_error(target_left_partition, target_right_partition, sample_count)
                split_error[cut] = error_split

        if len(split_error) == 0:
            # no split possible
            return -1, -1

        return min(split_error.items(), key=lambda x: x[1])

    def _get_possible_splits(self, features, feature_idx):
        """
        Returns a numpy array of the mean (midpoints) between values for a given array.
        """
        # get unique feature value occurances
        feature_vals = np.unique(features[:, feature_idx])
        return (feature_vals[:-1] + feature_vals[1:]) / 2.0

    def _get_after_split_error(self, left_target, right_target, sample_count):
        """
        Get the error value of the partitions that would be created by a split.
        """
        error_left_partition = self._calculate_prediction_error(left_target)
        error_right_partition = self._calculate_prediction_error(right_target)

        if left_target.shape[0] == 0:
            return error_right_partition
        elif right_target.shape[0] == 0:
            return error_left_partition
        else:
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
        left_f, left_t, right_f, right_t = self._split_samples(node.features, node.target,
                                                               feature_idx=node.split_feature_index,
                                                               cut_val=node.split_feature_cut_val)

        # adjust parameter values for childnodes
        params_left, params_right = self._calculate_new_parameters(node.parameters, node.split_feature_index,
                                                                   node.split_feature_cut_val)

        # calculate error for child nodes
        left_error = self._calculate_prediction_error(left_t)
        right_error = self._calculate_prediction_error(right_t)

        # create child nodes
        node.left = Node(params_left, left_f, left_t, node.depth + 1, self.node_count, left_error)
        node.right = Node(params_right, right_f, right_t, node.depth + 1, self.node_count + 1, right_error)
        self.node_count += 2
        self.node_count = self.node_count

        if node.depth + 1 > self._depth:
            self._depth = node.depth + 1

        # adjust leaf nodes
        if node.idx in self.leaf_nodes:
            del self.leaf_nodes[node.idx]
        self.leaf_nodes[node.left.idx] = node.left
        self.leaf_nodes[node.right.idx] = node.right

    def _split_samples(self, features, target, **kwargs):
        """
        Split Features and Targets according to Feature and its Cut Value.
        """
        if "feature_idx" not in kwargs or "cut_val" not in kwargs:
            log_error("Can't split samples without feature and cut value.")

        feature_idx, cut_val = kwargs["feature_idx"], kwargs["cut_val"]
        left_f = features[features[:, feature_idx] <= cut_val]
        left_t = target[features[:, feature_idx] <= cut_val]
        right_f = features[features[:, feature_idx] > cut_val]
        right_t = target[features[:, feature_idx] > cut_val]
        return left_f, left_t, right_f, right_t

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

    def _calculate_prediction_error(self, target):
        """
        Calculate the error value of a given node according to homogeneity metric.
        """
        if len(target) == 0:
            LOG.debug("Calculating error for empty partition.")
            return 0

        prediction = np.mean(target)
        if self.p.get("error_metric") == "mse":
            return np.mean((target - prediction) ** 2.0)
        elif self.p.get("error_metric") == "mae":
            return np.mean(np.absolute(target - prediction))
        else:
            log_error("Error metric {} not supported for DecisionTree".format(self.p.get("error_metric")))

    def _get_config_from_partition(self, node):
        """
        Given the node to sample from, randomly select a configuration from the node's partition space.

        Config format should be: ({'c1': 1, 'c2': 1, 'c3': 1}, {'c1': 1, 'c2': 1, 'c3': 1})
        """
        res = self._reconstruct_random_config(node.parameters)
        LOG.debug("Selected config: ()".format(res))
        return res

    def _reconstruct_random_config(self, parameters):
        """
        Randomly choose parameter values within the specified parameter thresholds.
        """
        sfc_config = []
        for vnf in parameters:
            vnf_config = {}
            for param in vnf.keys():
                vnf_config[param] = choice(vnf.get(param))
            sfc_config.append(vnf_config)
        return tuple(sfc_config)

    def select_next(self):
        """
        Return next configuration to be profiled.
        """
        if self._root is None:
            config = self._reconstruct_random_config(self.params_per_vnf)
        else:
            next_node = self._determine_node_to_sample()
            config = self._get_config_from_partition(next_node)
        return config

    def adapt_tree(self, sample):
        """
        Add new sample values (config and performance) to feature/target of node that has last been sampled and
        grow at that node. If no node has been sampled before, initialize tree.

        :param sample: A tuple of a flat config and a target value.
        """
        config, performance = sample[0], sample[1]

        curr_node = self.last_sampled_node
        curr_node.features = np.append(curr_node.features, [config], axis=0)
        curr_node.target = np.append(curr_node.target, performance)
        curr_node.error = self._calculate_prediction_error(curr_node.target)

        self._grow_tree_at_node(curr_node)

    def get_tree(self):
        """
        Return Decision Tree model.
        """
        return self._root

    def print_tree(self, node: Node, condition=""):
        """
        Print tree to STDOUT.
        """
        if node is not None:
            print(condition)
            print(str(node))
            if node.split_feature_index is not None:
                cond = ("X[%s] <= %s" % (node.split_feature_index, node.split_feature_cut_val))
                self.print_tree(node.left, ("if " + cond))
                self.print_tree(node.right, ("if not " + cond))


class ObliqueDecisionTree(DecisionTree):

    def _prepare_tree(self, feature, target):
        """
        Set root node, VNF count and Feature-index-to-name dictionary.
        """
        features = np.array([feature])
        target = np.array([target])
        self.stagnation_probability = 0.3 if self.p.get("p_stag") is None else self.p.get("p_stag")

        self._prepare_index_to_vnf_mapping(self.params_per_vnf)

        if self.p.get("config_space") is None:
            log_error("Configuration Space needs to be specified for oblique tree.")

        c_space = np.array(self.p.get("config_space"))
        err = self._calculate_prediction_error(target)
        self._root = ONode(c_space, features, target, 1, self.node_count, err)
        self.node_count += 1
        self.last_sampled_node = self._root
        self.leaf_nodes[self._root.idx] = self._root

    def _determine_best_split_of_node(self, node: ONode):
        """
        Determines oblique split line for a given node and sets it in the node object
        as well as the resulting improvement in homogeneity.

        :param node: The node to be split.
        """
        sample_count, feature_count = node.features.shape

        # get feature index with lowest error and its corresponding tuple of cut value and its error
        feature_idx, cut_val = self._get_feature_with_best_split(node.features, node.target)

        if feature_idx == -1 or cut_val == -1:
            node.split_improvement = 0
            node.split_vector = None
            LOG.info("It's not possible to split this node further.")
            return

        # initiate split_vector to enable the linear combination split, last elem is the cut value
        node.split_vector = np.zeros((feature_count + 1,))
        node.split_vector[-1] = cut_val
        node.split_vector[feature_idx] = 1

        # split partition by split vector
        left_f, right_f, left_t, right_t = self._split_samples(node.features, node.target,
                                                               split_vector=node.split_vector)

        if node.error is None:
            node.error = self._calculate_prediction_error(node.target)

        error_split = self._get_after_split_error(left_t, right_t, sample_count)
        node.split_improvement = max(node.split_improvement, (node.error - error_split))

        # perturb a random feature of the split vector 10 times
        for i in range(10):
            feature_idx = randint(0, feature_count - 1)
            # updates node's split vector and error improvement to the vector with minimal error
            self._perturb_hyperplane_coefficients(node, feature_idx)

    def _get_feature_with_best_split(self, features, target):
        """
        Return the feature index with minimal error value and its split value.
        """
        feature_idx = split_val = min_error = -1

        feature_count = features.shape[1]
        for index in range(feature_count):
            cut, split_error = self._get_best_split_of_feature(features, target, index)
            # if there is a possible split and if it's the best split so far
            if split_error != -1 and (min_error == -1 or split_error < min_error):
                min_error = split_error
                feature_idx, split_val = index, cut

        return feature_idx, split_val

    def _calculate_U_value(self, row, split_vector, feature_idx):
        """
        Calculate U_j values of OC1 algorithm.
        """
        upper = split_vector[feature_idx] * row[feature_idx] - self._check_config_position(row, split_vector)
        if upper == 0:
            return 0
        if row[feature_idx] == 0:
            return upper
        return upper / row[feature_idx]

    def _perturb_hyperplane_coefficients(self, node: ONode, feature_idx):
        # calculate all values of U with the current value for feature_idx in split_vector
        u_values = [[self._calculate_U_value(row, node.split_vector, feature_idx)] for row in node.features]
        u_values = np.array(sorted(u_values))

        # possible splits are the midpoints between Uj values
        possible_cuts = self._get_possible_splits(u_values, 0)

        # Find best split (minimal error) of U_j values in possible splits
        best_split_vector = np.array(node.split_vector)
        min_error = node.error
        for cut in possible_cuts:
            new_split_vector = np.array(node.split_vector)
            new_split_vector[feature_idx] = cut

            low_f, high_f, low_t, high_t = self._split_samples(node.features, node.target,
                                                               split_vector=new_split_vector)
            error_after_split = self._get_after_split_error(low_t, high_t, node.features.shape[1])
            if error_after_split < min_error:
                min_error = error_after_split
                best_split_vector = new_split_vector

        # set node properties to best split
        best_improvement = node.error - min_error
        if best_improvement > node.split_improvement:
            node.split_vector = best_split_vector
            node.split_improvement = node.error - min_error
        elif best_improvement == node.split_improvement and random() < self.stagnation_probability:
            # stagnation probability determines if hyperplane should still be perturbed
            node.split_vector = best_split_vector

    def _check_config_position(self, row, split_vector):
        """
        Calculates V_j of OC1 algorithm, the sign of the sample.
        Used to check if a configuration is above or below hyperplane.
        (split_vector[-1] is the cut value of the best feature split)
        """
        temp = np.multiply(row, split_vector[:-1])
        return np.sum(temp) - split_vector[-1]

    def _split_samples(self, features, target, **kwargs):
        """
        Split Features and Targets according to split vector.
        """
        if "split_vector" not in kwargs:
            log_error("Can't split samples without split vector.")

        split_vector = kwargs["split_vector"]
        sample_count, feature_count = features.shape

        samples = np.concatenate((features, target.reshape(-1, 1)), axis=1)
        samples_above = np.zeros((1, feature_count + 1))
        samples_below = np.zeros((1, feature_count + 1))

        for row in samples:
            config_pos = self._check_config_position(row[:-1], split_vector)
            if config_pos > 0:
                samples_above = np.vstack((samples_above, row))
            else:
                samples_below = np.vstack((samples_below, row))

        samples_above = np.delete(samples_above, 0, axis=0)
        samples_below = np.delete(samples_below, 0, axis=0)
        return samples_below[:, :-1], samples_above[:, :-1], samples_below[:, -1], samples_above[:, -1]

    def _split_node(self, node: ONode):
        """
        Split tree at given (leaf) node according to its defined split vector.
        Create two new leaf nodes with adjusted partition, feature and target values.
        """
        low_f, high_f, low_t, high_t = self._split_samples(node.features, node.target, split_vector=node.split_vector)
        partition_left, partition_right = self._split_config_space(node.config_partition, node.split_vector)

        # delete partition of parent
        node.config_partition = None

        # calculate error for child nodes
        left_error = self._calculate_prediction_error(low_t)
        right_error = self._calculate_prediction_error(high_t)

        node.left = ONode(partition_left, low_f, low_t, node.depth + 1, self.node_count, left_error)
        node.right = ONode(partition_right, high_f, high_t, node.depth + 1, self.node_count + 1, right_error)
        self.node_count += 2
        self.node_count = self.node_count

        if node.depth + 1 > self._depth:
            self._depth = node.depth + 1

        # adjust leaf nodes
        if node.idx in self.leaf_nodes:
            del self.leaf_nodes[node.idx]
        self.leaf_nodes[node.left.idx] = node.left
        self.leaf_nodes[node.right.idx] = node.right

    def _split_config_space(self, config_space_partition, split_vector):
        """
        Split the configuration space partition.
        """
        # Todo: Nicer way to do this?
        config_count, feature_count = config_space_partition.shape

        partition_above = np.zeros((1, feature_count))
        partition_below = np.zeros((1, feature_count))

        for row in config_space_partition:
            config_pos = self._check_config_position(row, split_vector)
            if config_pos > 0:
                partition_above = np.vstack((partition_above, row))
            else:
                partition_below = np.vstack((partition_below, row))

        partition_above = np.delete(partition_above, 0, axis=0)
        partition_below = np.delete(partition_below, 0, axis=0)
        return partition_below, partition_above

    def _get_config_from_partition(self, node):
        idx = np.random.randint(0, len(node.config_partition))
        selected_config_flat = node.config_partition[idx]

        sfc_config, feature_idx = [], 0
        for vnf in self.params_per_vnf:
            vnf_config = {}
            for param in vnf:
                vnf_config[param] = selected_config_flat[feature_idx]
                feature_idx += 1
            sfc_config.append(vnf_config)
        res = tuple(sfc_config)
        LOG.debug("Selected config: ()".format(res))
        return res

    def print_tree(self, node: ONode, condition=""):
        """
        Print tree to STDOUT.
        """
        if node is not None:
            print(condition)
            print(str(node))
            if node.split_vector is not None:
                split_cond = ""
                for i in range(len(node.split_vector) - 1):
                    if node.split_vector[i] != 0:
                        split_cond += "f{}*{} + ".format(i, node.split_vector[i])
                split_cond = split_cond[:-2] + "< {}".format(node.split_vector[-1])
                self.print_tree(node.left, ("if " + split_cond))
                self.print_tree(node.right, ("if not " + split_cond))
