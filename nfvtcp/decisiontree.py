import logging
import os
import numpy as np
import heapq

LOG = logging.getLogger(os.path.basename(__file__))


class Node:
    """
    Base Class for Decision Tree Nodes.
    """

    def __init__(self, params, features, target, depth):
        # Todo: Delete feature and target values if node is no leaf no save memory? (Can be recalculated for pruning)
        self.parameters = params  # values a feature can have
        self.features = features  # configuration values
        self.target = target  # performance values
        self.left = None
        self.right = None
        self.depth = depth
        self.homogeneity = None
        self.split_feature_index = None
        self.split_feature_cut_val = None
        self.split_improvement = 0.0
        self.pred_value = np.mean(target)


class DecisionTree:
    """
    Decision Tree Base Class.
    """

    def __init__(self, configs, parameters, features, target, regression='default', homog_metric='mse',
                 min_homogeneity_gain=0.05, max_depth=10,
                 min_samples_split=2):
        """
        Grow initial DT.
        """
        self._root = Node(parameters, features, target, depth=1)
        self.leaf_nodes = {self._root}  # needed for selection of config to profile
        self.max_depth = ((2 ** 31) - 1 if max_depth is None else max_depth)
        self.regression = regression  # default DT, oblique, svm?
        self.homog_metric = homog_metric
        self.config_space = configs  # should also be flat
        self.min_samples_split = min_samples_split  # minimum number of samples a node needs to have for split
        self.min_homogeneity_gain = min_homogeneity_gain  # minimum improvement to do a split
        # self.regression_technique = regression
        # self.min_samples_leaf = 1   # minimum required number of samples within one leaf
        # self.max_features_split = np.shape(features)    # consider only 30-40% of features for split search?

        # grow tree initially
        self._grow_tree_at_node(self._root)

    def _grow_tree_at_node(self, node):
        """
        Grow (sub)tree until defined termination definition is reached. Initially called for root node
        """
        if node.depth == self.max_depth or len(node.target) < self.min_samples_split:
            return  # stop growing

        if self.regression == 'default':
            self._determine_best_split_of_node(node)
        elif self.regression == 'oblique':
            pass  # Todo: simulated annealing? (statt x < 2, e.g. 2x + y > 3)
        else:
            # Todo: support more regression= split ways, e.g. svm?
            LOG.error("DT Regression technique '{}‘ not supported.".format(str(self.regression)))
            LOG.error("Exit!")
            exit(1)

        if node.split_improvement < self.min_homogeneity_gain:
            return  # stop growing

        self._split_node(node)

        # depth first approach, does it matter?
        self._grow_tree_at_node(node.left)
        self._grow_tree_at_node(node.right)

    def _determine_best_split_of_node(self, node):
        """
        Given a node, determine the best feature and the best feature value to split the node.
        Homogeneity improvement, best feature and split value are set in the node object.
        """
        node.homogeneity = self._calculate_node_homogeneity(node.target)
        feature_count = node.features.shape[1]
        sample_count = node.features.shape[0]

        # Todo: only evaluate 40% of features?
        for col in range(feature_count):
            # get all unique values for this feature in current node
            feature_vals = np.unique(node.features[:, col])
            # get all possible cuts for that feature (mean of two possible values) - assumes that features are sorted!
            cuts = (feature_vals[:-1] + feature_vals[1:]) / 2.0

            for cut in cuts:
                target_left_partition = node.target[node.features[:, col] <= cut]
                target_right_partition = node.target[node.features[:, col] > cut]

                homog_left_partition = self._calculate_node_homogeneity(target_left_partition)
                homog_right_partition = self._calculate_node_homogeneity(target_right_partition)

                left_percentage = float(target_left_partition.shape[0]) / sample_count
                right_percentage = 1 - left_percentage

                homogeneity_split = left_percentage * homog_left_partition + right_percentage * homog_right_partition
                homog_improvement = node.homogeneity - homogeneity_split
                if homog_improvement > node.split_improvement:
                    node.split_improvement = homog_improvement
                    node.split_feature_index = col
                    node.split_feature_cut_val = cut

    def _split_node(self, node):
        """
        Split tree at given (leaf) node according to its defined split-feature und split-threshold value.
        """
        # get all rows where the split value is less or equal than threshold and grow left node
        left_features = node.features[node.features[:, node.split_feature_index] <= node.split_feature_cut_val]
        left_target = node.target[node.features[:, node.split_feature_index] <= node.split_feature_cut_val]
        node.left = Node(left_features, left_target, node.depth + 1)

        # get all rows where the split value is greater than threshold and grow right node
        right_features = node.features[node.features[:, node.split_feature_index] > node.split_feature_cut_val]
        right_target = node.target[node.features[:, node.split_feature_index] > node.split_feature_cut_val]
        node.right = Node(right_features, right_target, node.depth + 1)

        self.leaf_nodes.remove(node)
        self.leaf_nodes.add(node.left)
        self.leaf_nodes.add(node.right)

    def _calculate_node_homogeneity(self, target):
        """
        Calculate the homogeneity of a given node according to homogeneity metric (self.homog_metric)
        """
        # Todo: more? Std deviation?
        if self.homog_metric == 'var-reduction':  # same as mse? Lowest value = best
            return np.mean((target - np.mean(target)) ** 2.0)

    def _determine_node_to_sample(self):  # Es wird immer an dem Node gesplitted, dem das neue Sample zugeordnet wird?
        # Todo: find node with lowest accuracy/homogeneity!
        # Todo: factor size of node in! (size = configs nicht samples!) --> can be calculated through parameters!!
        # Todo: ggf priority queue mit homogeneity werten --> heapq aber *(-1) da min heap
        pass

    def _get_config_from_partition(self, node):
        # Todo: traverse back to root and save/calculate feature intervals, and select config by randomly choosing param values in interval
        c = None
        return c

    def select_next(self):
        """
        Return next configuration to be profiled.
        """
        next_node = self._determine_node_to_sample()
        config = self._get_config_from_partition(next_node)
        return config

    def adapt_tree(self, sample):
        pass
        # Todo: take new sample, run (predict) through tree til leaf node --> grow at that node
        # Problem: recalculate all impurity values of decision nodes, da sample size größer? sample zu jedem feature/target space hinzufügen?

    def prune_tree(self):
        # Todo: Prune tree, called afterwards?
        pass

    def get_tree(self):
        """
        If tree is used again after initial selection process.

        :return: Decision Tree Model.
        """
        return self._root
