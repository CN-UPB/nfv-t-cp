import logging
import os
import numpy as np

LOG = logging.getLogger(os.path.basename(__file__))


class Node:
    """
    Base Class for Decision Tree Nodes.
    """
    def __init__(self, features, target, depth):
        # Todo: Delete feature and target values if node is no leaf no save memory? (Can be recalculated for pruning)
        self.features = features # configuration values
        self.target = target # performance values
        self.left = None
        self.right = None
        self.depth = depth
        self.split_feature_index = None
        self.split_feature_cut_val = None
        self.pred_value = None
        self.accuracy = 0


class DTree:
    """
    Decision Tree Base Class.
    """

    def __init__(self, configs, features, target, split_metric='mse', max_depth=10, regression='linear', prune=False):
        """
        Grow initial DT.
        :param features: Dictionary of features and possible values.
        :param intial_samples: Already profiled configurations and their performance value.
        :param split_metric: Homogeneity metric.
        :param prune: Prune Tree?
        :param max_depth: Maximal depth of tree.
        """
        self._root = Node(features, target, depth=1)
        self.leaf_nodes = {self._root}
        self.max_depth = ((2 ** 31) - 1 if max_depth is None else max_depth)
        self.split_metric = split_metric
        self.regression_technique = regression
        self.config_space = configs # should also be flat

    def _grow_tree_at_node(self, node):
        # Todo: grow tree initially
        # Todo: check if node.depth == max_depth
        # Todo: find best feature/cut/gain in leaf nodes here
        pass

    def _regression(self, node, regression_technique='linear'):
        # Todo: implement more regression techniques: svm...
        # Todo: use regression to find more interesting configs in partition?
        # Todo: umbennen?
        if regression_technique == 'linear':
            return np.mean(node.target)

    def _find_best_split(self, r='linear'):
        # Todo: check if split necessary (is there improvement?)
        # Todo: support multiple split metrics (MSE, Variance Reduction)
        # Todo: ggf. simulated annealing for better split line
        pass
        # get all performance values out of samples
        # check if there is only one possible value, if so just return
        # else calculate mean of all target values
        # berechne impurity (mse)

    def _calculate_split_quality(self, target):
        """ target are available performance values in node """
        if self.split_metric == 'mse':
            return np.mean((target - np.mean(target)) ** 2.0)
        if self.split_metric == 'var-reduction':
            # Todo: calculate var reduction, ggf more
            pass

    def _split_node(self, node):
        """
        Split tree at given (leaf) node.
        """
        # get all rows where the split value is less or equal than threshold and grow left node
        left_features = node.features[node.features[:, node.split_feature_index] <= node.split_feature_cut_val]
        left_target = node.target_vals[node.features[:, node.split_feature_index] <= node.split_feature_cut_val]
        node.left = Node(left_features, left_target, node.depth + 1)

        # get all rows where the split value is greater than threshold and grow right node
        right_features = node.features[node.features[:, node.split_feature_index] > node.split_feature_cut_val]
        right_target = node.target_vals[node.features[:, node.split_feature_index] > node.split_feature_cut_val]
        node.right = Node(right_features, right_target, node.depth + 1)

        self.leaf_nodes.remove(node)
        self.leaf_nodes.add(node.left)
        self.leaf_nodes.add(node.right)

        self._grow_tree_at_node(node.left)
        self._grow_tree_at_node(node.right)

    def adapt_tree(self):
        pass
        # Todo: iterate over leaves and adapt if necessary
        # Todo: split most promising leaf node and calculate new regression
        # calculate split quality of each node/ feature
        # only adapt node that was recently split? NO

    def _prune_tree(self):
        # Todo: Prune nodes / subtree
        pass

    def select_next(self):
        pass
        # todo: go through leaf nodes, Choose most promising node(s) and select config at random

    def get_tree(self):
        """
        If tree is used again after initial selection process.

        :return: Decision Tree Model.
        """
        return self._root
