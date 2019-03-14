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
        self.pred_value = np.mean(target)   # Todo: ggf. andere regression Werte? How?
        self.accuracy = 0


class DTree:
    """
    Decision Tree Base Class.
    """
    # Todo: Wie Zugriff auf nicht geprofilete Configs aus Partition?

    def __init__(self, configs, features, target, homog_metric='mse', max_depth=10, min_samples_split=2):
        """
        Grow initial DT.
        """
        self._root = Node(features, target, depth=1)
        self.leaf_nodes = {self._root} # needed for selection of config to profile
        self.max_depth = ((2 ** 31) - 1 if max_depth is None else max_depth)
        self.homog_metric = homog_metric
        self.config_space = configs # should also be flat
        self.min_samples_split = min_samples_split # minimum number of samples a node needs to have in order to split it - answer: Can node be split?
        #self.regression_technique = regression
        #self.min_samples_leaf = 1   # minimum required number of samples within one leaf
        #self.max_features_split = np.shape(features)    # consider only 30-40% of features for split search?

        # grow tree initially
        self._grow_tree_at_node(self._root)

    def _grow_tree_at_node(self, node):
        """
        Grow (sub)tree until defined termination definition is reached. Initially called for root node
        """
        if node.depth == self.max_depth or len(node.target) < self.min_samples_split:
            # stop growing # Todo: check if split necessary (is there improvement?)
            return

        # Todo: calculate best feature/cut/gain for given leaf node
        self._split_node(node)


    def _find_best_node_to_split(self): # nicht ganz korrekt? es wird immer an dem Node gesplitted, dem das neue Sample zugeordnet wird?
        # Todo: find node with lowest accuracy/homogeneity! --> factor size of node in!
        # Todo: ggf. simulated annealing for better split line
        # Todo: ggf priority queue mit homogeneity werten?
        pass

    def _calculate_node_homogeneity(self, node):
        """
        Calculate the homogeneity of a given node according to homogeneity metric (self.homog_metric)
        """
        # Todo: more? Std deviation?
        if self.homog_metric == 'var-reduction':    # same as mse? Lowest value = best
            return np.mean((node.target - np.mean(node.target)) ** 2.0)

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

        #self._grow_tree_at_node(node.left)
        #self._grow_tree_at_node(node.right)

    def adapt_tree(self):
        pass
        # Todo: iterate over leaves and adapt if necessary --> wait: take new samply, run (predict) through tree til leaf node --> grow at that node
        # Todo: split most promising leaf node and calculate new regression
        # calculate split quality of each node/ feature
        # only adapt node that was recently split? NO

    def prune_tree(self):
        # Todo: Prune tree, called afterwards?
        pass

    def get_config_partition(self, node):
        # traverse back to root and save partition of corresponding configs to randomly select from?
        pass

    def select_next(self):
        pass
        # todo: go through leaf nodes, Choose most promising node(s) and select config at random
        # todo: find most interesting config in partitions?

    def get_tree(self):
        """
        If tree is used again after initial selection process.

        :return: Decision Tree Model.
        """
        return self._root
