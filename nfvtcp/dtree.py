import logging
import os
import numpy as np

LOG = logging.getLogger(os.path.basename(__file__))


class Node:
    """
    Base Class for Decision Tree Nodes.
    """
    def __init__(self, features, target_vals):
        self.features = features
        self.target_vals = target_vals
        self.left_dec_true = None
        self.right_dec_false = None
        self.is_leaf = True
        self.regr_value = None




class DTree:
    """
    Decision Tree Base Class.
    """
    def __init__(self, features, intial_samples, split_metric='mse', prune=False, max_depth=10):
        """
        Grow initial DT.
        :param features: Dictionary of features and possible values.
        :param intial_samples: Already profiled configurations and their performance value.
        :param split_metric: Homogeneity metric.
        :param prune: Prune Tree?
        :param max_depth: Maximal depth of tree.
        """
        # Todo: stop criterion is max_number of samples reached or no improvement
        self._root = Node(intial_samples)
        self.leaf_nodes = {self._root}
        self.max_depth = max_depth

    def _grow_tree(self, samples, split_metric='mse'):
        pass
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        # Todo: grow tree initially

    def _find_best_split(self):
        pass
        # get all performance values out of samples
        # check if there is only one possible value, if so just return
        # else calculate mean of all target values
        # berechne impurity (mse)

    def _calculate_split_quality(self, target, split_metric='mse'):
        """ target are available performance values in node """
        # Todo: support more homogeneity metrics
        if split_metric == 'mse':
            return np.mean((target - np.mean(target)) ** 2.0)
        if split_metric == 'var-reduction':
            return None # TODO

    def _prune_tree(self):
        # Todo: Prune nodes / subtree
        pass

    def adapt_tree(self):
        pass
        # Todo: iterate over leaves and adapt if necessary
        # Todo: split most promising leaf node and calculate new regression
        # calculate split quality of each node/ feature
        # only adapt node that was recently split? NO

    def select_next(self, split_metric='mse'):
        pass
        # todo: go through leaf nodes, Choose most promising node(s) and select config at random
        # todo: support multiple split metrics (MSE, Variance Reduction)

    def get_tree(self):
        """
        If tree is used again after initial selection process.

        :return: Decision Tree Model.
        """
        return self._root


