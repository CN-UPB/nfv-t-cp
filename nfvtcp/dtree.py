import logging
import os
import numpy as np

LOG = logging.getLogger(os.path.basename(__file__))


class DTree:
    """
    Decision Tree Base Class.
    """
    def __init__(self, intial_samples=[], criterion='mse', prune=False, max_depth=10):
        pass

    def _grow_tree(self, samples):
        pass
        # Todo: grow tree intially

    def adapt_tree(self):
        pass
        # Todo: iterate over leaves and adapt if necessary
        # Todo: split most promising leaf node and calculate new regression
        # calculate split quality of each node/ feature
        # only adapt node that was recently split? NO

    def select_next(self):
        pass
        # todo: go through leaf nodes, Choose most promising node(s) and select config at random


