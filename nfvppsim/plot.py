"""
Copyright (c) 2017 Manuel Peuster
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
import logging
import os
import re
import seaborn as sns


LOG = logging.getLogger(os.path.basename(__file__))


def get_by_name(name):
    if name == "Boxplot":
        return Boxplot
    raise NotImplementedError("'{}' not implemented".format(name))


class Boxplot(object):
    """
    Simple boxplots.
    """

    @classmethod
    def generate(cls, conf):
        """
        Generate list of plotter objects. One for each plotter config.
        """
        return [cls(**conf)]

    def __init__(self, **kwargs):
        # apply default params
        p = {"title": "MSE",
             "path": "plots",
             "x": "k_samples",
             "y": "error_value",
             "n_plots": "degree"}
        p.update(kwargs)
        self.params = p
        LOG.debug("Initialized plotter: {}".format(self))

    def __repr__(self):
        return "{}({})".format(self.name, self.params)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def short_name(self):
        return re.sub('[^A-Z]', '', self.name)

    def plot(self, df):
        """
        Create a simple boxplot using Pandas default boxplot method.
        """
        sns.set()
        sns.set_context("paper")
        for p_arg in list(set(df[self.params.get("n_plots")])):
            dff = df[df[self.params.get("n_plots")] == p_arg]
            ax = dff.boxplot(self.params.get("y"), self.params.get("x"))
            fig = ax.get_figure()
            fig.suptitle(self.params.get("title"))
            ax.set_title("{}: {}".format(self.params.get("n_plots"), p_arg))
            ax.set_ylabel(self.params.get("y"))
            ax.set_xlabel(self.params.get("x"))
            ax.set_ylim([0, .2])
            path = os.path.join(self.params.get("path"), "plot_{}-{}.pdf"
                                .format(self.params.get("n_plots"), p_arg))
            fig.savefig(path, bbox_inches="tight")
            LOG.info("Wrote plot: {}".format(path))
