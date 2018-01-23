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
import itertools
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from nfvppsim.helper import cartesian_product


LOG = logging.getLogger(os.path.basename(__file__))


def get_by_name(name):
    if name == "Boxplot":
        return Boxplot
    if name == "Lineplot":
        return Lineplot
    raise NotImplementedError("'{}' not implemented".format(name))


class BasePlot(object):
    """
    Base class for plots. Place to put common helpers.
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
             "n_plots": ["degree"],
             "fig_width": 16,
             "fig_height": 12,
             "fig_dpi": 300,
             "y_lim_min": None,
             "y_lim_max": None}
        p.update(kwargs)
        self.params = p
        self.marker = itertools.cycle(
            ("s", "o", "^", "v", "X", "P", "D", "*", "H"))
        LOG.debug("Initialized plotter: {}".format(self))

    def __repr__(self):
        return "{}({})".format(self.name, self.params)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def short_name(self):
        return re.sub('[^A-Z]', '', self.name)

    def _get_plot_name(self, filter_dict):
        """
        Generate a plot name based on title and filter.
        Filter values are encoded in the name.
        """
        r = ""
        # title to name prefix
        t = str(self.params.get("title"))
        t = t.replace(" ", "_")
        r += t + "_"
        # unroll filter dict to sting
        for k in self.params.get("n_plots"):
            r += "{}-{}_".format(k, filter_dict.get(k))
        return r.strip("-_. ")

    def _filter_to_string(self, filter_dict):
        r = ""
        for k, v in filter_dict.items():
            r += "{}={} ".format(k, v)
        return r.strip("-_. ")
        
    def _generate_filters(self, df, column_lst):
        """
        config defines arbitrary column names over which we want to iterate
        to create multiple plots, we fetch the possible values of each column
        from the dataset, and compute a float list (cartesian_product) of
        configuration combinations to be plotted
        """
        # extract possible values
        filter_dict = dict()
        for column in column_lst:
            filter_dict[column] = list(set(df[column]))
        # all combinations
        return cartesian_product(filter_dict)

    def _filter_df_by_dict(self, df, filter_dict):
        """
        do some Pandas magic to dynamically filter df by given dict
        filter_dict = {"column1": "value", "column2": ...}
        """
        return df.loc[
            (df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)]

    def plot(self, df):
        if self.params.get("disabled"):
            LOG.info("'{}' disabled. Skipping plot.".format(self.name))
            return
        return self._plot(df)

    def _plot(self, df):
        """
        Plot method to be overwritten.
        """
        LOG.warning("BasePlot.plot() not implemented.")


class Boxplot(BasePlot):
    """
    Simple boxplots.
    """

    def _plot(self, df):
        """
        Create a simple boxplot using Pandas default boxplot method.
        """
        # plot setup
        sns.set()
        sns.set_context("paper")
        # generate filters (on filter per plot)
        filter_dict_list = self._generate_filters(
            df, self.params.get("n_plots"))
        # iterate over all filters
        for f in filter_dict_list:
            # select data to be plotted
            dff = self._filter_df_by_dict(df, f)
            if len(dff) < 1:
                continue  # nothing to plot
            fig, ax = plt.subplots(
                figsize=(self.params.get("fig_width"),
                         self.params.get("fig_height")),
                dpi=self.params.get("fig_dpi"))
            dff.boxplot(self.params.get("y"), self.params.get("x"), ax=ax)
            fig.suptitle(self.params.get("title"))
            ax.set_title(self._get_plot_name(f))
            ax.set_ylabel(self.params.get("y"))
            ax.set_xlabel(self.params.get("x"))
            ax.set_ylim([self.params.get("y_lim_min"),
                         self.params.get("y_lim_max")])
            path = os.path.join(self.params.get("path"), "plot_{}.pdf"
                                .format(self._get_plot_name(f)))
            fig.savefig(path, bbox_inches="tight")
            LOG.info("Wrote plot: {}".format(path))
            plt.close()


class Lineplot(BasePlot):
    """
    Simple lineplot.
    """

    def _plot(self, df):
        """
        Create a simple boxplot using Pandas default boxplot method.
        """
        # plot setup
        sns.set()
        sns.set_context("paper")
        # generate filters (one filter per plot)
        filter_dict_list = self._generate_filters(
            df, self.params.get("n_plots"))
        # iterate over all filters to gen. diff. plots
        for f in filter_dict_list:
            # select data to be plotted
            dff = self._filter_df_by_dict(df, f)
            if len(dff) < 1:
                continue  # nothing to plot
            fig, ax = plt.subplots(
                figsize=(self.params.get("fig_width"),
                         self.params.get("fig_height")),
                dpi=self.params.get("fig_dpi"))

            # generate filters (one filter per plot line)
            filter_dict_list2 = self._generate_filters(
                dff, self.params.get("line_plots"))
            # iterate over line keys
            for f2 in filter_dict_list2:
                # select data for one line
                dff2 = self._filter_df_by_dict(dff, f2)
                if len(dff2) < 1:
                    continue  # nothing to plot
                # calculate mean over all repetitions
                grp = dff2.groupby(
                    ["conf_id"])[self.params.get("x"),
                                 self.params.get("y")].mean()
                # print(grp)
                # plot
                ax = grp.plot(ax=ax,
                              kind='line',
                              linewidth=1.0,
                              marker=next(self.marker),
                              x=self.params.get("x"),
                              y=self.params.get("y"),
                              label=self._filter_to_string(f2))
            # create legend
            ax.legend(loc='best')
            fig.suptitle(self.params.get("title"))
            ax.set_title(self._get_plot_name(f))
            ax.set_ylabel(self.params.get("y"))
            ax.set_xlabel(self.params.get("x"))
            ax.set_ylim([self.params.get("y_lim_min"),
                         self.params.get("y_lim_max")])
            path = os.path.join(self.params.get("path"), "plot_{}.pdf"
                                .format(self._get_plot_name(f)))
            fig.savefig(path, bbox_inches="tight")
            LOG.info("Wrote plot: {}".format(path))
            plt.close()
