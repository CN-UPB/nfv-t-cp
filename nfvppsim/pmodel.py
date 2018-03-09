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
import math
import random
import numpy as np
import networkx as nx
import itertools as it
from nfvppsim.helper import cartesian_product

LOG = logging.getLogger(os.path.basename(__file__))


# global cache vars
CACHE_C_SPACE = dict()  # model -> cache


def get_by_name(name):
    if name == "CrossValidationModel":
        return CrossValidationModel
    if name == "ExampleModel":
        return ExampleModel
    if name == "NFVSDN17Model":
        return NFVSDN17Model
    if name == "PanicTerrasortModel":
        return PanicTerrasortModel
    if name == "TCPaperModel4VNF":
        return TCPaperModel4VNF
    if name == "TCPaperModel5VNFSimple":
        return TCPaperModel5VNFSimple
    if name == "RandomSyntheticModel":
        return RandomSyntheticModel
    raise NotImplementedError("'{}' not implemented".format(name))


class VnfPerformanceModel(object):
    
    def __init__(self, vnf_id, name, parameter, func):
        self.vnf_id = vnf_id  # identification
        self.name = name  # just for humans
        # list of parameter tuples (name, lst_of_values)
        self.parameter = parameter
        # function to evaluate the VNF's performance
        # (p1, ... pn) -> performance
        self.func = func
        LOG.debug("Generated VNF {} with vnf_id={}".format(name, vnf_id))

    def reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        """
        pass
        
    def evaluate(self, c):
        """
        Calculate the resulting performance of the VNF
        for the given configuration vector.
        :param c: Configuration parameter list (m)
        :return: single performance value (e.g. throughput)
        """
        # LOG.debug("eval VNF={} cf={}".format(self.name, c))
        return self.func(c)


class SfcPerformanceModel(object):

    @classmethod
    def generate(cls, conf):
        """
        Generate list of model objects. One for each conf. to be tested.
        """
        n_model_instances = 1
        if conf is not None:
            n_model_instances = conf.get("n_model_instances", 1)
        model_lst = list()
        # can generate n_model_instances (useful for randomization)
        for mid in range(0, n_model_instances):
            parameter, vnf_lst = cls.generate_vnfs(conf)
            sfc_graph = cls.generate_sfc_graph(conf, vnf_lst)
            LOG.info("Generated SFC graph with nodes={} and edges={}"
                     .format(sfc_graph.nodes(), sfc_graph.edges()))
            pm_obj = cls(parameter=parameter,
                         vnfs=vnf_lst,
                         sfc_graph=sfc_graph,
                         conf=conf,
                         mid=mid)
            pm_obj.initialize()
            model_lst.append(pm_obj)
        return model_lst
    
    @classmethod
    def generate_vnfs(cls, conf):
        LOG.error("VNF generation not implemented in base class.")
        
    @classmethod
    def generate_sfc_graph(cls, conf):
        LOG.error("Service graph generation not implemented in base class.")
    
    def __init__(self, **kwargs):
        self.parameter = kwargs.get("parameter", {})
        self.vnfs = kwargs.get("vnfs", [])
        self.sfc_graph = kwargs.get("sfc_graph")
        self.conf = kwargs.get("conf", {})
        self.mid = kwargs.get("mid", 0)
        LOG.info("Initialized performance model: '{}' with {} VNFs ...".format(
            self, len(self.vnfs)))
        LOG.info("\t ... the SFC graph has {} nodes and {} edges ...".format(
            len(self.sfc_graph.nodes()),  len(self.sfc_graph.edges())))
        LOG.info("\t ... each VNF has {} possible configurations ...".format(
            len(self.get_conf_space_vnf())))
        LOG.info("\t ... the SFC has {} possible configurations.".format(
            len(self.get_conf_space())))

    def initialize(self):
        """
        Called after generation.
        """
        pass

    def reinitialize(self, repetition_id):
        """
        Called once for each experiment repetition.
        Can be used to re-initialize data structures for each repetition.
        """
        for v in self.vnfs:
            v.reinitialize(repetition_id)

    def __repr__(self):
        return "{}".format(
            self.name)

    @property
    def name(self):
        return "{}.{}".format(
            self.__class__.__name__,
            self.mid)

    @property
    def short_name(self):
        return "{}.{}".format(
            re.sub('[^A-Z]', '', self.name),
            self.mid)

    def get_results(self):
        """
        Getter for global result collection.
        :return: dict for result row
        """
        r = {"pmodel": self.short_name}
        return r
    
    def get_conf_space_vnf(self, modified_parameter=None):
        """
        Return the configuration space for a single VNF.
        :return: list of configuration dicts
        """
        if modified_parameter:
            return cartesian_product(modified_parameter)
        return cartesian_product(self.parameter)

    def get_conf_space(self, modified_parameter=None, no_cache=False):
        """
        Return the COMPLETE configuration space for this model.
        :return: list of configuration tuples of one dict per VNF of graph
        """
        global CACHE_C_SPACE
        if CACHE_C_SPACE.get(self.name) is not None and no_cache is False:
            LOG.debug("Using configuration space from cache.")
            return CACHE_C_SPACE.get(self.name)
        # config space for one VNF
        cf = self.get_conf_space_vnf(modified_parameter)
        # config space for n VNFs in the SFC
        cs = list(it.product(cf, repeat=len(self._get_vnfs_from_sg())))
        if no_cache is False:
            CACHE_C_SPACE[self.name] = cs
        return cs

    def _get_vnfs_from_sg(self):
        """
        Return VNF objects from SG.
        """
        vnfs = list()
        for n, nd in self.sfc_graph.nodes(data=True):
            if nd.get("vnf") is not None:
                vnfs.append(nd.get("vnf"))
        return vnfs

    def evaluate(self, c):
        """
        Calculate the resulting performance of SG
        for the given configuration vector.
        :param c: Configuration parameter tuple (n x m)
        :return: single performance value (e.g. throughput)
        """
        G = self.sfc_graph
        # 1. compute/assign capacity to each node
        # LOG.debug("eval cs={}".format(c))
        for n, nd in G.nodes.data():
            if nd.get("vnf") is not None:
                nd["capacity"] = nd.get("vnf").evaluate(c[n])
        # 2. transform graph: reduce to classic max_flwo problem
        #    (each VNF node becomes vin - vout)
        G_new = nx.DiGraph()
        # for each node add in and out with capacity edge
        for (n, nd) in G.nodes.data():
            G_new.add_node("{}_in".format(n), **nd)
            G_new.add_node("{}_out".format(n), **nd)
            G_new.add_edge("{}_in".format(n),
                           "{}_out".format(n),
                           capacity=nd.get("capacity", float("inf")))
        # replicate each edge from old graph in new graph
        for (u, v) in G.edges():
            G_new.add_edge("{}_out".format(u), "{}_in".format(v))
        # debugging
        # LOG.debug("Nodes:", G_new.nodes())
        # for e in G_new.edges(data=True):
        #    LOG.debug("Edge: {}".format(e))
        # calculate service throughput (solve max_flow problem)
        mf_value, mf_path = nx.maximum_flow(G_new, "s_in", "t_out")
        # LOG.debug("MaxFlow calculation (value/path): {} / {}".format(
        #    mf_value, mf_path))
        self.sfc_graph_reduced = G_new
        return mf_value


class CrossValidationModel(SfcPerformanceModel):
    """
    Model from Jupyter prototype. Used for cross validation.
    """

    @classmethod
    def generate_vnfs(cls, conf):
        # define parameters
        # dict of lists defining possible configuration parameters
        p = {"c0": list(np.linspace(0.01, 1.0, num=20))}
        # create vnfs
        # function: config_list -> performance
        # REQUIREMENT: vnf_ids of objects == idx in list
        vnf0 = VnfPerformanceModel(0, "vnf_0", p,
                                   lambda c: (c["c0"] ** 2
                                              + (c["c0"] * 2) + 0.1))
        vnf1 = VnfPerformanceModel(1, "vnf_1", p,
                                   lambda c: (c["c0"] ** 4 + (.5 * c["c0"])))
        # return parameters, list of vnfs
        return p, [vnf0, vnf1]

    @classmethod
    def generate_sfc_graph(cls, conf, vnfs):
        # create a directed graph
        G = nx.DiGraph()
        # add nodes and assign VNF objects
        G.add_node(0, vnf=vnfs[0])
        G.add_node(1, vnf=vnfs[1])
        # G.add_node(2, vnf=vnfs[1])
        G.add_node("s", vnf=None)
        G.add_node("t", vnf=None)
        # simple linear: s -> 0 -> 1 ->  -> t
        G.add_edges_from([("s", 0), (0, 1), (1, "t")])
        return G


class NFVSDN17Model(SfcPerformanceModel):
    """
    Model based on single node measurements of NFV-SDN'17 paper.
    VNF1: Nginx
    VNF2: Socat
    VNF3: Squid
    """

    @classmethod
    def generate_vnfs(cls, conf):
        # define parameters
        # dict of lists defining possible configuration parameters
        p = {"c1": list(np.linspace(0.01, 1.0, num=20))}
        # create vnfs
        # function: config_list -> performance
        # REQUIREMENT: vnf_ids of objects == idx in list
        vnf0 = VnfPerformanceModel(0, "vnf_0", p,
                                   lambda c: (c["c1"] * 9.0))
        vnf1 = VnfPerformanceModel(1, "vnf_1", p,
                                   lambda c: (c["c1"] * 3.3))
        vnf2 = VnfPerformanceModel(2, "vnf_2", p,
                                   lambda c: (c["c1"] * 1.2))
        # return parameters, list of vnfs
        return p, [vnf0, vnf1, vnf2]

    @classmethod
    def generate_sfc_graph(cls, conf, vnfs):
        # create a directed graph
        G = nx.DiGraph()
        # add nodes and assign VNF objects
        G.add_node(0, vnf=vnfs[0])
        G.add_node(1, vnf=vnfs[1])
        G.add_node(2, vnf=vnfs[2])
        G.add_node("s", vnf=None)
        G.add_node("t", vnf=None)
        # simple linear: s -> 0 -> 1 -> 2 -> t
        G.add_edges_from([("s", 0), (0, 1), (1, 2), (2, "t")])
        return G


class PanicTerrasortModel(SfcPerformanceModel):
    """
    Model based on single node measurements of PANIC paper.
    Attention: Smaller result is better (runntime)
    VNF1: Terrasort
    """

    @classmethod
    def generate_vnfs(cls, conf):
        # define parameters
        # dict of lists defining possible configuration parameters
        p = {"cluster_size": list(range(2, 11)),
             "data_size": list(range(10, 60, 10)),
             "cores": [1, 2, 4]}
        # create vnfs
        # function: config_list -> performance
        # REQUIREMENT: vnf_ids of objects == idx in list
        vnf0 = VnfPerformanceModel(0, "vnf_0", p,
                                   lambda c: (((-50 * c["cluster_size"] + 700)
                                               + (12.75 * c["data_size"]
                                                  - 37.5)) / 2 * c["cores"]))
        # return parameters, list of vnfs
        return p, [vnf0]

    @classmethod
    def generate_sfc_graph(cls, conf, vnfs):
        # create a directed graph
        G = nx.DiGraph()
        # add nodes and assign VNF objects
        G.add_node(0, vnf=vnfs[0])
        G.add_node("s", vnf=None)
        G.add_node("t", vnf=None)
        # simple linear: s -> 0 -> t
        G.add_edges_from([("s", 0), (0, "t")])
        return G


class TCPaperModel4VNF(SfcPerformanceModel):
    """
               - (v2)
    (s) - (v1)         (v4) - (t)
               - (v3)
    """

    @classmethod
    def generate_vnfs(cls, conf):
        # define parameters
        # dict of lists defining possible configuration parameters
        # use normalized inputs for now
        p = {"cpu": list(np.linspace(0.1, 1.0, num=4)),
             "mem": list(np.linspace(0.1, 1.0, num=5)),
             "sriov": [0, 1]}

        # create vnfs
        # function: config_list -> performance
        # REQUIREMENT: vnf_ids of objects == idx in list
        vnf0 = VnfPerformanceModel(0, "vnf0", p,
                                   lambda c: (c["cpu"] * 9.0
                                              + c["mem"] * 0.5
                                              + c["sriov"] * 0.0))
        vnf1 = VnfPerformanceModel(0, "vnf1", p,
                                   lambda c: (c["cpu"] * .8
                                              + c["mem"] * 2.5
                                              + c["sriov"] * 0.0))
        vnf2 = VnfPerformanceModel(0, "vnf2", p,
                                   lambda c: (c["cpu"] * .8
                                              + c["mem"] * 2.5
                                              + c["sriov"] * 0.0))
        vnf3 = VnfPerformanceModel(0, "vnf3", p,
                                   lambda c: (c["cpu"] * 2.0
                                              + c["mem"] * 0.0
                                              + c["sriov"] * 2.0))
        # return parameters, list of vnfs
        return p, [vnf0, vnf1, vnf2, vnf3]

    @classmethod
    def generate_sfc_graph(cls, conf, vnfs):
        # create a directed graph
        G = nx.DiGraph()
        # add nodes and assign VNF objects
        G.add_node(0, vnf=vnfs[0])
        G.add_node(1, vnf=vnfs[1])
        G.add_node(2, vnf=vnfs[2])
        G.add_node(3, vnf=vnfs[3])
        G.add_node("s", vnf=None)
        G.add_node("t", vnf=None)
        # s -> 0 -> 1,2 -> 3 -> 4 -> t
        G.add_edges_from([("s", 0),
                          (0, 1), (0, 2),
                          (1, 3), (2, 3),
                          (3, "t")])
        return G


class TCPaperModel5VNFSimple(SfcPerformanceModel):
    """
               - (v2)
    (s) - (v1)         (v4) - (v5) - (t)
               - (v3)

    Simple version with only one parameter per VNF!
    """

    @classmethod
    def generate_vnfs(cls, conf):
        # define parameters
        # dict of lists defining possible configuration parameters
        # use normalized inputs for now
        p = {"p1": list(np.linspace(0.0, 1.0, num=10))}

        # create vnfs
        # function: config_list -> performance
        # REQUIREMENT: vnf_ids of objects == idx in list
        vnf0 = VnfPerformanceModel(0, "vnf0", p,
                                   lambda c: (c["p1"] * 0.2))
        vnf1 = VnfPerformanceModel(0, "vnf1", p,
                                   lambda c: (c["p1"] * 0.1))
        vnf2 = VnfPerformanceModel(0, "vnf2", p,
                                   lambda c: (c["p1"] * 0.1))
        vnf3 = VnfPerformanceModel(0, "vnf3", p,
                                   lambda c: (c["p1"] * 0.4))
        vnf4 = VnfPerformanceModel(0, "vnf4", p,
                                   lambda c: (c["p1"] * 0.2))

        # return parameters, list of vnfs
        return p, [vnf0, vnf1, vnf2, vnf3, vnf4]

    @classmethod
    def generate_sfc_graph(cls, conf, vnfs):
        # create a directed graph
        G = nx.DiGraph()
        # add nodes and assign VNF objects
        G.add_node(0, vnf=vnfs[0])
        G.add_node(1, vnf=vnfs[1])
        G.add_node(2, vnf=vnfs[2])
        G.add_node(3, vnf=vnfs[3])
        G.add_node(4, vnf=vnfs[4])
        G.add_node("s", vnf=None)
        G.add_node("t", vnf=None)
        # s -> 0 -> 1,2 -> 3 -> 4 -> t
        G.add_edges_from([("s", 0),
                          (0, 1), (0, 2),
                          (1, 3), (2, 3),
                          (3, 4),
                          (4, "t")])
        return G


class ExampleModel(SfcPerformanceModel):
    """
    Playground model.
    """

    @classmethod
    def generate_vnfs(cls, conf):
        # define parameters
        # dict of lists defining possible configuration parameters
        p = {"c1": list(np.linspace(0.01, 1.0, num=5)),
             "c2": list(np.linspace(0.01, 1.0, num=5)),
             "c3": list(np.linspace(0.01, 1.0, num=5))}
        # create vnfs
        # function: config_list -> performance
        # REQUIREMENT: vnf_ids of objects == idx in list
        vnf0 = VnfPerformanceModel(0, "vnf_0", p,
                                   lambda c: (c["c1"] * 8.0
                                              + c["c2"] * 1.5
                                              + c["c3"] * 0.5))
        vnf1 = VnfPerformanceModel(1, "vnf_1", p,
                                   lambda c: (c["c1"]
                                              + c["c2"] * 4.0))
        # return parameters, list of vnfs
        return p, [vnf0, vnf1]

    @classmethod
    def generate_sfc_graph(cls, conf, vnfs):
        # create a directed graph
        G = nx.DiGraph()
        # add nodes and assign VNF objects
        G.add_node(0, vnf=vnfs[0])
        G.add_node(1, vnf=vnfs[1])
        # G.add_node(2, vnf=vnfs[1])
        G.add_node("s", vnf=None)
        G.add_node("t", vnf=None)
        # simple linear: s -> 0 -> 1 ->  -> t
        G.add_edges_from([("s", 0), (0, 1), (1, "t")])
        return G


class RandomSyntheticModel(SfcPerformanceModel):
    """
    Randomized Performance Model
    - 1 prameter per VNF
    - f synthetic performance functions
    - TODO 1...4 VNFs
    - TODO topologies: l1-l4, d2-d4
    - TODO random bias
    func. source: http://ieeexplore.ieee.org/document/8257924/

    Conf. Parameter:
    - a1_range = [0.1, 2.0]
    - func_set = [1, 2, 3, 4, 5, 6, 7, 8]  (or a subset of these)
    """
    @classmethod
    def grf(cls, conf):
        """
        Get Random Function
        """
        # prepare synthetic func. generation
        assert(conf is not None)
        assert("a1_range" in conf)
        assert("func_set" in conf)
        # randomly set global coefficient(s)
        a1 = random.uniform(
            conf.get("a1_range")[0],
            conf.get("a1_range")[1])
        fnum = random.choice(conf.get("func_set"))
        assert(fnum >= 1 and fnum <= 8)
        LOG.debug("Selected synthetic function no. {}".format(fnum))

        # synthetic functions
        def f1(x):
            return a1*x

        def f2(x):
            return a1*x**2

        def f3(x):
            return math.exp(f1(x))

        def f4(x):
            return math.exp(abs(f1(x)))

        def f5(x):
            return math.exp(-(f1(x)**2))

        def f6(x):
            return math.exp(-(f2(x)))

        def f7(x):
            return math.cos(f1(x)) * f3(x)

        def f8(x):
            return f2(x) * f6(x)

        # ok, how to use Pyhton's getattr for netsed functions?
        # a = getattr(cls.grf, "f2")(1) does not work.
        # use a manual selection, even if its ugly:
        if fnum == 1:
            return f1
        elif fnum == 2:
            return f2
        elif fnum == 3:
            return f3
        elif fnum == 4:
            return f4
        elif fnum == 5:
            return f5
        elif fnum == 6:
            return f6
        elif fnum == 7:
            return f7
        elif fnum == 8:
            return f8
        else:
            raise BaseException("Function not found!")
        return None

    @classmethod
    def generate_vnfs(cls, conf):
        # define parameters
        # dict of lists defining possible configuration parameters
        # use normalized inputs for now
        p = {"p1": list(np.linspace(0.0, 1.0, num=2))}

        # randomly pick synthetic functions
        fn0 = cls.grf(conf)
        fn1 = cls.grf(conf)
        fn2 = cls.grf(conf)
        fn3 = cls.grf(conf)
        fn4 = cls.grf(conf)

        # create vnfs and assign functions
        vnf0 = VnfPerformanceModel(0, "vnf0", p,
                                   lambda c: fn0(c["p1"]))
        vnf1 = VnfPerformanceModel(0, "vnf1", p,
                                   lambda c: fn1(c["p1"]))
        vnf2 = VnfPerformanceModel(0, "vnf2", p,
                                   lambda c: fn2(c["p1"]))
        vnf3 = VnfPerformanceModel(0, "vnf3", p,
                                   lambda c: fn3(c["p1"]))
        vnf4 = VnfPerformanceModel(0, "vnf4", p,
                                   lambda c: fn4(c["p1"]))

        # return parameters, list of vnfs
        return p, [vnf0, vnf1, vnf2, vnf3, vnf4]

    @classmethod
    def generate_sfc_graph(cls, conf, vnfs):
        # create a directed graph
        G = nx.DiGraph()
        # add nodes and assign VNF objects
        G.add_node(0, vnf=vnfs[0])
        G.add_node(1, vnf=vnfs[1])
        G.add_node(2, vnf=vnfs[2])
        G.add_node(3, vnf=vnfs[3])
        G.add_node(4, vnf=vnfs[4])
        G.add_node("s", vnf=None)
        G.add_node("t", vnf=None)
        # s -> 0 -> 1,2 -> 3 -> 4 -> t
        G.add_edges_from([("s", 0),
                          (0, 1), (0, 2),
                          (1, 3), (2, 3),
                          (3, 4),
                          (4, "t")])
        return G
