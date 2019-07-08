"""
Copyright (c) 2019 Heidi Neuhäuser (Modifications)
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
import itertools as it


LOG = logging.getLogger(os.path.basename(__file__))


def cartesian_product(p_dict):
    """
    Compute Cartesian product on parameter dict:
    In:
        {"number": [1,2,3], "color": ["orange","blue"] }
    Out:
        [ {"number": 1, "color": "orange"},
          {"number": 1, "color": "blue"},
          {"number": 2, "color": "orange"},
          {"number": 2, "color": "blue"},
          {"number": 3, "color": "orange"},
          {"number": 3, "color": "blue"}
        ]
    """
    p_names = sorted(p_dict)
    return [dict(zip(p_names, prod)) for prod in it.product(
        *(p_dict[n] for n in p_names))]


def flatten_conf(cl):
    """
    Takes configuration space and returns list of dictionary values for each config.
    In:
        [({"a": 1, "b": 1}, {"a": 2, "b": 1}), ({"a": 2, "b": 2}, {"a": 3, "b": 1})]
    Out:
        [[1, 1, 2, 1], [2, 2, 3, 1]]
    """
    r = list()
    for ct in cl:
        tmp = list()
        for d in ct:
            tmp += d.values()
        r.append(tmp)
    return r


def dict_to_short_str(d):
    return "-".join(
        "{!s}={!r}".format(key, val) for (key, val) in d.items())


def compress_keys(d):
    manual_compressor = {
        "border_point_mode": "bpm",
        "border_point_mode_panic": "bpmp",
        "p_samples_per_vnf": "pspv",
        "sampling_mode_maxmin": "smmm",
        "max_border_points": "mbp",
        "alpha": "a",
        "degree": "dgr",
        "epsilon": "e",
        "max_tree_depth": "mtd"
    }
    r = dict()
    for k, v in d.items():
        r[manual_compressor.get(k, k)] = v
    return r
