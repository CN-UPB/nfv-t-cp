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
import yaml
import sys

LOG = logging.getLogger(os.path.basename(__file__))


def read_config(path):
    try:
        with open(path, "r") as f:
            conf = yaml.load(f)
            # do some basic validation on config
            assert("name" in conf)
            assert("author" in conf)
            assert("version" in conf)
            assert("sim_t_max" in conf)
            assert("pmodel" in conf)
            assert("selector" in conf)
            assert("predictor" in conf)
            assert("error" in conf)
    except AssertionError as e:
        LOG.exception("Couldn't parse config '{}' {}".format(path, e))
        LOG.error("Abort.")
        sys.exit(1)
    except:
        LOG.error("Couldn't open config '{}'. Abort.".format(path))
        sys.exit(1)
    return conf
