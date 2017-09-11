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
import coloredlogs
import os

from nfvppsim.experiment import Experiment
from nfvppsim.config import read_config


LOG = logging.getLogger(os.path.basename(__file__))


def logging_setup():
    os.environ["COLOREDLOGS_LOG_FORMAT"] \
        = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"


def main():
    # TODO CLI interface
    logging_setup()
    print("")
    print("*" * 64)
    print("nfv-pp-sim by Manuel Peuster <manuel@peuster.de>")
    print("*" * 64)
    coloredlogs.install(level="DEBUG")
    # TODO replace this with configuration runner module
    # initialize and configure involved modules
    conf = read_config("example_experiment.yaml")
    e = Experiment(conf)
    e.prepare()
    e.run()
    e.store_result(conf.get("result_path"))
    return

