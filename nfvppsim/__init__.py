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
import sys
import argparse
import time

from nfvppsim.experiment import Experiment
from nfvppsim.config import read_config


LOG = logging.getLogger(os.path.basename(__file__))


def logging_setup():
    os.environ["COLOREDLOGS_LOG_FORMAT"] \
        = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"


def parse_args():
    # TODO add a "silent" flag
    parser = argparse.ArgumentParser(
        description="nfv-pp-sim")

    parser.add_argument(
        "-c",
        "--config",
        help="Experiment configuration file.",
        required=False,
        default="example_experiment.yaml",
        dest="config_path")

    parser.add_argument(
        "-v",
        "--verbose",
        help="Output debug messages.",
        required=False,
        default=False,
        dest="verbose",
        action="store_true")

    parser.add_argument(
        "--no-prepare",
        help="Stop before prepare step.",
        required=False,
        default=False,
        dest="no_prepare",
        action="store_true")

    parser.add_argument(
        "--no-run",
        help="Stop before run step.",
        required=False,
        default=False,
        dest="no_run",
        action="store_true")

    parser.add_argument(
        "--plot",
        help="Plot the given results.",
        required=False,
        default=None,
        dest="plot")
    
    parser.add_argument(
        "--result-print",
        help="Print results as Pandas table.",
        required=False,
        default=False,
        dest="result_print",
        action="store_true")

    parser.add_argument(
        "--result-path",
        help="Path for results file (overwrites config)",
        required=False,
        default=None,
        dest="result_path")
    return parser.parse_args()


def show_welcome():
    print("""*****************************************************
**          Welcome to nfv-pp-sim                  **
**                                                 **
** (c) 2017 by Manuel Peuster (manuel@peuster.de)  **
*****************************************************""")


def show_byebye(t_start=None, rc=0):
    print("*****************************************************")
    print("Simulation done!")
    if t_start:
        print("Runtime: {0:.3f}s".format(time.time() - t_start))
    print("")
    sys.exit(rc)

    
def main():
    t_start = time.time()
    # CLI interface
    args = parse_args()
    # configure logging
    logging_setup()
    if args.verbose:
        coloredlogs.install(level="DEBUG")
    else:
        coloredlogs.install(level="INFO")
    # show welcome screen
    show_welcome()
    # read experiment configuration
    conf = read_config(args.config_path)
    # get result path
    rpath = conf.get("result_path")
    if args.result_path:
        rpath = args.result_path
    # initialize experiment
    e = Experiment(conf)
    # prepare experiment
    if args.no_prepare:
        show_byebye(t_start)
    e.prepare()
    # plot only (just plot existing Pikle file)
    if args.plot is not None:
        e.plot(args.plot)
        show_byebye(t_start)
    # run experiment
    if args.no_run:
        show_byebye(t_start)
    e.run()
    # store results
    e.store_result(rpath)
    if args.result_print:
        e.print_results()
    # show bye bye screen
    show_byebye(t_start)

