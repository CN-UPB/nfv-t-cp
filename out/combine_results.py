"""
Copyright (c) 2018 Manuel Peuster
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
import glob
import argparse
import pandas as pd

LOG = logging.getLogger(os.path.basename(__file__))


def logging_setup():
    os.environ["COLOREDLOGS_LOG_FORMAT"] \
        = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"


def parse_args():
    # TODO add a "silent" flag
    parser = argparse.ArgumentParser(
        description="Pandas DF combiner")

    parser.add_argument(
        "-i",
        "--input",
        help="Path w. wildcards e.g. -i '*.pkl'",
        required=False,
        default="experiment_*.pkl",
        dest="input")

    parser.add_argument(
        "-o",
        "--output",
        help="Path of combined DF pickle file.",
        required=False,
        default="combined.pkl",
        dest="output")

    parser.add_argument(
        "-v",
        "--verbose",
        help="Output debug messages.",
        required=False,
        default=False,
        dest="verbose",
        action="store_true")

    return parser.parse_args()


def main():
    # CLI interface
    args = parse_args()
    # configure logging
    logging_setup()
    if args.verbose:
        coloredlogs.install(level="DEBUG")
    else:
        coloredlogs.install(level="INFO")
    LOG.info("Pattern: {}".format(args.input))
    f_lst = glob.glob(args.input)

    LOG.info("Files found:")
    for f in f_lst:
        LOG.info(f)

    df_lst = list()
    for f in f_lst:
        LOG.info("Loading file: {}". format(f))
        df = pd.read_pickle(f, compression="bz2")
        print(df.info())
        df_lst.append(df)
    df_full = pd.concat(df_lst, ignore_index=True)
    # print(df_full)
    LOG.info("Concatenated all data frames.")
    print(df_full.info())
    LOG.info("Writing result to {}".format(args.output))
    df_full.to_pickle(args.output, compression="bz2")
    LOG.info("DONE!")


if __name__ == "__main__":
    main()
