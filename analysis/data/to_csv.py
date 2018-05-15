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
import argparse
import pandas as pd

LOG = logging.getLogger(os.path.basename(__file__))


def logging_setup():
    os.environ["COLOREDLOGS_LOG_FORMAT"] \
        = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"


def parse_args():
    # TODO add a "silent" flag
    parser = argparse.ArgumentParser(
        description="Pandas DF converter")

    parser.add_argument(
        "-i",
        "--input",
        help="Path",
        required=False,
        default="combined.pkl",
        dest="input")

    parser.add_argument(
        "-o",
        "--output",
        help="Path of umcompressed DF pickle file.",
        required=False,
        default="uncompressed.pkl",
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
    LOG.info("Reading input: {}".format(args.input))
    df = pd.read_pickle(args.input) # , compression="bz2"
    print(df.info())
    LOG.info("Writing result to {}".format(args.output))
    df.to_csv(args.output)
    LOG.info("DONE!")


if __name__ == "__main__":
    main()
