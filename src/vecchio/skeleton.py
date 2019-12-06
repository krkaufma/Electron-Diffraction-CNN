#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import argparse
import sys
import logging

import make_manifest
from vecchio import __version__
import train

_logger = logging.getLogger(__name__)


def setup_parser():
    """Parse command line parameters

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="A Machine Learning Pipeline CLI")
    parser.add_argument(
        '--version',
        action='version',
        version='Electron-Diffraction-CNN {ver}'.format(ver=__version__))
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)

    subparser = parser.add_subparsers()

    train_parser = subparser.add_parser(name='train', help='Train a ML Model')
    train.make_parser(train_parser)

    make_manifest_parser = subparser.add_parser('manifest', help='Make a manifest file')
    make_manifest.make_parser(make_manifest_parser)

    return parser


def setup_logging(log_level):
    """Setup basic logging

    Args:
      log_level (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=log_level, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    parser = setup_parser()
    args = parser.parse(args)

    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
