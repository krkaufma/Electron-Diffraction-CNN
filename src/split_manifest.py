import argparse
import typing

import pandas as pd
import sklearn.model_selection

__author__ = 'alex@aira.io'

SPLITTERS = {
    'shuffle': sklearn.model_selection.ShuffleSplit,
    'stratified-shuffle': sklearn.model_selection.StratifiedShuffleSplit,
}


def split_dataset(args: argparse.Namespace) -> None:
    """Adds `_split` column to a CSV file which assigns each row to an experiment group (train, test, or validation).

    Args:
        args: parse arguments from :func:`~split_manifest.make_parser`.
    """

    df = pd.read_csv(args.manifest)

    assert args.val_size + args.test_size <= 1, 'Validation and Test sizes must sum to at most one!'

    adjusted_validation_ratio = args.val_size / (1 - args.test_size)

    tst_splitter = SPLITTERS[args.split_method](test_size=args.test_size, random_state=args.seed)
    val_splitter = SPLITTERS[args.split_method](test_size=adjusted_validation_ratio, random_state=args.seed)

    rest_sp, test_sp = tst_splitter.split(df[args.label_column]).__next__()
    train_sp, val_sp = val_splitter.split(df.iloc[rest_sp][args.label_column]).__next__()

    n = len(df[args.label_column])

    def assign_to_group(idx: int) -> str:
        """Test group for `idx` via scope. Returns label of group."""
        if idx in test_sp:
            return 'test'
        if idx in val_sp:
            return 'validation'
        return 'train'

    df = df.assign(_split=pd.Series((assign_to_group(idx) for idx in range(n))).values)

    dest = args.manifest
    if args.output is not None:
        dest = args.output

    df.to_csv(dest, index=False, header=True)


def make_parser(_parser: typing.Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """The CLI parser for this module, i.e. making a manifest file.

    Args:
        _parser: (optional) an existing argument parser. Can be used to extend the functionality of a larger CLI.

    Returns:
        An argument parser that can process CLI args necessary for marking a manifest file.
    """
    if not _parser:
        _parser = argparse.ArgumentParser(description='Produce an EBSD data manifest file from a root directory.')

    _parser.add_argument('manifest',
                         type=str,
                         help='path/to/manifest.csv')

    _parser.add_argument('label_column',
                         type=str,
                         help='Name of column considered the label or `y`.')

    _parser.add_argument('-ts',
                         '--test-size',
                         dest='test_size',
                         type=float,
                         default=0.20,
                         help='Ratio of dataset to include in test set')

    _parser.add_argument('-vs', '--validation-size',
                         dest='val_size',
                         type=float, default=0.20,
                         help='Ratio of dataset to include in validation set')

    _parser.add_argument('-s', '--seed',
                         dest='seed',
                         type=int, default=None,
                         help='Random seed')

    _parser.add_argument('-sm', '--split-method',
                         dest='split_method',
                         choices=list(SPLITTERS.keys()),
                         default=list(SPLITTERS.keys())[0])

    _parser.add_argument('-o', '--output',
                         dest='output',
                         type=str, default=None,
                         help='(optional) /path/to/mutated/copy/of/manifest.csv')
    return _parser


if __name__ == '__main__':
    import sys

    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    split_dataset(args)
