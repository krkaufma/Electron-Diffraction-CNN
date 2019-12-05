"""
This module parses a root file and produces a manifest of EBSD data and metadata.

Note
    The EBSD system persists recordings as tiff images in a standard format and folder structure.
    (Meta)data is encoded in both the file structure and in Tab-Separated-Value (TSV) format files.

"""
import typing
import argparse
import os
import pandas as pd

__author__ = 'alex@aira.io'

METADATA_FILENAME = 'Grain List.txt'


def make_manifest(args: argparse.Namespace) -> None:
    """Executor function: traverses file structure, merges data, and writes the output to an output CSV file.

    Args:
        args: Parsed arguments from :func:`make_manifest.make_parser`.
    """
    df = pd.DataFrame()

    def merge_data(path):
        nonlocal df
        df = populate_dataframe(df, path)

    traverse(args.source_directory, merge_data)

    df.to_csv(args.output, index=False, header=True)


def populate_dataframe(df: pd.DataFrame, path: str, target_file: str = METADATA_FILENAME) -> pd.DataFrame:
    """Modifies an input DataFrame by populating it with data from the file structure (path).

    If the input DataFrame is empty, this function populates the DataFrame with initial data.
    If the input DataFrame contains data, append to it more data from the filesystem.

    This function does not mutate the input data. Instead, it returns a new DataFrame with the updates (new data).

    Args:
        df (pd.DataFrame): Input dataframe -- either empty or containing previous filesystem data.
        path (str): path/to/leaf/directory/
        target_file (str): Name of TSV file in the current directory.

    Returns:
        New DataFrame with data from the current directory (`target_file` and all `*.tiff` files in the directory).

    """
    cp = pd.DataFrame(df, copy=True)

    files = os.listdir(path)
    tiff_files = list(filter(lambda x: str(x).endswith('.tiff'), files))
    read = pd.read_csv(path + os.sep + target_file, sep='\t')

    mapped_input_file = mutate_input_data(read, path, tiff_files)

    # TODO: write unit tests to prove this fixes a NAN bug
    input_with_images = mapped_input_file.dropna(axis=0, subset=['path'])

    if df.empty:
        cp = input_with_images
    else:
        cp = cp.append(input_with_images)

    return cp


def mutate_input_data(df: pd.DataFrame, root: str, tiff_files: typing.List[str]) -> pd.DataFrame:
    """Modify the input metadata file given the path to the file and list of associated images.

    At minimum, this adds a 'path' column to the input dataframe that contains a path to the 'tiff' file.

    Args:
        df (pd.DataFrame): Read in TSV metadata file.
        root (str): path/to/metadata_file.txt
        tiff_files ([str]): List of names of tiff files

    Returns:
        A new DataFrame with the target column-set and filled in data. Must contain a 'path' column with a path to
        the associated image file.
    """
    cp = pd.DataFrame(df, copy=True)

    def associate_id_with_file(x: str) -> typing.Optional[str]:
        file_num = int(x)

        items = list(filter(lambda y: (_get_image_number(y) - 1) == file_num, tiff_files))

        if items:
            return root.strip().rstrip(os.sep) + os.sep + items[0]

        return None

    cp['path'] = cp['ID'].apply(associate_id_with_file)

    # TODO: check if we want to include ID
    # cp = cp.drop('ID', axis=1)

    return cp


def _get_image_number(image_file_name: str) -> int:
    """Get the number of the image file with a particular pattern at a particular site.

    Args:
        image_file_name (str): Name of the file

    Examples:
        >>> _get_image_number('Ni Patterns 0 Deformation Specimen 1 Speed2 Map Data 2_0001.tiff')
        1
        >>> _get_image_number('Ni Patterns 0 Deformation Specimen 1 Speed2 Map Data 2_0100.tiff')
        100

    Returns:
        (int) The number of the image (related to a pattern's ID) in the filename.

    """
    return int(str(image_file_name).rsplit('_', 1)[1].split('.', 1)[0])


def traverse(root: str, handle_leaf: typing.Callable[[str], None], target_dir=METADATA_FILENAME) -> None:
    """Iterates through the filesystem via depth-first search, applying a consumer function to the target directories.

    Args:
        root (str): Starting position on the filesystem. Will explore all subdirectories of the root.
        handle_leaf: Function that takes in the root filesystem path, may produce side effects.
        target_dir: Directory to apply `handle_leaf` function. Defaults to module constant.
    """
    # Base case: bad input
    if not os.path.exists(root):
        return

    # Base case: missed target
    if os.path.isfile(root):
        return

    if os.path.isdir(root):
        directory_children = os.listdir(root)

        # Base case: target reached
        if target_dir in directory_children:
            handle_leaf(root)

        # Recursive case
        for child in directory_children:
            traverse(root + os.sep + child, handle_leaf)


def make_parser(_parser: typing.Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """The CLI parser for this module, i.e. making a manifest file.

    Args:
        _parser: (optional) an existing argument parser. Can be used to extend the functionality of a larger CLI.

    Returns:
        An argument parser that can process CLI args necessary for marking a manifest file.
    """
    if not _parser:
        _parser = argparse.ArgumentParser(description='Produce an EBSD data manifest file from a root directory.')

    _parser.add_argument('source_directory',
                         type=str,
                         help='Root directory to begin parsing data')

    _parser.add_argument('-o',
                         '--output',
                         type=str,
                         default='manifest.csv',
                         help='/path/to/output.csv Default: `manifest.csv`')

    return _parser


if __name__ == '__main__':
    import sys

    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    make_manifest(args)
