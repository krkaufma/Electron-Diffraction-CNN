import importlib
import os
import typing


def dirfmt(d: str) -> str:
    """Normalize a string of a path or directory.

    Args:
        d (str): A string that represents a path.


    Examples:
        >>> dirfmt('path/to/file')
        'path/to/file/'
        >>> dirfmt('path/to/file/')
        'path/to/file/'
        >>> dirfmt(' path/with/whitespace ')
        'path/with/whitespace/'

    Returns:
        A string with a consistent format for directories.

    """
    return d.strip().rstrip(os.sep) + os.sep


def try_mkdir(dirname: str) -> None:
    """Make directory if it does not exist.

    Will create all directories and subdirectories in the tree of the path should any part not exist.

    Args:
        dirname: Directory to check or bring into existance.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def import_from_directory(target_path: str = os.sep.join(['src', 'vecchio', 'models']),
                          package='vecchio.models') -> typing.Iterator:
    """Dynamically import modules from a directory. Defaults to importing models.

    Args:
        target_path (str): Target directory for import
        package (str): that modules belong to.

    Returns:
        A series of dynamically imported modules.
    """
    modules_files = os.listdir(target_path)
    module_names = map(lambda m: m.split('.')[0], modules_files[1:])

    return map(lambda m: importlib.import_module(package + '.' + m), module_names)

