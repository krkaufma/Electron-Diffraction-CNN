import os
import typing as t

import abc
import collections.abc as cabc
import imageio
import keras
import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from vecchio.file_utils import dirfmt


def scale_dims(max_dim, h, w):
    if w > h:
        r = max_dim / w
    else:
        r = max_dim / h
    return int(r * h), int(r * w)


MAX_DIM = 299
HEIGHT, WIDTH = scale_dims(MAX_DIM, 512, 622)
DEPTH = 1


class EBSDSequence(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, ):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return (np.array([filename_to_img(filename) for filename in batch_x]),
                self.prepare_y(batch_y))

    @abc.abstractmethod
    def prepare_y(self, batch_y: cabc.Sequence) -> cabc.Sequence:
        pass


class ClassificationEBSDSequence(EBSDSequence):

    def __init__(self, *args, num_classes=None):
        super().__init__(*args)
        self.num_classes = num_classes

    def prepare_y(self, batch_y: cabc.Sequence) -> cabc.Sequence:
        return keras.utils.to_categorical(batch_y, num_classes=self.num_classes)


class RegressionEBSDSequence(EBSDSequence):
    def prepare_y(self, batch_y: cabc.Sequence) -> cabc.Sequence:
        return batch_y


class MultiLabelRegressionEBSDSequence(RegressionEBSDSequence):
    def __init__(self, *args, n_labels=None):
        super().__init__(*args)

        self.n_labels = 3 if n_labels is None else n_labels

    def prepare_y(self, batch_y: cabc.Sequence) -> cabc.Sequence:
        return np.array(batch_y, dtype=np.float32)


def filename_to_img(filename):
    img = imageio.imread(filename)

    if len(img.shape) < 3:
        img = img[:, :, np.newaxis]

    img_resized = resize(img, (HEIGHT, WIDTH), mode='constant', anti_aliasing=True)
    img_float = img_resized.astype(dtype=np.float32)

    return img_float


def present_manifest(manifest_path: str, label_columns: t.Union[t.Sequence[str], str]) -> \
        t.Mapping[str, t.List[pd.Series]]:
    """Present pre-split data from the manifest in standard way

    Expects split to be `test`, `train`, `validation`, but will work with arbitrary named groups.
    Supports single or multi-label y-values.

    Args:
        manifest_path: /path/to/manifest.csv
        label_columns: one or a list of strings corresponding to the column names from the manifest file.

    Returns:
        A dictionary whose keys are the split group, and values are X, y tuples.
    """
    man_df = pd.read_csv(manifest_path)

    splits = {}

    split_set = man_df['_split'].unique()

    for sp in split_set:
        curr_split = man_df[man_df['_split'] == sp]
        splits[sp] = [curr_split['path'], curr_split[label_columns]]

    return splits


def get_filepaths(data_dir, test_size=0.20, val_size=0.20, *, seed=42):
    """Return a list of filepaths and classes, split into train, test, and validation sets

    Test and train data are split first. Then validation taken, taking a percentage of the train data.

    Args:
        data_dir: source directory for data
        test_size: percentage of data to put in test bucket
        val_size: percentage of data to put in validation bucket
        seed: seed for random partition of the data. Needs to be fixed so model_train and model_eval will work on the
            same test/training data

    Returns:
        a dictionary of tuples. The left value of each tuple is the data, the right is the label. The keys of the
        dictionary are 'train', 'test' and 'validation'.
    """
    dir_contents = os.listdir(data_dir)
    class_folders = [folder for folder in dir_contents if os.path.isdir(dirfmt(data_dir) + dirfmt(folder))]

    img_files = []
    img_labels = []
    c = 0
    for klass in class_folders:
        path = dirfmt(data_dir) + dirfmt(klass) + dirfmt('Images')
        files = os.listdir(path)

        imgs = [path + img for img in files if '.tiff' in img]
        img_files.extend(imgs)
        img_labels.extend([c for _ in range(len(imgs))])

        c += 1

    x_train, x_test, y_train, y_test = train_test_split(img_files, img_labels, test_size=test_size, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed)

    # Lists are preferred over tuples in order to prevent copying the data
    return {'train': [x_train, y_train], 'test': [x_test, y_test], 'validation': [x_val, y_val]}
