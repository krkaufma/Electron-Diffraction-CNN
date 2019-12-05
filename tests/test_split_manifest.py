import argparse
import os
import typing as t

import numpy as np
import pandas as pd

import src.split_manifest
from src.make_manifest import METADATA_FILENAME


class TestHappyPath:

    @classmethod
    def setup_class(cls):
        grain_list = pd.read_csv('tests' + os.sep + METADATA_FILENAME, sep='\t')
        eg_manifest = grain_list.copy(deep=True)
        eg_manifest.assign(path=pd.Series(map(str, np.random.randn(len(grain_list['ID'])))))

        cls.output_name = 'test_manifest.csv'
        eg_manifest.to_csv(cls.output_name, index=False, header=True)

    @classmethod
    def teardown_class(cls):
        os.remove(cls.output_name)

    def try_split(self, label_column='Phase', test_size=0.2, val_size=0.2, seed=None, split_method='shuffle',
                  output=None) -> t.Optional[pd.DataFrame]:

        args = argparse.Namespace(manifest=self.output_name,
                                  label_column=label_column, test_size=test_size, val_size=val_size, seed=seed,
                                  split_method=split_method, output=output)

        src.split_manifest.split_dataset(args)

        target = self.output_name
        if output is not None:
            target = output

        return pd.read_csv(target)

    def split_test(self, val_size=0.2, test_size=0.2):

        df = self.try_split(val_size=val_size, test_size=test_size)
        if df is None:
            return

        assert '_split' in list(df.columns)

        split_series = df['_split']

        total = split_series.count()
        hist = split_series.value_counts()

        assert len(hist) == 3
        assert within_tol(hist['test'] / total, test_size)
        assert within_tol(hist['validation'] / total, val_size)
        assert within_tol(hist['train'] / total, 1 - val_size - test_size)

    def test_default_split(self):
        self.split_test()

    def test_small_val_split(self):
        self.split_test(val_size=0.1)

    def test_impossible_split(self):
        try:
            self.split_test(val_size=0.51, test_size=0.51)
        except AssertionError:
            assert True
        else:
            assert False


def within_tol(actual: float, expected: float, tol=0.1) -> bool:
    return abs(actual - expected) <= tol
