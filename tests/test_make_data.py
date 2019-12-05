import os
import typing as t
import pandas as pd
from src.vecchio.make_data import present_manifest
from tests.mixins import SetupDummyManifest


class TestPresentManifest(SetupDummyManifest):

    def test_single_label_mulit_group(self):

        splits = present_manifest(self.manifest_path, 'label_0')

        for sp in self.groups:
            assert sp in splits

            x, y = splits[sp]

            assert len(y.shape) == 1

    def test_multi_label_mulit_group(self):
        splits = present_manifest(self.manifest_path, ['label_0', 'label_1'])

        for sp in self.groups:
            assert sp in splits

            x, y = splits[sp]

            assert len(y.shape) == 2
            assert 2 in y.shape

