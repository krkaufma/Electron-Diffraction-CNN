import os
import itertools

import pandas as pd


class SetupDummyManifest:
    manifest_path = ''
    N = 10
    groups = ['train', 'test', 'validation']

    @classmethod
    def setup_class(cls):
        cls.manifest_path = 'manifest.csv'

        manf = pd.DataFrame({'path': list(itertools.islice(itertools.repeat('tests{sep}sample_ebsd.tiff'.format(sep=os.sep)), cls.N)),
                             '_split': list(itertools.islice(itertools.cycle(cls.groups), cls.N)),
                             'label_0': [x % 2 + 1 for x in range(cls.N)],
                             'label_1': [x % 5 + 2 for x in range(cls.N)]})

        manf.to_csv(cls.manifest_path)

    @classmethod
    def teardown_class(cls):
        os.remove(cls.manifest_path)
