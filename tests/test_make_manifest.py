import os
import pandas as pd
from src import make_manifest

TARGET_FILENAME = make_manifest.METADATA_FILENAME
NOT_TARGET_FILENAME = 'another_file.txt'


class TestTraversalSingleTarget:
    TARGET_DIR = os.sep.join(['a', 'b', 'c'])
    NOT_TARGET_DIR = os.sep.join(['a', 'c', 'b'])

    @classmethod
    def setup_class(cls):
        os.makedirs(cls.TARGET_DIR)
        os.makedirs(cls.NOT_TARGET_DIR)

        with open(cls.TARGET_DIR + os.sep + TARGET_FILENAME, 'w') as f:
            f.write('dummy target file')

        with open(cls.NOT_TARGET_DIR + os.sep + NOT_TARGET_FILENAME, 'w') as f:
            f.write('not target file')

    @classmethod
    def teardown_class(cls):
        os.remove(cls.TARGET_DIR + os.sep + TARGET_FILENAME)
        os.remove(cls.NOT_TARGET_DIR + os.sep + NOT_TARGET_FILENAME)
        os.removedirs(cls.TARGET_DIR)
        os.removedirs(cls.NOT_TARGET_DIR)

    def test_traversal_general(self):
        path = []

        def log_path(root):
            path.append(root)

        make_manifest.traverse('a', log_path)

        assert path == [self.TARGET_DIR]


class TestTraversalMultiTarget:
    DIR0 = os.sep.join(['a', 'b', 'c'])
    DIR1 = os.sep.join(['a', 'c', 'b'])
    DIR2 = DIR0.rsplit(os.sep, 1)[0]

    @classmethod
    def setup_class(cls):
        os.makedirs(cls.DIR0)
        os.makedirs(cls.DIR1)

        with open(cls.DIR0 + os.sep + TARGET_FILENAME, 'w') as f:
            f.write('dummy target file')

        with open(cls.DIR2 + os.sep + TARGET_FILENAME, 'w') as f:
            f.write('dummy target file')

        with open(cls.DIR1 + os.sep + NOT_TARGET_FILENAME, 'w') as f:
            f.write('not target file')

    @classmethod
    def teardown_class(cls):
        os.remove(cls.DIR0 + os.sep + TARGET_FILENAME)
        os.remove(cls.DIR1 + os.sep + NOT_TARGET_FILENAME)
        os.remove(cls.DIR2 + os.sep + TARGET_FILENAME)
        os.removedirs(cls.DIR0)
        os.removedirs(cls.DIR1)

    def test_traversal_general(self):
        path = []

        def log_path(root):
            path.append(root)

        make_manifest.traverse('a', log_path)

        assert path == [self.DIR2, self.DIR0]


class TestPopulateDataframe:
    DIR0 = os.sep.join(['a', 'b', 'c'])
    DIR1 = os.sep.join(['a', 'g'])
    test_df = None
    source_df = None

    TARGET_DIRS = [DIR0, DIR1]

    @classmethod
    def setup_class(cls):

        try:
            cls.source_df = pd.read_csv('Grain List.txt', sep='\t')
        except FileNotFoundError:
            cls.source_df = pd.read_csv('tests/Grain List.txt', sep='\t')

        for dir in cls.TARGET_DIRS:
            os.makedirs(dir)

            cls.source_df.to_csv(dir + os.sep + TARGET_FILENAME, sep='\t')

            for i in range(10):
                with open(dir + os.sep + 'target site 1 image number_{0}.tiff'.format(i), 'w') as f:
                    f.write('dummy tiff file')

    @classmethod
    def teardown_class(cls):
        for dir in cls.TARGET_DIRS:
            os.remove(dir + os.sep + TARGET_FILENAME)
            for i in range(10):
                os.remove(dir + os.sep + 'target site 1 image number_{0}.tiff'.format(i))
            os.removedirs(dir)

    def base_df_assertions(self, df: pd.DataFrame):
        # Dataframe should never be empty
        assert not df.empty

        # Dataframe should have (nearly) all the columns from the Grain List TSV
        for c in filter(lambda x: 'ID' not in x, self.source_df.columns):
            assert c in df

        # There should be at least one column with a path to the tiff file
        assert any(['path' in str(c).lower() for c in self.test_df.columns])

        for c in self.test_df.columns:
            if 'path' in str(c).lower():
                series = self.test_df[c]

                # All values should either be None or be a path to a .tiff
                assert all(series.apply(lambda x: '.tiff' in str(x) or x is None))

                # All values should either be None or have the path to the directory
                assert all(series.apply(lambda x: self.DIR0 in str(x) or x is None))

                # Make sure that there are more one string (i.e. a string a None) in the column.
                assert len(series.value_counts()) > 2

    def test_empty_dataframe(self):
        self.test_df = pd.DataFrame()

        assert self.test_df.empty

        self.test_df = make_manifest.populate_dataframe(self.test_df, self.DIR0)

        self.base_df_assertions(self.test_df)

    def test_add_to_existing_dataframe(self):
        self.test_empty_dataframe()

        original_count = len(self.test_df)

        assert original_count > 0

        new_df = make_manifest.populate_dataframe(self.test_df, self.DIR1)

        self.base_df_assertions(new_df)

        current_count = len(new_df)

        assert current_count > original_count
