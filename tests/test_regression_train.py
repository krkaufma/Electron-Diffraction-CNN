import typing
import os
import argparse

from src.regression_train import create_training_sequences, make_parser, prepare_model_dest_dir
from tests.mixins import SetupDummyManifest
from vecchio.types import Model


class TestParseArguments(SetupDummyManifest):

    def test_single_label(self):
        parser = make_parser()

        actual_ns = parser.parse_args('MultiLabelLinearRegressor {manpath} label_0'.format(manpath=self.manifest_path).split())

        assert actual_ns.label_columns is str or len(actual_ns.label_columns) == 1

    def test_multi_label(self):
        parser = make_parser()

        actual_ns = parser.parse_args('MultiLabelLinearRegressor {manpath} label_0 label_1'.format(manpath=self.manifest_path).split())

        assert len(actual_ns.label_columns) == 2


class TestCreateTrainingSequences(SetupDummyManifest):

    def test_single_label(self):
        ns = argparse.Namespace(manifest=self.manifest_path, label_columns='label_0', batch_size=3)

        seqs = create_training_sequences(ns)

        for sq in seqs:
            assert len(sq.y.shape) == 1

            X, y = sq[0]

            assert len(y.shape) == 1

    def test_multi_label(self):
        ns = argparse.Namespace(manifest=self.manifest_path, label_columns=['label_0', 'label_1'], batch_size=3)

        seqs = create_training_sequences(ns)

        for sq in seqs:
            assert len(sq.y.shape) == 2
            assert 2 in sq.y.shape

            X, y = sq[0]

            assert len(y.shape) == 2
            assert 2 in y.shape


class TestPrepareOutputDir:

    def execute(self, out: typing.Optional[str] = None, mdl: typing.Optional[Model] = None) -> str:

        ns = argparse.Namespace(output=out)

        if mdl is None:
            class Dummy(Model):

                @property
                def version(self) -> str:
                    return '0.0.0'

                def create_model(self, weights_filename: typing.Optional[str] = None):
                    return None

            mdl = Dummy()

        return prepare_model_dest_dir(ns, mdl)

    def test_ns_has_output(self):
        actual = self.execute('test_out')
        assert 'test_out' == actual

    def test_no_ns_output(self):
        actual = self.execute()
        assert 'models{sep}Dummy-0.0.0'.format(sep=os.sep) == actual.rsplit('/', 2)[0]


