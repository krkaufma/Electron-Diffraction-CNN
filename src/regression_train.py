import argparse
import multiprocessing as mp
import os
import time
import typing
import typing as t

import keras

import vecchio
import vecchio.file_utils
import vecchio.make_data
import vecchio.models
import vecchio.types
from vecchio.make_data import RegressionEBSDSequence


def do_train(args: argparse.Namespace) -> None:
    train_sequence, validation_sequence = create_training_sequences(args)

    model_factory = args.models[args.model](n_labels=len(args.label_columns))
    model = model_factory.create_model()

    output = prepare_model_dest_dir(args, model_factory)
    os.makedirs(output, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=args.min_delta,
                                      patience=args.patience,
                                      verbose=0,
                                      restore_best_weights=True),

        keras.callbacks.ModelCheckpoint(filepath=output + 'model_checkpoint.h5',
                                        verbose=0,
                                        save_weights_only=False,
                                        save_best_only=True),

        keras.callbacks.CSVLogger(filename=output + 'training_log.csv', separator=',', append=True)
    ]

    model.fit_generator(generator=train_sequence,
                        validation_data=validation_sequence,
                        epochs=args.epochs,
                        steps_per_epoch=len(train_sequence.x) / args.batch_size,
                        verbose=2,
                        use_multiprocessing=True,
                        workers=mp.cpu_count() + 1,
                        max_queue_size=args.batch_size,
                        callbacks=callbacks)

    # model.save(output + 'final_model')


def create_training_sequences(args: argparse.Namespace) -> t.Tuple[RegressionEBSDSequence, RegressionEBSDSequence]:
    data_split = vecchio.make_data.present_manifest(args.manifest, args.label_columns)

    x_train, y_train = data_split['train']
    x_val, y_val = data_split['validation']

    train_sequence = vecchio.make_data.MultiLabelRegressionEBSDSequence(x_train, y_train, args.batch_size)
    validation_sequence = vecchio.make_data.MultiLabelRegressionEBSDSequence(x_val, y_val, args.batch_size)

    return train_sequence, validation_sequence


def prepare_model_dest_dir(args: argparse.Namespace, model: vecchio.types.Model) -> str:
    output = args.output
    if output is None:
        output = vecchio.file_utils.dirfmt('models{sep}{name}{sep}{time}'
                .format(sep=os.sep, name=model.canonical_name, time=time.strftime("%Y-%m-%d-%I-%M")))

    return output


def make_parser(_parser: typing.Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if not _parser:
        _parser = argparse.ArgumentParser(description='Train a Regression ML Model')

    models = {n.__name__: n for n in vecchio.types.RegressionModel.__subclasses__()}

    _parser.add_argument('model',
                         choices=list(models.keys()),
                         help='The model to use for regression.')

    _parser.add_argument('manifest',
                         type=str,
                         help='path/to/manifest.csv')

    _parser.add_argument('label_columns',
                         type=str,
                         nargs='+',
                         help='Column(s) name for label (y) from manifest file.')

    _parser.add_argument('-bs', '--batch-size',
                         dest='batch_size',
                         type=int,
                         default=32,
                         help='Training batch size (default: 32)')

    _parser.add_argument('-e', '--epochs',
                         dest='epochs',
                         type=int,
                         default=10000,
                         help='Number of epochs (default: 10000)')

    _parser.add_argument('-md', '--min-delta',
                         dest='min_delta',
                         type=float,
                         default=0.001,
                         help='Minimum change int the monitored quantity to qualify as an improvement. Default: 0.001')

    _parser.add_argument('-p', '--patience',
                         dest='patience',
                         type=int,
                         default=25,
                         help='Number of epochs with no improvement after which training will be stopped Default: 25')

    _parser.add_argument('-o', '--output',
                         dest='output',
                         type=str,
                         default=None,
                         help='(optional) path/to/output/directory/')

    _parser.set_defaults(models=models)

    return _parser


if __name__ == '__main__':
    import sys

    do_train(make_parser().parse_args(sys.argv[1:]))
