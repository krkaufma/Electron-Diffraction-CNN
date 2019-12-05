import argparse
import os
import time
import typing
from multiprocessing import cpu_count

import keras
import numpy as np
import sklearn
import vecchio
import vecchio.models
from vecchio import make_data
import vecchio.types
import vecchio as v
from vecchio.file_utils import dirfmt, import_from_directory


def do_train(args: argparse.Namespace) -> None:
    """Train models with a consistent process, determined by CLI parameters.

    Args:
        args: arguments from :func:`train.make_parser`
    """

    data_split = v.make_data.present_manifest(args.manifest, args.label_column)

    x_train, y_train = data_split['train']
    x_val, y_val = data_split['validation']

    model_factory = args.models[args.model]()
    model = model_factory.create_model()

    train_sequence = v.make_data.ClassificationEBSDSequence(x_train, y_train, args.batch_size, num_classes=model_factory.n_classes)
    validation_sequence = v.make_data.ClassificationEBSDSequence(x_val, y_val, args.batch_size, num_classes=model_factory.n_classes)

    output = args.output
    if output is None:
        output = dirfmt('models' + os.sep + model_factory.canonical_name + os.sep + time.strftime('%Y-%m-%d-%I-%M'))

    os.makedirs(output, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=args.min_delta,
                                      patience=args.patience,
                                      verbose=0),

        keras.callbacks.ModelCheckpoint(filepath=output + 'model_checkpoint.h5',
                                        verbose=0,
                                        save_weights_only=False,
                                        save_best_only=True),

        keras.callbacks.CSVLogger(filename=output + 'training_log.csv', separator=',', append=True)
    ]

    y_weights = None
    if args.should_weight_classes:
        y_weights = get_balanced_class_weight(y_train)

    model.fit_generator(generator=train_sequence,
                        validation_data=validation_sequence,
                        class_weight=y_weights,
                        epochs=args.epochs,
                        steps_per_epoch=len(x_train) / args.batch_size,
                        verbose=2,
                        use_multiprocessing=True,
                        workers=cpu_count() + 1,
                        max_queue_size=args.batch_size,
                        callbacks=callbacks)


def get_balanced_class_weight(labels):
    weight_vec = sklearn.utils.compute_class_weight('balanced', np.unique(labels), labels)
    weight_dict = dict(enumerate(weight_vec))
    return weight_dict


def make_parser(_parser: typing.Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if not _parser:
        _parser = argparse.ArgumentParser(description='Train a Classification ML Model')

    models = {n.__name__: n for n in vecchio.types.ClassificationModel.__subclasses__()}

    _parser.add_argument('model',
                         choices=list(models.keys()),
                         help='The model to use for classification.')

    _parser.add_argument('manifest',
                         type=str,
                         help='path/to/manifest.csv')

    _parser.add_argument('label_column',
                         type=str,
                         help='Column name for label (y) from manifest file.')

    _parser.add_argument('--weight-classes',
                         dest='should_weight_classes',
                         action='store_true',
                         default=False,
                         help='Flag that turns on class balancing (should not be used with regression).')

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

