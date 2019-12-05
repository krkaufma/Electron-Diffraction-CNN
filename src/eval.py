import typing
import typing as t
import argparse
import vecchio
import vecchio.types
import vecchio.file_utils
import vecchio.make_data
import vecchio.models
from multiprocessing import cpu_count
import pandas as pd


def do_eval(args: argparse.Namespace) -> None:
    """Evaluate Classification Model"""

    # Get and construct user specified model
    found_mdl = find_model(args.trained_model, list(args.models))

    if found_mdl is None:
        print("Couldn't find model that matches path {}".format(args.trained_model))
        return

    mdl_factory = found_mdl()

    model = mdl_factory.create_model(args.trained_model)

    # Create iterator over the test data.
    data_split = vecchio.make_data.present_manifest(args.manifest, args.label_column)

    test = data_split['test']
    x_test, y_test = test

    test_sequence = vecchio.make_data.ClassificationEBSDSequence(x_test, y_test, args.batch_size, num_classes=mdl_factory.n_classes)


    # Perform inference, yield output
    y_pred = model.predict_generator(test_sequence, use_multiprocessing=True, workers=(cpu_count() + 1),
                                     max_queue_size=args.batch_size)

    results = pd.concat([x_test, pd.DataFrame(y_pred, index=x_test.index), y_test], axis=1, join_axes=[x_test.index])

    if args.output is None:
        print(results)
    else:
        results.to_csv(args.output, index=False, header=True)

    # Print model metrics.
    metrics = model.evaluate_generator(test_sequence, use_multiprocessing=True, workers=(cpu_count() + 1),
                                       max_queue_size=args.batch_size)

    print(model.metrics_names)
    print(metrics)


def find_model(path: str, candidates: typing.List[vecchio.types.Model.__class__]) -> \
        typing.Optional[vecchio.types.Model.__class__]:
    for m in candidates:
        mdl = m()
        if mdl.version in path and mdl.__class__.__name__ in path:
            return m

    return None


def make_parser(_parser: typing.Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if not _parser:
        _parser = argparse.ArgumentParser(description='Evaluate a Classification ML Model')

    models = {n: n for n in
              [*vecchio.types.ClassificationModel.__subclasses__()]}

    _parser.add_argument('trained_model',
                         type=str,
                         help='/path/to/model_checkpoint.h5')

    _parser.add_argument('manifest',
                         type=str,
                         help='path/to/manifest.csv')

    _parser.add_argument('label_column',
                         type=str,
                         help='Column name for label (y) from manifest file.')

    _parser.add_argument('-o', '--output',
                         dest='output',
                         type=str,
                         default='eval_output.csv',
                         help='(optional) path/to/output/directory/')

    _parser.add_argument('-bs', '--batch-size',
                         dest='batch_size',
                         type=int,
                         default=32,
                         help='Training batch size (default: 32)')

    _parser.set_defaults(models=models)

    return _parser


if __name__ == '__main__':
    import sys

    do_eval(make_parser().parse_args(sys.argv[1:]))
