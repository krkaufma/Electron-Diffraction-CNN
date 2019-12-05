import abc
import typing

import keras


class Model(abc.ABC):
    def __init__(self, *args, height=246, width=299, depth=1, n_labels=1, **kwargs):
        self.height = height
        self.width = width
        self.depth = depth
        self.shape = (self.height, self.width, self.depth)
        self.n_labels = n_labels

    @abc.abstractmethod
    def create_model(self, weights_filename: typing.Optional[str] = None):
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def version(self) -> str:
        raise NotImplementedError()

    @property
    def canonical_name(self) -> str:
        return '{name}-{version}'.format(name=self.name, version=self.version)


class ClassificationModel(Model, abc.ABC):

    def __init__(self, *args, n_classes=14,
                 metrics=[keras.metrics.categorical_accuracy, keras.metrics.categorical_crossentropy],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = metrics
        self.n_classes = n_classes


class RegressionModel(Model, abc.ABC):
    def __init__(self, *args, metrics=['mean_squared_error'], **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = metrics

