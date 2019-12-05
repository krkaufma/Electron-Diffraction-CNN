from keras.applications.xception import Xception
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
import keras as k

import vecchio.types


class ResNet50Model(vecchio.types.ClassificationModel):

    @property
    def version(self) -> str:
        return '0.0.0'

    def create_model(self, weights_filename=None):
        model = ResNet50(weights=None, input_shape=[self.height, self.width, self.depth],
                         include_top=True, classes=self.n_classes)

        if weights_filename is not None:
            model.load_weights(weights_filename)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=self.metrics)

        return model


class Xception(vecchio.types.ClassificationModel):

    @property
    def version(self) -> str:
        return '0.1.0'

    def create_model(self, weights_filename=None):
        model = Xception(weights=None, input_shape=[self.height, self.width, self.depth],
                         include_top=True, classes=self.n_classes)

        if weights_filename is not None:
            model.load_weights(weights_filename)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=self.metrics)

        return model


