import tensorflow as tf
import numpy as np
import os


class CnnFeatureExtractor:
    def __init__(self, output_layer='avg_pool', model_name='ResNet50', imagenet=None):
        self.model_name = model_name
        self.output_layer = output_layer
        self.imagenet = imagenet

        if self.model_name == 'ResNet50':
            self.model = tf.keras.applications.ResNet50()
        elif self.model_name == 'VGG19':
            self.model = tf.keras.applications.VGG19()
        elif self.model_name == 'ResNet152':
            self.model = tf.keras.applications.ResNet152()
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def classify(self, sample):
        image, filename = sample
        output = self.model.predict(image, batch_size=1)

        return {'ImageID': os.path.splitext(filename)[0],
                'ClassStr': self.imagenet[int(np.argmax(output))],
                'ClassNum': np.argmax(output),
                'Prob': np.amax(output)}

    def extract_feature(self, sample):
        image, filename = sample
        output = tf.keras.Model(self.model.input,
                                self.model.get_layer(self.output_layer).output)(image, training=False)

        return output
