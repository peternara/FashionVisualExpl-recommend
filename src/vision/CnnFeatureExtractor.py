import tensorflow as tf
import numpy as np
import cv2
import os


class CnnFeatureExtractor:
    def __init__(self, gram_output_layers, resize_gram, output_layer='avg_pool', model_name='ResNet50', imagenet=None):
        self.model_name = model_name
        self.output_layer = output_layer
        self.imagenet = imagenet
        self.gram_output_layers = gram_output_layers
        self.resize_gram = resize_gram

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

    def extract_texture(self, sample):
        image, filename = sample
        gram_matrices = np.empty(shape=[len(self.gram_output_layers), np.prod(self.resize_gram)])

        for i, layer in enumerate(self.gram_output_layers):
            # extract feature maps
            current_model = tf.keras.Model(self.model.input, self.model.get_layer(layer).output)
            current_f_maps = current_model(image, training=False)
            del current_model

            # create gram matrix
            current_f_maps = tf.reshape(current_f_maps,
                                        [current_f_maps.shape[3], current_f_maps.shape[1] * current_f_maps.shape[2]])
            current_gram_matrix = (tf.matmul(current_f_maps,
                                             current_f_maps,
                                             transpose_b=True) / np.prod(current_f_maps.shape)).numpy()
            del current_f_maps
            gram_matrices[i] = cv2.resize(current_gram_matrix,
                                          dsize=self.resize_gram,
                                          interpolation=cv2.INTER_CUBIC).flatten()

        return gram_matrices.flatten()
