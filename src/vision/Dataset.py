from PIL import Image
from config.configs import *
import tensorflow as tf
import numpy as np
import os


class Dataset:
    def __init__(self, dataset, model_name, resize=None):
        self.directory = images_path.format(dataset)
        self.filenames = os.listdir(self.directory)
        self.filenames.sort(key=lambda x: int(x.split(".")[0]))
        self.num_samples = len(self.filenames)
        self.model_name = model_name
        self.resize = resize

    def resize_and_normalize(self, sample):
        res_sample = sample.resize(self.resize, resample=Image.BICUBIC)

        if self.model_name == 'ResNet50':
            norm_sample = tf.keras.applications.resnet.preprocess_input(np.array(res_sample))
        elif self.model_name == 'VGG19':
            norm_sample = tf.keras.applications.vgg19.preprocess_input(np.array(res_sample))
        elif self.model_name == 'ResNet152':
            norm_sample = tf.keras.applications.resnet.preprocess_input(np.array(res_sample))
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

        return norm_sample

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = Image.open(self.directory + self.filenames[idx])

        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        # norm_sample = self.resize_and_normalize(sample)

        # return np.expand_dims(norm_sample, axis=0), np.array(sample), self.filenames[idx]
        return np.array(sample), self.filenames[idx]
