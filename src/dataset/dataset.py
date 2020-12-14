from config.configs import *
from PIL import Image
import tensorflow as tf
import numpy as np


class DataLoader(object):
    """
    Load train and test dataset
    """

    def __init__(self, params):
        """
        Constructor of DataLoader
        :param params: all input parameters
        """
        self.params = params

        self.path_train_data = training_path.format(self.params.dataset)
        self.path_validation_data = None
        if self.params.validation:
            self.path_validation_data = validation_path.format(self.params.dataset)
        self.path_test_data = test_path.format(self.params.dataset)

        self.num_users, self.num_items = self.get_length()

        # train
        self.training_list = []
        self.load_list('train')

        # validation
        self.validation_list = []
        if self.params.validation:
            self.load_list('val')

        # test
        self.test_list = []
        self.load_list('test')

    def get_length(self):
        with open(dataset_info.format(self.params.dataset), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 2:
                    us = int(line.split(': ')[1])
                if i == 3:
                    it = int(line.split(': ')[1])
                    break
        return us, it

    def load_list(self, train_val_test):
        # Get number of users and items
        u_ = 0
        items = []
        read_path = self.path_train_data if train_val_test == 'train' \
            else self.path_validation_data if train_val_test == 'val' \
            else self.path_test_data
        with open(read_path, "r") as f:
            line = f.readline()
            index = 0
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    if train_val_test == 'train':
                        self.training_list.append(items)
                    elif train_val_test == 'val':
                        self.validation_list.append(items)
                    else:
                        self.test_list.append(items)
                    items = []
                    u_ += 1
                index += 1
                items.append(i)
                line = f.readline()
        if train_val_test == 'train':
            self.training_list.append(items)
        elif train_val_test == 'val':
            self.validation_list.append(items)
        else:
            self.test_list.append(items)

    def all_triple_batches(self):
        r_int = np.random.randint
        actual_used_samples = (self.num_users // self.params.batch_size) * self.params.batch_size
        user_input, pos_input, neg_input = [], [], []

        for ep in range(self.params.epochs):
            for ab in range(actual_used_samples):
                u = r_int(self.num_users)
                ui = set(self.training_list[u])
                lui = len(ui)
                if lui == self.num_items:
                    continue
                i = list(ui)[r_int(lui)]

                j = r_int(self.num_items)
                while j in ui:
                    j = r_int(self.num_items)
                user_input.append(np.array(u))
                pos_input.append(np.array(i))
                neg_input.append(np.array(j))

        return user_input, pos_input, neg_input,

    def next_triple_batch(self):
        all_triples = self.all_triple_batches()
        data = tf.data.Dataset.from_tensor_slices(all_triples)
        data = data.batch(batch_size=self.params.batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    def next_triple_batch_pipeline(self):
        def load_func(u, p, n):
            b = tf.py_function(
                self.read_images_triple,
                (u, p, n,),
                (np.int32, np.int32, np.float32, np.int32, np.float32)
            )
            return b

        all_triples = self.all_triple_batches()
        data = tf.data.Dataset.from_tensor_slices(all_triples)
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=self.params.batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    def read_images_triple(self, user, pos, neg):
        # load positive and negative item images
        im_pos = Image.open(edges_path.format(self.params.dataset) + str(pos.numpy()) + '.tiff')
        im_neg = Image.open(edges_path.format(self.params.dataset) + str(neg.numpy()) + '.tiff')

        try:
            im_pos.load()
        except ValueError:
            print(f'Image at path {pos}.jpg was not loaded correctly!')

        try:
            im_neg.load()
        except ValueError:
            print(f'Image at path {neg}.jpg was not loaded correctly!')

        if im_pos.mode != 'RGB':
            im_pos = im_pos.convert(mode='RGB')
        if im_neg.mode != 'RGB':
            im_neg = im_neg.convert(mode='RGB')

        im_pos = np.array(im_pos.resize((224, 224))) / np.float32(255)
        im_neg = np.array(im_neg.resize((224, 224))) / np.float32(255)
        return user.numpy(), pos.numpy(), im_pos, neg.numpy(), im_neg
