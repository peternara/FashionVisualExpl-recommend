import logging
import os
from abc import ABC

import numpy as np
import tensorflow as tf
import random

from PIL import Image

from dataset.visual_loader_mixin import VisualLoader
from recommender.models.BPRMF import BPRMF
from recommender.models.cnn import CNN
from config.configs import *

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ExplVBPR(BPRMF, VisualLoader, ABC):

    def __init__(self, data, params):
        super(ExplVBPR, self).__init__(data, params)

        self.embed_k = self.params.embed_k
        self.embed_d_v = self.params.embed_d_v
        self.embed_d_t = self.params.embed_d_t
        self.learning_rate = self.params.lr
        self.l_e = self.params.l_e
        self.l_f = self.params.l_f

        self.process_visual_features()
        self.process_expl_visual_features()

        # Initialize Model Parameters

        # CNN's visual features
        self.Bpf = tf.Variable(
            self.initializer(shape=[self.dim_visual_feature, 1]), name='Bpf', dtype=tf.float32)
        self.Tuf = tf.Variable(
            self.initializer(shape=[self.num_users, self.embed_d_f]),
            name='Tuf', dtype=tf.float32)
        self.F = tf.Variable(
            self.visual_features,
            name='F', dtype=tf.float32, trainable=False)
        self.Ef = tf.Variable(
            self.initializer(shape=[self.dim_visual_feature, self.embed_d_f]),
            name='Ef', dtype=tf.float32)

        # color features
        self.Bpc = tf.Variable(
            self.initializer(shape=[self.dim_color_features, 1]), name='Bpc', dtype=tf.float32)
        self.Tuc = tf.Variable(
            self.initializer(shape=[self.num_users, self.color_features]),
            name='Tuc', dtype=tf.float32)
        self.C = tf.Variable(
            self.color_features,
            name='C', dtype=tf.float32, trainable=False)

        # texture features
        self.Bpt = tf.Variable(
            self.initializer(shape=[self.dim_texture_features, 1]), name='Bpt', dtype=tf.float32)
        self.Tut = tf.Variable(
            self.initializer(shape=[self.num_users, self.embed_d_t]),
            name='Tut', dtype=tf.float32)
        self.T = tf.Variable(
            self.texture_features,
            name='T', dtype=tf.float32, trainable=False)
        self.Et = tf.Variable(
            self.initializer(shape=[self.dim_texture_features, self.embed_d_t]),
            name='Et', dtype=tf.float32)

        # edge features
        self.cnn = CNN(self.embed_k)
        self.Bpe = tf.Variable(
            self.initializer(shape=[self.embed_k, 1]), name='Bpe', dtype=tf.float32)
        self.Tue = tf.Variable(
            self.initializer(shape=[self.num_users, self.embed_k]),
            name='Tue', dtype=tf.float32)
        self.Edge = np.empty(shape=[self.num_items, self.embed_k], dtype=np.float32)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def call(self, inputs, training=None, mask=None):
        """
        Generates prediction for passed users and items indices

        Args:
            inputs: user, item (batch)
            training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).

        Returns:
            prediction and extracted model parameters
        """
        user, item, edges = inputs

        # USER
        # user collaborative profile
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        # user visual features profile
        theta_u_f = tf.squeeze(tf.nn.embedding_lookup(self.Tuf, user))
        # user color features profile
        theta_u_c = tf.squeeze(tf.nn.embedding_lookup(self.Tuc, user))
        # user texture features profile
        theta_u_t = tf.squeeze(tf.nn.embedding_lookup(self.Tut, user))
        # user edge features profile
        theta_u_e = tf.squeeze(tf.nn.embedding_lookup(self.Tue, user))

        # ITEM
        # item collaborative profile
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        # item visual features profile
        feature_i = tf.squeeze(tf.nn.embedding_lookup(self.F, item))
        # item color features profile
        color_i = tf.squeeze(tf.nn.embedding_lookup(self.C, item))
        # item texture features profile
        texture_i = tf.squeeze(tf.nn.embedding_lookup(self.T, item))
        # item edge features profile
        edge_i = self.cnn(edges, training=True)

        # BIASES
        # item collaborative bias
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))

        # score prediction
        xui = beta_i + \
              tf.reduce_sum(gamma_u * gamma_i, 1) + \
              tf.reduce_sum(theta_u_f * tf.matmul(feature_i, self.E), 1) + \
              tf.reduce_sum(theta_u_c * color_i, 1) + \
              tf.reduce_sum(theta_u_t * tf.matmul(texture_i, self.T), 1) + \
              tf.reduce_sum(theta_u_e * edge_i, 1) + \
              tf.squeeze(tf.matmul(feature_i, self.Bpf)) + \
              tf.squeeze(tf.matmul(color_i, self.Bpc)) + \
              tf.squeeze(tf.matmul(texture_i, self.Bpt)) + \
              tf.squeeze(tf.matmul(edge_i, self.Bpe))

        return xui, \
               gamma_u, \
               gamma_i, \
               feature_i, \
               theta_u_f, \
               color_i, \
               theta_u_c, \
               texture_i, \
               theta_u_t, \
               edge_i, \
               theta_u_e, \
               beta_i

    def predict_all(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        edges_list = os.listdir(edges_path.format(self.dataset_name))
        edges_list.sort(key=lambda x: int(x.split(".")[0]))
        for index, item in enumerate(edges_list):
            im = Image.open(edges_path.format(self.dataset_name) + item)
            try:
                im.load()
            except ValueError:
                print(f'Image at path {images_path.format(self.dataset_name) + item} was not loaded correctly!')
            if im.mode != 'RGB':
                im = im.convert(mode='RGB')
            im = np.reshape(np.array(im.resize((224, 224))) / np.float32(255), (1, 224, 224, 3))
            phi = self.cnn(im, training=False)
            self.Edge[index, :] = phi

        return self.Bi + \
               tf.matmul(self.Gu, self.Gi, transpose_b=True) + \
               tf.matmul(self.Tuf, tf.matmul(self.F, self.Ef), transpose_b=True) + \
               tf.squeeze(tf.matmul(self.F, self.Bpf)) + \
               tf.matmul(self.Tuc, self.C, transpose_b=True) + \
               tf.squeeze(tf.matmul(self.C, self.Bpc)) + \
               tf.matmul(self.Tut, tf.matmul(self.T, self.Et), transpose_b=True) + \
               tf.squeeze(tf.matmul(self.T, self.Bpt)) + \
               tf.matmul(self.Tue, tf.Variable(self.Edge), transpose_b=True) + \
               tf.squeeze(tf.matmul(self.Tue, self.Bpe))

    def train_step(self, batch):
        """
        Apply a single training step on one batch.

        Args:
            batch: batch used for the current train step

        Returns:
            loss value at the current batch
        """
        user, pos, edge_pos, neg, edge_neg = batch
        with tf.GradientTape() as t:
            # Clean Inference
            xu_pos, \
                gamma_u, \
                gamma_i_pos, \
                feature_i_pos, \
                theta_u_f, \
                color_i_pos, \
                theta_u_c, \
                texture_i_pos, \
                theta_u_t, \
                edge_i_pos, \
                theta_u_e, \
                beta_i_pos = self(inputs=(user, pos, edge_pos), training=True)

            xu_neg, \
                _, \
                gamma_i_neg, \
                feature_i_neg, \
                _, \
                color_i_neg, \
                _, \
                texture_i_neg, \
                _, \
                edge_i_neg, \
                _, \
                beta_i_neg = self(inputs=(user, neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_i_pos),
                                                 tf.nn.l2_loss(gamma_i_neg),
                                                 tf.nn.l2_loss(theta_u_f),
                                                 tf.nn.l2_loss(theta_u_c),
                                                 tf.nn.l2_loss(theta_u_t),
                                                 tf.nn.l2_loss(theta_u_e)]) + \
                       self.l_b * tf.nn.l2_loss(beta_i_pos) + \
                       self.l_b * tf.nn.l2_loss(beta_i_neg) / 10 + \
                       self.l_e * tf.reduce_sum([tf.nn.l2_loss(self.Ef),
                                                 tf.nn.l2_loss(self.Et),
                                                 tf.nn.l2_loss(self.Bp)]) + \
                       self.l_f * tf.reduce_sum([tf.nn.l2_loss(layer)
                                                 for layer in self.cnn.trainable_variables
                                                 if 'bias' not in layer.name])

            # Loss to be optimized
            loss += reg_loss

        params = [
            self.Gu, self.Gi,
            self.Bi, self.Bpf, self.Bpc, self.Bpt, self.Bpe,
            self.Ef, self.Et,
            self.Tuf, self.Tuc, self.Tut, self.Tue,
            *[layer for layer in self.cnn.trainable_variables]
        ]
        grads = t.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return loss.numpy()
