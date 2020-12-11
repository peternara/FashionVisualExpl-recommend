import logging
from time import time
import os
from abc import ABC
from copy import deepcopy

import numpy as np
import tensorflow as tf
import random

from PIL import Image

from dataset.visual_loader_mixin import VisualLoader
from recommender.models.BPRMF import BPRMF
from recommender.models.cnn import CNN
from utils.write import save_obj
from config.configs import *

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)


class CompVBPR(BPRMF, VisualLoader, ABC):

    def __init__(self, data, params):
        self.initializer = tf.initializers.GlorotUniform()
        super(CompVBPR, self).__init__(data, params)

        self.embed_k = self.params.embed_k
        self.embed_d = self.params.embed_d
        self.activated_components = self.params.activated_components
        self.learning_rate = self.params.lr
        self.l_e = self.params.l_e
        self.l_f = self.params.l_f

        # Initialize model parameters
        if self.activated_components[0]:
            self.process_visual_features()
            self.semantic_weights = dict()
            self.create_semantic_weights()
        if self.activated_components[1]:
            self.process_color_visual_features()
            self.color_weights = dict()
            self.create_color_weights()
        if self.activated_components[2]:
            self.edges_weights = dict()
            self.create_edges_weights()
        if self.activated_components[3]:
            self.process_texture_visual_features()
            self.texture_weights = dict()
            self.create_texture_weights()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def create_semantic_weights(self):
        self.semantic_weights['Bps'] = tf.Variable(
            self.initializer(shape=[self.dim_semantic_feature, 1]), name='Bps', dtype=tf.float32)
        self.semantic_weights['Tus'] = tf.Variable(
            self.initializer(shape=[self.num_users, self.embed_d]),
            name='Tus', dtype=tf.float32)
        self.semantic_weights['Fs'] = tf.Variable(
            self.semantic_features,
            name='Fs', dtype=tf.float32, trainable=False)
        self.semantic_weights['Es'] = tf.Variable(
            self.initializer(shape=[self.dim_semantic_feature, self.embed_d]),
            name='Es', dtype=tf.float32)

    def create_color_weights(self):
        self.color_weights['Bpc'] = tf.Variable(
            self.initializer(shape=[self.dim_color_features, 1]), name='Bpc', dtype=tf.float32)
        self.color_weights['Tuc'] = tf.Variable(
            self.initializer(shape=[self.num_users, self.embed_d]),
            name='Tuc', dtype=tf.float32)
        self.color_weights['Fc'] = tf.Variable(
            self.color_features,
            name='Fc', dtype=tf.float32, trainable=False)
        self.color_weights['Ec'] = tf.Variable(
            self.initializer(shape=[self.dim_color_features, self.embed_d]),
            name='Ec', dtype=tf.float32)

    def create_texture_weights(self):
        self.texture_weights['Bpt'] = tf.Variable(
            self.initializer(shape=[self.dim_texture_features, 1]), name='Bpt', dtype=tf.float32)
        self.texture_weights['Tut'] = tf.Variable(
            self.initializer(shape=[self.num_users, self.embed_d]),
            name='Tut', dtype=tf.float32)
        self.texture_weights['Ft'] = tf.Variable(
            self.texture_features,
            name='Ft', dtype=tf.float32, trainable=False)
        self.texture_weights['Et'] = tf.Variable(
            self.initializer(shape=[self.dim_texture_features, self.embed_d]),
            name='Et', dtype=tf.float32)

    def create_edges_weights(self):
        self.edges_weights['cnn'] = CNN(self.embed_d)
        self.edges_weights['Bpe'] = tf.Variable(
            self.initializer(shape=[self.embed_d, 1]), name='Bpe', dtype=tf.float32)
        self.edges_weights['Tue'] = tf.Variable(
            self.initializer(shape=[self.num_users, self.embed_d]),
            name='Tue', dtype=tf.float32)
        self.edges_weights['Fe'] = np.empty(shape=[self.num_items, self.embed_d], dtype=np.float32)

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
        if self.activated_components[2]:
            user, item, edges = inputs
        else:
            user, item = inputs
            edges = 0

        # USER
        # user collaborative profile
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        if self.activated_components[0]:
            # user semantic features profile
            theta_u_s = tf.nn.embedding_lookup(self.semantic_weights['Tus'], user)
        else:
            theta_u_s = 0

        if self.activated_components[1]:
            # user color features profile
            theta_u_c = tf.nn.embedding_lookup(self.color_weights['Tuc'], user)
        else:
            theta_u_c = 0

        if self.activated_components[2]:
            # user edge features profile
            theta_u_e = tf.nn.embedding_lookup(self.edges_weights['Tue'], user)
        else:
            theta_u_e = 0

        if self.activated_components[3]:
            # user texture features profile
            theta_u_t = tf.nn.embedding_lookup(self.texture_weights['Tut'], user)
        else:
            theta_u_t = 0

        # ITEM
        # item collaborative profile
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        if self.activated_components[0]:
            # item semantic features profile
            semantic_i = tf.squeeze(tf.nn.embedding_lookup(self.semantic_weights['Fs'], item))
            theta_i_s = tf.matmul(semantic_i, self.semantic_weights['Es'])
        else:
            semantic_i = 0
            theta_i_s = 0

        if self.activated_components[1]:
            # item color features profile
            color_i = tf.squeeze(tf.nn.embedding_lookup(self.color_weights['Fc'], item))
            theta_i_c = tf.matmul(color_i, self.color_weights['Ec'])
        else:
            color_i = 0
            theta_i_c = 0

        if self.activated_components[2]:
            # item edge features profile
            theta_i_e = self.edges_weights['cnn'](edges, training=True)
        else:
            theta_i_e = 0

        if self.activated_components[3]:
            # item texture features profile
            texture_i = tf.squeeze(tf.nn.embedding_lookup(self.texture_weights['Ft'], item))
            theta_i_t = tf.matmul(texture_i, self.texture_weights['Et'])
        else:
            texture_i = 0
            theta_i_t = 0

        # BIASES
        # item collaborative bias
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))

        # score prediction
        xui = beta_i + \
              tf.reduce_sum(gamma_u * gamma_i, 1) + \
              (tf.reduce_sum(theta_u_s * theta_i_s, -1) if self.activated_components[0] else 0) + \
              (tf.reduce_sum(theta_u_c * theta_i_c, -1) if self.activated_components[1] else 0) + \
              (tf.reduce_sum(theta_u_e * theta_i_e, -1) if self.activated_components[2] else 0) + \
              (tf.reduce_sum(theta_u_t * theta_i_t, -1) if self.activated_components[3] else 0) + \
              (tf.squeeze(tf.matmul(semantic_i, self.semantic_weights['Bps']))
               if self.activated_components[0] else 0) + \
              (tf.squeeze(tf.matmul(color_i, self.color_weights['Bpc'])) if self.activated_components[1] else 0) + \
              (tf.squeeze(tf.matmul(theta_i_e, self.edges_weights['Bpe'])) if self.activated_components[2] else 0) + \
              (tf.squeeze(tf.matmul(texture_i, self.texture_weights['Bpt'])) if self.activated_components[3] else 0)

        return xui, \
               gamma_u, \
               gamma_i, \
               (semantic_i if semantic_i else None), \
               (theta_u_s if theta_u_s else None), \
               (color_i if color_i else None), \
               (theta_u_c if theta_u_c else None), \
               (texture_i if texture_i else None), \
               (theta_u_t if theta_u_t else None), \
               (theta_i_e if theta_i_e else None), \
               (theta_u_e if theta_u_e else None), \
               beta_i

    def train_step(self, batch):
        """
        Apply a single training step on one batch.

        Args:
            batch: batch used for the current train step

        Returns:
            loss value at the current batch
        """
        if self.activated_components[2]:
            user, pos, edge_pos, neg, edge_neg = batch
        else:
            user, pos, neg = batch
            edge_pos = 0
            edge_neg = 0

        with tf.GradientTape() as t:
            # Clean Inference
            xu_pos, \
                gamma_u, \
                gamma_i_pos, \
                semantic_i_pos, \
                theta_u_s, \
                color_i_pos, \
                theta_u_c, \
                texture_i_pos, \
                theta_u_t, \
                edge_i_pos, \
                theta_u_e, \
                beta_i_pos = self(inputs=(user,
                                          pos,
                                          edge_pos) if self.activated_components[2] else (user, pos), training=True)

            xu_neg, \
                _, \
                gamma_i_neg, \
                semantic_i_neg, \
                _, \
                color_i_neg, \
                _, \
                texture_i_neg, \
                _, \
                edge_i_neg, \
                _, \
                beta_i_neg = self(inputs=(user,
                                          neg,
                                          edge_neg) if self.activated_components[2] else (user, neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u)] + \
                                                [tf.nn.l2_loss(gamma_i_pos)] + \
                                                [tf.nn.l2_loss(gamma_i_neg)] + \
                                                ([tf.nn.l2_loss(theta_u_s)] if theta_u_s else []) + \
                                                ([tf.nn.l2_loss(theta_u_c)] if theta_u_c else []) + \
                                                ([tf.nn.l2_loss(theta_u_t)] if theta_u_t else []) + \
                                                ([tf.nn.l2_loss(theta_u_e)] if theta_u_e else [])) * 2 + \
                       self.l_b * tf.nn.l2_loss(beta_i_pos) * 2 + \
                       self.l_b * tf.nn.l2_loss(beta_i_neg) * 2 / 10 + \
                       self.l_e * tf.reduce_sum(
                        ([tf.nn.l2_loss(self.semantic_weights['Es'])] if self.activated_components[0] else []) + \
                        ([tf.nn.l2_loss(self.color_weights['Ec'])] if self.activated_components[1] else []) + \
                        ([tf.nn.l2_loss(self.texture_weights['Et'])] if self.activated_components[3] else []) + \
                        ([tf.nn.l2_loss(self.semantic_weights['Bps'])] if self.activated_components[0] else []) + \
                        ([tf.nn.l2_loss(self.color_weights['Bpc'])] if self.activated_components[1] else []) + \
                        ([tf.nn.l2_loss(self.texture_weights['Bpt'])] if self.activated_components[3] else []) + \
                        ([tf.nn.l2_loss(self.edges_weights['Bpe'])] if self.activated_components[2] else [])) * 2 + \
                       self.l_f * tf.reduce_sum(
                                        [
                                             tf.nn.l2_loss(layer)
                                             for layer in self.edges_weights['cnn'].trainable_variables
                                             if 'bias' not in layer.name] if self.activated_components[2] else []) * 2

            # Loss to be optimized
            loss += reg_loss

        params = [self.Gu] + [self.Gi] + [self.Bi] + \
            ([self.semantic_weights['Bps']] if self.activated_components[0] else []) + \
            ([self.color_weights['Bpc']] if self.activated_components[1] else []) + \
            ([self.texture_weights['Bpt']] if self.activated_components[3] else []) + \
            ([self.edges_weights['Bpe']] if self.activated_components[2] else []) + \
            ([self.semantic_weights['Es']] if self.activated_components[0] else []) + \
            ([self.color_weights['Ec']] if self.activated_components[1] else []) + \
            ([self.texture_weights['Et']] if self.activated_components[3] else []) + \
            ([self.semantic_weights['Tus']] if self.activated_components[0] else []) + \
            ([self.color_weights['Tuc']] if self.activated_components[1] else []) + \
            ([self.texture_weights['Tut']] if self.activated_components[3] else []) + \
            ([self.edges_weights['Tue']] if self.activated_components[2] else []) + \
            ([layer for layer in self.edges_weights['cnn'].trainable_variables] if self.activated_components[2] else [])
        grads = t.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return loss.numpy()

    def train(self):
        # initialize the max_hr to memorize the best result
        max_hr = 0
        best_model = self
        best_epoch = self.restore_epochs
        results = {}
        if self.activated_components[2]:
            next_batch = self.data.next_triple_batch_pipeline()
        else:
            next_batch = self.data.next_triple_batch()
        steps = 0
        loss = 0
        it = 1
        steps_per_epoch = int(self.data.num_users // self.params.batch_size)

        start_ep = time()

        print('***************************')
        print('Start training...')
        print('***************************')
        for batch in next_batch:
            steps += 1
            loss_batch = self.train_step(batch)
            loss += loss_batch
            # print('\tBatches: %d/%d - Loss: %f' % (steps, steps_per_epoch, loss_batch))

            # epoch is over
            if steps == steps_per_epoch:
                epoch_text = 'Epoch {0}/{1} \tLoss: {2:.3f}'.format(it, self.params.epochs, loss / steps)
                self.evaluator.eval(it, results, epoch_text, start_ep)

                # Print and Log the best result (HR@k)
                if max_hr < results[it]['hr']:
                    max_hr = results[it]['hr']
                    best_epoch = it
                    best_model = deepcopy(self)

                if (it % self.verbose == 0 or it == 1) and self.verbose != -1:
                    self.saver_ckpt.save(f'{weight_dir}/{self.params.dataset}/{self.params.rec}/' + \
                                         f'weights-{it}-{self.learning_rate}-'
                                         f'{list(self.params.activated_components)}')
                start_ep = time()
                it += 1
                loss = 0
                steps = 0

        print('***************************')
        print('Training end...')
        print('***************************')
        self.evaluator.store_recommendation(path=f'{results_dir}/{self.params.dataset}/{self.params.rec}/' + \
                                                 f'recs-{it}-{self.learning_rate}-'
                                                 f'{list(self.params.activated_components)}.tsv')
        save_obj(results,
                 f'{results_dir}/{self.params.dataset}/{self.params.rec}'
                 f'/results-metrics-{self.learning_rate}-{list(self.params.activated_components)}')

        # Store the best model
        print("Store Best Model at Epoch {0}".format(best_epoch))
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=best_model)
        saver_ckpt.save(f'{weight_dir}/{self.params.dataset}/{self.params.rec}/' + \
                        f'best-weights-{best_epoch}-{self.learning_rate}-{list(self.params.activated_components)}')
        best_model.evaluator.store_recommendation(
            path=f'{results_dir}/{self.params.dataset}/{self.params.rec}/' + \
                 f'best-recs-{best_epoch}-{self.learning_rate}-{list(self.params.activated_components)}.tsv')

    def predict_all(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        if self.activated_components[2]:
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
                phi = self.edges_weights['cnn'](im, training=False)
                self.edges_weights['Fe'][index, :] = phi

        if self.activated_components[0]:
            theta_i_s = tf.matmul(self.semantic_weights['Fs'], self.semantic_weights['Es'])
        else:
            theta_i_s = 0

        if self.activated_components[1]:
            theta_i_c = tf.matmul(self.color_weights['Fc'], self.color_weights['Ec'])
        else:
            theta_i_c = 0

        if self.activated_components[2]:
            theta_i_e = tf.Variable(self.edges_weights['Fe'])
        else:
            theta_i_e = 0

        if self.activated_components[3]:
            theta_i_t = tf.matmul(self.texture_weights['Ft'], self.texture_weights['Et'])
        else:
            theta_i_t = 0

        return self.Bi + \
               tf.matmul(self.Gu, self.Gi, transpose_b=True) + \
               (tf.matmul(self.semantic_weights['Tus'], theta_i_s,
                          transpose_b=True) if self.activated_components[0] else 0) + \
               (tf.matmul(self.color_weights['Tuc'], theta_i_c,
                          transpose_b=True) if self.activated_components[1] else 0) + \
               (tf.matmul(self.edges_weights['Tue'], theta_i_e,
                          transpose_b=True) if self.activated_components[2] else 0) + \
               (tf.matmul(self.texture_weights['Tut'], theta_i_t,
                          transpose_b=True) if self.activated_components[3] else 0) + \
               (tf.squeeze(
                   tf.matmul(self.semantic_weights['Fs'],
                             self.semantic_weights['Bps']
                             )
               ) if self.activated_components[0] else 0) + \
               (tf.squeeze(
                   tf.matmul(self.color_weights['Fc'],
                             self.color_weights['Bpc']
                             )
               ) if self.activated_components[1] else 0) + \
               (tf.squeeze(
                   tf.matmul(self.edges_weights['Fe'],
                             self.edges_weights['Bpe']
                             )
               ) if self.activated_components[2] else 0) + \
               (tf.squeeze(
                   tf.matmul(self.texture_weights['Ft'],
                             self.texture_weights['Bpt']
                             )
               ) if self.activated_components[3] else 0)
