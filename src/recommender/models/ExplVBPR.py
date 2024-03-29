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


class ExplVBPR(BPRMF, VisualLoader, ABC):

    def __init__(self, data, params):
        self.initializer = tf.initializers.RandomNormal(mean=0, stddev=0.01)
        self.initializer_attentive = tf.initializers.GlorotUniform()
        super(ExplVBPR, self).__init__(data, params)

        self.embed_k = self.params.embed_k
        self.embed_d = self.params.embed_d
        self.attention_layers = self.params.attention_layers
        self.learning_rate = self.params.lr
        self.l_e = self.params.l_e
        self.l_f = self.params.l_f

        self.process_visual_features()
        self.process_color_visual_features()
        self.process_texture_visual_features()

        # Initialize model parameters
        self.semantic_weights = dict()
        self.color_weights = dict()
        self.edges_weights = dict()
        self.texture_weights = dict()

        # Create model parameters
        self.create_semantic_weights()
        self.create_color_weights()
        self.create_edges_weights()
        self.create_texture_weights()

        # Initialize attention parameters
        self.attention_network = dict()
        for layer in range(len(self.attention_layers)):
            self.attention_network['l_{}'.format(layer + 1)] = dict()

        # Create attention parameters
        self.create_attention_weights()

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

    def create_attention_weights(self):
        for layer in range(len(self.attention_layers)):
            if layer == 0:
                self.attention_network['l_{}'.format(layer + 1)]['W'] = tf.Variable(
                    self.initializer_attentive(shape=[self.embed_d, self.attention_layers[layer]]),
                    name='W_{}'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['l_{}'.format(layer + 1)]['b'] = tf.Variable(
                    self.initializer_attentive(shape=[self.attention_layers[layer]]),
                    name='b_{}'.format(layer + 1),
                    dtype=tf.float32
                )
            else:
                self.attention_network['l_{}'.format(layer + 1)]['W'] = tf.Variable(
                    self.initializer_attentive(shape=[self.attention_layers[layer - 1], self.attention_layers[layer]]),
                    name='W_{}'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['l_{}'.format(layer + 1)]['b'] = tf.Variable(
                    self.initializer_attentive(shape=[self.attention_layers[layer]]),
                    name='b_{}'.format(layer + 1),
                    dtype=tf.float32
                )

    def propagate_attention(self, theta_i_s, theta_i_c, theta_i_t, theta_i_e):
        b_s, b_c, b_t, b_e = theta_i_s, theta_i_c, theta_i_t, theta_i_e

        for layer in range(len(self.attention_layers)):
            b_s = tf.matmul(b_s, self.attention_network['l_{}'.format(layer + 1)]['W']) + \
                  self.attention_network['l_{}'.format(layer + 1)]['b']
            b_c = tf.matmul(b_c, self.attention_network['l_{}'.format(layer + 1)]['W']) + \
                  self.attention_network['l_{}'.format(layer + 1)]['b']
            b_t = tf.matmul(b_t, self.attention_network['l_{}'.format(layer + 1)]['W']) + \
                  self.attention_network['l_{}'.format(layer + 1)]['b']
            b_e = tf.matmul(b_e, self.attention_network['l_{}'.format(layer + 1)]['W']) + \
                  self.attention_network['l_{}'.format(layer + 1)]['b']
            if layer == 0:
                b_s = tf.nn.relu(b_s)
                b_c = tf.nn.relu(b_c)
                b_t = tf.nn.relu(b_t)
                b_e = tf.nn.relu(b_e)

        all_b = tf.concat([b_s, b_c, b_t, b_e], axis=1)
        all_beta = tf.nn.softmax(all_b, axis=1)
        return all_beta

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
        # user semantic features profile
        theta_u_s = tf.expand_dims(tf.nn.embedding_lookup(self.semantic_weights['Tus'], user), axis=1)
        # user color features profile
        theta_u_c = tf.expand_dims(tf.nn.embedding_lookup(self.color_weights['Tuc'], user), axis=1)
        # user texture features profile
        theta_u_t = tf.expand_dims(tf.nn.embedding_lookup(self.texture_weights['Tut'], user), axis=1)
        # user edge features profile
        theta_u_e = tf.expand_dims(tf.nn.embedding_lookup(self.edges_weights['Tue'], user), axis=1)
        all_theta_u = tf.transpose(
            tf.concat([theta_u_s, theta_u_c, theta_u_t, theta_u_e], axis=1),
            perm=[1, 0, 2]
        )

        # ITEM
        # item collaborative profile
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        # item semantic features profile
        semantic_i = tf.squeeze(tf.nn.embedding_lookup(self.semantic_weights['Fs'], item))
        theta_i_s = tf.expand_dims(tf.matmul(semantic_i, self.semantic_weights['Es']), axis=1)
        # item color features profile
        color_i = tf.squeeze(tf.nn.embedding_lookup(self.color_weights['Fc'], item))
        theta_i_c = tf.expand_dims(tf.matmul(color_i, self.color_weights['Ec']), axis=1)
        # item texture features profile
        texture_i = tf.squeeze(tf.nn.embedding_lookup(self.texture_weights['Ft'], item))
        theta_i_t = tf.expand_dims(tf.matmul(texture_i, self.texture_weights['Et']), axis=1)
        # item edge features profile
        theta_i_e = tf.expand_dims(self.edges_weights['cnn'](edges, training=True), axis=1)
        all_theta_i = tf.concat([theta_i_s, theta_i_c, theta_i_t, theta_i_e], axis=1)
        # attention network
        all_attention = self.propagate_attention(theta_i_s,
                                                 theta_i_c,
                                                 theta_i_t,
                                                 theta_i_e)

        # BIASES
        # item collaborative bias
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))

        # score prediction
        xui = beta_i + \
              tf.reduce_sum(gamma_u * gamma_i, 1) + \
              tf.reduce_sum(tf.matmul(all_theta_u, tf.transpose(tf.multiply(all_attention, all_theta_i),
                                                                perm=[1, 2, 0])), 0) + \
              tf.squeeze(tf.matmul(semantic_i, self.semantic_weights['Bps'])) + \
              tf.squeeze(tf.matmul(color_i, self.color_weights['Bpc'])) + \
              tf.squeeze(tf.matmul(texture_i, self.texture_weights['Bpt'])) + \
              tf.squeeze(tf.matmul(theta_i_e, self.edges_weights['Bpe']))

        return xui, \
               gamma_u, \
               gamma_i, \
               semantic_i, \
               theta_u_s, \
               color_i, \
               theta_u_c, \
               texture_i, \
               theta_u_t, \
               theta_i_e, \
               theta_u_e, \
               beta_i

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
                semantic_i_pos, \
                theta_u_s, \
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
                semantic_i_neg, \
                _, \
                color_i_neg, \
                _, \
                texture_i_neg, \
                _, \
                edge_i_neg, \
                _, \
                beta_i_neg = self(inputs=(user, neg, edge_neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_i_pos),
                                                 tf.nn.l2_loss(gamma_i_neg),
                                                 tf.nn.l2_loss(theta_u_s),
                                                 tf.nn.l2_loss(theta_u_c),
                                                 tf.nn.l2_loss(theta_u_t),
                                                 tf.nn.l2_loss(theta_u_e)]) * 2 + \
                       self.l_b * tf.nn.l2_loss(beta_i_pos) * 2 + \
                       self.l_b * tf.nn.l2_loss(beta_i_neg) * 2 / 10 + \
                       self.l_e * tf.reduce_sum([tf.nn.l2_loss(self.semantic_weights['Es']),
                                                 tf.nn.l2_loss(self.color_weights['Ec']),
                                                 tf.nn.l2_loss(self.texture_weights['Et']),
                                                 tf.nn.l2_loss(self.semantic_weights['Bps']),
                                                 tf.nn.l2_loss(self.color_weights['Bpc']),
                                                 tf.nn.l2_loss(self.texture_weights['Bpt']),
                                                 tf.nn.l2_loss(self.edges_weights['Bpe'])]) * 2 + \
                       self.l_f * tf.reduce_sum([tf.nn.l2_loss(layer)
                                                 for layer in self.edges_weights['cnn'].trainable_variables
                                                 if 'bias' not in layer.name]) * 2

            # Loss to be optimized
            loss += reg_loss

        params = [
            self.Gu, self.Gi,
            self.Bi,
            self.semantic_weights['Bps'], self.color_weights['Bpc'],
            self.texture_weights['Bpt'], self.edges_weights['Bpe'],
            self.semantic_weights['Es'], self.color_weights['Ec'], self.texture_weights['Et'],
            self.semantic_weights['Tus'], self.color_weights['Tuc'],
            self.texture_weights['Tut'], self.edges_weights['Tue'],
            *[layer for layer in self.edges_weights['cnn'].trainable_variables]
        ]
        grads = t.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return loss.numpy()

    def train(self):
        max_metrics = {'hr': 0, 'p': 0, 'r': 0, 'auc': 0, 'ndcg': 0}
        best_model = self
        best_epoch = self.restore_epochs
        best_epoch_print = 'No best epoch found!'
        results = {}
        next_batch = self.data.next_triple_batch_pipeline()
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

            # epoch is over
            if steps == steps_per_epoch:
                epoch_text = 'Epoch {0}/{1} \tLoss: {2:.3f}'.format(it, self.params.epochs, loss / steps)
                epoch_print = self.evaluator.eval(it, results, epoch_text, start_ep)

                for metric in max_metrics.keys():
                    if max_metrics[metric] <= results[it][metric]:
                        max_metrics[metric] = results[it][metric]
                        if metric == self.params.best_metric:
                            best_epoch, best_model, best_epoch_print = it, deepcopy(self), epoch_print

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
        print(best_epoch_print)
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=best_model)
        saver_ckpt.save(f'{weight_dir}/{self.params.dataset}/{self.params.rec}/' + \
                        f'best-weights-{best_epoch}-{self.learning_rate}-{list(self.params.activated_components)}')
        best_model.evaluator.store_recommendation(
            path=f'{results_dir}/{self.params.dataset}/{self.params.rec}/' + \
                 f'best-recs-{best_epoch}-{self.learning_rate}-{list(self.params.activated_components)}.tsv')
        print('End Store Best Model!')

        print('Best Values for Each Metric:\nHR\tPrec\tRec\tAUC\tnDCG\n{}\t{}\t{}\t{}\t{}\n'.format(
            max_metrics['hr'],
            max_metrics['p'],
            max_metrics['r'],
            max_metrics['auc'],
            max_metrics['ndcg']
        ))

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
            phi = self.edges_weights['cnn'](im, training=False)
            self.edges_weights['Fe'][index, :] = phi

        all_theta_u = tf.transpose(tf.concat([
            tf.expand_dims(self.semantic_weights['Tus'], axis=1),
            tf.expand_dims(self.color_weights['Tuc'], axis=1),
            tf.expand_dims(self.texture_weights['Tut'], axis=1),
            tf.expand_dims(self.edges_weights['Tue'], axis=1)
        ], axis=1), perm=[1, 0, 2])

        theta_i_s = tf.expand_dims(tf.matmul(self.semantic_weights['Fs'], self.semantic_weights['Es']), axis=1)
        theta_i_c = tf.expand_dims(tf.matmul(self.color_weights['Fc'], self.color_weights['Ec']), axis=1)
        theta_i_t = tf.expand_dims(tf.matmul(self.texture_weights['Ft'], self.texture_weights['Et']), axis=1)
        theta_i_e = tf.expand_dims(tf.Variable(self.edges_weights['Fe']), axis=1)
        all_theta_i = tf.concat([
            theta_i_s,
            theta_i_c,
            theta_i_t,
            theta_i_e
        ], axis=1)

        all_attention = self.propagate_attention(theta_i_s,
                                                 theta_i_c,
                                                 theta_i_t,
                                                 theta_i_e)

        return self.Bi + \
               tf.matmul(self.Gu, self.Gi, transpose_b=True) + \
               tf.reduce_sum(tf.matmul(all_theta_u, tf.transpose(tf.multiply(all_attention, all_theta_i),
                                                                    perm=[1, 2, 0])), 0) + \
               tf.squeeze(tf.matmul(self.semantic_weights['Fs'], self.semantic_weights['Bps'])) + \
               tf.squeeze(tf.matmul(self.color_weights['Fc'], self.color_weights['Bpc'])) + \
               tf.squeeze(tf.matmul(self.texture_weights['Ft'], self.texture_weights['Bpt'])) + \
               tf.squeeze(tf.matmul(self.edges_weights['Fe'], self.edges_weights['Bpe']))
