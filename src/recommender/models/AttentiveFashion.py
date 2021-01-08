import logging
from time import time
from abc import ABC
from copy import deepcopy

import numpy as np
import tensorflow as tf
import random

from dataset.visual_loader_mixin import VisualLoader
from recommender.models.BPRMF import BPRMF
from utils.write import save_obj
from config.configs import *

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)


class AttentiveFashion(BPRMF, VisualLoader, ABC):

    def __init__(self, data, params):
        self.initializer = tf.initializers.RandomNormal(mean=0, stddev=0.01)
        self.initializer_attentive = tf.initializers.GlorotUniform()
        super(AttentiveFashion, self).__init__(data, params)

        self.embed_k = self.params.embed_k
        self.embed_d = self.params.embed_d
        self.attention_layers = self.params.attention_layers
        self.learning_rate = self.params.lr

        self.process_edge_visual_features()
        self.process_color_visual_features()

        # Initialize model parameters
        self.color_weights = dict()
        self.edges_weights = dict()

        # Create model parameters
        self.create_color_weights()
        self.create_edges_weights()

        # Initialize attention parameters
        self.attention_network = dict()

        # Create attention parameters
        self.create_attention_weights()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def create_color_weights(self):
        self.color_weights['Fc'] = tf.Variable(
            self.color_features,
            name='Fc', dtype=tf.float32, trainable=False)
        self.color_weights['E1c'] = tf.Variable(
            self.initializer(shape=[self.dim_color_features, self.embed_d]),
            name='E1c', dtype=tf.float32)
        self.color_weights['E2c'] = tf.Variable(
            self.initializer(shape=[self.embed_d, self.embed_k]),
            name='E2c', dtype=tf.float32)

    def create_edges_weights(self):
        self.edges_weights['Fe'] = tf.Variable(
            self.edge_features,
            name='Fe', dtype=tf.float32, trainable=False)
        self.edges_weights['E1e'] = tf.Variable(
            self.initializer(shape=[self.dim_edge_features, self.embed_d]),
            name='E1e', dtype=tf.float32)
        self.edges_weights['E2e'] = tf.Variable(
            self.initializer(shape=[self.embed_d, self.embed_k]),
            name='E2e', dtype=tf.float32)

    def create_attention_weights(self):
        for layer in range(len(self.attention_layers)):
            if layer == 0:
                self.attention_network['l_{}_Wu'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self.embed_k, self.attention_layers[layer]]),
                    name='W_{}'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['l_{}_Wf'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self.embed_k, self.attention_layers[layer]]),
                    name='W_{}'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['l_{}_b'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self.attention_layers[layer]]),
                    name='b_{}'.format(layer + 1),
                    dtype=tf.float32
                )
            else:
                self.attention_network['l_{}_W'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self.attention_layers[layer - 1], self.attention_layers[layer]]),
                    name='W_{}'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['l_{}_b'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self.attention_layers[layer]]),
                    name='b_{}'.format(layer + 1),
                    dtype=tf.float32
                )

    def propagate_attention(self, inputs):
        g_u, b_c, b_e = inputs['gamma_u'], inputs['theta_i_c'], inputs['theta_i_e']

        for layer in range(len(self.attention_layers)):
            if layer == 0:
                b_c = tf.matmul(g_u, self.attention_network['l_{}_Wu'.format(layer + 1)]) + \
                      tf.squeeze(tf.matmul(b_c, self.attention_network['l_{}_Wf'.format(layer + 1)])) + \
                      self.attention_network['l_{}_b'.format(layer + 1)]
                b_e = tf.matmul(g_u, self.attention_network['l_{}_Wu'.format(layer + 1)]) + \
                      tf.squeeze(tf.matmul(b_e, self.attention_network['l_{}_Wf'.format(layer + 1)])) + \
                      self.attention_network['l_{}_b'.format(layer + 1)]
                b_c = tf.nn.relu(b_c)
                b_e = tf.nn.relu(b_e)
            else:
                b_c = tf.matmul(b_c, self.attention_network['l_{}_W'.format(layer + 1)]) + \
                      self.attention_network['l_{}_b'.format(layer + 1)]
                b_e = tf.matmul(b_e, self.attention_network['l_{}_W'.format(layer + 1)]) + \
                      self.attention_network['l_{}_b'.format(layer + 1)]

        all_b = tf.concat([b_c, b_e], axis=1)
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
        user, item = inputs

        # USER
        # user collaborative profile
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))

        # ITEM
        # item collaborative profile
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        # item color features profile
        color_i = tf.squeeze(tf.nn.embedding_lookup(self.color_weights['Fc'], item))
        theta_i_c = tf.matmul(color_i, self.color_weights['E1c'])
        theta_i_c = tf.matmul(theta_i_c, self.color_weights['E2c'])
        # item edge features profile
        edges_i = tf.squeeze(tf.nn.embedding_lookup(self.edges_weights['Fe'], item))
        theta_i_e = tf.matmul(edges_i, self.edges_weights['E1e'])
        theta_i_e = tf.matmul(theta_i_e, self.edges_weights['E2e'])
        all_theta_i = tf.concat([
            tf.expand_dims(theta_i_c, axis=1),
            tf.expand_dims(theta_i_e, axis=1)
        ], axis=1)

        # attention network
        attention_inputs = {
            'gamma_u': gamma_u,
            'theta_i_c': theta_i_c,
            'theta_i_e': theta_i_e,
        }
        all_attention = self.propagate_attention(attention_inputs)

        # score prediction
        xui = tf.reduce_sum(gamma_u * (gamma_i * tf.reduce_sum(tf.multiply(
            tf.expand_dims(all_attention, axis=2),
            all_theta_i
        ), axis=1)), axis=1)

        return xui, \
               gamma_u, \
               gamma_i, \
               color_i, \
               edges_i, \
               theta_i_c, \
               theta_i_e, \
               all_attention

    def train_step(self, batch):
        """
        Apply a single training step on one batch.

        Args:
            batch: batch used for the current train step

        Returns:
            loss value at the current batch
        """
        user, pos, neg = batch
        with tf.GradientTape() as t:
            # Clean Inference
            xu_pos, \
                gamma_u, \
                gamma_i_pos, \
                color_i_pos, \
                edge_i_pos, \
                theta_i_c_pos, \
                theta_i_e_pos, \
                attention_pos = self(inputs=(user, pos), training=True)

            xu_neg, \
                _, \
                gamma_i_neg, \
                color_i_neg, \
                edge_i_neg, \
                theta_i_c_neg, \
                theta_i_e_neg, \
                attention_neg = self(inputs=(user, neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.reg * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_i_pos),
                                                 tf.nn.l2_loss(gamma_i_neg)]) * 2 + \
                       self.reg * tf.reduce_sum([tf.nn.l2_loss(self.color_weights['E1c']),
                                                 tf.nn.l2_loss(self.edges_weights['E1e']),
                                                 tf.nn.l2_loss(self.color_weights['E2c']),
                                                 tf.nn.l2_loss(self.edges_weights['E2e'])]) * 2

            # Loss to be optimized
            loss += reg_loss

        params = [
            self.Gu, self.Gi,
            self.color_weights['E1c'], self.edges_weights['E1e'],
            self.color_weights['E2c'], self.edges_weights['E2e'],
            *[value for _, value in self.attention_network.items()]
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
        next_batch = self.data.next_triple_batch()
        steps = 0
        loss = 0
        it = 1
        steps_per_epoch = sum([len(pos) for pos in self.data.training_list]) // self.params.batch_size

        start_ep = time()

        directory_parameters = f'batch_{self.params.batch_size}' \
                               f'-D_{self.params.embed_d}' \
                               f'-K_{self.params.embed_k}' \
                               f'-lr_{self.params.lr}' \
                               f'-reg_{self.params.reg}' \
                               f'-attlayers_{list(self.params.attention_layers)}'

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
                epoch_print = self.evaluator.eval(it, results, epoch_text, start_ep, attentive=True)

                for metric in max_metrics.keys():
                    if max_metrics[metric] <= results[it][metric + '_v']:
                        max_metrics[metric] = results[it][metric + '_v']
                        if metric == self.params.best_metric:
                            best_epoch, best_model, best_epoch_print = it, deepcopy(self), epoch_print

                if (it % self.verbose == 0 or it == 1) and self.verbose != -1:
                    self.saver_ckpt.save(f'{weight_dir}/{self.params.dataset}/{self.params.rec}/' + \
                                         f'weights-{it}-{directory_parameters}')
                start_ep = time()
                it += 1
                loss = 0
                steps = 0

        print('***************************')
        print('Training end...')
        print('***************************')
        self.evaluator.store_recommendation_attention(path=f'{results_dir}/{self.params.dataset}/{self.params.rec}/' + \
                                                           f'recs-{it}-{directory_parameters}.tsv')
        save_obj(results,
                 f'{results_dir}/{self.params.dataset}/{self.params.rec}'
                 f'/results-metrics-{directory_parameters}')

        # Store the best model
        print("Store Best Model at Epoch {0}".format(best_epoch))
        print(best_epoch_print)
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=best_model)
        saver_ckpt.save(f'{weight_dir}/{self.params.dataset}/{self.params.rec}/' + \
                        f'best-weights-{best_epoch}-{directory_parameters}')
        best_model.evaluator.store_recommendation_attention(
            path=f'{results_dir}/{self.params.dataset}/{self.params.rec}/' + \
                 f'best-recs-{best_epoch}-{directory_parameters}.tsv')
        print('End Store Best Model!')

        print('Best Values for Each Metric (Validation):\nHR\tPrec\tRec\tAUC\tnDCG\n{}\t{}\t{}\t{}\t{}\n'.format(
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

        theta_i_c = tf.matmul(self.color_weights['Fc'], self.color_weights['E1c'])
        theta_i_c = tf.matmul(theta_i_c, self.color_weights['E2c'])
        theta_i_e = tf.matmul(self.edges_weights['Fe'], self.edges_weights['E1e'])
        theta_i_e = tf.matmul(theta_i_e, self.edges_weights['E2e'])
        all_theta_i = tf.concat([
            tf.expand_dims(theta_i_c, axis=1),
            tf.expand_dims(theta_i_e, axis=1)
        ], axis=1)

        step = 100
        all_predictions = []
        all_attentions = []
        steps = (self.num_items // step) * step
        reminder = self.num_items % step

        for u in range(self.num_users):
            current_predictions = []
            current_attentions = []
            gamma_u = tf.repeat(tf.expand_dims(self.Gu[u], 0), repeats=step, axis=0)

            for sv in range(0, steps, step):
                attention_inputs = {
                    'gamma_u': gamma_u,
                    'theta_i_c': theta_i_c[sv:sv + step],
                    'theta_i_e': theta_i_e[sv:sv + step]
                }
                attention = self.propagate_attention(attention_inputs)
                current_attentions += attention.numpy().tolist()
                gamma_i = self.Gi[sv:sv + step]
                theta_i = all_theta_i[sv:sv + step]
                xui = tf.reduce_sum(gamma_u * (gamma_i * tf.reduce_sum(tf.multiply(
                    tf.expand_dims(attention, axis=2),
                    theta_i
                ), axis=1)), axis=1)
                current_predictions += xui.numpy().tolist()

            # reminder
            attention_inputs = {
                'gamma_u': tf.repeat(tf.expand_dims(self.Gu[u], 0), repeats=reminder, axis=0),
                'theta_i_c': theta_i_c[steps:steps + reminder],
                'theta_i_e': theta_i_e[steps:steps + reminder]
            }
            attention = self.propagate_attention(attention_inputs)
            current_attentions += attention.numpy().tolist()
            gamma_u = tf.repeat(tf.expand_dims(self.Gu[u], 0), repeats=reminder, axis=0)
            gamma_i = self.Gi[steps:steps + reminder]
            theta_i = all_theta_i[steps:steps + reminder]
            xui = tf.reduce_sum(gamma_u * (gamma_i * tf.reduce_sum(tf.multiply(
                tf.expand_dims(attention, axis=2),
                theta_i
            ), axis=1)), axis=1)
            current_predictions += xui.numpy().tolist()
            all_predictions.append(current_predictions)
            all_attentions.append(current_attentions)

        return np.array(all_predictions), np.array(all_attentions)
