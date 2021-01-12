import logging
from time import time
from abc import ABC
from copy import deepcopy

import numpy as np
import tensorflow as tf
import concurrent.futures
import random

from dataset.visual_loader_mixin import VisualLoader
from recommender.models.BPRMF import BPRMF
from utils.write import save_obj
from config.configs import *

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)


class GradFashion(BPRMF, VisualLoader, ABC):

    def __init__(self, data, params):
        self.initializer = tf.initializers.GlorotUniform()
        super(GradFashion, self).__init__(data, params)

        self.embed_k = self.params.embed_k
        self.embed_d = self.params.embed_d
        self.attention_layers = self.params.attention_layers
        self.learning_rate = self.params.lr

        self.process_edge_visual_features()
        self.process_color_visual_features()

        # Initialize model parameters
        self.color_weights = dict()
        self.edges_weights = dict()

        # Initialize embedding projection
        self.visual_profile = dict()
        self.create_visual_profile()

        # Create model parameters
        self.create_color_features()
        self.create_edges_features()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def create_color_features(self):
        self.color_weights['Fc'] = tf.Variable(
            self.color_features,
            name='Fc', dtype=tf.float32, trainable=False)

    def create_edges_features(self):
        self.edges_weights['Fe'] = tf.Variable(
            self.edge_features,
            name='Fe', dtype=tf.float32, trainable=False)

    def create_visual_profile(self):
        self.visual_profile['Bp'] = tf.Variable(
            self.initializer(shape=[self.dim_color_features + self.dim_edge_features, 1]), name='Bps', dtype=tf.float32)
        self.visual_profile['E'] = tf.Variable(
            self.initializer(shape=[self.dim_color_features + self.dim_edge_features, self.embed_d]),
            name='E', dtype=tf.float32)
        self.visual_profile['Tu'] = tf.Variable(
            self.initializer(shape=[self.num_users, self.embed_d]),
            name='Tu', dtype=tf.float32)

    def call(self, inputs, training=True, mask=None):
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

        if not training:
            raise NotImplemented('Call in inference mode has not been implemented yet!')
        else:
            # USER
            # user collaborative profile
            gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
            theta_u = tf.squeeze(tf.nn.embedding_lookup(self.visual_profile['Tu'], user))

            # ITEM
            # item collaborative profile
            gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
            color_i = tf.squeeze(tf.nn.embedding_lookup(self.color_weights['Fc'], item))
            edges_i = tf.squeeze(tf.nn.embedding_lookup(self.edges_weights['Fe'], item))
            visual_features_i = tf.concat([color_i, edges_i], axis=1)
            theta_i = tf.matmul(visual_features_i, self.visual_profile['E'])

            # BIASES
            # item collaborative bias
            beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))

            # score prediction
            xui = beta_i + \
                  tf.reduce_sum(gamma_u * gamma_i, 1) + \
                  tf.reduce_sum(theta_u * theta_i, 1) + \
                  tf.squeeze(tf.matmul(visual_features_i, self.visual_profile['Bp']))
            return xui, \
                   gamma_u, \
                   gamma_i, \
                   color_i, \
                   edges_i, \
                   theta_u, \
                   theta_i, \
                   beta_i

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
            theta_u, \
            theta_i_pos, \
            beta_pos = self(inputs=(user, pos), training=True)

            xu_neg, \
            _, \
            gamma_i_neg, \
            color_i_neg, \
            edge_i_neg, \
            _, \
            theta_i_neg, \
            beta_neg = self(inputs=(user, neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.reg * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_i_pos),
                                                 tf.nn.l2_loss(gamma_i_neg),
                                                 tf.nn.l2_loss(theta_u)]) * 2 + \
                       self.reg * tf.reduce_sum([tf.nn.l2_loss(beta_pos),
                                                 tf.nn.l2_loss(beta_neg)]) * 2 + \
                       self.reg * tf.reduce_sum([tf.nn.l2_loss(self.visual_profile['E']),
                                                 tf.nn.l2_loss(self.visual_profile['Bp'])]) * 2

            # Loss to be optimized
            loss += reg_loss

        params = [
            self.Gu, self.Gi, self.Bi,
            self.visual_profile['Tu'], self.visual_profile['E'], self.visual_profile['Bp']
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
                epoch_print = self.evaluator.eval(it, results, epoch_text, start_ep)

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
        # STANDARD STORE RECOMMENDATION ON LAST EPOCH
        self.evaluator.store_recommendation(path=f'{results_dir}/{self.params.dataset}/{self.params.rec}/' + \
                                                 f'recs-{it - 1}-{directory_parameters}.tsv')
        # GRADS STORE RECOMMENDATION ON LAST EPOCH
        self.evaluator.store_recommendation_grads(path=f'{results_dir}/{self.params.dataset}/{self.params.rec}/' + \
                                                       f'recs-{it - 1}-{directory_parameters}.tsv')
        save_obj(results,
                 f'{results_dir}/{self.params.dataset}/{self.params.rec}'
                 f'/results-metrics-{directory_parameters}')

        # Store the best model
        print("Store Best Model at Epoch {0}".format(best_epoch))
        print(best_epoch_print)
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=best_model)
        saver_ckpt.save(f'{weight_dir}/{self.params.dataset}/{self.params.rec}/' + \
                        f'best-weights-{best_epoch}-{directory_parameters}')
        # STANDARD STORE RECOMMENDATION ON BEST EPOCH
        best_model.evaluator.store_recommendation(
            path=f'{results_dir}/{self.params.dataset}/{self.params.rec}/' + \
                 f'best-recs-{best_epoch}-{directory_parameters}.tsv')
        # GRADS STORE RECOMMENDATION ON BEST EPOCH
        best_model.evaluator.store_recommendation_grads(
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

    def predict_ui_grads(self, inputs):
        u, i = inputs
        with tf.GradientTape() as t:
            color_i = tf.Variable(tf.expand_dims(self.color_weights['Fc'][i], 0))
            edges_i = tf.Variable(tf.expand_dims(self.edges_weights['Fe'][i], 0))
            visual_features_i = tf.concat([color_i, edges_i], axis=1)
            theta_i = tf.matmul(visual_features_i, self.visual_profile['E'])

            # score prediction
            preds = tf.expand_dims(self.Bi[i], 0) + \
                    tf.matmul(tf.expand_dims(self.Gu[u], 0),
                              tf.expand_dims(self.Gi[i], 0), transpose_b=True) + \
                    tf.matmul(tf.expand_dims(self.visual_profile['Tu'][u], 0), theta_i, transpose_b=True) + \
                    tf.squeeze(tf.matmul(visual_features_i, self.visual_profile['Bp']))

        grads = t.gradient(preds, [color_i, edges_i])
        final_grads = [grads[0] * color_i, grads[1] * edges_i]
        final_grads = [tf.reduce_sum(g, axis=1) for g in final_grads]
        current_grads = tf.concat([tf.expand_dims(final_grads[0], axis=1),
                                   tf.expand_dims(final_grads[1], axis=1)], axis=1)
        return current_grads

    def get_grads_top_k_user(self, u, top_k_items):
        inputs = [(u, i) for i in top_k_items]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            outputs = executor.map(self.predict_ui_grads, inputs)

        current_grads = zip(*outputs)
        current_grads = [cg for cg in current_grads]

        return np.array(current_grads)[0]

    def predict_all(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        visual_features_i = tf.concat([self.color_weights['Fc'], self.edges_weights['Fe']], axis=1)
        theta_i = tf.matmul(visual_features_i, self.visual_profile['E'])
        preds = self.Bi + \
                tf.matmul(self.Gu, self.Gi, transpose_b=True) + \
                tf.matmul(self.visual_profile['Tu'], theta_i, transpose_b=True) + \
                tf.squeeze(tf.matmul(visual_features_i, self.visual_profile['Bp']))
        return preds
