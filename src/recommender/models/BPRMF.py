from abc import ABC
from copy import deepcopy
from time import time

import tensorflow as tf
import numpy as np
import logging
import random

from config.configs import *
from recommender.Evaluator import Evaluator
from recommender.RecommenderModel import RecommenderModel
from utils.write import save_obj

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)


class BPRMF(RecommenderModel, ABC):

    def __init__(self, data, params):
        """
        Create a BPR-MF instance.
        (see https://arxiv.org/pdf/1205.2618 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        super(BPRMF, self).__init__(data, params)
        self.initializer = tf.initializers.GlorotUniform()
        self.embed_k = self.params.embed_k
        self.learning_rate = self.params.lr
        self.reg = self.params.reg

        self.evaluator = Evaluator(self, data, params.top_k)

        # Initialize Model Parameters
        self.Bi = tf.Variable(tf.zeros(self.num_items), name='Bi', dtype=tf.float32)
        self.Gu = tf.Variable(self.initializer(shape=[self.num_users, self.embed_k]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self.num_items, self.embed_k]), name='Gi', dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
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
        user, item = inputs
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))

        xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, 1)

        return xui, beta_i, gamma_u, gamma_i

    def predict_all(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        return self.Bi + tf.matmul(self.Gu, self.Gi, transpose_b=True)

    def train_step(self, batch):
        """
        Apply a single training step on one batch.

        Args:
            batch: batch used for the current train step

        Returns:
            loss value at the current batch
        """
        user, pos, neg = batch
        with tf.GradientTape() as tape:

            # Clean Inference
            xu_pos, beta_pos, gamma_u, gamma_pos = self(inputs=(user, pos), training=True)
            xu_neg, beta_neg, _, gamma_neg = self(inputs=(user, neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))

            # Regularization Component
            reg_loss = self.reg * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg)]) * 2 \
                       + self.reg * tf.nn.l2_loss(beta_pos) * 2 \
                       + self.reg * tf.nn.l2_loss(beta_neg) * 2 / 10

            # Loss to be optimized
            loss += reg_loss

        params = [
            self.Bi,
            self.Gu,
            self.Gi
        ]
        grads = tape.gradient(loss, params)
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

        print('Start training...')
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

        print('Training end...')
        # STANDARD STORE RECOMMENDATION ON LAST EPOCH
        self.evaluator.store_recommendation(path=f'{results_dir}/{self.params.dataset}/{self.params.rec}/' + \
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
        print('End Store Best Model!')

        print('Best Values for Each Metric:\nHR\tPrec\tRec\tAUC\tnDCG\n{}\t{}\t{}\t{}\t{}\n'.format(
            max_metrics['hr'],
            max_metrics['p'],
            max_metrics['r'],
            max_metrics['auc'],
            max_metrics['ndcg']
        ))
