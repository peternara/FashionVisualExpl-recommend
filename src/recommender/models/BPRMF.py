from abc import ABC
from copy import deepcopy
from time import time

import tensorflow as tf
import numpy as np
import os
import logging
import random

from config.configs import *
from recommender.Evaluator import Evaluator
from recommender.RecommenderModel import RecommenderModel
from utils.read import find_checkpoint
from utils.write import save_obj

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        self.embed_k = self.params.embed_k
        self.learning_rate = self.params.lr
        self.l_w = self.params.l_w
        self.l_b = self.params.l_b

        self.evaluator = Evaluator(self, data, params.k)

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

    def one_epoch(self, batches):
        """
        Train recommender model for one epoch.
        Args:
            batches: list of batches to train on
        Returns:
            average loss over epoch
        """
        loss = 0
        steps = 0
        for batch in zip(*batches):
            steps += 1
            loss += self.train_step(batch)
        return loss/steps

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
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg)]) \
                       + self.l_b * tf.nn.l2_loss(beta_pos) \
                       + self.l_b * tf.nn.l2_loss(beta_neg)/10

            # Loss to be optimized
            loss += reg_loss

        grads = tape.gradient(loss, [self.Bi, self.Gu, self.Gi])
        self.optimizer.apply_gradients(zip(grads, [self.Bi, self.Gu, self.Gi]))

        return loss.numpy()

    def train(self):
        if self.restore():
            self.restore_epochs += 1
        else:
            print("Training from scratch...")

        # initialize the max_ndcg to memorize the best result
        max_hr = 0
        best_model = self
        best_epoch = self.restore_epochs
        results = {}

        for self.epoch in range(self.restore_epochs, self.epochs + 1):
            start_ep = time()
            batches = self.data.shuffle(self.batch_size)
            loss = self.one_epoch(batches)
            epoch_text = 'Epoch {0}/{1} \tLoss: {2:.3f}'.format(self.epoch, self.epochs, loss)
            self.evaluator.eval(self.epoch, results, epoch_text, start_ep)

            # print and log the best result (HR@10)
            if max_hr < results[self.epoch]['hr']:
                max_hr = results[self.epoch]['hr']
                best_epoch = self.epoch
                best_model = deepcopy(self)

            if self.epoch % self.verbose == 0 or self.epoch == 1:
                self.saver_ckpt.save(f'{weight_dir}/{self.params.dataset}/' + \
                                     f'weights-{self.epoch}-{self.learning_rate}-{self.__class__.__name__}')

        self.evaluator.store_recommendation(path=f'{results_dir}/{self.params.dataset}/' + \
                                            f'recs-{self.epoch}-{self.learning_rate}-{self.__class__.__name__}.tsv')
        save_obj(results, f'{results_dir}/{self.params.dataset}/results-metrics-{self.learning_rate}')

        # Store the best model
        print("Store Best Model at Epoch {0}".format(best_epoch))
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=best_model)
        saver_ckpt.save(f'{weight_dir}/{self.params.dataset}/' + \
                        f'best-weights-{best_epoch}-{self.learning_rate}-{self.__class__.__name__}')
        best_model.evaluator.store_recommendation(path=f'{results_dir}/{self.params.dataset}/' + \
                                                       f'best-recs-{best_epoch}-{self.learning_rate}-' + \
                                                       f'{self.__class__.__name__}.tsv')

    def restore(self):
        if self.restore_epochs > 1:
            try:
                checkpoint_file = find_checkpoint(weight_dir, self.restore_epochs, self.epochs,
                                                  self.rec)
                self.saver_ckpt.restore(checkpoint_file)
                print("Model correctly Restored at Epoch: {0}".format(self.restore_epochs))
                return True
            except Exception as ex:
                print("Error in model restoring operation! {0}".format(ex))
        else:
            print("Restore Epochs Not Specified")
        return False
