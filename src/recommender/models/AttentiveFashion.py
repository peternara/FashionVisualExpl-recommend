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
        super(AttentiveFashion, self).__init__(data, params)
        self.initializer = tf.initializers.RandomNormal(mean=0, stddev=0.01)
        self.initializer_attentive = tf.initializers.GlorotUniform()
        self.attention_layers = self.params.attention_layers

        del self.Gi

        self.process_color_visual_features()
        self.process_class_visual_features()

        # Initialize input features encoders
        self.color_encoder = None
        self.edges_encoder = None
        self.class_encoder = None

        # Create model parameters
        self.color_embedding = None
        self.class_embedding = None
        self.create_color_weights()
        self.create_edges_weights()
        self.create_class_weights()

        # Initialize attention parameters
        self.attention_network = dict()

        # Create attention parameters
        self.create_attention_weights()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def create_color_weights(self):
        self.color_embedding = tf.Variable(
            self.color_features,
            name='FeatColor', dtype=tf.float32, trainable=False
        )
        self.color_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1024, activation='relu', input_dim=self.color_embedding.shape[1]),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=self.embed_k, use_bias=False)
        ])

    def create_edges_weights(self):
        self.edges_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(padding='same'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=256, activation='relu', input_dim=self.color_embedding.shape[1]),
            tf.keras.layers.Dense(units=self.embed_k, use_bias=False)
        ])

    def create_class_weights(self):
        self.class_embedding = tf.Variable(
            self.class_features,
            name='FeatClass', dtype=tf.float32, trainable=False
        )
        self.class_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1024, activation='relu', input_dim=self.class_embedding.shape[1]),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=self.embed_k, use_bias=False)
        ])

    def create_attention_weights(self):
        for layer in range(len(self.attention_layers)):
            if layer == 0:
                self.attention_network['W_{}_u'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self.embed_k, self.attention_layers[layer]]),
                    name='W_{}_u'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['W_{}_f'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self.embed_k, self.attention_layers[layer]]),
                    name='W_{}_f'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['b_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self.attention_layers[layer]]),
                    name='b_{}'.format(layer + 1),
                    dtype=tf.float32
                )
            else:
                self.attention_network['W_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self.attention_layers[layer - 1], self.attention_layers[layer]]),
                    name='W_{}'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['b_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self.attention_layers[layer]]),
                    name='b_{}'.format(layer + 1),
                    dtype=tf.float32
                )

    def propagate_attention(self, inputs):
        g_u, colors, edges, classes = inputs['gamma_u'], inputs['colors'], inputs['edges'], inputs['classes']
        all_a_i_l = None

        for layer in range(len(self.attention_layers)):
            if layer == 0:
                all_a_i_l = tf.expand_dims(tf.matmul(g_u, self.attention_network['W_{}_u'.format(layer + 1)]), 1) + \
                            tf.tensordot(tf.concat([colors, edges, classes], axis=1),
                                         self.attention_network['W_{}_f'.format(layer + 1)], axes=[[2], [0]]) + \
                            self.attention_network['b_{}'.format(layer + 1)]
                all_a_i_l = tf.nn.relu(all_a_i_l)
            else:
                all_a_i_l = tf.tensordot(all_a_i_l, self.attention_network['W_{}'.format(layer + 1)], axes=[[2], [0]]) + \
                            self.attention_network['b_{}'.format(layer + 1)]

        all_alpha = tf.nn.softmax(all_a_i_l, axis=1)
        return all_alpha

    def call(self, inputs, training=None, mask=None):
        user, item, edges = inputs

        # USER
        # user collaborative profile
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))

        # ITEM
        # item color features
        color_i = tf.expand_dims(self.color_encoder(tf.squeeze(tf.nn.embedding_lookup(self.color_embedding, item))), 1)
        # item edge features
        edges_i = tf.expand_dims(self.edges_encoder(edges), 1)
        # item class features
        class_i = tf.expand_dims(self.class_encoder(tf.squeeze(tf.nn.embedding_lookup(self.class_embedding, item))), 1)

        # attention network
        attention_inputs = {
            'gamma_u': gamma_u,
            'colors': color_i,
            'edges': edges_i,
            'classes': class_i
        }
        all_attention = self.propagate_attention(attention_inputs)

        # score prediction
        xui = tf.reduce_sum(gamma_u * (tf.reduce_sum(tf.multiply(
            all_attention,
            tf.concat([color_i, edges_i, class_i], axis=1)
        ), axis=1)), axis=1)

        return xui, \
               gamma_u, \
               color_i, \
               edges_i, \
               class_i, \
               all_attention

    def train_step(self, batch):
        user, pos, edges_pos, neg, edges_neg = batch
        with tf.GradientTape() as t:
            # Clean Inference
            xu_pos, \
            gamma_u, \
            color_i_pos, \
            edge_i_pos, \
            class_i_pos, \
            attention_pos = self(inputs=(user, pos, edges_pos), training=True)

            xu_neg, \
            _, \
            color_i_neg, \
            edge_i_neg, \
            class_i_neg, \
            attention_neg = self(inputs=(user, neg, edges_neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.reg * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(color_i_pos), tf.nn.l2_loss(color_i_neg),
                                                 tf.nn.l2_loss(edge_i_pos), tf.nn.l2_loss(edge_i_neg),
                                                 tf.nn.l2_loss(class_i_pos), tf.nn.l2_loss(class_i_neg)]) * 2

            # Loss to be optimized
            loss += reg_loss

        params = [
            self.Gu,
            *self.color_encoder.trainable_weights,
            *self.edges_encoder.trainable_weights,
            *self.class_encoder.trainable_weights,
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
        next_batch = self.data.next_triple_batch_pipeline()
        steps = 0
        loss = 0
        it = 1
        steps_per_epoch = sum([len(pos) for pos in self.data.training_list]) // self.params.batch_size

        start_ep = time()

        directory_parameters = f'batch_{self.params.batch_size}' \
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

    def predict_all_batch(self, step, next_image):
        all_predictions = []
        all_attentions = []
        reminder = self.num_items % step

        all_colors = self.color_encoder(self.color_embedding)
        all_class = self.class_encoder(self.class_embedding)

        for u in range(self.num_users):
            current_predictions = []
            current_attentions = []
            gamma_u = tf.repeat(tf.expand_dims(self.Gu[u], 0), repeats=step, axis=0)
            sv = 0
            for id_im, im in next_image:
                if self.data.num_items == id_im.numpy()[-1] + 1:
                    gamma_u = gamma_u[:reminder]
                edges = tf.expand_dims(self.edges_encoder(im), 1)
                colors = tf.expand_dims(
                    all_colors[sv:(sv + step if self.data.num_items != id_im.numpy()[-1] + 1 else sv + reminder)], 1)
                classes = tf.expand_dims(
                    all_class[sv:(sv + step if self.data.num_items != id_im.numpy()[-1] + 1 else sv + reminder)], 1)
                # attention network
                attention_inputs = {
                    'gamma_u': gamma_u,
                    'colors': colors,
                    'edges': edges,
                    'classes': classes
                }

                all_attention = self.propagate_attention(attention_inputs)

                # score prediction
                xui = tf.reduce_sum(gamma_u * (tf.reduce_sum(tf.multiply(
                    all_attention,
                    tf.concat([colors, edges, classes], axis=1)
                ), axis=1)), axis=1)
                current_predictions += xui.numpy().tolist()
                current_attentions += all_attention.numpy()[:, :, 0].tolist()
                sv = sv + step
            all_predictions.append(current_predictions)
            all_attentions.append(current_attentions)

        return np.array(all_predictions), np.array(all_attentions)
