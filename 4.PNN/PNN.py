#coding: utf-8
########################

#网络: PNN
#结构: 串行

########################


import sys, os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class PNN(object):
    def __init__(self, feature_size, field_size, embedding_size, deep_layer, deep_init_size, dropout_deep,
                 deep_layer_activation, epoch, batch_size, learning_rate, optimizer_type, batch_norm, batch_norm_decay,
                 l2_reg, random_seed = 2019, loss_type = 'logloss', use_inner = False):

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.deep_layer = deep_layer
        self.deep_init_size = deep_init_size
        self.dropout_deep = dropout_deep
        self.deep_layer_activation = deep_layer_activation
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.l2_reg = l2_reg
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.use_inner = use_inner

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.fea_idx = tf.placeholder(dtype= tf.int32, shape= [None, None], name= 'fea_idx')
            self.fea_val = tf.placeholder(dtype= tf.float32, shape= [None, None], name = 'fea_val')
            self.dropout_deep_keep = tf.placeholder(dtype = tf.float32, shape=[None], name = 'dropout_deep_keep')

            self.label = tf.placeholder(dtype= tf.float32, shape= [None, 1], name = 'label')

            self.weights = self._init_weight()

            #embedding

            self.embedding = tf.nn.embedding_lookup(self.weights['embeddings'], self.fea_idx)
            fea_val = tf.reshape(self.fea_val, shape= [-1, self.field_size, 1])
            self.embedding = tf.multiply(self.embedding, fea_val)

            #linear layer
            linear_output = []
            for i in range(self.deep_init_size):
                linear_output.append(
                    tf.reshape(
                        tf.reduce_sum(
                            tf.multiply(self.embedding, tf.expand_dims(self.weights['product_linear_weight'][i], 0)),
                            axis = [1, 2]
                        ),
                        shape = [-1, 1])

                )

            self.lz = tf.concat(linear_output, axis = 1) #batch * deep_init

            #quardatic layer
            quardatic_output = []
            if self.use_inner:
                for i in range(self.deep_init_size):
                    theta = tf.multiply(
                        self.embedding,
                        tf.reshape(
                            self.weights['product_quadratic_inner'][i],
                            shape = [1, -1, 1]
                        )
                    )
                    quardatic_output.append(
                        tf.reshape(
                            tf.norm(
                                tf.reduce_sum(theta,
                                              axis = 1
                                              ),
                                axis = 1
                            ),
                            shape = [-1, 1]
                        )
                    )
            else:
                embedding_sum = tf.reduce_sum(self.embedding, axis = 1)
                p = tf.matmul(tf.expand_dims(embedding_sum, 2), tf.expand_dims(embedding_sum, 1))
                for i in range(self.deep_init_size):
                    quardatic_output.append(
                        tf.reshape(
                            tf.reduce_sum(
                                tf.multiply(
                                    p,
                                    tf.expand_dims(
                                        self.weights['product_quadratic_outer'][i],
                                        0
                                    )
                                ),
                                axis = [1, 2]
                            ),
                            shape = [-1, 1]
                        )
                    )

            self.lp = tf.concat(quardatic_output, axis = 1)

            self.y_deep = self.deep_layer_activation(tf.add(tf.add(self.lz, self.lp), self.weights['product_bias']))
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_deep_keep[0])

            for i in range(len(self.deep_layer)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights['weight_layer_%d' % i]), self.weights['bias_layer_%d' % i])
                self.y_deep = self.deep_layer_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_deep_keep[i + 1])

            self.out = tf.add(tf.matmul(self.y_deep, self.weights['output_weight']), self.weights['output_bias'])

            self.output = tf.nn.sigmoid(self.out)

            if self.loss_type == 'logloss':
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= self.out, labels= self.label))
            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(self.label, self.out)

            #if self.l2_reg > 0:
            #    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['output_weight'])
            #    for i in range(len(self.deep_layer)):
            #        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['weight_layer_%d' % i])

                #for i in range(self.deep_init_size):
            #    if self.use_inner:
            #        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['product_quadratic_inner'])
            #    else:
            #        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['product_quadratic_outer'])

            #    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['product_linear_weight'])

            #tf.summary.scalar('loss', self.loss)

            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate, beta1 = 0.9, beta2= 0.999,
                                                        epsilon= 1e-8).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate= self.learning_rate,
                                                           initial_accumulator_value= 1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate= self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'momentum':
                self.optimizer_type = tf.train.MomentumOptimizer(learning_rate= self.learning_rate, momentum= 0.95).minimize(self.loss)

            saver = tf.train.Saver()
            #self.merged = tf.summary.merge_all()


            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.writer = tf.summary.FileWriter('./logs/', self.sess.graph)

    def _init_weight(self):
        weights = {}

        #embedding layer
        weights['embeddings'] = tf.get_variable(name = 'embeddings',
                                                shape = [self.feature_size, self.embedding_size],
                                                initializer = tf.random_normal_initializer(0, 0.01))

        #product layer
        if self.use_inner:
            weights['product_quadratic_inner'] = tf.get_variable(name = 'product_quadratic_inner',
                                                                 shape = [self.deep_init_size, self.field_size],
                                                                 initializer = tf.random_normal_initializer(0, 0.01))
        else:
            weights['product_quadratic_outer'] = tf.get_variable(name = 'product_quadratic_outer',
                                                                 shape = [self.deep_init_size, self.embedding_size, self.embedding_size],
                                                                 initializer= tf.random_normal_initializer(0, 0.01))

        weights['product_linear_weight'] = tf.get_variable(name = 'product_linear_weight',
                                                           shape = [self.deep_init_size, self.field_size, self.embedding_size],
                                                           initializer= tf.random_normal_initializer(0, 0.01))
        weights['product_bias'] = tf.get_variable(name = 'product_bias',
                                                         shape = [self.deep_init_size,],
                                                         initializer= tf.random_normal_initializer(0, 1.0))

        #deep layer
        input_size = self.deep_init_size
        deep_init = tf.glorot_normal_initializer()

        weights['weight_layer_0'] = tf.get_variable(name = 'weight_layer_0',
                                                    shape = [input_size, self.deep_layer[0]],
                                                    initializer= deep_init)
        weights['bias_layer_0'] = tf.get_variable(name = 'bias_layer_0',
                                                  shape = [1, self.deep_layer[0]],
                                                  initializer= deep_init)

        for i in range(1, len(self.deep_layer)):
            weights['weight_layer_%d' % i] = tf.get_variable(name = 'weight_layer_%d' % i,
                                                              shape = [self.deep_layer[i -1], self.deep_layer[i]],
                                                              initializer= deep_init)
            weights['bias_layer_%d' % i] = tf.get_variable(name = 'bias_layer_%d' % i,
                                                           shape = [1, self.deep_layer[i]],
                                                           initializer= deep_init)

        #output
        weights['output_weight'] = tf.get_variable(name = 'output_weight',
                                                   shape = [self.deep_layer[-1], 1],
                                                   initializer= deep_init)
        weights['output_bias'] = tf.get_variable(name = 'output_bias',
                                                 shape = [1],
                                                 initializer= tf.constant_initializer(0.01))


        #tf.summary.histogram('out_weight', weights['output_weight'])

        return weights


    def _shuffle_data(self, train_xi, train_xv, train_y):
        r_state = np.random.get_state()
        np.random.shuffle(train_xi)
        np.random.set_state(r_state)
        np.random.shuffle(train_xv)
        np.random.set_state(r_state)
        np.random.shuffle(train_y)

    def _get_batch(self, train_xi, train_xv, train_y, batch_size, idx):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        end = end if end < len(train_y) else len(train_y)
        return train_xi[start: end], train_xv[start: end], [[y] for y in train_y[start: end]]

    def fit(self, train_xi, train_xv, train_y, valid_xi = None, valid_xv = None, valid_y_ = None):
        for epoch in range(self.epoch):
            self._shuffle_data(train_xi, train_xv, train_y)
            total_batch = int(len(train_y) * 1.0 / self.batch_size)
            for i in range(total_batch):
                xi_batch, xv_batch, y_batch = self._get_batch(train_xi, train_xv, train_y, self.batch_size, i)
                #print xi_batch, len(xi_batch)
                #print xv_batch, len(xv_batch)
                loss_, _= self.sess.run([self.loss, self.optimizer],
                              feed_dict = {
                                  self.fea_idx: xi_batch,
                                  self.fea_val: xv_batch,
                                  self.label: y_batch,
                                  self.dropout_deep_keep: self.dropout_deep,
                                }
                             )
                #self.writer.add_summary(mr, i)
                #print loss_
                #print 'epoch: %d, batch: %d, loss: %.4f' % (epoch, i, loss_)

            if valid_xi:
                valid_y_ = np.array(valid_y_).reshape(-1, 1)
                valid_loss = self.sess.run(self.loss,
                                feed_dict = {
                                    self.fea_idx: valid_xi,
                                    self.fea_val: valid_xv,
                                    self.label: valid_y_,
                                    self.dropout_deep_keep: [1.0] * len(self.dropout_deep),
                                }
                            )
                print 'epoch: %d, loss on valid set: %.4f' % (epoch, valid_loss)

    def predict(self, test_xi, test_xv, test_y):
        total_batch = int(len(test_y) * 1.0 / self.batch_size)
        out_list = None
        for i in range(total_batch):
            xi_batch, xv_batch, y_batch = self._get_batch(test_xi, test_xv, test_y, self.batch_size, i)
            y_dummy = np.ones_like(y_batch)
            y_predict = self.sess.run(self.output,
                        feed_dict = {
                            self.fea_idx: xi_batch,
                            self.fea_val: xv_batch,
                            self.label: y_dummy,
                            self.dropout_deep_keep: [1.0] * len(self.dropout_deep),
                        }
                    )

            y_real = np.array(y_batch)

            if i == 0:
                out_list = np.c_[y_predict, y_real]
            else:
                out_list = np.r_[out_list, np.c_[y_predict, y_real]]


        auc = roc_auc_score(out_list[:, 1], out_list[:, 0])
        print 'final auc: %.3f' % auc
