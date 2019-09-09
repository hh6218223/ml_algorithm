#coding: utf-8

import sys, os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class DCN(object):
    def __init__(self, cate_fea_size, field_size, numeric_fea_size, embedding_size, deep_layer, dropout_deep,
                 deep_layer_activation, epoch, batch_size, optimizer_type, batch_norm, batch_norm_decay, cross_layer_num, learning_rate,
                 loss_type, l2_reg = 'logloss', random_seed = 2019):
        self.cate_fea_size = cate_fea_size
        self.field_size = field_size
        self.numeric_fea_size = numeric_fea_size
        self.embedding_size = embedding_size
        self.deep_layer = deep_layer
        self.dropout_deep = dropout_deep
        self.deep_layer_activation = deep_layer_activation
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.cross_layer_num = cross_layer_num
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.l2_reg = l2_reg
        self.random_seed = random_seed
        self.total_size = self.embedding_size * self.field_size + self.numeric_fea_size

        self._init_graph()



    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.fea_idx = tf.placeholder(tf.int32, [None, None], name = 'fea_idx')
            self.fea_val = tf.placeholder(tf.float32, [None, None], name = 'fea_val')
            self.numeic_val = tf.placeholder(tf.float32, [None, None], name = 'numeric_val')
            self.label = tf.placeholder(tf.float32, [None, 1], name = 'label')

            self.dropout_deep_keep = tf.placeholder(tf.float32, [None], name = 'dropout_deep_keep')

            self.weights = self._init_weight()

            #embedding
            self.embeddings = tf.nn.embedding_lookup(self.weights['embeddings'], self.fea_idx)
            fea_val = tf.reshape(self.fea_val, shape = [-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, fea_val)

            self.x0 = tf.concat([self.numeic_val, tf.reshape(self.embeddings, shape = [-1, self.embedding_size * self.field_size])], axis = 1)

            #deep part

            self.y_deep = tf.nn.dropout(self.x0, self.dropout_deep_keep[0])

            for i in range(len(self.deep_layer)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights['deep_weight_%d' % i]), self.weights['deep_bias_%d' % i])
                self.y_deep = self.deep_layer_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_deep_keep[i])

            #cross part
            self.x1 = tf.reshape(self.x0, shape = [-1, self.total_size, 1])
            x_l = self.x1
            for i in range(self.cross_layer_num):
                x_l = tf.tensordot(tf.matmul(self.x1, x_l, transpose_b = True), self.weights['cross_weight_%d' % i], 1) + \
                            self.weights['cross_bias_%d' % i] + x_l

            self.cross_out = tf.reshape(x_l, shape = [-1, self.total_size])

            self.concat_input = tf.concat([self.cross_out, self.y_deep], axis = 1)

            #concat part
            self.out = tf.add(tf.matmul(self.concat_input, self.weights['concat_weight']), self.weights['concat_bias'])

            self.output = tf.nn.sigmoid(self.out)

            if self.loss_type == 'logloss':
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.out, labels = self.label))
            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            if self.l2_reg > 0.0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['concat_weight'])
                for i in range(len(self.deep_layer)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['deep_weight_%d' % i])

                for i in range(self.cross_layer_num):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['cross_weight_%d' % i])

            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate, beta1= 0.9, beta2= 0.999, epsilon = 1e-8).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate= self.learning_rate, initial_accumulator_value= 1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate= self.learning_rate).minimize(self.loss)
            elif self.optimizer == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate= self.learning_rate, momentum= 0.95).minimize(self.loss)

            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _init_weight(self):
        weights = {}

        #embedding
        weights['embeddings'] = tf.get_variable(name = 'embeddings',
                                                shape = [self.cate_fea_size, self.embedding_size],
                                                dtype = tf.float32,
                                                initializer = tf.random_normal_initializer(0.0, 0.01))
        #deep layer
        deep_init = tf.glorot_normal_initializer()
        weights['deep_weight_0'] = tf.get_variable(name = 'deep_weight_0',
                                                   shape = [self.total_size, self.deep_layer[0]],
                                                   dtype = tf.float32,
                                                   initializer = deep_init)
        weights['deep_bias_0'] = tf.get_variable(name = 'deep_bias_0',
                                                shape = [1, self.deep_layer[0]],
                                                dtype = tf.float32,
                                                initializer = deep_init)

        for i in range(1, len(self.deep_layer)):
            weights['deep_weight_%d' % i] = tf.get_variable(name = 'deep_weight_%d' % i,
                                                            shape = [self.deep_layer[i - 1], self.deep_layer[i]],
                                                            dtype = tf.float32,
                                                            initializer = deep_init)
            weights['deep_bias_%d' % i] = tf.get_variable(name = 'deep_bias_%d' % i,
                                                          shape = [1, self.deep_layer[i]],
                                                          dtype = tf.float32,
                                                          initializer = deep_init)
        #cross layer
        for i in range(self.cross_layer_num):
            weights['cross_weight_%d' % i] = tf.get_variable(name = 'cross_weight_%d' % i,
                                                             shape = [self.total_size, 1],
                                                             dtype = tf.float32,
                                                             initializer = deep_init)
            weights['cross_bias_%d' % i] = tf.get_variable(name = 'cross_bias_%d' % i,
                                                           shape = [self.total_size, 1],
                                                           dtype = tf.float32,
                                                           initializer = deep_init)

        input_size = self.total_size + self.deep_layer[-1]
        weights['concat_weight'] = tf.get_variable(name = 'concat_weight',
                                                   shape = [input_size, 1],
                                                   dtype = tf.float32,
                                                   initializer = deep_init)
        weights['concat_bias'] = tf.get_variable(name = 'concat_bias',
                                                shape = [1],
                                                dtype = tf.float32,
                                                initializer = tf.constant_initializer(0.01))

        return weights


    def _get_shuffle(self, data_xi, data_xv, data_numeric, data_y):
        r_state = np.random.get_state()
        np.random.shuffle(data_xi)
        np.random.set_state(r_state)
        np.random.shuffle(data_xv)
        np.random.set_state(r_state)
        np.random.shuffle(data_numeric)
        np.random.set_state(r_state)
        np.random.shuffle(data_y)

    def _get_batch(self, data_xi, data_xv, data_numeric, data_y, batch_size, idx):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        end = end if end < len(data_y) else len(data_y)
        return data_xi[start: end], data_xv[start: end], data_numeric[start: end], [[y] for y in data_y[start: end]]


    def fit(self, train_xi, train_xv, train_nx, train_y, valid_xi = None, valid_xv = None, valid_nv = None, valid_y = None):
        batch_num = len(train_xi) / self.batch_size
        for epoch in range(self.epoch):
            self._get_shuffle(train_xi, train_xv, train_nx, train_y)
            for it in range(batch_num):
                batch_xi, batch_xv, batch_xn, batch_y = self._get_batch(train_xi, train_xv, train_nx, train_y, self.batch_size, it)

                #print batch_xi
                #print batch_xv
                #print batch_xn
                #print batch_y
                #print self.dropout_deep
                self.sess.run([self.optimizer, self.loss],
                              feed_dict= {
                                  self.fea_idx: batch_xi,
                                  self.fea_val: batch_xv,
                                  self.numeic_val: batch_xn,
                                  self.label: batch_y,
                                  self.dropout_deep_keep: self.dropout_deep,
                              })

            if valid_xi:
                valid_y_ = np.array(valid_y).reshape(-1, 1)
                valid_loss = self.sess.run(self.loss,
                                           feed_dict= {
                                               self.fea_idx: valid_xi,
                                               self.fea_val: valid_xv,
                                               self.numeic_val: valid_nv,
                                               self.label: valid_y_,
                                               self.dropout_deep_keep: [1.0] * len(self.dropout_deep)
                                          })
                print "Epoch: %dth, loss: %.3f" % (epoch, valid_loss)



    def predict(self, test_xi, test_xv, test_xn, test_y):
        total_batch = int(len(test_y) * 1.0 / self.batch_size)
        out_list = None
        for i in range(total_batch):
            batch_xi, batch_xv, batch_xn, batch_y = self._get_batch(test_xi, test_xv, test_xn, test_y, self.batch_size, i)
            y_dummy = np.ones_like(batch_y)
            y_predict = self.sess.run(self.output,
                                feed_dict= {
                                  self.fea_idx: batch_xi,
                                  self.fea_val: batch_xv,
                                  self.numeic_val: batch_xn,
                                  self.label: batch_y,
                                  self.dropout_deep_keep: [1.0] * len(self.dropout_deep),
                              })

            y_real = np.array(batch_y)

            if i == 0:
                out_list = np.c_[y_predict, y_real]
            else:
                out_list = np.r_[out_list, np.c_[y_predict, y_real]]

        auc = roc_auc_score(out_list[:, 1], out_list[:, 0])
        print 'final auc: %.3f' % auc


