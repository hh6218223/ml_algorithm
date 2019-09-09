#coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

class DeepFM(object):
    def __init__(self, feature_size, field_size, embedding_size, dropout_fm, deep_layer, dropout_deep, deep_layer_activation, epoch, 
                batch_size, learning_rate, optimizer, batch_norm, batch_norm_decay, l2_reg, loss_type = 'logloss', random_seed = 2019):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.dropout_fm_keep = dropout_fm
        self.deep_layer = deep_layer
        self.dropout_deep_keep = dropout_deep
        self.deep_layer_activation = deep_layer_activation
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.l2_reg = l2_reg
        self.loss_type = loss_type
        self.random_seed = random_seed
        
        self._init_graph()
        
        
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            
            self.fea_idx = tf.placeholder(tf.int32, [None, None], name = 'fea_idx')
            self.fea_val = tf.placeholder(tf.float32, [None, None], name = 'fea_val')
            self.label = tf.placeholder(tf.float32, [None, 1], name = 'label')
            
            self.dropout_fm = tf.placeholder(tf.float32, [None], name = 'dropout_fm')
            self.dropout_deep = tf.placeholder(tf.float32, [None], name = 'dropout_deep')
            
            self.weights = self._init_weight()
            
            self.embedding = tf.nn.embedding_lookup(self.weights['feature_embedding'], self.fea_idx)
            fea_val = tf.reshape(self.fea_val, shape = [-1, self.field_size, 1])
            self.embedding = tf.multiply(self.embedding, fea_val)
            
            #FM part
            #first order
            self.first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.fea_idx)
            self.first_order = tf.reshape(self.first_order, shape = [-1, self.field_size])
            self.first_order = tf.nn.dropout(self.first_order, self.dropout_fm[0])
            
            #second order
            self.sum_feature = tf.reduce_sum(self.embedding, axis = 1)
            self.sum_feature_square = tf.square(self.sum_feature)
            
            self.square_feature = tf.square(self.embedding)
            self.square_feature_sum = tf.reduce_sum(self.square_feature, axis = 1)
            
            self.second_order = 0.5 * tf.subtract(self.sum_feature_square, self.square_feature_sum)
            
            self.second_order = tf.nn.dropout(self.second_order, self.dropout_fm[1])
            
            #deep part
            self.deep = tf.reshape(self.embedding, shape = [-1, self.field_size * self.embedding_size])
            self.deep = tf.nn.dropout(self.deep, self.dropout_deep[0])
            
            for i in range(len(self.deep_layer)):
                self.deep = tf.add(tf.matmul(self.deep, self.weights['layer_weight_%d' % i]), self.weights['layer_bias_%d' % i])
                self.deep = self.deep_layer_activation(self.deep)
                self.deep = tf.nn.dropout(self.deep, self.dropout_deep[i + 1])
            
            #concat_size = self.field_size + self.feature_size + self.deep_layer[-1]
            self.concat_input = tf.concat([self.first_order, self.second_order, self.deep], axis = 1)
            self.output = tf.add(tf.matmul(self.concat_input, self.weights['concat_weight']), self.weights['concat_bias'])
            self.out = tf.sigmoid(self.output)
            
            if self.loss_type == 'logloss':
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.output, labels = self.label))
            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.output))
                
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['concat_weight'])
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['concat_bias'])
                for i in range(len(self.deep_layer)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['layer_weight_%d' % i])
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['layer_bias_%d' % i])
                    
            
                
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = 0.9, beta2 = 0.999,
                                                      epsilon = 1e-8).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.learning_rate,
                                                           initial_accumulator_value = 1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, 
                                                            momentum = 0.95).minimizer(self.loss)
                
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
    
    def _init_weight(self):
        weights = {}
        
        #FM layer
        weights['feature_embedding'] = tf.get_variable(name = 'feature_embedding', 
                                                    shape = [self.feature_size, self.embedding_size],
                                                    dtype = tf.float32,
                                                    initializer = tf.random_normal_initializer(0.0, 0.01))
        weights['feature_bias'] = tf.get_variable(name = 'feature_bias', 
                                                  shape = [self.feature_size, 1], 
                                                  dtype = tf.float32,
                                                  initializer = tf.random_normal_initializer(0.0, 1.0))
        
        #deep_layer
        input_size = self.embedding_size * self.field_size
        num_layer = len(self.deep_layer)
        deep_init = tf.glorot_normal_initializer()
        
        weights['layer_weight_0'] = tf.get_variable(name = 'layer_weight_0',
                                                 shape = [input_size, self.deep_layer[0]],
                                                 dtype = tf.float32,
                                                 initializer = deep_init)
        weights['layer_bias_0'] = tf.get_variable(name = 'layer_bias_0',
                                                 shape = [1, self.deep_layer[0]],
                                                 dtype = tf.float32,
                                                 initializer = deep_init)
        for i in range(1, len(self.deep_layer)):
            weights['layer_weight_%d' % i] = tf.get_variable(name = 'layer_weight_%d' % i,
                                                 shape = [self.deep_layer[i - 1], self.deep_layer[i]],
                                                 dtype = tf.float32,
                                                 initializer = deep_init)
            weights['layer_bias_%d' % i] = tf.get_variable(name = 'layer_bias_%d' % i,
                                                 shape = [1, self.deep_layer[i]],
                                                 dtype = tf.float32,
                                                 initializer = deep_init)
            
        #concat layer
        input_size = self.field_size + self.embedding_size + self.deep_layer[-1]
        weights['concat_weight'] = tf.get_variable(name = 'concat_weight',
                                                  shape = [input_size, 1],
                                                  dtype = tf.float32,
                                                  initializer = deep_init)
        weights['concat_bias'] = tf.get_variable(name = 'concat_bias',
                                                  shape = [1],
                                                  dtype = tf.float32,
                                                  initializer = tf.constant_initializer(0.01))
        
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
    
    def fit(self, train_xi, train_xv, train_y, valid_xi = None, valid_xv = None , valid_y_ = None):

        for epoch in range(self.epoch):
            self._shuffle_data(train_xi, train_xv, train_y)
            total_batch = int(len(train_y) * 1.0 / self.batch_size)
            for i in range(total_batch):
                xi_batch, xv_batch, y_batch = self._get_batch(train_xi, train_xv, train_y, self.batch_size, i)
                #print xi_batch, len(xi_batch)
                #print xv_batch, len(xv_batch)
                loss_, _ = self.sess.run([self.loss, self.optimizer], 
                              feed_dict = {
                                  self.fea_idx: xi_batch,
                                  self.fea_val: xv_batch,
                                  self.label: y_batch,
                                  self.dropout_fm: self.dropout_fm_keep,
                                  self.dropout_deep: self.dropout_deep_keep,
                                }
                             )
                #print loss_
                #print 'epoch: %d, batch: %d, loss: %.4f' % (epoch, i, loss_)

            if valid_xi:
                valid_y_ = np.array(valid_y_).reshape(-1, 1)
                valid_loss = self.sess.run(self.loss,
                                feed_dict = {
                                    self.fea_idx: valid_xi,
                                    self.fea_val: valid_xv,
                                    self.label: valid_y_,
                                    self.dropout_fm: [1.0] * len(self.dropout_fm_keep),
                                    self.dropout_deep: [1.0] * len(self.dropout_deep_keep),
                                }
                            )
                print 'epoch: %d, loss on valid set: %.4f' % (epoch, valid_loss)

    def predict(self, test_xi, test_xv, test_y):
        total_batch = int(len(test_y) * 1.0 / self.batch_size)
        out_list = None
        for i in range(total_batch):
            xi_batch, xv_batch, y_batch = self._get_batch(test_xi, test_xv, test_y, self.batch_size, i)
            y_dummy = np.ones_like(y_batch)
            y_predict = self.sess.run(self.out,
                        feed_dict = {
                            self.fea_idx: xi_batch,
                            self.fea_val: xv_batch,
                            self.label: y_dummy,
                            self.dropout_fm: [1.0] * len(self.dropout_fm_keep),
                            self.dropout_deep: [1.0] * len(self.dropout_deep_keep),
                        }
                    )

            y_real = np.array(y_batch)

            if i == 0:
                out_list = np.c_[y_predict, y_real]
            else:
                out_list = np.r_[out_list, np.c_[y_predict, y_real]]


        auc = roc_auc_score(out_list[:, 1], out_list[:, 0])
        print 'final auc: %.3f' % auc

            #out_list = np.r_[out_list, np.c_[y_predict, y_real]] if out_list else np.c_[y_predict, y_real]



