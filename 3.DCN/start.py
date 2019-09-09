#coding: utf-8


import tensorflow as tf
import numpy as np
import pandas as pd
import os, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from DCN import *
from sklearn.preprocessing import StandardScaler

continus_columns = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
bin_columns = ['workclass','education','marital-status','occupation',
                'relationship','race','sex','native-country']

param = {
    'embedding_size': 8,
    'deep_layer': [40, 20],
    'dropout_deep': [0.5, 0.5, 0.5],
    'deep_layer_activation': tf.nn.relu,
    'epoch': 30,
    'batch_size': 64,
    'learning_rate': 0.01,
    'optimizer_type': 'adagrad',
    'batch_norm': 1,
    'batch_norm_decay': 0.995,
    'l2_reg': 0.1,
    'loss_type': 'logloss',
    'cross_layer_num': 3,
}

def load_data(train_file, test_file):
    columns = ['age','workclass','fnlwgt','education','education-num',
               'marital-status','occupation','relationship','race','sex',
               'capital-gain','capital-loss','hours-per-week','native-country','label']
    train_data = pd.read_csv(train_file, names = columns, header = None, sep = ',')
    test_data = pd.read_csv(test_file, names = columns, header = None, sep = ',')

    train_data['label'] = train_data['label'].map(lambda x: 1 if x.strip() == '>50K' else 0)
    test_data['label'] = test_data['label'].map(lambda x: 1 if x.strip() == '>50K.' else 0)

    train_y = train_data['label']
    train_x = train_data.drop(['label'], axis = 1)

    test_y = test_data['label']
    test_x = test_data.drop(['label'], axis = 1)

    for col in continus_columns:
        ss = StandardScaler()
        train_x[col] = ss.fit_transform(train_x[[col]])
        test_x[col] = ss.transform(test_x[[col]])

    all_data = pd.concat([train_x, test_x], axis = 0)
    fea_dict, fea_size = gen_dict(all_data, columns)

    param['cate_fea_size'] = fea_size
    param['field_size'] = len(bin_columns)
    param['numeric_fea_size'] = len(continus_columns)

    train_di, train_dv, train_numeric_xv = to_id(train_x, fea_dict)
    test_di, test_dv, test_numeric_xv = to_id(test_x, fea_dict)
    return train_x.values.tolist(), train_y.values.tolist(), train_di.values.tolist(), train_dv.values.tolist(), train_numeric_xv, \
        test_di.values.tolist(), test_dv.values.tolist(), test_y.values.tolist(), test_numeric_xv

def gen_dict(df, columns):
    fea_dict = {}

    idx = 0
    for col in columns:
        if col in continus_columns:
            continue
        elif col in bin_columns:
            fea = df[col].unique()
            fea_dict[col] = dict(zip(fea, range(idx, idx + len(fea))))
            idx += len(fea)
        else:
            continue

    return fea_dict, idx

def to_id(df, fea_dict):
    dfi = df.copy()

    numeric_xv = df[continus_columns].values.tolist()
    dfi.drop(continus_columns, axis = 1, inplace = True)
    dfv = dfi.copy()
    for col in dfi.columns:
        if col in continus_columns:
            continue
        elif col in bin_columns:
            dfi[col] = dfi[col].map(lambda x: fea_dict[col][x])
            dfv[col] = 1.

    return dfi, dfv, numeric_xv

train_file = './data/adult.data.txt'
test_file = './data/adult.test.txt'
train_x, train_y, train_di, train_dv, train_numeric_xv, test_di, test_dv, test_y, test_numeric_xv = load_data(train_file, test_file)

fold = list(StratifiedKFold(n_splits = 3, shuffle = True, random_state = 2018).split(train_x, train_y))

_get = lambda x, y: [x[i] for i in y]

for i, (train_idx, valid_idx) in enumerate(fold):
    print "the %dth fold start" % (i + 1)
    train_xi, train_xv, train_nx, train_y_ = _get(train_di, train_idx), _get(train_dv, train_idx), _get(train_numeric_xv, train_idx), _get(train_y, train_idx)
    valid_xi, valid_xv, valid_nx, valid_y_ = _get(train_di, valid_idx), _get(train_dv, valid_idx), _get(train_numeric_xv, valid_idx), _get(train_y, valid_idx)

    dcn = DCN(**param)
    dcn.fit(train_xi, train_xv, train_nx, train_y_, valid_xi, valid_xv, valid_nx, valid_y_)

print 'Fit all train data......'
dcn = DCN(**param)
dcn.fit(train_di, train_dv, train_numeric_xv, train_y)
dcn.predict(test_di, test_dv, test_numeric_xv, test_y)



