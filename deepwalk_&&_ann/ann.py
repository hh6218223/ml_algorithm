import os, sys
import gensim
from annoy import AnnoyIndex
from time import time
import logging

logging.basicConfig(level= logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
model = gensim.models.Word2Vec.load('./model/w2v.model')
word_vectors = model.wv

vocab = word_vectors.vocab.keys()

t1 = time()
t = AnnoyIndex(64)

if not os.path.exists('./ann_index/t_index_bulid10.index'):
    for key in vocab:
    #print int(key)
    #print list(word_vectors[key])
    #print type(word_vectors[key])
        t.add_item(int(key), word_vectors[key].tolist())

    t.build(16)

    t2 = time()

    t.save('./ann_index/t_index_bulid10.index')
    print '[Annoy] time used: %f' % (t2 - t1)

t.load('./ann_index/t_index_bulid10.index')

t3 = time()

with open('ann_output.file', 'w') as fout:
    for key in vocab:
        ann_list = map(lambda x: str(x), t.get_nns_by_item(int(key), 20))
        fout.write('\t'.join(ann_list) + '\n')

t4 = time()

print '[Search Ann] time used: %f' % (t4 - t3)
