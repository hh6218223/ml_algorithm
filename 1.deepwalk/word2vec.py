import os, sys
import gensim
from multiprocessing import cpu_count
import logging
from time import time

class randomWalk(object):
    def __init__(self, file_list):
        self.file_list = file_list

    def __iter__(self):
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    yield line.split()

def process():
    filelist = ['/home/tanzhengan/11.tools/deepwalk/5.random_walk/script/path.file']
    model = gensim.models.Word2Vec(sentences = randomWalk(filelist), size = 64, window = 5, min_count = 3, sg=1, hs=1, workers = cpu_count())

    #filelist = ['./tmp']
    #model = gensim.models.Word2Vec(sentences = randomWalk(filelist), size = 10, window = 5, min_count = 5, sg=1, hs=1)
    model.save('./model/w2v.model')

if __name__ =='__main__':
    logging.basicConfig(level= logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    t1 = time()
    process()
    logging.info('[W2V] time used: %f' % (time() - t1))
