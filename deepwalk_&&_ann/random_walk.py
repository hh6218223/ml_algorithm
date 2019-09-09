#coding: gbk

import sys, os
import random
from time import time

def load_dict(fpath):
    node_dict = {}
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue

            data_list = line.split('\t')
            node = int(data_list[0])
            vertex_list = [int(x) for x in data_list[1:]]

            node_dict[node] = vertex_list
    return node_dict

def walk(node_dict, path_len, rand = random.Random(0), shuffle_size = 500):
    #pd.read_csv(fpath, sep = '\t', )
    nodes = list(node_dict.keys())
    loop_cnt = len(nodes) / shuffle_size
    i = 0
    while i < loop_cnt:
        random_list = nodes[i * shuffle_size : (i + 1) * shuffle_size]
        rand.shuffle(random_list)
        for node in random_list:
            yield random_walk(node, node_dict, path_len, rand)

        i += 1

    random_list = nodes[i * shuffle_size:]
    rand.shuffle(random_list)
    for node in random_list:
        yield random_walk(node, node_dict, path_len, rand)

    #for node in nodes:
    #    yield random_walk(node, node_dict, path_len, rand)
    
def random_walk(start, node_dict, path_len, rand = random.Random(0)):
    path = [start]
    while len(path) < path_len:
        cur = path[-1]
        if cur not in node_dict:
            break
        if len(node_dict[cur]) > 0:
            path.append(rand.choice(node_dict[cur]))
        else:
            break
    return [str(node) for node in path]

if __name__ == '__main__':
    fpath = sys.argv[1]
    fout = sys.argv[2]

    pathlen = 6
    shuffle_size = 300
    epoch = 1

    t0 = time()
    node_dict = load_dict(fpath)

    t1 = time()
    print '[Load Dict] time used: %f' % (t1 - t0)

    with open(fout, 'w+') as f:
        for i in range(epoch):
            for path in walk(node_dict, path_len = pathlen, shuffle_size = shuffle_size):
                if len(path) < pathlen:
                    continue
                f.write(u"{}\n".format(u" ".join(v for v in path)))

    print '[Gen pathfile] time used: %f' % (time() - t1)
