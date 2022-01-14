import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import *

class NodeMinibatchIterator:
    def __init__(self, 
                 graphs,                 # all snapshots from dataset                    : testing snapshot at (t+1) modified: has (t) edges
                 adjs,
                 start_time=0,
                 batch_size=300,                                    
        ):
        self.graphs = graphs
        self.adjs = adjs
        self.batch_size = batch_size
        self.start_time = start_time
                
        self.num_graphs = len(graphs)
        self.num_nodes = [len(graph.nodes) for graph in graphs]
        self.node_inds = [np.random.permutation(num_nodes) for num_nodes in self.num_nodes]

        self.cur_train_step = 0
        self.batch_num = 0
        print('Initializae NodeMinibatchIterator')

    def end(self, ):
        # current graph is the last graph and
        # current mini-batch is the last mini-batch
        return (self.cur_train_step + 1 >= self.num_graphs) and (self.batch_num * self.batch_size >= self.num_nodes[-1])

    def next_minibatch_feed_dict(self,):

        train_start = self.start_time
        train_end = self.cur_train_step + 1 + self.start_time

        start_idx = self.batch_num * self.batch_size
        end_idx   = (self.batch_num +1) * self.batch_size
        batch_nodes = self.node_inds[self.cur_train_step][start_idx : end_idx]

        # if the next mini-batch is out of the current graph size

        self.batch_num += 1
        if (self.batch_num) * self.batch_size >= self.num_nodes[self.cur_train_step] and (self.cur_train_step + 1 < self.num_graphs):
            self.cur_train_step += 1 # at least 1
            self.batch_num = 0

        return train_start, train_end, batch_nodes

    def shuffle(self, ):
        self.cur_train_step = 0
        self.batch_num = 0
        self.node_inds = [np.random.permutation(num_nodes) for num_nodes in self.num_nodes]

