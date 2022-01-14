import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import *

class NodeMinibatchIterator:
    def __init__(self, 
                 graphs,                 # all snapshots from dataset                    : testing snapshot at (t+1) modified: has (t) edges
                 adjs,
                 negative_mult_training,
                 batch_size=300,
                 start_time=0
        ):
        self.graphs = graphs
        self.adjs = adjs
        self.negative_mult_training = negative_mult_training
        self.batch_size = batch_size # number of edges
        self.start_time = start_time
                
        self.num_graphs = len(graphs)
        self.num_nodes = [len(graph.nodes) for graph in graphs]
            
        self.edges, self.num_edges = get_all_positive_edges(self.adjs[1:], self.num_nodes[:-1])
        
        self.cur_train_step = 0
        self.batch_num = 0
        
        print('Initializae NodeMinibatchIterator')
        
    def end(self, ):
        return (self.cur_train_step + 1 >= self.num_graphs-1) and (self.batch_num * self.batch_size >= self.num_edges[-1])

    def next_minibatch_feed_dict(self,):
        train_start = self.start_time
        train_end = self.cur_train_step + self.start_time + 1
        
        start_idx = self.batch_num * self.batch_size
        end_idx   = (self.batch_num +1) * self.batch_size
        
        pos_edges = self.edges[self.cur_train_step][:, start_idx : end_idx] # [2, batch_size]
        num_nodes = self.num_nodes[self.cur_train_step]
        
        pos_edges, neg_edges = get_non_existing_edges(pos_edges, num_nodes, self.negative_mult_training)
        
        self.batch_num += 1
        if (self.batch_num) * self.batch_size >= self.num_edges[self.cur_train_step] and (self.cur_train_step + 1 < self.num_graphs-1):
            self.cur_train_step += 1 # at least 1
            self.batch_num = 0
            
        return train_start, train_end, pos_edges, neg_edges 

    def shuffle(self, ):
        self.cur_train_step = 0
        self.batch_num = 0
        self.edges = [edges[:, np.random.permutation(num_edges)] for num_edges, edges in zip(self.num_edges, self.edges)]

##########################################################################
##########################################################################
##########################################################################
def get_non_existing_edges(pos_edges, num_nodes, negative_mult_training):

    adj_row = pos_edges[0, :]
    adj_col = pos_edges[1, :]
    edge_indices = set(adj_row * num_nodes + adj_col)
    
    num_edges = len(adj_row)
            
    # want to sample negative edges that are connected to the nodes in the positive edge set 
    neg_samples = negative_mult_training * num_edges

    # the maximum of edges would be all edges that don't exist between nodes that have edges
    neg_samples = min(neg_samples, num_nodes * (num_nodes-1) - len(edge_indices))

    # sample some negative candidates then remove unsatisfied ones
    edges = sample_edges(num_nodes, pos_edges, 4*neg_samples) 
    edge_ids = edges[0, :] * num_nodes + edges[1, :]
    
    out_ids = set()
    num_sampled = 0
    sampled_indices = []
    for i in range(neg_samples*4):
        eid = edge_ids[i]
        # ignore if any of these conditions happen
        if eid in out_ids or edges[0, i] < edges[1, i] or eid in edge_indices:
            continue

        #add the eid and the index to a list
        out_ids.add(eid)
        sampled_indices.append(i)
        num_sampled += 1

        #if we have sampled enough edges break
        if num_sampled >= neg_samples:
            break

    neg_edges = edges[:, sampled_indices]
    
    return pos_edges.astype(np.int32), neg_edges.astype(np.int32)

##########################################################################
##########################################################################
##########################################################################
# https://github.com/IBM/EvolveGCN/blob/90869062bbc98d56935e3d92e1d9b1b4c25be593/taskers_utils.py#L208
def sample_edges(num_nodes, pos_edges, neg_samples):
    
    existing_nodes = np.unique(pos_edges)
    num_existing_nodes = len(existing_nodes)
    
    prob = np.zeros(num_nodes)
    prob[np.random.choice(num_nodes, size=num_existing_nodes, replace=False)] = 1
    prob[existing_nodes] = 5
    prob = prob/np.sum(prob)

    ###
    from_id = np.random.choice(num_nodes, size=neg_samples, p=prob, replace=True)
    to_id = np.random.choice(num_nodes, size=neg_samples, p=prob, replace=True)

    edges = np.stack([from_id, to_id])
    return edges
    
##########################################################################
##########################################################################
##########################################################################
def get_all_positive_edges(all_adjs, all_visible_nodes):
    all_edges = []
    all_num_edges = []
    for t, adj in enumerate(all_adjs):
        
        visible_nodes = all_visible_nodes[t]
        adj = adj[:visible_nodes, :][:, :visible_nodes].tocoo()
        num_nodes = adj.shape[0]

        adj_row, adj_col = adj.row, adj.col
        pos_edges, num_edges = [], 0
        for row_, col_ in zip(adj_row, adj_col):
            if row_ > col_:
                pos_edges.append([row_, col_])
                num_edges += 1

        pos_edges = np.transpose(np.array(pos_edges)) # [2, num_edges]
        all_edges.append(pos_edges)
        all_num_edges.append(num_edges)
        
    return all_edges, all_num_edges

