import numpy as np
import networkx as nx
import scipy.sparse as sp
from utilities import *
from preprocess import *

import torch
import torch.nn as nn
import torch.nn.functional as F

######### test on torch.sparse ############

# WINDOW = -1 # window for temporal attention: if -1, apply on whole time axis

# NEG_SAMPLE_SIZE = 10 # number of negative samples 

# NUM_TIME_STEPS = 3 # use 2 snapshots to train, the 3rd one to predict

# graphs, adjs = load_graphs("enron_raw")
# feats = load_feats("enron_raw")

# adj_train = []
# feats_train = []
# num_features_nonzero = []
# loaded_pairs = False

# # Load training context pairs (or compute them if necessary)
# context_pairs_train = get_context_pairs(graphs, NUM_TIME_STEPS)

# # Load evaluation data: picking edges constrainted in nodeset from last training snapshot and from next snapshot edges
# train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_evaluation_data(adjs, NUM_TIME_STEPS, "enron_raw")

# # Create the adj_train so that it includes nodes from (t+1) but only edges from t: this is for the purpose of
# # inductive testing. 
# # new_G has all nodes in predicting snapshots but only edges from last training snapshot

# new_G = nx.MultiGraph()
# new_G.add_nodes_from(graphs[NUM_TIME_STEPS - 1].nodes(data=True))

# for e in graphs[NUM_TIME_STEPS - 2].edges():
#     new_G.add_edge(e[0], e[1])

# graphs[NUM_TIME_STEPS - 1] = new_G
# adjs[NUM_TIME_STEPS - 1] = nx.adjacency_matrix(new_G)

# # make graphs, adjs has its predicting snapshot has some of its edges removed to do inductive testing

# print("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))

# # Normalize and convert adj. to sparse tuple format (to provide as input via SparseTensor)
# adj_train = list(map(lambda adj: normalize_graph_gcn(adj), adjs))

# num_features = feats[0].shape[1]

# feats_train = list(map(lambda feat: preprocess_features(feat)[1], feats))

# num_features_nonzero = [x[1].shape[0] for x in feats_train]

# a = torch.sparse_coo_tensor(adj_train[0][0].T, adj_train[0][1], adj_train[0][2])
# print(a)
# #########