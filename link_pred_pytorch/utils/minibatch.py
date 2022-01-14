import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import *

"""
Class:
    - NodeMinibatchIterator
        Funcs:
        - construct_degs
        - end
        - batch_feed_dict
        - num_training_batches
        - next_minibatch_feed_dict
        - shuffle
        - test_reset
"""

#np.random.seed(123)

##### Current test: 18, 23, 24 # of nodes in 1st, 2nd(training), 3rd(testing) snapshots

class NodeMinibatchIterator:
    """
    This minibatch iterator iterates over nodes to sample context pairs for a batch of nodes.

    graphs -- list of networkx graphs
    adjs -- list of adj matrices (of the graphs)
    num_time_steps -- number of graphs to train +1
    context_pairs -- list of (target, context) pairs obtained from random walk sampling.
    batch_size -- size of the minibatches (# nodes)
    """
    def __init__(self, 
                 window,                 # window for temporal attention: if -1, apply on whole time axis
                 neg_sample_size,        # number of negative samples 
                 graphs,                 # all snapshots from dataset                    : testing snapshot at (t+1) modified: has (t) edges
                 context_pairs=None, 
                 batch_size=100          # number of nodes to sample per batch
        ):
        self.window = window
        self.neg_sample_size = neg_sample_size

        self.graphs = graphs
        
        assert window == len(graphs)
        
        self.batch_size = batch_size
        self.batch_num = 0
        self.degs = self.construct_degs()       # [ list of arrays of degrees for graph at time t ]

        self.context_pairs = context_pairs
        self.max_positive = neg_sample_size

        # this part, we assume that the last snapshot has the most amount of nodes
        self.train_nodes = np.array(self.graphs[-1].nodes()) # all nodes in the graph at predicting snapshot
        print('Initializae NodeMinibatchIterator')
        print ("# train nodes", len(self.train_nodes))

    def construct_degs(self):
        """ 
            Compute node degrees in each graph snapshot.
            Output: list of np arrays of node degrees (index as node number, value as degree)
        """
        degs = []
        for graph in self.graphs:
            num_nodes = len(graph.nodes())
            deg = np.zeros((num_nodes,))
            for nodeid in graph.nodes():
                neighbors = np.array(list(graph.neighbors(nodeid)))
                deg[nodeid] = len(neighbors)
            degs.append(deg)
        return degs

    def end(self):
        """
        check whether sampling reaches end
        """
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes):
        """ 
            Return node pairs
        """
        node_1_all = []
        node_2_all = []
            
        for t in range(self.window):
            node_1 = []
            node_2 = []

            for n in batch_nodes:
                if len(self.context_pairs[t][n]) > self.max_positive:
                    node_1.extend([n]* self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][n], self.max_positive, replace=False))
                else:
                    node_1.extend([n]* len(self.context_pairs[t][n]))
                    node_2.extend(self.context_pairs[t][n])

            assert len(node_1) == len(node_2)
            assert len(node_1) <= self.batch_size * self.max_positive

            node_1_all.append(node_1)
            node_2_all.append(node_2)

        return node_1_all, node_2_all

    def num_training_batches(self):
        """ 
            Compute the number of training batches (using batch size)
        """
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        """ 
            Return the info for the next minibatch (in the current epoch) with random shuffling
        """
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        node_1_all, node_2_all = self.batch_feed_dict(batch_nodes)

        proximity_neg_samples = get_proximity_negative_sample(degrees=self.degs, 
                                                              num_samples=self.neg_sample_size, 
                                                              distortion=0.75)

        return node_1_all, node_2_all, proximity_neg_samples

    def shuffle(self):
        """ 
            Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def test_reset(self):
        """ 
            Reset batch number
        """
        self.train_nodes = np.array(self.graphs[-1].nodes()) 
        self.batch_num = 0

##########################################################################
##########################################################################
##########################################################################
def fixed_unigram_candidate_sampler(
    num_samples,            # number of idx to sample
    range_max,              # sampling range (idx)
    distortion,             # unigrams distortion
    unigrams,               # list; should have length == range_max
    replacement=True,       # true or false: sample with/without replacement
):
    """
    basic unigram_candidate_sampler
    return: list of shape [num_samples]
    """
    assert len(unigrams) == range_max
    unigrams = np.array(unigrams)
    if distortion != 1.:
        unigrams = unigrams.astype(np.float64) ** distortion
    result = np.zeros(num_samples, dtype=np.int64)
    result = torch.utils.data.WeightedRandomSampler(
        weights=unigrams,
        num_samples=num_samples,
        replacement=replacement,
    )
    return list(result)


def get_proximity_negative_sample(
    degrees,        # [T, Nt]
    num_samples,    # number of samples
    distortion,     # sampling distortion
):
    """
    return list of shape [T-1, num_samples]
    """
    proximity_neg_samples = []
    for t in range(len(degrees)):
        proximity_neg_samples.append(
            fixed_unigram_candidate_sampler(
                num_samples=num_samples,            # number of idx to sample
                range_max=len(degrees[t]),          # sampling range (idx)
                distortion=distortion,              # unigrams distortion
                unigrams=degrees[t].tolist(),       # list; should have length == range_max
            )
        )
    return proximity_neg_samples

################################ unit test for NodeMinibatchIterator() ################################

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

# minibatchIterator = NodeMinibatchIterator(
#     graphs, feats_train, adj_train,
#     NUM_TIME_STEPS, batch_size=512,
#     context_pairs=context_pairs_train
# )

# minibatchIterator.shuffle()
# node_1_all, node_2_all, features, adjs, batch_nodes = minibatchIterator.next_minibatch_feed_dict()

# print(features[0])
# print(adjs[0])
# print(batch_nodes)

# print(features[1])
# print(adjs[1])
# print(batch_nodes)

# print(features[2])
# print(adjs[2])
# print(batch_nodes)