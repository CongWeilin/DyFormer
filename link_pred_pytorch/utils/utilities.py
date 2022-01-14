from __future__ import print_function
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from .random_walk import Graph_RandomWalk

"""
Functions:
    - to_one_hot
    - sample_mask
    - run_random_walks_n2v
"""

def to_one_hot(labels, N, multilabel=False):
    """
        In: list of (nodeId, label) tuples, #nodes N
        Out: N * |label| numpy matrix
    """
    ids, labels = zip(*labels)
    lb = MultiLabelBinarizer()
    if not multilabel:
        labels = [[x] for x in labels]
    lbs = lb.fit_transform(labels)
    encoded = np.zeros((N, lbs.shape[1]))
    for i in range(len(ids)):
        encoded[ids[i]] = lbs[i]
    return encoded


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


"""Random walk-based pair generation."""

def run_random_walks_n2v(graph, nodes, num_walks=10, walk_len=40):
    """ 
        In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using the sampling strategy of node2vec (deepwalk)
        Out: 
    """
    nx_G = nx.Graph()
    adj = nx.adjacency_matrix(graph)

    # add all edges from input graph to nx_G graph
    for e in graph.edges():
        nx_G.add_edge(e[0], e[1])

    # set weight for all edges in nx_G graph
    for edge in graph.edges():
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_len)

    # now we have num_walks * number of nodes = 180 walks of length 20 (for the one we are examining, 18 nodes in total) type: list
    WINDOW_SIZE = 10
    pairs = defaultdict(lambda: [])
    pairs_cnt = 0
    for walk in walks:
        # for each walk
        for word_index, word in enumerate(walk):
            # for each node in a given walk
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs

#################### unit test for run_random_walks_n2v() ####################

# walk = [9, 0, 0, 10, 0, 5, 11, 5]
# WINDOW_SIZE = 10
# pairs = defaultdict(lambda: [])
# pairs_cnt = 0
# for word_index, word in enumerate(walk):
#     # for each node in a given walk
#     for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
#         if nb_word != word:
#             pairs[word].append(nb_word)
#             pairs_cnt += 1

# print(pairs)
#########################################################################