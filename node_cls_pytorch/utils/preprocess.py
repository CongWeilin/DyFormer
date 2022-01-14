from __future__ import print_function
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
from .utilities import run_random_walks_n2v
import dill
import pickle

#np.random.seed(123)

"""
Funcs:
    - load_graphs
    - load_feats
    - sparse_to_tuple
    - tuple_to_tensor
    - tuple_to_sparse
    - preprocess_features
    - normalize_graph_gcn
    - get_context_pairs
    - create_data_splits
    - get_evaluation_data
"""

def load_graphs(dataset_str):
    """
        Load graph snapshots given the name of dataset (all Gs, all Adjs)
        output: graphs: np_array of nx.multigraph
                adj_matrices: list of csr_matrix
    """
    graphs = np.load("./data/{}/{}".format(dataset_str, "graphs.npz"), allow_pickle=True)['graph']

    graphs = list(graphs)
    print("Loaded {} graphs ".format(len(graphs)))
    adj_matrices = list(map(lambda x: nx.adjacency_matrix(x), graphs))
    return graphs, adj_matrices


def load_feats(dataset_str):
    """ 
        Load node attribute snapshots given the name of dataset (all features)
        output: features: list of csr_matrix
        
    """
    features = np.load("./data/{}/{}".format(dataset_str, "features.npz"), allow_pickle=True)['feats']
    print("Loaded {} X matrices ".format(len(features)))
    return features

def sparse_to_tuple(sparse_mx):
    """
        Convert single scipy sparse matrix to tuple representation.
        Out: (<np array of non-zero indexes>,<np array of non-zero elements>,<pair of mx shape>)
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

#################### unit test for sparse_to_tuple() ####################

# a = sp.csc_matrix([[1, 0, 0, 0], [0, 0, 10, 11], [0, 0, 0, 99]])
# print(sparse_to_tuple(a))

#########################################################################

# def tuple_to_tensor(tuple):
#     """
#     convert tuple to pytorch dense tensor
#     """
#     return torch.sparse_coo_tensor(tuple[0].T, tuple[1], tuple[2]).to_dense()

def tuple_to_sparse(tuple, dtype):
    """
    convert tuple to pytorch sparse tensor
    """
    return torch.sparse_coo_tensor(tuple[0].T, tuple[1], tuple[2], dtype=dtype)

def preprocess_features(features):
    """
        Row-normalize (single) feature matrix and convert to tuple representation
        output: (dense_matrix,(<np array of non-zero indexes>,<np array of non-zero elements>,<pair of mx shape>))
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

#################### unit test for preprocess_features() ####################

# fea = load_feats("enron_raw")
# print(preprocess_features(fea[0]))

#########################################################################


def normalize_graph_gcn(adj):
    """
        GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format
        Out: (<np array of non-zero indexes>,<np array of non-zero elements>,<pair of mx shape>)
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def get_context_pairs(graphs_train, eval_time_step, window_size, dataset, force_regen=False):
    """ 
        Load/generate context pairs for each snapshot through random walk sampling.
        if does not present, generate for all snapshots
        input: load_graphs() first output
        output: [{node: [list of neighbors]; node: [list of neighbors]}, {node: [list of neighbors]; node: [list of neighbors]}] 
        len(output) = num_time_steps; len({}) = # of nodes in particular snapshot
    """
    load_path = "./data/{}/train_pairs_n2v_{}_{}.pkl".format(dataset, str(eval_time_step), str(window_size))
    
    if force_regen == False:
        try:
            context_pairs_train = dill.load(open(load_path, 'rb'))
            print("Loaded context pairs from pkl file directly")
        except (TypeError, IOError, EOFError):
            print("Computing training pairs ...")
            context_pairs_train = []
            for graph in graphs_train:
                context_pairs = run_random_walks_n2v(graph, graph.nodes())
                # print(len(graph.nodes), len(graph.edges), sum(context_pairs.keys()), min(context_pairs.keys()), max(context_pairs.keys()))
                context_pairs_train.append(context_pairs)
            dill.dump(context_pairs_train, open(load_path, 'wb'))
            print ("Saved pairs") 
    else:
        print("Recomputing training pairs ...")
        context_pairs_train = []
        for graph in graphs_train:
            context_pairs = run_random_walks_n2v(graph, graph.nodes())
            # print(len(graph.nodes), len(graph.edges), sum(context_pairs.keys()), min(context_pairs.keys()), max(context_pairs.keys()))
            context_pairs_train.append(context_pairs)
        dill.dump(context_pairs_train, open(load_path, 'wb'))
        print ("Re-saved pairs")
     
    return context_pairs_train

    
#################### unit test for get_context_pairs() ####################

# graphs, _ = load_graphs("enron_raw")
# o = get_context_pairs(graphs, 2)
# print(type(o), len(o))
# print(o[0].keys(), len(o[0]))

#########################################################################

def create_data_splits(adj, next_adj, val_mask_fraction=0.2, test_mask_fraction=0.6):
    """
        In: (adj, next_adj) along with test and val fractions. For link prediction (on all links), all links in
        next_adj are considered positive examples.
        Out: list of positive and negative pairs for link prediction (train/val/test)
    """
    # all edges in testing snapshot as indexes
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # Remove diagonal elements
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges()))) # all edges in testing snapshot 
    edges = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    edges = np.array(edges)

    # now, edges contains all edges in testing snapshots along with last training snapshot nodes

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)

    if len(all_edge_idx) < 5:
        num_test = len(all_edge_idx)
        num_val = len(all_edge_idx)
        val_edge_idx = all_edge_idx
        test_edge_idx = all_edge_idx
        test_edges = edges
        val_edges = edges
        train_edges = edges
    else:
        num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
        num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    
    # Create train edges.
    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    if len(all_edge_idx) >= 5:
        assert ~ismember(test_edges_false, edges_all)
        assert ~ismember(val_edges_false, edges_all)
        assert ~ismember(val_edges, train_edges)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)
    print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

    return list(train_edges), train_edges_false, list(val_edges), val_edges_false, list(test_edges), test_edges_false

def get_evaluation_data(last_train_adj, eval_adj, eval_time_step, dataset, force_regen=False):
    """ 
        Load train/val/test examples to evaluate link prediction performance
        input: load_graphs() 2nd output; num_time_steps; dataset string: enron_raw
        output: all returns are list of arrays [[],[],[]] as list of positive/negative edges
    """
    eval_path = "./data/{}/eval_{}.npz".format(dataset, str(eval_time_step))
    
    if force_regen == False:
        try:
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
                np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
            print("Loaded eval data")
        except (OSError, IOError):
            print("Generating and saving eval data ....")
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
                create_data_splits(last_train_adj, eval_adj, val_mask_fraction=0.2, test_mask_fraction=0.6)
            np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false, \
                test_edges, test_edges_false], dtype=object))
    else:
        print("Re-generating and saving eval data ....")
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            create_data_splits(last_train_adj, eval_adj, val_mask_fraction=0.2, test_mask_fraction=0.6)
        np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false, \
            test_edges, test_edges_false], dtype=object))

    return np.array(train_edges), np.array(train_edges_false), np.array(val_edges), np.array(val_edges_false), np.array(test_edges), np.array(test_edges_false)

#################### unit test for get_evaluation_data() ####################
# graphs, adjs = load_graphs("enron_raw")
# print(type(graphs),type(adjs))
# train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_evaluation_data(adjs, 2, "enron_raw")
# print(type(train_edges), len(train_edges))
# print(type(train_edges[0]), len(train_edges[0]))
#########################################################################

from scipy.sparse import csr_matrix

def get_feats(adjs_train, num_nodes_graph_eval, train_start_time, eval_time, FLAGS):
    
    if FLAGS.feature_less == True:
        feats = []
        for adj in adjs_train:
            cur_num_nodes = adj.shape[0]
            cur_node_ids = np.arange(cur_num_nodes)
            cur_feats = csr_matrix((np.ones_like(cur_node_ids), (cur_node_ids, cur_node_ids)), 
                                   shape=(cur_num_nodes, num_nodes_graph_eval), dtype=np.float32)    
            feats.append(cur_feats)
    else:
        feats = load_feats(FLAGS.dataset)
        feats = feats[train_start_time: eval_time]
    return feats

def update_eval_graph(graph_train, graph_eval):
    new_graph_eval = nx.MultiGraph()
    new_graph_eval.add_nodes_from(graph_eval.nodes(data=True)) 
    for e in graph_train.edges():                      
        new_graph_eval.add_edge(e[0], e[1])
    
    return new_graph_eval, nx.adjacency_matrix(new_graph_eval)