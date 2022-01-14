from __future__ import print_function
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
from .utilities import run_random_walks_n2v
import dill

#np.random.seed(123)

def cls_load_graphs(dataset_str):
    """
        Load graph snapshots given the name of dataset (all Gs, all Adjs)
        output: graphs: np_array of nx.multigraph
                adj_matrices: list of csr_matrix: no self-loop, adj weight are # of edges
    """
    graphs = np.load("./data/{}/{}".format(dataset_str, "graphs.npz"), allow_pickle=True)['graph']
    print("Loaded {} graphs ".format(len(graphs)))
    adj_matrices = list(map(lambda x: nx.adjacency_matrix(x), graphs))
    return graphs, adj_matrices


def cls_load_feats(dataset_str):
    """ 
        Load node attribute snapshots given the name of dataset (all features)
        output: features: list of csr_matrix
        
    """
    features = np.load("./data/{}/{}".format(dataset_str, "features.npz"), allow_pickle=True)['feats']
    print("Loaded {} X matrices ".format(len(features)))
    return features


def cls_sparse_to_tuple(sparse_mx):
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


def cls_tuple_to_tensor(tuple):
    """
    convert tuple to pytorch dense tensor
    """
    return torch.sparse_coo_tensor(tuple[0].T, tuple[1], tuple[2]).to_dense()


def cls_tuple_to_sparse(tuple, dtype):
    """
    convert tuple to pytorch sparse tensor
    """
    return torch.sparse_coo_tensor(tuple[0].T, tuple[1], tuple[2], dtype=dtype)


def cls_preprocess_features(features):
    """
        Row-normalize (single) feature matrix and convert to tuple representation
        output: (dense_matrix,(<np array of non-zero indexes>,<np array of non-zero elements>,<pair of mx shape>))
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), cls_sparse_to_tuple(features)


def cls_normalize_graph_gcn(adj, self_loop=True):
    """
        GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format
        Out: (<np array of non-zero indexes>,<np array of non-zero elements>,<pair of mx shape>)
    """
    adj = sp.coo_matrix(adj)
    if self_loop:
        adj_ = adj + sp.eye(adj.shape[0])
    else:
        adj_ = adj
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return cls_sparse_to_tuple(adj_normalized)


def cls_get_context_pairs(graphs_train, eval_time_step, window_size, dataset, force_regen=False):
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
    

def cls_create_data_splits(evaluation_graph, num_time_steps, val_mask_fraction=0.2, test_mask_fraction=0.2):
    """
        In: graphs_eval and the evaluation step (index from 1) along with test and val fractions.
        Node classification task returns the corresponding nodes to be used for train/val/test as node ids
        Out: ids, and associated labels for node classification (train/val/test)
    """
    ids, labels = cls_extract_node_labels(evaluation_graph)
    pos_inds, neg_inds = np.where(labels == 0), np.where(labels == 1)
    all_positives_ids = ids[np.where(labels == 0)]
    all_negatives_ids = ids[np.where(labels == 1)]
    # some assertions to make sure the code is right
    assert np.array_equal(pos_inds[0], all_positives_ids)
    assert np.array_equal(neg_inds[0], all_negatives_ids)
    # reconstruct dataset: ignore label == -1 class
    new_ids = np.concatenate((all_positives_ids, all_negatives_ids)) # new reconstructed index list for nodes
    new_labels = np.concatenate((np.zeros_like(all_positives_ids), np.ones_like(all_negatives_ids)))
    from sklearn.model_selection import train_test_split
    # split train / (val + test)
    train_frac = 1 - val_mask_fraction - test_mask_fraction # default 0.6
    x_train, x_test, y_train, y_test = train_test_split(new_ids, new_labels, stratify=new_labels, train_size=train_frac)
    # split val / test
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, stratify=y_test, train_size=val_mask_fraction/(val_mask_fraction+test_mask_fraction))
    
#     def balance(x, y):
#         num_neg = np.sum(y==0)
#         num_pos = np.sum(y==1)
#         select_num = np.min([num_neg, num_pos])
        
#         neg_inds = np.random.choice(np.where(y==0)[0], size=select_num, replace=False)
#         pos_inds = np.random.choice(np.where(y==1)[0], size=select_num, replace=False)
#         inds = np.concatenate([pos_inds, neg_inds])
        
#         return x[inds], y[inds]
    
#     x_train, y_train = balance(x_train, y_train)
#     x_val, y_val     = balance(x_val, y_val)
#     x_test, y_test   = balance(x_test, y_test)
    
    # print stats
    print("# train examples (cls): ", len(x_train))
    print("# val examples (cls):",    len(x_val))
    print("# test examples (cls):",   len(x_test))
    
    return x_train, y_train, x_val, y_val, x_test, y_test



def cls_get_evaluation_data(evaluation_graph, num_time_steps, dataset, force_regen=False):
    """ 
        Load train/val/test examples to evaluate node classification performance
        input: load_graphs() 2nd output; num_time_steps; dataset string: wiki_classification
        output: numpy arrays of x_train, y_train, x_val, y_val, x_test, y_test as node ids
    """
    eval_path = "./data/{}/eval_{}.npz".format(dataset, str(num_time_steps))
    
    if force_regen == False:
        try:
            x_train, y_train, x_val, y_val, x_test, y_test = \
                np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
            print("Loaded eval data")
        except (OSError, IOError):
            print("Generating and saving eval data (cls)....")
            x_train, y_train, x_val, y_val, x_test, y_test = \
                cls_create_data_splits(evaluation_graph, num_time_steps, val_mask_fraction=0.2, test_mask_fraction=0.2)
            np.savez(eval_path, data=np.array([x_train, y_train, x_val, y_val, x_test, y_test]))
    else:
        print("Re-generating and saving eval data (cls)....")
        x_train, y_train, x_val, y_val, x_test, y_test = \
            cls_create_data_splits(evaluation_graph, num_time_steps, val_mask_fraction=0.2, test_mask_fraction=0.2)
        np.savez(eval_path, data=np.array([x_train, y_train, x_val, y_val, x_test, y_test]))
 
    return x_train, y_train, x_val, y_val, x_test, y_test


def cls_process_graph(G):
    """
    from loaded multigraph to regular graph with:
        1. self-loops
        2. averaged edge features per snapshot
        3. un-normalized weights
        4. node labels
    """
    # what we want: a graph with self loops, that self loop has averaged 1-hop neighbor features
    new_G = nx.Graph()
    new_G.add_nodes_from(G.nodes(data=True))

    # merge multiedge to single edge
    for u, v, data in G.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        feat = np.array(data['feat'])
        edge_dim = feat.shape[0]
        if new_G.has_edge(u, v):
            new_G[u][v]['weight'] += w
            new_G[u][v]['feat'] += feat
        else:
            new_G.add_edge(u, v, weight=w, feat=feat)

    # set proper edge feature
    for u, v, data in new_G.edges(data=True):
        new_G[u][v]['feat'] = new_G[u][v]['feat'] / new_G[u][v]['weight']

    # add self-loops and set edge feature: generate tuple for edges
    for u, data in new_G.nodes(data=True):
        weights = np.array([data['weight'] for (i, j, data) in new_G.edges(u, data=True)])
        feats = np.array([data['feat'] for (i, j, data) in new_G.edges(u, data=True)])
        if len(weights) != 0:
            feat = np.average(feats, axis=0, weights=weights)
        else:
            feat = np.zeros(edge_dim)
        new_G.add_edge(u, u, weight=1, feat=feat)

    return new_G


def cls_extract_edge_features_dense(G):
    '''
    extract all edge features in a 'complete' fashion (align to adjacency matrix view) for new_G:
        0. usage: coupled after process_graph function
        1. values are normalized
    '''
    adj = cls_normalize_graph_gcn(nx.adjacency_matrix(G), self_loop=False)
    edges = G.edges(data=True)
    coords = adj[0]
    # values = adj[1]
    feats = np.array([G[coord[0]][coord[1]]['feat'] for coord in coords])
    # shape = adj[2]
    return feats # coords, values, shape, 


def cls_extract_node_labels(G):
    """
    extract node labels for a particluar new_G:
        0. usage: coupled after process_graph function
        1. -1 indicates item node label
        2. 0/1 as binary user node label
    """
    nodes = G.nodes(data=True)
    ids = np.array([node[0] for node in nodes])
    labels = np.array([node[1]['label'] for node in nodes])
    return ids, labels



#####################################################################


from scipy.sparse import csr_matrix

def cls_get_feats(adjs_train, num_nodes_graph_eval, train_start_time, eval_time, FLAGS):
    
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
        feats = cls_load_feats[train_start_time: eval_time]
    return feats