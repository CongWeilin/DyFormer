import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import os
import sys
import dill

def simple_graph():
    G = nx.MultiGraph()
    G.add_nodes_from([(0, {"label": -1}), (1, {"label": -1}), (2, {"label": 1}), (3, {"label": 0})])
    G.add_edges_from([(0, 1, {"feat": [0, 0, 1]}), (1, 2, {"feat": [1, 1, 2]}), \
    (1, 0, {"feat": [0, 1, 0]}), (1, 3, {"feat": [1, 1, 3]})])
    return G

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

def normalize_graph_gcn(adj, self_loop=True):
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
    print(adj_normalized)
    return sparse_to_tuple(adj_normalized)

G = simple_graph()
G_adj = nx.adjacency_matrix(G)

print('original adj: ', G_adj)
adj = normalize_graph_gcn(G_adj)
print('original normalized adj: ', adj)

def process_graph(G):
    """
    from multigraph to graph with self-loops and edge features and weights
    also with node labels
    """
    # what we want: a graph with self loops, that self loop has averaged 1-hop neighbor features
    new_G = nx.Graph()
    new_G.add_nodes_from(G.nodes(data=True))

    # merge multiedge to single edge
    for u, v, data in G.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        feat = np.array(data['feat'])
        if new_G.has_edge(u, v):
            new_G[u][v]['weight'] += w
            new_G[u][v]['feat'] += feat
        else:
            new_G.add_edge(u, v, weight=w, feat=feat)

    # set proper edge feature
    for u, v, data in new_G.edges(data=True):
        new_G[u][v]['feat'] = new_G[u][v]['feat'] / new_G[u][v]['weight']

    # add self-loops and set edge feature: 7 edges in total: generate tuple for edges
    for u, data in new_G.nodes(data=True):
        weights = np.array([data['weight'] for (i, j, data) in new_G.edges(u, data=True)])
        feats = np.array([data['feat'] for (i, j, data) in new_G.edges(u, data=True)])
        feat = np.average(feats, axis=0, weights=weights)
        new_G.add_edge(u, u, weight=1, feat=feat)

    return new_G

# extract edge features to tuple format
new_G = process_graph(G)
# print('#################################')
# print(new_G.nodes(data=True))
# print(new_G.edges(data=True))
# print('#################################')


def extract_edge_features(G):
    '''
    extract all edge features in a compact version: values here are not normalized (thus not returned)
    '''
    edges = G.edges(data=True)
    coords = np.array([(edge[0], edge[1]) for edge in edges])
    # values = np.array([edge[2]['weight'] for edge in edges])
    feats = np.array([edge[2]['feat'] for edge in edges])
    return coords, feats

def extract_edge_features_dense(G):
    '''
    extract all edge features in a 'complete' fashion (align to adjacency matrix view): values are also normalized
    '''
    adj = normalize_graph_gcn(nx.adjacency_matrix(G), self_loop=False)
    edges = G.edges(data=True)
    coords = adj[0]
    values = adj[1]
    feats = np.array([G[coord[0]][coord[1]]['feat'] for coord in coords])
    shape = adj[2]
    return coords, values, shape, feats 

def extract_node_labels(G):
    """
    extract node labels
    """
    nodes = G.nodes(data=True)
    ids = np.array([node[0] for node in nodes])
    labels = np.array([node[1]['label'] for node in nodes])
    return ids, labels

# print(extract_edge_features(new_G))
# print(nx.adjacency_matrix(new_G))
# print(normalize_graph_gcn(nx.adjacency_matrix(new_G), self_loop=False))
# print(extract_edge_features_dense(new_G))
# print('######################')
# print(extract_node_labels(new_G))

def tuple_to_sparse(tuple, dtype):
    """
    convert tuple to pytorch sparse tensor
    """
    return torch.sparse_coo_tensor(tuple[0].T, tuple[1], tuple[2], dtype=dtype)


# new tests
h = np.array([[0.1, 0.2],[1.1, 1.2],[2.1, 2.2],[3.1, 3.2]])
new_adj = nx.adjacency_matrix(new_G)
new_adj = normalize_graph_gcn(new_adj, self_loop=False)
print('new G adj: ', new_adj)

# convert to torch
new_adj = tuple_to_sparse(new_adj, torch.float32)
adj_mat = new_adj.coalesce()
edge_idxs = new_adj._indices()                                                  # [2, E]
edge_w = new_adj._values()                                                      # [E]
print(edge_idxs)
print(edge_w)

# try to add edge features to calculate attention
h = torch.from_numpy(h)
edge_h = torch.cat((h[edge_idxs[0, :], :], h[edge_idxs[1, :], :]), dim=1).T    # [2F, E]
print(edge_h)

# check edge features #
print("########################")
print(extract_edge_features_dense(new_G))
print("########################")
# add edge feature in
coords, values, shape, edge_feats = extract_edge_features_dense(new_G)
edge_feats = torch.from_numpy(edge_feats)
edge_h = torch.cat((h[edge_idxs[0, :], :], edge_feats, h[edge_idxs[1, :], :]), dim=1).T    # [2F, E]
print(edge_h)