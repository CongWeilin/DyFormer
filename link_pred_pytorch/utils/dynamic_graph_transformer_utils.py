import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix, coo_matrix
import pickle
import os
import numpy as np
import time

# import cupy as cp
# from cupyx.scipy.sparse import csr_matrix as csr_gpu

##########################################################################
##########################################################################
##########################################################################
"""
Generate node ane edge encodings
"""

def row_normalize(mx): # Checked
    """Row-normalize sparse matrix"""
    mx = mx - sp.diags(mx.diagonal()) + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.divide(np.ones_like(rowsum), rowsum,
                      out=np.zeros_like(rowsum), where=rowsum != 0)
    r_inv = r_inv.flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    

def generate_compressed_graphs(graphs, alpha = 0.15, max_dist=5): # Checked
    print('max_distance', max_dist)
    
    window_size = len(graphs)
    all_nodes = list(graphs[-1].nodes)
    num_nodes = len(all_nodes)
    
    edge_encode_set_all = set()
    edge_encode_all_row, edge_encode_all_col, edge_encode_all_data = [], [], []

    #### edge_encode
    edge_encode_list = []
    for t, G in enumerate(graphs[::-1]):
        edge_encode_row, edge_encode_col, edge_encode_data = [], [], []
        edge_encode_set = set()
    
        for edge in G.edges:
            edge_ind = num_nodes * edge[0] + edge[1]

            if edge_ind not in edge_encode_set:
                edge_encode_set.add(edge_ind)
                edge_encode_row.append(edge[0])
                edge_encode_col.append(edge[1])
                edge_encode_data.append(t)

            if edge_ind not in edge_encode_set_all:
                edge_encode_all_row.append(edge[0])
                edge_encode_all_col.append(edge[1])
                edge_encode_all_data.append(1)
        edge_encode_list.append(
            csr_matrix((edge_encode_data, (edge_encode_row, edge_encode_col)), shape=(num_nodes, num_nodes))
        )
    
    #### edge_dist_encode
    st = time.time()
    adj = nx.to_scipy_sparse_matrix(graphs[-1])
    edge_dist_encode = compute_shortest_path_dist(adj, max_dist)
    print('Compute Edge encoding takes time', time.time() - st)

    #### PPR
    st = time.time()
    sparse_edge_exist = csr_matrix((edge_encode_all_data, (edge_encode_all_row, edge_encode_all_col)), shape=(num_nodes, num_nodes))
    G = nx.from_scipy_sparse_matrix(sparse_edge_exist)
    PPR = np.array(nx.google_matrix(G, alpha), dtype=np.float32)
    print('Compute PPR takes time', time.time() - st)    

    
    return edge_encode_list, PPR, edge_dist_encode

def compute_shortest_path_dist(adj, dmax):
    # self connect dist = 0
    # if not neighbors within dmax, set distance as dmax+1
    A = adj.astype(np.int32)
    num_nodes = adj.shape[0]
    A = A - sp.diags(A.diagonal()) + sp.eye(A.shape[0])
    A[A != 0] = 1
    
    x = A.toarray()
    x_prev = np.zeros_like(x)
    short_path_dist_row  = []
    short_path_dist_col  = []
    short_path_dist_data = []
    
    # check if a node is the k-hop neighbor of another node
    for i in range(dmax):
        
        row, col = np.where(x - x_prev !=0)
        short_path_dist_row.append(row)
        short_path_dist_col.append(col)
        short_path_dist_data.append(np.ones_like(row)*(i+1))
        
        x_prev = x
        x = A.dot(x)
        x[x!=0]=1
        
    short_path_dist_row  = np.concatenate(short_path_dist_row)
    short_path_dist_col  = np.concatenate(short_path_dist_col)
    short_path_dist_data = np.concatenate(short_path_dist_data)
    
    short_path_dist = csr_matrix((short_path_dist_data, (short_path_dist_row, short_path_dist_col)),  shape=(num_nodes, num_nodes))
    short_path_dist = short_path_dist - sp.diags(short_path_dist.diagonal())
    return short_path_dist

def compute_node_edge_encoding(graphs_train, FLAGS, force_regen=False):
    compact_edges_dir = "./data/%s/compact_edges_et%d_wz%d.pkl"%(FLAGS.dataset, FLAGS.eval_time, FLAGS.cur_window)
    
    if force_regen and os.path.exists(compact_edges_dir):
        os.remove(compact_edges_dir)
        print('Re-generate new encodings.')
        
    if os.path.exists(compact_edges_dir):
        with open(compact_edges_dir, 'rb') as f:
            data_dict = pickle.load(f)
        print('Load encoding from', compact_edges_dir)
    else:
        # same start point with diff end point [eval_graph, 1], [0, 2], ..., [0, eval_time-1]
        print('Compute node edge encoding')
        num_graphs = len(graphs_train)
        data_dict = dict()
        for i in range(1, num_graphs+1):
            edge_encode, PPR, edge_dist_encode = generate_compressed_graphs(graphs_train[0 : i], max_dist=FLAGS.max_dist)
            dict_name = 'st_%d_et_%d'%(FLAGS.train_start_time, FLAGS.train_start_time+i)
            print(dict_name)
            data_dict[dict_name] = [edge_encode, PPR, edge_dist_encode]
        with open(compact_edges_dir, 'wb') as f:
            pickle.dump(data_dict, f)
        print('Save to dir:', compact_edges_dir)
    return data_dict

##########################################################################
##########################################################################
##########################################################################

def translate(edges, node_trans_dict, shape):
        new_edges = []
        
        if shape == 'Nx2':
            for i in range(edges.shape[0]):
                new_edges.append([node_trans_dict[edges[i,0]], node_trans_dict[edges[i,1]]])
            new_edges = np.array(new_edges)
        elif shape == '2xN':
            for i in range(edges.shape[1]):
                new_edges.append([node_trans_dict[edges[0,i]], node_trans_dict[edges[1,i]]])
            new_edges = np.transpose(np.array(new_edges))
        else:
            edges_shape = np.shape(edges)
            edges_flatten = np.flatten(edges)
            for i in range(edges_flatten.shape[0]):
                new_edges.append(node_trans_dict[edges_flatten[i]])
            new_edges = np.reshape(np.array(edges_flatten), edges_shape)
            
        return new_edges
    
def get_new_indices(edges_list, transpose=False):
    
    if transpose: # [from Nx2 to 2xN]
        edges_list = [np.transpose(edges) for edges in edges_list]
        
    node_trans_dict = dict()
    for i, node in enumerate(np.unique(np.concatenate(edges_list, axis=1))):
        node_trans_dict[node] = i
    
    edges_list = [translate(edges, node_trans_dict) for edges in edges_list]
    
    if transpose: # [from 2xN to Nx2]
        edges_list = [np.transpose(edges) for edges in edges_list]
    return edges_list, node_trans_dict

def sample_joint_neighbors(active_nodes, PPR, deterministic=True, two_stream_structure=True, max_neighbors=-1):
    num_active_nodes = len(active_nodes)
    
    if max_neighbors >= 0:
        num_neighbors = min(num_neighbors, max_neighbors)
    
    if two_stream_structure:
        num_active_nodes = len(active_nodes)
        num_neighbors = PPR.shape[0] - len(active_nodes)
        
        # no bigger than num_neighbors
        if max_neighbors >= 0:
            num_neighbors = min(num_active_nodes, max_neighbors)
        else:
            num_neighbors = num_active_nodes

        
        sampling_prbs = np.array(np.sum(PPR[active_nodes, :], axis=0)).reshape(-1)
        sampling_prbs[active_nodes] = 0.
        sampling_prbs = sampling_prbs/np.sum(sampling_prbs)
        num_neighbors = min(num_neighbors, np.sum(sampling_prbs>0))
        if deterministic:
            shared_neighbors = np.argsort(sampling_prbs)[-num_neighbors:]
        else:
            shared_neighbors = np.random.choice(PPR.shape[0], p=sampling_prbs, size=num_neighbors, replace=False)
            
        num_shared_neighbors = len(shared_neighbors)
        if num_shared_neighbors < num_active_nodes:
            select = np.random.permutation(num_active_nodes)[:num_active_nodes-num_shared_neighbors]
            shared_neighbors = np.concatenate([shared_neighbors, active_nodes[select]])
        num_shared_neighbors = len(shared_neighbors)
        
        assert num_shared_neighbors == num_active_nodes
        
        all_nodes = np.concatenate([active_nodes, shared_neighbors])
        target_node_size = len(active_nodes)
        context_nodes_size = len(shared_neighbors)
    else:
        all_nodes = active_nodes
        target_node_size = len(active_nodes)
        context_nodes_size = 0
        
    new_index = np.arange(len(all_nodes))
    all_nodes_to_new_index = dict(zip(all_nodes, new_index))
    return all_nodes, all_nodes_to_new_index, target_node_size, context_nodes_size

def generate_temporal_edges(edge_encode_np, target_node_size, neg_sample_size):
    time_steps = edge_encode_np.shape[-1]

    # edge_encode_np [num_nodes, num_nodes, time_steps]
    edge_encode_np = edge_encode_np[:target_node_size, :target_node_size, :]
    sampled_temporal_edges = []
    for t in range(time_steps):
        # positive
        pos_row, pos_col = np.where(edge_encode_np[:, :, t]%2==1)
        pos_edges = np.stack([pos_row, pos_col])
        num_pos_samples = len(pos_row)

        # negative
        neg_row, neg_col = np.where(edge_encode_np[:, :, t]%2==0)
        num_neg_samples = len(neg_row)
        select = np.random.permutation(num_neg_samples)[:neg_sample_size*num_pos_samples]
        neg_row, neg_col = neg_row[select], neg_col[select]
        neg_edges = np.stack([neg_row, neg_col])

        sampled_temporal_edges.append([pos_edges, neg_edges])
    
    return sampled_temporal_edges