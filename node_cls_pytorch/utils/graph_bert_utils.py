import abc
import networkx as nx
import pickle
import hashlib
import numpy as np
import os

from scipy.sparse import eye
from scipy.sparse.linalg import inv

import torch

from .dynamic_graph_transformer_utils import row_normalize
    
##########################################################################
##########################################################################
##########################################################################

class WLNodeColoring:
    def __init__(self, max_iter=2):
        self.max_iter = max_iter
            
    def create_dicts(self, ):
        self.node_color_dict = {}
        self.node_neighbor_dict = {}
        
    def get_WLNodeColoring(self, G):
        self.create_dicts()
        undirct_G = nx.Graph(G)
        node_list = undirct_G.nodes
        link_list = undirct_G.edges
        self.setting_init(node_list, link_list) 
        self.WL_recursion(node_list)
        return self.node_color_dict
    
    def setting_init(self, node_list, link_list):
        for node in node_list:
            self.node_color_dict[node] = 1
            self.node_neighbor_dict[node] = {}

        for pair in link_list:
            u1, u2 = pair
            if u1 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u1] = {}
            if u2 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u2] = {}
            self.node_neighbor_dict[u1][u2] = 1
            self.node_neighbor_dict[u2][u1] = 1
        
    def WL_recursion(self, node_list):
        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in node_list:
                neighbors = self.node_neighbor_dict[node]
                neighbor_color_list = [self.node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(self.node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if self.node_color_dict == new_color_dict or iteration_count == self.max_iter:
                return
            else:
                self.node_color_dict = new_color_dict
            iteration_count += 1

    
##########################################################################
##########################################################################
##########################################################################


class GraphBatching:
    def __init__(self, k=5, alpha=0.15):
        self.k = k
        self.alpha = alpha
        
    def graph_batching(self, G):
        
        undirct_G = nx.Graph(G)
        num_nodes = len(undirct_G)
        A = nx.adjacency_matrix(undirct_G) + eye(num_nodes)
        DA = row_normalize(A)

        PPR = eye(num_nodes) - (1-self.alpha) * DA
        PPR = self.alpha * inv(PPR.tocsc())

        batch_dict = {}
        for node_id in range(num_nodes):
            s_row = PPR[node_id, :].toarray().reshape(-1)
            s_row[node_id] = -1000.0
            top_k_neighbor_index = np.argsort(s_row)[-self.k:][::-1]
            batch_dict[node_id] = []
            for neighbor_id in top_k_neighbor_index:
                batch_dict[node_id].append((neighbor_id, s_row[neighbor_id]))

        hop_dict = {}
        for node in batch_dict:
            if node not in hop_dict: hop_dict[node] = {}
            for neighbor, score in batch_dict[node]:
                try:
                    hop = nx.shortest_path_length(G, source=node, target=neighbor)
                except:
                    hop = 99
                hop_dict[node][neighbor] = hop
        
        return batch_dict, hop_dict
    
    
# graph_batching = GraphBatching()
# graph_batching.graph_batching(G)

##########################################################################
##########################################################################
##########################################################################


def save_encoding_data(graphs, FLAGS, force_regen=False):
    data_dict_path = "./data/%s/data_dict.pkl"%(FLAGS.dataset)
    
    if force_regen and os.path.exists(data_dict_path):
        os.remove(data_dict_path)
        print('Re-generate new encodings.')
        
    if os.path.exists(data_dict_path):
        with open(data_dict_path, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        wl_node_coloring = WLNodeColoring()
        graph_batching = GraphBatching()

        data_dict = {}
        for i, G in enumerate(graphs):
            wl_dict = wl_node_coloring.get_WLNodeColoring(G)
            batch_dict, hop_dict = graph_batching.graph_batching(G)
            data_dict[i] = [wl_dict, batch_dict, hop_dict]


        with open(data_dict_path, 'wb') as f:
            pickle.dump(data_dict_path, f)
    
    return data_dict

##########################################################################
##########################################################################
##########################################################################

def get_encodings(features, wl_dict, batch_dict, hop_dict, device):

    num_nodes = features.shape[0]

    raw_feature_list = []
    role_ids_list = []
    position_ids_list = []
    hop_ids_list = []

    for node in range(num_nodes):
        neighbors_list = batch_dict[node]

        raw_feature = [features[node, :]]
        role_ids = [wl_dict[node]]
        position_ids = range(len(neighbors_list) + 1)
        hop_ids = [0]
        for neighbor, intimacy_score in neighbors_list:
            raw_feature.append(features[neighbor])
            role_ids.append(wl_dict[neighbor])
            if neighbor in hop_dict[node]:
                hop_ids.append(hop_dict[node][neighbor])
            else:
                hop_ids.append(99)
        raw_feature_list.append(raw_feature)
        role_ids_list.append(role_ids)
        position_ids_list.append(position_ids)
        hop_ids_list.append(hop_ids)
        
    raw_embeddings = torch.FloatTensor(raw_feature_list).to(device)
    wl_embedding   = torch.LongTensor(role_ids_list).to(device)
    hop_embeddings = torch.LongTensor(hop_ids_list).to(device)
    int_embeddings = torch.LongTensor(position_ids_list).to(device)
    
    return raw_embeddings, wl_embedding, hop_embeddings, int_embeddings