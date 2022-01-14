from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

import numpy as np
import networkx as nx

import torch


def get_normalized_eigenvalues(graphs, num_eigen=20, which='LM'):
    spectrums = []
    activity_vecs = []  #eigenvector of the largest eigenvalue

    max_num_nodes = len(graphs[-1].nodes)
    for G in graphs:
        if len(G.nodes) < max_num_nodes:
            for i in range(len(G.nodes), max_num_nodes):
                G.add_node(-1 * i) #add empty node with no connectivity (zero padding)

        if nx.is_directed(G):
            L = nx.directed_laplacian_matrix(G)
        else:
            L = nx.laplacian_matrix(G)
            L = L.asfptype()

        u, s, v = svds(L, k=num_eigen, which=which)
        spectrums.append(np.asarray(s))
    
    spectrums = np.asarray(spectrums).real
    spectrums = spectrums.reshape((len(graphs),-1))
    spectrums= normalize(spectrums, norm='l2')

    return spectrums
    

def prepare_train_spectrums(temporal_spectrums, time_step, device):
    spectrums_list = []

    for t in range(time_step-1):
        spectrums = np.concatenate([temporal_spectrums[t, :], temporal_spectrums[time_step-1, :]])
        spectrums_list.append(spectrums)

    spectrums_list = np.array(spectrums_list)
    spectrums_list = torch.from_numpy(spectrums_list).float()
    return spectrums_list.to(device)
