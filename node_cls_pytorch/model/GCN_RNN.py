import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch import Tensor
from torch_geometric.nn.inits import glorot

##########################################################################
##########################################################################
##########################################################################

class GCN_RNN(nn.Module):
    def __init__(self, n_feats, n_hid, n_layers, dropout, RNN_type='LSTM'):
        super(GCN_RNN, self).__init__()
        self.n_layers = n_layers
        self.n_hid = n_hid
        self.dropout = dropout
        
        # spatial model
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConv(n_feats,  n_hid))
        for _ in range(n_layers-1):
            self.gcs.append(GraphConv(n_hid,  n_hid))
        
        # temporal model
        if RNN_type=='LSTM':
            self.rnn = nn.LSTM(input_size=n_hid, hidden_size=n_hid, num_layers=1)
        elif RNN_type=='GRU':
            self.rnn = nn.GRU(input_size=n_hid, hidden_size=n_hid, num_layers=1)
        
    def forward_spatial(self, x, adj, ell):
        x = self.gcs[ell](x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        
    def forward(self, x_list, adj_list, device):
        total_steps = len(adj_list)
        temporal_seq = []
        
        for t in range(total_steps):
            h, adj = x_list[t], adj_list[t]
            for ell in range(self.n_layers):
                h = self.forward_spatial(h, adj, ell)
            temporal_seq.append(h)
            
        # pad pack structural output embeddings across snapshots; then stack
        for t in range(total_steps):
            zero_padding = torch.zeros(
                size=(temporal_seq[-1].shape[0] - temporal_seq[t].shape[0], 
                      temporal_seq[t].shape[1]),  # this is for N_T - N_t compensation
            ).to(temporal_seq[t])
            temporal_seq[t] = torch.cat((temporal_seq[t], zero_padding), 0)
        temporal_seq = torch.stack(temporal_seq)
        
        output, _ = self.rnn(temporal_seq, None)
        return output.permute(1, 0, 2)
    
##########################################################################
##########################################################################
##########################################################################

class GCN_RNN_v2(nn.Module):
    def __init__(self, n_feats, n_hid, n_layers, dropout, RNN_type='LSTM'):
        super(GCN_RNN_v2, self).__init__()
        self.n_layers = n_layers
        self.n_hid = n_hid
        self.dropout = dropout
        
        # spatial model
        self.gcs = nn.ModuleList()
        
        self.gcs.append(GraphConv(n_feats,  n_hid))
        for _ in range(n_layers-1):
            self.gcs.append(GraphConv(n_hid,  n_hid))
        
        # temporal model
        self.rnn = nn.ModuleList()
        
        for _ in range(n_layers):            
            if RNN_type=='LSTM':
                self.rnn.append(nn.LSTM(input_size=n_hid, hidden_size=n_hid, num_layers=1))
            elif RNN_type=='GRU':
                self.rnn.append(nn.GRU(input_size=n_hid, hidden_size=n_hid, num_layers=1))
        
    def forward_spatial(self, x, adj, ell):
        x = self.gcs[ell](x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        
    def forward(self, x_list, adj_list, device):
        total_steps = len(adj_list)
        num_nodes = [x.size(0) for x in x_list]
        
        for ell in range(self.n_layers):
            temporal_seq = []
            for t in range(total_steps):
                if ell == 0:
                    h, adj = x_list[t], adj_list[t]
                else:
                    h, adj = output[t][:num_nodes[t]], adj_list[t]
                h = self.forward_spatial(h, adj, ell)
                temporal_seq.append(h)
                
            # pad pack structural output embeddings across snapshots; then stack
            for t in range(total_steps):
                zero_padding = torch.zeros(
                    size=(temporal_seq[-1].shape[0] - temporal_seq[t].shape[0], 
                          temporal_seq[t].shape[1]),  # this is for N_T - N_t compensation
                ).to(temporal_seq[t])
                temporal_seq[t] = torch.cat((temporal_seq[t], zero_padding), 0)
            
            temporal_seq = torch.stack(temporal_seq)
            output, _ = self.rnn[ell](temporal_seq, None)
            
        return output.permute(1, 0, 2)
    
##########################################################################
##########################################################################
##########################################################################

class GraphConv(nn.Module):
    def __init__(self, n_in, n_out):
        super(GraphConv, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.w = nn.Parameter(torch.Tensor(n_in, n_out))        
        self.reset_parameters()
        
    def forward(self, x, adj):
        
        if isinstance(x, Tensor):
            h = torch.mm(x, self.w)      
        else:
            h = torch.sparse.mm(x, self.w)
            
        out = torch.sparse.mm(adj, h)
        return out
    
    def reset_parameters(self):
        glorot(self.w)
        
        