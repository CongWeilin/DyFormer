import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor

from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter


##########################################################################
##########################################################################
##########################################################################

class DyTransformer_compact(nn.Module):
    def __init__(self, n_feats, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.2):
        super(DyTransformer_compact, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.n_feats   = n_feats
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.dropout   = nn.Dropout(dropout)
        
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(n_feats, n_hid))
            
        for l in range(n_layers - 1):
            self.gcs.append(DyTransformer_compact_layer(n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=True))
        self.gcs.append(DyTransformer_compact_layer(n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=False))

    def forward(self, node_feature, node_type, edge_index, edge_val, edge_type, device):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            node_emb = self.adapt_ws[t_id](node_feature[idx])
            res[idx] = torch.tanh(node_emb)
        res = self.dropout(res)
        for gc in self.gcs:
            res = gc(res, node_type, edge_index, edge_val, edge_type)
        return res 
    
##########################################################################
##########################################################################
##########################################################################

class DyTransformer_compact_layer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm=True):
        super(DyTransformer_compact_layer, self).__init__()
        
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
                        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            self.norms.append(nn.LayerNorm(out_dim))
                
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.dropout        = nn.Dropout(dropout)
        
        self.emb_temporal   = RelTemporalEncoding(in_dim, max_len=3)
        self.emb_spatial    = RelTemporalEncoding(in_dim, max_len=3)
         
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
    
    def forward(self, node_emb, node_type, edge_index, edge_val, edge_type):
        edge_index_dst, edge_index_src = edge_index[:, 0], edge_index[:, 1]
        node_emb_dst, node_emb_src = node_emb[edge_index_dst], node_emb[edge_index_src]
        node_type_dst, node_type_src = node_type[edge_index_dst], node_type[edge_index_src]
        # j: source, i: target; <j, i>
        data_size = edge_index.size(0)
        
        # Create Attention and Message tensor
        res_att = torch.zeros(data_size, self.n_heads).to(node_emb)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_emb)
        
        for source_type in range(self.num_types): # node type
            sb = (node_type_src == int(source_type))
            k_linear = self.k_linears[source_type] # select the weight according to source node type
            v_linear = self.v_linears[source_type] 
            
            for target_type in range(self.num_types):
                tb = (node_type_dst == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                
                for relation_type in range(self.num_relations): # idx is all the edges with meta relation <source_type, relation_type, target_type>
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                        
                    # Get the corresponding input node representations by idx.
                    target_node_vec = node_emb_dst[idx]
                    source_node_vec = node_emb_src[idx]
                    source_node_vec = source_node_vec + self.emb_temporal(source_node_vec, edge_val[idx, 0]) + self.emb_spatial(source_node_vec, edge_val[idx, 1])

                    # Apply attention
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    
                    # Message passing
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        

        att = softmax(res_att, edge_index_dst, dim=0)
        res_msg = res_msg * att.view(-1, self.n_heads, 1)
        res_msg = res_msg.view(-1, self.out_dim)
        
        aggr_out = scatter(res_msg, edge_index_dst, reduce='sum', dim=0, dim_size=node_emb.size(0))
        
        # Target-specific Aggregation. x = W[node_type] * Agg(x) + x
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_emb.device)
        
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue

            trans_out = self.dropout(self.a_linears[target_type](aggr_out[idx]))
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            alpha = torch.sigmoid(self.skip[target_type])
                        
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_emb[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_emb[idx] * (1 - alpha)
        return res
    
##########################################################################
##########################################################################
##########################################################################

class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 100, dropout = 0.2):
        self.max_len = max_len
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
        
    def forward(self, x, t):
        # print(x.shape, t.shape)
        return self.lin(self.emb(t))