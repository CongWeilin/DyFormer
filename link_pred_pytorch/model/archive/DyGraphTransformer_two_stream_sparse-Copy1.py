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

class DyGraphTransformer(nn.Module):
    def __init__(
        self, 
        num_features,        ## original feature vector dim: D
        num_heads,           ## number of attention heads for Transformer: H_s
        num_hids,            ## number of hidden units for Transformer: also as output embedding for Transformer: F_s
        num_layers,
        attn_drop,           ## dropout % for attn layer
        feat_drop,
        edge_encode_num, edge_dist_encode_num, window_size, 
        use_unsupervised_loss,
        neighbor_sampling_size
    ):
        super(DyGraphTransformer, self).__init__()
        self.num_features   = num_features
        self.num_heads      = num_heads
        self.num_hids       = num_hids
        self.num_layers     = num_layers
        
        self.attn_drop      = attn_drop
        self.feat_drop      = feat_drop
        
        self.edge_encode_num      = edge_encode_num
        self.edge_dist_encode_num = edge_dist_encode_num
        self.window_size          = window_size
        
        self.use_unsupervised_loss = use_unsupervised_loss
        self.neighbor_sampling_size = neighbor_sampling_size
        
        self.edge_embedding      = nn.Embedding(self.edge_encode_num*self.window_size, self.num_hids, max_norm=True)
        self.edge_dist_embedding = nn.Embedding(self.edge_dist_encode_num, self.num_hids, max_norm=True)
        
        self.transformer_layer_list   = nn.ModuleList()
        for ell in range(self.num_layers):

            block = EncoderLayer(self.num_hids, self.feat_drop, self.attn_drop, self.num_heads)
            self.transformer_layer_list.append(block)
            
        self.node_feats_fc           = nn.Linear(self.num_features, self.num_hids)
        self.edge_encode_fc          = nn.Linear(self.num_hids, self.num_heads)
        self.edge_dist_encode_num_fc = nn.Linear(self.num_hids, self.num_heads)
        
        if self.use_unsupervised_loss:
            self.decoder_heads = nn.ModuleList()
            for t in range(self.window_size):
                self.decoder_heads.append(nn.Linear(self.num_hids, self.num_hids))
                
    def get_encodings(self, edge_encodes, edge_dist_encodes, target_node_size, 
                      context_sparse_row, context_sparse_col,
                      target_sparse_row, target_sparse_col):
        ### attn_bias_context
        context_edge_encodes = edge_encodes[target_node_size:, :target_node_size, :]
        attn_bias_1_context = torch.mean(self.edge_embedding(context_edge_encodes[context_sparse_row, context_sparse_col, :]), axis=-2)
        attn_bias_1_context = self.edge_encode_fc(attn_bias_1_context)

        context_edge_dist_encodes = edge_dist_encodes[target_node_size:, :target_node_size]
        attn_bias_2_context = self.edge_dist_embedding(context_edge_dist_encodes[context_sparse_row, context_sparse_col])
        attn_bias_2_context = self.edge_dist_encode_num_fc(attn_bias_2_context)

        attn_bias_context = attn_bias_1_context + attn_bias_2_context

        ### attn_bias_target
        target_edge_encodes = edge_encodes[:target_node_size, target_node_size:, :]
        attn_bias_1_target = torch.mean(self.edge_embedding(target_edge_encodes[target_sparse_row, target_sparse_col, :]), axis=-2)
        attn_bias_1_target = self.edge_encode_fc(attn_bias_1_target)

        target_edge_dist_encodes = edge_dist_encodes[:target_node_size:, target_node_size:]
        attn_bias_2_target = self.edge_dist_embedding(target_edge_dist_encodes[target_sparse_row, target_sparse_col])
        attn_bias_2_target = self.edge_dist_encode_num_fc(attn_bias_2_target)

        attn_bias_target = attn_bias_1_target + attn_bias_2_target
        
        return attn_bias_context, attn_bias_target
    
    @torch.no_grad()
    def get_encodings_using_cached_memory(self, edge_encodes, edge_dist_encodes, target_node_size, 
                      context_sparse_row, context_sparse_col,
                      target_sparse_row, target_sparse_col, device):
        
        #### step 1.0: attn_bias_context
        num_attn_compute        = context_sparse_row.size(0)
        attn_compute_ind        = torch.arange(num_attn_compute).to(device)
        attn_compute_ind_splits = torch.split(attn_compute_ind, 4096)
        
        context_edge_encodes = edge_encodes[target_node_size:, :target_node_size, :]
        context_edge_dist_encodes = edge_dist_encodes[target_node_size:, :target_node_size]
        
        attn_bias_1_context_list, attn_bias_2_context_list = [], []
        for cur_attn_compute_ind in attn_compute_ind_splits:
            cur_context_sparse_row = context_sparse_row[cur_attn_compute_ind]
            cur_context_sparse_col = context_sparse_col[cur_attn_compute_ind]
            
            attn_bias_1_context = torch.mean(self.edge_embedding(context_edge_encodes[cur_context_sparse_row, cur_context_sparse_col, :]), axis=-2)
            attn_bias_1_context_list.append(self.edge_encode_fc(attn_bias_1_context))
            
            attn_bias_2_context = self.edge_dist_embedding(context_edge_dist_encodes[cur_context_sparse_row, cur_context_sparse_col])
            attn_bias_2_context_list.append(self.edge_dist_encode_num_fc(attn_bias_2_context))
            
        attn_bias_1_context = torch.cat(attn_bias_1_context_list, dim=0)
        attn_bias_2_context = torch.cat(attn_bias_2_context_list, dim=0)
        del attn_bias_1_context_list, attn_bias_2_context_list
        
        attn_bias_context = attn_bias_1_context + attn_bias_2_context

        #### step 2.0: attn_bias_target
        num_attn_compute        = target_sparse_row.size(0)
        attn_compute_ind        = torch.arange(num_attn_compute).to(device)
        attn_compute_ind_splits = torch.split(attn_compute_ind, 4096)
        
        target_edge_encodes = edge_encodes[:target_node_size, target_node_size:, :]
        target_edge_dist_encodes = edge_dist_encodes[:target_node_size:, target_node_size:]
        
        attn_bias_1_target_list, attn_bias_2_target_list = [], []
        for cur_attn_compute_ind in attn_compute_ind_splits:
            cur_target_sparse_row = target_sparse_row[cur_attn_compute_ind]
            cur_target_sparse_col = target_sparse_col[cur_attn_compute_ind]
            
            attn_bias_1_target = torch.mean(self.edge_embedding(target_edge_encodes[cur_target_sparse_row, cur_target_sparse_col, :]), axis=-2)
            attn_bias_1_target_list.append(self.edge_encode_fc(attn_bias_1_target))
            
            attn_bias_2_target = self.edge_dist_embedding(target_edge_dist_encodes[cur_target_sparse_row, cur_target_sparse_col])
            attn_bias_2_target_list.append(self.edge_dist_encode_num_fc(attn_bias_2_target))
            
        attn_bias_1_target = torch.cat(attn_bias_1_target_list, dim=0)
        attn_bias_2_target = torch.cat(attn_bias_2_target_list, dim=0)
        del attn_bias_1_target_list, attn_bias_2_target_list
        attn_bias_target = attn_bias_1_target + attn_bias_2_target
        
        return attn_bias_context, attn_bias_target
        
        
    def forward(self, x, edge_encodes, edge_dist_encodes, 
                target_node_size, context_node_size, device):
        
        # for each node, random sample some neighbors
        target_sparse_row, target_sparse_col = [], []
        context_sparse_row, context_sparse_col = [], []
        
        if self.training:
            context_sample_size = math.ceil(context_node_size*self.neighbor_sampling_size)
            target_sample_size  = math.ceil(target_node_size *self.neighbor_sampling_size)
        
            for i in range(target_node_size):
                target_sparse_row.append(torch.ones(context_sample_size)*i)
                target_sparse_col.append(torch.randperm(context_node_size)[:context_sample_size])


            for i in range(context_node_size):
                context_sparse_row.append(torch.ones(target_sample_size)*i)
                context_sparse_col.append(torch.randperm(target_node_size)[:target_sample_size])
        else:
            for i in range(target_node_size):
                target_sparse_row.append(torch.ones(context_node_size)*i)
                target_sparse_col.append(torch.arange(context_node_size))


            for i in range(context_node_size):
                context_sparse_row.append(torch.ones(target_node_size)*i)
                context_sparse_col.append(torch.arange(target_node_size))
            
        target_sparse_row = torch.cat(target_sparse_row).long().to(device)
        target_sparse_col = torch.cat(target_sparse_col).long().to(device)        
        context_sparse_row = torch.cat(context_sparse_row).long().to(device)
        context_sparse_col = torch.cat(context_sparse_col).long().to(device)
        
        #######################################################
        #######################################################
        #######################################################
        x = self.node_feats_fc(x) 
        
        if self.training:
            attn_bias_context, attn_bias_target = self.get_encodings(edge_encodes, edge_dist_encodes, target_node_size, 
                                                                     context_sparse_row, context_sparse_col,
                                                                     target_sparse_row, target_sparse_col)
        else:
            attn_bias_context, attn_bias_target = self.get_encodings_using_cached_memory(edge_encodes, edge_dist_encodes, target_node_size, 
                                                                     context_sparse_row, context_sparse_col,
                                                                     target_sparse_row, target_sparse_col, device)
            
        # x = x + node_temp_encode
        for ell in range(self.num_layers):
            x = self.transformer_layer_list[ell](x, target_node_size, context_node_size, 
                                                 attn_bias_target, attn_bias_context, 
                                                 context_sparse_row, context_sparse_col, target_sparse_row, target_sparse_col, device)
        x = x.squeeze(0)

        if self.use_unsupervised_loss and self.training:
            temporal_output = []
            for decoder_head in self.decoder_heads:
                temporal_output.append(decoder_head(x))
            x_unsupervised = torch.stack(temporal_output)
            return x, x_unsupervised
        else:
            return x, None

##########################################################################
##########################################################################
##########################################################################

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention_1 = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_2 = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, hidden_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x_all, target_node_size, context_node_size, 
                attn_bias_target, attn_bias_context,
                context_sparse_row, context_sparse_col,
                target_sparse_row, target_sparse_col, device):
        
        y_all = self.self_attention_norm(x_all)

        y_target, y_context = torch.split(y_all, [target_node_size, context_node_size], dim=0)
        
        if self.training:
            y_context_ = self.self_attention_1(q=y_target, k=y_context, v=y_context, 
                                               attn_bias=attn_bias_target, 
                                               sparse_row = target_sparse_row, 
                                               sparse_col = target_sparse_col, 
                                               attn_shape = (target_node_size, context_node_size))  # attn_bias_target  = [target_size,  context_size]

            y_target_  = self.self_attention_2(q=y_context, k=y_target,  v=y_target,  
                                               attn_bias=attn_bias_context, 
                                               sparse_row = context_sparse_row, 
                                               sparse_col = context_sparse_col,
                                               attn_shape = (context_node_size, target_node_size)) # attn_bias_context = [context_size, context_size]
        else:
            y_context_ = self.self_attention_1.forward_using_cached_memory(
                                               q=y_target, k=y_context, v=y_context, 
                                               attn_bias  = attn_bias_target, 
                                               sparse_row = target_sparse_row, 
                                               sparse_col = target_sparse_col, 
                                               attn_shape = (target_node_size, context_node_size), device = device)  # attn_bias_target  = [target_size,  context_size]

            y_target_  = self.self_attention_2.forward_using_cached_memory(
                                               q=y_context, k=y_target,  v=y_target,  
                                               attn_bias  = attn_bias_context, 
                                               sparse_row = context_sparse_row, 
                                               sparse_col = context_sparse_col,
                                               attn_shape = (context_node_size, target_node_size), device = device) # attn_bias_context = [context_size, context_size]
            
        y_all = torch.cat([y_target_, y_context_], dim=0)
        
        y_all = self.self_attention_dropout(y_all)
        x_all = x_all + y_all

        y_all = self.ffn_norm(x_all)
        y_all = self.ffn(y_all)
        y_all = self.ffn_dropout(y_all)
        x_all = x_all + y_all
        return x_all

##########################################################################
##########################################################################
##########################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias, sparse_row, sparse_col, attn_shape):
        
        ##########################

        d_k = self.att_size
        d_v = self.att_size
        
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(-1, self.head_size, d_k)
        k = self.linear_k(k).view(-1, self.head_size, d_k)
        v = self.linear_v(v).view(-1, self.head_size, d_v)
                
        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        x = torch.sum(q[sparse_row, :, :] * k[sparse_col, :, :], dim=-1) * self.scale  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = softmax(values=x, indices=sparse_row)
        x = self.att_dropout(x)
        
        x = x.view(-1, self.head_size, 1)
        x = scatter(x * v[sparse_col, :, :], sparse_row, reduce='sum', dim=0)
        x = x.reshape(-1, self.head_size * d_v)
        
        # borrow implementation from DySAT, seems like it waste more memory and takes more time
#         x = torch.exp(x).view(-1, self.head_size)
#         edge_inds = torch.stack([sparse_row, sparse_col])
#         x_head_outputs = []
#         for head in range(self.head_size):
#             x_sparse = torch.sparse.FloatTensor(edge_inds, x[:, head], attn_shape)  
#             x_sparse_rowsum = torch.sparse.sum(x_sparse, dim=1).to_dense().unsqueeze(1)
#             x_sparse = torch.sparse.FloatTensor(edge_inds, self.att_dropout(x[:, head]), attn_shape)  
#             x_ = torch.sparse.mm(x_sparse, v[:, head, :]).div(x_sparse_rowsum)
#             x_head_outputs.append(x_)
#         x = torch.cat(x_head_outputs, dim=-1)
        
        
        
        # faster version
        
        
        x = self.output_layer(x)
        
        return x
    
    # used for large-scale inference
    @torch.no_grad()
    def forward_using_cached_memory(self, q, k, v, attn_bias, sparse_row, sparse_col, attn_shape, device):
        ##########################
        d_k = self.att_size
        d_v = self.att_size
        ##########################
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(-1, self.head_size, d_k)
        k = self.linear_k(k).view(-1, self.head_size, d_k)
        v = self.linear_v(v).view(-1, self.head_size, d_v)
        
        ##########################
        
        num_attn_compute        = sparse_row.size(0)
        
        attn_compute_ind        = torch.arange(num_attn_compute).to(device)
        attn_compute_ind_splits = torch.split(attn_compute_ind, 1024)
        
        x = []
        for cur_attn_compute_ind in attn_compute_ind_splits:
            # Scaled Dot-Product Attention.
            # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
            cur_sparse_row = sparse_row[cur_attn_compute_ind]
            cur_sparse_col = sparse_col[cur_attn_compute_ind]
            x.append(attn_bias[cur_attn_compute_ind, :] + torch.sum(q[cur_sparse_row, :, :] * k[cur_sparse_col, :, :], dim=-1) * self.scale)
        x = torch.cat(x, dim=0)
        
        x = softmax(values=x, indices=sparse_row)
        x = self.att_dropout(x)
        
        # borrow implementation from DySAT, seems like it waste more memory and takes more time
        x = x.view(-1, self.head_size)
        edge_inds = torch.stack([sparse_row, sparse_col])
        
        x_head_outputs = []
        for head in range(self.head_size):
            x_, src_mask = prune_weak_attn(values=x[:, head], indices=sparse_row, thresh=5e-4)
            x_sparse = torch.sparse.FloatTensor(edge_inds[:, src_mask], x_, attn_shape) 
            # x_sparse = torch.sparse.FloatTensor(edge_inds, x[:, head], attn_shape)     
            x_ = torch.sparse.mm(x_sparse, v[:, head, :]) 
            x_head_outputs.append(x_)
        x = torch.cat(x_head_outputs, dim=-1)
        
        x = self.output_layer(x)
        
        return x
        
##########################################################################
##########################################################################
##########################################################################

def softmax(values, indices):

    src_max = scatter(values, indices, reduce='max', dim=0)
    out = (values-src_max[indices, :]).exp()
    out_sum = scatter(out, indices, reduce='sum', dim=0) 

    return out / (out_sum[indices] + 1e-16)

def average(values, indices):
    values = values.flatten()
    indices = indices.flatten()
    values_sum = scatter(values, indices, reduce='sum') 
    return values / (values_sum[indices] + 1e-16)

def prune_weak_attn(values, indices, thresh): 
    # the smaller thresh, the more nodes left
    values = values.flatten()
    indices = indices.flatten()
    
    src_mean = scatter(values, indices, reduce='mean')
    src_mean = src_mean[indices]
    
    src_mask = torch.zeros_like(values).bool()
    src_mask[values > thresh * src_mean] = True
    
    return average(values[src_mask], indices[src_mask]), src_mask

##########################################################################
##########################################################################
##########################################################################

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        # x = self.dropout(x)
        x = self.layer2(x)
        # x = self.dropout(x)
        return x
    
