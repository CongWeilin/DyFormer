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
        node_encode_num, edge_encode_num, edge_dist_encode_num, window_size, # 2*FLAGS.cur_window, 2*FLAGS.cur_window, FLAGS.max_dist+2
    ):
        super(DyGraphTransformer, self).__init__()
        self.num_features   = num_features
        self.num_heads      = num_heads
        self.num_hids       = num_hids
        self.num_layers     = num_layers
        
        self.attn_drop      = attn_drop
        self.feat_drop      = feat_drop
        
        self.node_encode_num      = node_encode_num
        self.edge_encode_num      = edge_encode_num
        self.edge_dist_encode_num = edge_dist_encode_num
        self.window_size          = window_size
        
        # self.node_embedding      = nn.Embedding(self.node_encode_num*self.window_size, self.num_hids, max_norm=True)
        self.edge_embedding      = nn.Embedding(self.edge_encode_num*self.window_size, self.num_hids, max_norm=True)
        self.edge_dist_embedding = nn.Embedding(self.edge_dist_encode_num, self.num_hids, max_norm=True)
        
        self.transformer_layer_list   = nn.ModuleList()
        for ell in range(self.num_layers):

            block = EncoderLayer(self.num_hids, self.feat_drop, self.attn_drop, self.num_heads)
            self.transformer_layer_list.append(block)
            
        self.node_feats_fc           = nn.Linear(self.num_features, self.num_hids)
        self.edge_encode_fc          = nn.Linear(self.num_hids, self.num_heads)
        self.edge_dist_encode_num_fc = nn.Linear(self.num_hids, self.num_heads)
                
    def forward(self, x, node_encodes, edge_encodes, edge_dist_encodes, 
                target_node_size, context_nodes_size, device):
        
        x = self.node_feats_fc(x) 
        x = x.unsqueeze(0)
        
        # node_temp_encode = torch.mean(self.node_embedding(node_encodes), axis=-2)
        # node_temp_encode = node_temp_encode.unsqueeze(0)
        
        edge_attn_bias_1 = torch.mean(self.edge_embedding(edge_encodes), axis=-2)
        edge_attn_bias_1 = self.edge_encode_fc(edge_attn_bias_1)
        
        # edge_attn_bias_2 = torch.mean(self.edge_dist_embedding(edge_dist_encodes), axis=-2)
        edge_attn_bias_2 = self.edge_dist_embedding(edge_dist_encodes[:, :, 0])
        edge_attn_bias_2 = self.edge_dist_encode_num_fc(edge_attn_bias_2)
        
        edge_attn_bias = edge_attn_bias_1 + edge_attn_bias_2
        edge_attn_bias = edge_attn_bias.unsqueeze(0)
        edge_attn_bias = edge_attn_bias.permute(0, 3, 1, 2)
                
        # x = x + node_temp_encode
        for ell in range(self.num_layers):
            x = self.transformer_layer_list[ell](x, edge_attn_bias)
        return x.squeeze(0)

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

    def forward(self, q, k, v, attn_bias=None):
        
        ##########################
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)
        
        assert x.size() == orig_q_size
        return x

##########################################################################
##########################################################################
##########################################################################

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, hidden_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


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
        x = self.dropout(x)
        x = self.layer2(x)
        # x = self.dropout(x)
        return x
    
