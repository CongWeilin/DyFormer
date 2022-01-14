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
        node_encode_num, edge_encode_num, edge_dist_encode_num, window_size, memory_size, # 2*FLAGS.cur_window, 2*FLAGS.cur_window, FLAGS.max_dist+2
        use_memory_net
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
        self.memory_size          = memory_size
        self.use_memory_net       = use_memory_net
        
        self.node_embedding      = nn.Embedding(self.node_encode_num*self.window_size, self.num_hids, max_norm=True)
        self.edge_embedding      = nn.Embedding(self.edge_encode_num*self.window_size, self.num_hids, max_norm=True)
        self.edge_dist_embedding = nn.Embedding(self.edge_dist_encode_num, self.num_hids, max_norm=True)
        
        self.transformer_layer_list   = nn.ModuleList()
        for ell in range(self.num_layers):

            block = EncoderLayer(self.num_hids, self.feat_drop, self.attn_drop, self.num_heads, self.memory_size, self.use_memory_net)
            self.transformer_layer_list.append(block)
            
        self.node_feats_fc           = nn.Linear(self.num_features, self.num_hids)
        self.edge_encode_fc          = nn.Linear(self.num_hids, self.num_heads)
        self.edge_dist_encode_num_fc = nn.Linear(self.num_hids, self.num_heads)
                
    def forward(self, x, node_encodes, edge_encodes, edge_dist_encodes, 
                target_node_size, context_nodes_size, device):
        
        x = self.node_feats_fc(x) 
        x = x.unsqueeze(0)
        
        node_temp_encode = torch.mean(self.node_embedding(node_encodes), axis=-2)
        node_temp_encode = node_temp_encode.unsqueeze(0)
        
        edge_attn_bias_1 = torch.mean(self.edge_embedding(edge_encodes), axis=-2)
        edge_attn_bias_1 = self.edge_encode_fc(edge_attn_bias_1)
        
        edge_attn_bias_2 = self.edge_dist_embedding(edge_dist_encodes[:, :, 0])
        edge_attn_bias_2 = self.edge_dist_encode_num_fc(edge_attn_bias_2)
        
        edge_attn_bias = edge_attn_bias_1 + edge_attn_bias_2
        edge_attn_bias = edge_attn_bias.unsqueeze(0)
        edge_attn_bias = edge_attn_bias.permute(0, 3, 1, 2)
        
        x = x + node_temp_encode
        target_node_size = x.size(1) - edge_attn_bias.size(2)
        context_node_size = edge_attn_bias.size(2)
        
        if target_node_size == 0:
            target_x, context_x = None, x
        else:
            target_x, context_x = torch.split(x, (target_node_size, context_node_size), dim=1)
            
        for ell in range(self.num_layers):
            target_x, context_x = self.transformer_layer_list[ell](target_x  = target_x, 
                                                                   context_x = context_x,
                                                                   attn_bias = edge_attn_bias)
            
        if target_node_size == 0:
            return context_x.squeeze(0)
        else:
            return target_x.squeeze(0), context_x.squeeze(0)

##########################################################################
##########################################################################
##########################################################################

class MemoryLowRankAttention(nn.Module):
    def __init__(self, hidden_size, head_size, memory_size, attention_dropout_rate, use_memory_net):
        super(MemoryLowRankAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head_size   = head_size
        self.memory_size = memory_size
        self.use_memory_net = use_memory_net
        
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer  = nn.Linear(head_size * att_size, hidden_size)
        
        self.memory = nn.Parameter(torch.Tensor(1, memory_size, hidden_size))
        self.memory_merge_attn = nn.Linear(hidden_size*2, 1)
        self.reset_parameters()
        
    def forward(self, target_x, context_x, attn_bias=None):
        
        ##########################
        d_k = self.att_size
        d_v = self.att_size
        batch_size = context_x.size(0) # batch_size=1
        assert batch_size == 1

        ########################## [Nxd] -> [Mxd]. head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        memory_norm = self.memory.norm(p=2, dim=1, keepdim=True) # similar to nn.Embedding with max_norm=1.0
        memory_norm[memory_norm<1.0] = 1.0
        self.memory.data = self.memory.div(memory_norm).data 

        q = self.linear_q(self.memory).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(context_x).view(batch_size, -1, self.head_size, d_k)
        v = k # v = self.linear_v(context_x).view(batch_size, -1, self.head_size, d_v)
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        
        # Scaled Dot-Product Attention. Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        att = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            edge_bias_proj = torch.matmul(att, attn_bias)
            att = att + edge_bias_proj
        att = torch.softmax(att, dim=3)
        att = self.att_dropout(att)
        
        memory_aug = att.matmul(v).contiguous()  # [b, h, q_len, attn]
        memory_aug = memory_aug.view(batch_size, -1, self.head_size * d_v)
        
        memory_merge_alpha = torch.sigmoid(self.memory_merge_attn(torch.cat([self.memory, memory_aug], dim=2))) # [b, n, d]
        
        if self.use_memory_net and self.training:
            self.memory = (1-memory_merge_alpha) * self.memory +  memory_merge_alpha * memory_aug
            augmented_memory = self.memory
        else:
            augmented_memory = (1-memory_merge_alpha) * self.memory +  memory_merge_alpha * memory_aug
        
        ########################## [Mxd] -> [Nxd]. head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        if target_x is None:
            x = context_x
        else:
            x = torch.cat([target_x, context_x], dim=1)
        q = self.linear_q(x).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(augmented_memory).view(batch_size, -1, self.head_size, d_k)
        v = k # v = self.linear_v(augmented_memory).view(batch_size, -1, self.head_size, d_v)
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        
        # Scaled Dot-Product Attention. Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        att = torch.matmul(q, k)  # [b, h, q_len, k_len]
        
        if attn_bias is not None:
            edge_bias_proj = torch.matmul(attn_bias, att)
            att = att + edge_bias_proj
        att = torch.softmax(att, dim=3)
        
        x_aug = att.matmul(v).contiguous()  # [b, h, q_len, attn]
        x_aug = x_aug.view(batch_size, -1, self.head_size * d_v)
        
        x_merge_alpha = torch.sigmoid(self.memory_merge_attn(torch.cat([x, x_aug], dim=2)))
        augmented_x = (1-x_merge_alpha) * x + x_merge_alpha * x_aug
        ##########################

        augmented_x = self.output_layer(augmented_x)
        if target_x is None:
            target_x, context_x = None, augmented_x
        else:
            target_x, context_x = torch.split(augmented_x, (target_x.size(1), context_x.size(1)), dim=1)
        ##########################
    
        return target_x, context_x
    
    def reset_parameters(self):
        glorot(self.memory)
##########################################################################
##########################################################################
##########################################################################

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout_rate, attention_dropout_rate, head_size, memory_size, use_memory_net):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MemoryLowRankAttention(hidden_size, head_size, memory_size, attention_dropout_rate, use_memory_net)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, hidden_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, target_x, context_x, attn_bias=None):
        if target_x is None:
            target_y = None
        else:
            target_y  = self.self_attention_norm(target_x)
        context_y = self.self_attention_norm(context_x)
        
        target_y, context_y = self.self_attention(target_y, context_y, attn_bias)
        
        if target_y is None:
            target_x = None
        else:
            target_y  = self.self_attention_dropout(target_y)
            target_x  = target_x + target_y
            target_y = self.ffn_norm(target_x)
            target_y = self.ffn(target_y)
            target_y = self.ffn_dropout(target_y)
            target_x = target_x + target_y
            
        context_y = self.self_attention_dropout(context_y)
        context_x = context_x + context_y
        context_y = self.ffn_norm(context_x)
        context_y = self.ffn(context_y)
        context_y = self.ffn_dropout(context_y)
        context_x = context_x + context_y
        
        return target_x, context_x


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
        x = self.dropout(x)
        return x
    
