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
from .DyGraphTransformer import FeedForwardNetwork
from .DyGraphTransformer import MultiHeadAttention

class DyGraphTransformer(nn.Module):
    def __init__(
        self, 
        num_features,        ## original feature vector dim: D
        num_heads,           ## number of attention heads for Transformer: H_s
        num_hids,            ## number of hidden units for Transformer: also as output embedding for Transformer: F_s
        num_layers,
        attn_drop,           ## dropout % for attn layer
        feat_drop,
        node_encode_num, edge_encode_num, edge_dist_encode_num, window_size# 2*FLAGS.cur_window, 2*FLAGS.cur_window, FLAGS.max_dist+2
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
                target_node_size, context_node_size, device):
        
        x = self.node_feats_fc(x) 
        x = x.unsqueeze(0)
        
        # node_temp_encode = torch.mean(self.node_embedding(node_encodes), axis=-2)
        # node_temp_encode = node_temp_encode.unsqueeze(0)
        
        #######################################################
        #######################################################
        #######################################################
        attn_bias_1_context = torch.mean(self.edge_embedding(edge_encodes[target_node_size:, :target_node_size, :]), axis=-2)
        attn_bias_1_context = self.edge_encode_fc(attn_bias_1_context)
        
        # attn_bias_2 = torch.mean(self.edge_dist_embedding(edge_dist_encodes), axis=-2)
        attn_bias_2_context = self.edge_dist_embedding(edge_dist_encodes[target_node_size:, :target_node_size, 0])
        attn_bias_2_context = self.edge_dist_encode_num_fc(attn_bias_2_context)
        
        attn_bias_context = attn_bias_1_context + attn_bias_2_context
        attn_bias_context = attn_bias_context.unsqueeze(0)
        attn_bias_context = attn_bias_context.permute(0, 3, 1, 2)
        
        #######################################################
        #######################################################
        #######################################################
        attn_bias_1_target = torch.mean(self.edge_embedding(edge_encodes[:target_node_size, target_node_size:, :]), axis=-2)
        attn_bias_1_target = self.edge_encode_fc(attn_bias_1_target)
        
        # attn_bias_2 = torch.mean(self.edge_dist_embedding(edge_dist_encodes), axis=-2)
        attn_bias_2_target = self.edge_dist_embedding(edge_dist_encodes[:target_node_size, target_node_size:, 0])
        attn_bias_2_target = self.edge_dist_encode_num_fc(attn_bias_2_target)
        
        attn_bias_target = attn_bias_1_target + attn_bias_2_target
        attn_bias_target = attn_bias_target.unsqueeze(0)
        attn_bias_target = attn_bias_target.permute(0, 3, 1, 2)
            
        # x = x + node_temp_encode
        for ell in range(self.num_layers):
            x = self.transformer_layer_list[ell](x, target_node_size, context_node_size, 
                                                 attn_bias_target, attn_bias_context)
        return x.squeeze(0)

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

    def forward(self, x_all, target_node_size, context_node_size, attn_bias_target, attn_bias_context):
        y_all = self.self_attention_norm(x_all)
        # print(target_node_size, context_node_size, target_node_size+context_node_size, y_all.shape)
        y_target, y_context = torch.split(y_all, [target_node_size, context_node_size], dim=1)

        y_context_ = self.self_attention_1(q=y_target,  k=y_context, v=y_context, attn_bias=attn_bias_target)  # attn_bias_target  = [target_size,  context_size]
        y_target_  = self.self_attention_2(q=y_context, k=y_target,  v=y_target,  attn_bias=attn_bias_context) # attn_bias_context = [context_size, context_size]
        y_all = torch.cat([y_target_, y_context_], dim=1)
        
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