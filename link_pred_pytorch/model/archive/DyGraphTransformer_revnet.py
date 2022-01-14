import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor

from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_sparse import SparseTensor

from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

from .DyGraphTransformer_sparse import MultiHeadAttention, FeedForwardNetwork
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
        
        self.node_embedding      = nn.Embedding(self.node_encode_num*self.window_size, self.num_hids, max_norm=True)
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
        
        k_hop_row, k_hop_col = torch.where(edge_dist_encodes[:, :, 0] <= 2)  # window attention
        
        disjoint_row, disjoint_col = torch.where(edge_dist_encodes[:, :, 0] > 2)
        
        select = disjoint_row > disjoint_col
        disjoint_row = disjoint_row[select]
        disjoint_col = disjoint_col[select]
        
        disjoint_select = torch.randperm(disjoint_row.size(0))[:2*k_hop_row.size(0)].to(device) # sparse attention
        disjoint_row = disjoint_row[disjoint_select]
        disjoint_col = disjoint_col[disjoint_select]
        
        sparse_row = torch.cat([k_hop_row, disjoint_row, disjoint_col])
        sparse_col = torch.cat([k_hop_col, disjoint_col, disjoint_row])
        
        k_hop_edge_encodes = edge_encodes[sparse_row, sparse_row, :]
        k_hop_edge_dists   = edge_dist_encodes[sparse_row, sparse_row, 0]
        
        x = self.node_feats_fc(x) 
        
        node_temp_encode = torch.mean(self.node_embedding(node_encodes), axis=-2)
        
        edge_attn_bias_1 = torch.mean(self.edge_embedding(k_hop_edge_encodes), axis=-2)
        edge_attn_bias_1 = self.edge_encode_fc(edge_attn_bias_1)
        
        edge_attn_bias_2 = self.edge_dist_embedding(k_hop_edge_dists)
        edge_attn_bias_2 = self.edge_dist_encode_num_fc(edge_attn_bias_2)
        
        edge_attn_bias = edge_attn_bias_1 + edge_attn_bias_2
        
        
        x = x + node_temp_encode
        x = _ReversibleFunction.apply(x, edge_attn_bias, sparse_row, sparse_col, self.transformer_layer_list)
        
        return x.squeeze(0)

##########################################################################
##########################################################################
##########################################################################


class EncoderLayer_F(nn.Module):
    def __init__(self, hidden_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer_F, self).__init__()
    
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, attn_bias, sparse_row, sparse_col):
        x = self.self_attention_norm(x)
        x = self.self_attention(x, x, x, attn_bias, sparse_row, sparse_col)
        x = self.self_attention_dropout(x)
        return x
    
class Deterministic_F(nn.Module):
    def __init__(self, net):
        super(Deterministic_F, self).__init__()
        
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, x):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(x)

    def forward(self, x, attn_bias, sparse_row, sparse_col,
                record_rng = False, set_rng = False):
        
        if record_rng:
            self.record_rng(x)

        if not set_rng:
            return self.net(x, attn_bias, sparse_row, sparse_col)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(x, attn_bias, sparse_row, sparse_col)
        
##########################################################################
##########################################################################
##########################################################################

class EncoderLayer_G(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(EncoderLayer_G, self).__init__()
    
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, hidden_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        return x
    
class Deterministic_G(nn.Module):
    def __init__(self, net):
        super(Deterministic_G, self).__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, x):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(x)

    def forward(self, x, record_rng = False, set_rng = False):
        
        if record_rng:
            self.record_rng(x)

        if not set_rng:
            return self.net(x)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(x)
        
##########################################################################
##########################################################################
##########################################################################
        
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()
        
        assert hidden_size % 2 == 0
        
        self.f = Deterministic_F(EncoderLayer_F(int(hidden_size/2), dropout_rate, attention_dropout_rate, head_size))
        self.g = Deterministic_G(EncoderLayer_G(int(hidden_size/2), dropout_rate))

    def forward(self, x, attn_bias, sparse_row, sparse_col):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, attn_bias, sparse_row, sparse_col, record_rng=self.training)
            y2 = x2 + self.g(y1, record_rng=self.training)

        return torch.cat([y1, y2], dim=-1)

    def backward_pass(self, y, dy, attn_bias, sparse_row, sparse_col):
        y1, y2 = torch.chunk(y, 2, dim=-1)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=-1)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, attn_bias, sparse_row, sparse_col, set_rng=True)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=-1)
            dx = torch.cat([dx1, dx2], dim=-1)

        return x, dx

##########################################################################
##########################################################################
##########################################################################

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, attn_bias, sparse_row, sparse_col, blocks):
        ctx.attn_bias  = attn_bias
        ctx.sparse_row = sparse_row
        ctx.sparse_col = sparse_col
        for block in blocks:
            x = block(x, attn_bias, sparse_row, sparse_col)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y          = ctx.y
        attn_bias  = ctx.attn_bias
        sparse_row = ctx.sparse_row
        sparse_col = ctx.sparse_col
        
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, attn_bias, sparse_row, sparse_col)
        return dy, None, None, None, None
    
##########################################################################
##########################################################################
##########################################################################


    
