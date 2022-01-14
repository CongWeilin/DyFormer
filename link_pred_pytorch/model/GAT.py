import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor

from torch_geometric.nn.inits import glorot

class GAT(nn.Module):
    def __init__(self, num_features, num_hiddens, num_layers, num_heads, spatial_drop, feat_drop, 
                use_residual, use_rnn):
        super(GAT, self).__init__()
        
        # hyper-parameters
        self.num_features   = num_features        
        self.num_hiddens    = num_hiddens
        self.num_layers     = num_layers
        self.num_heads      = num_heads
        self.spatial_drop   = spatial_drop
        self.feat_drop      = feat_drop

        self.use_residual   = use_residual
        self.use_rnn        = use_rnn

        if use_rnn:
            self.rnn = nn.GRU(input_size=num_hiddens, hidden_size=num_hiddens, num_layers=2)

        # model structure
        self.structural_attention_layer_list = nn.ModuleList()
        
        self.encoder = nn.Linear(num_features, num_hiddens)
        
        for i in range(num_layers):
            self.structural_attention_layer_list.append(
                StructuralAttentionLayer(n_heads=num_heads,
                                         input_dim=num_hiddens,
                                         output_dim=num_hiddens,
                                         attn_drop=spatial_drop)
            )

            
    def forward_single_timestep(self, x, adj, device):

        for ell in range(self.num_layers):
            out = self.structural_attention_layer_list[ell](x, adj)
            
            if ell != self.num_layers-1: # Don't apply dropout activation to last layer
                out = F.elu(out)
                out = F.dropout(out, self.feat_drop, training=self.training)

            if ell > 0 and self.use_residual:
                x = out + x
            else:
                x = out

        return x

    def forward(self, x, adj, device):
        x = [self.encoder(x_) for x_ in x]
        
        structural_att_outputs = [self.forward_single_timestep(x_, adj_, device) for x_, adj_ in zip(x, adj)]

        for t in range(len(structural_att_outputs)):
            zero_padding = torch.zeros(
                size=(structural_att_outputs[-1].shape[0] - structural_att_outputs[t].shape[0], 
                        structural_att_outputs[t].shape[1]),  # this is for N_T - N_t compensation
                dtype=torch.float32,
                device=device,
            )
            structural_att_outputs[t] = torch.cat((structural_att_outputs[t], zero_padding), 0)

        structure_outputs = torch.stack(structural_att_outputs)
        x = structure_outputs
        
        if self.use_rnn:
            x, _ = self.rnn(x, None)

        return x.permute(1, 0, 2)


###############################################################################
###############################################################################
###############################################################################

class StructuralAttentionLayer(nn.Module):
    """
    Structural Attention Block (multi-head)
    """
    def __init__(
        self, 
        n_heads,            # H
        input_dim,          # D
        output_dim,         # F
        attn_drop,          # normalized attention coef dropout
    ):
        super(StructuralAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attn_drop = attn_drop
        
        # construct parameters according to n_heads and input_dim, output_dim
        self.W_list = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.input_dim, self.output_dim // self.n_heads))
            for head in range(self.n_heads)  # [D, F]
        ])
        self.a_list = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, self.output_dim * 2 // self.n_heads)) 
            for head in range(self.n_heads)  # [1, 2F]
        ])         
        self.leakyRelu = nn.LeakyReLU(0.2)
        
        self._reset_parameters()
        
        
    def forward(
        self,
        x,                              # [N, D]: input node features
        adj_mat,                        # [N, N]: weighted adj matrix
    ):
        """
        forward function for GAT; applied to all graph snapshots
        return: [N, F]
        """
        out = []
        for head in range(self.n_heads):
            out.append(self._single_head_attn(x, adj_mat,
                                              w=self.W_list[head],
                                              a=self.a_list[head]))
        out = torch.cat(out, dim=1)      
        return out
        
    def _single_head_attn(self, x, adj_mat, w, a):
        if isinstance(x, Tensor):
            h = torch.mm(x, w)      
        else:
            h = torch.sparse.mm(x, w)
        
        # edge info extraction
        adj_mat = adj_mat.coalesce()
        edge_idxs = adj_mat._indices()                                                  # [2, E]
        edge_vals = adj_mat._values()                                                   # [E]
        edge_h = torch.cat((h[edge_idxs[0, :], :], 
                            h[edge_idxs[1, :], :]), dim=1).T     # [2F, E]
        edge_e = torch.exp(self.leakyRelu(edge_vals * a.mm(edge_h).squeeze())) 
        
        # prepare and compute rowsum for softmax 
        edge_e_sp = torch.sparse.FloatTensor(
            edge_idxs,
            edge_e,
            adj_mat.size()
        )                                                                               # logical [N, N]
        e_rowsum = torch.sparse.sum(edge_e_sp, dim=1).to_dense().unsqueeze(1)           # [N, 1]
        
        # attention dropout
        edge_e = F.dropout(edge_e, self.attn_drop, training=self.training)                   # [E]
        edge_e_sp = torch.sparse.FloatTensor(
            edge_idxs,
            edge_e,
            adj_mat.size()
        )

        # graph convolution
        h_prime = torch.sparse.mm(edge_e_sp, h)                                         # [N, F]
        
        # softmax divide
        h_prime = h_prime.div(e_rowsum)                                                 # [N, F]
        return h_prime
        
    def _reset_parameters(self, ):
        for head in range(self.n_heads):
            glorot(self.W_list[head])
            glorot(self.a_list[head])