import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor

from torch_geometric.nn.inits import glorot

##########################################################################
##########################################################################
##########################################################################

class DyTransformer(nn.Module):
    def __init__(
        self, 
        num_features,                                   ## original feature vector dim: D
        num_time_steps,                                 ## number of trainina snapshots + 1: T+1
        spatial_drop,                                   ## dropout % for structural layer
        temporal_drop,                                  ## dropout % for temporal layer
        num_structural_heads_list,                      ## number of attention heads for GAT: H_s
        num_structural_hids_list,                       ## number of hidden units for GAT: also as output embedding for GAT: F_s
        num_temporal_heads_list,                        ## number of attention heads for temporal block: H_t
        num_temporal_hids_list,                         ## number of hidden units for temporal block: F_t
    ):
        super(DyTransformer, self).__init__()
        
        # hyper-parameters
        self.num_features   = num_features
        self.num_time_steps = num_time_steps
        
        self.spatial_drop  = spatial_drop
        self.temporal_drop = temporal_drop
        
        self.num_structural_heads_list = num_structural_heads_list
        self.num_structural_hids_list  = num_structural_hids_list
        
        self.num_temporal_heads_list = num_temporal_heads_list
        self.num_temporal_hids_list  = num_temporal_hids_list
        
        assert len(num_structural_hids_list) == len(num_temporal_hids_list)
        self.num_layers = len(num_structural_hids_list)
        
        # model structure
        self.structural_attention_layer_list = nn.ModuleList()
        self.temporal_attention_layer_list   = nn.ModuleList()
        
        for i in range(len(self.num_structural_hids_list)):
            if i == 0:
                input_dim=self.num_features
            else:
                input_dim=self.num_structural_hids_list[i-1]
                
            self.structural_attention_layer_list.append(
                StructuralAttentionLayer(
                    n_heads=self.num_structural_heads_list[i],                      # H
                    input_dim=input_dim,                                            # D
                    output_dim=self.num_structural_hids_list[i],                    # F     
                    attn_drop=self.spatial_drop,                                    # normalized attention coef dropout
                ))
        
        for i in range(len(self.num_temporal_hids_list)): 
            self.temporal_attention_layer_list.append(
                TemporalAttentionLayer(
                    n_heads=self.num_temporal_heads_list[i],                        # G
                    input_dim=self.num_temporal_hids_list[i-1],                     # F
                    output_dim=self.num_temporal_hids_list[i],                      # F'
                    num_time_steps=self.num_time_steps,                             # T = # training snapshots + 1
                    attn_drop=self.temporal_drop,                                   # normalized attention coef dropout
                ))
            
    def forward(
        self,
        features,           # tuple format of sparse feature matrix [T, Nt, D]
        adjs,               # tuple format of sparse adj matrix [T, Nt, Nt]
        device,
    ):
        """
        T: number of time steps (t for training + 1)
        N: number of sampled nodes
        D: initial feature dimension
        output: tensor of shape [N, T, F]
        """
        # 1. structural attention layer forward
        # input: [T, Nt, D] features; [T, Nt, Nt] weighted adj matrixs
        structural_inputs = features
        num_nodes = [structural_input.size(0) for structural_input in structural_inputs]
        
        for ell in range(self.num_layers):
            structural_att_outputs = []
            for t in range(self.num_time_steps):
                # input: [Nt, D] -> [Nt, F/H]; after concat: [Nt, F]
                x_t = structural_inputs[t]
                if ell > 0:
                    x_t = x_t[:num_nodes[t]]
                x = self.structural_attention_layer_list[ell](x_t, adjs[t])
                structural_att_outputs.append(x)
        
            # 2. pack structural output embeddings across snapshots; then stack and transpose
            # input: [T, Nt, F] structural_att_outputs
            # output: [T, NT, F] -> [N, T, F]
            for t in range(self.num_time_steps):
                zero_padding = torch.zeros(
                    size=(structural_att_outputs[-1].shape[0] - structural_att_outputs[t].shape[0], 
                          structural_att_outputs[t].shape[1]),  # this is for N_T - N_t compensation
                    dtype=torch.float32,
                    device=device,
                )
                structural_att_outputs[t] = torch.cat((structural_att_outputs[t], zero_padding), 0)

            structure_outputs = torch.stack(structural_att_outputs)
            structure_outputs = structure_outputs.permute(1, 0, 2)

            # 3. temporal attention layer forward
            # input: [N, T, F]: [24, 3, 128] -> [N, T, F']
            temporal_inputs = structure_outputs
            outputs = self.temporal_attention_layer_list[ell](temporal_inputs, device)
            structural_inputs = outputs.permute(1, 0, 2)

        return outputs
    
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
        out = F.elu(out)
        out = F.dropout(out, self.attn_drop, training=self.training)      
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

###############################################################################
###############################################################################
###############################################################################

class TemporalAttentionLayer(nn.Module):
    """
    Temporal Attention Block (Single head)
    """
    def __init__(
        self, 
        n_heads,                        # G
        input_dim,                      # F
        output_dim,                     # F'
        num_time_steps,                 # T = # training snapshots + 1
        attn_drop,                      # normalized attention coef dropout
        use_residual=True,
    ):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_time_steps = num_time_steps
        self.attn_drop = attn_drop
        self.use_residual = use_residual
        
        self.Wq = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))  # [F, F']
        self.Wk = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))  # [F, F']
        self.Wv = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))  # [F, F']
        self.Wp = nn.Parameter(torch.Tensor(self.num_time_steps, self.input_dim)) # [T, F]
            
        self._reset_parameters()

    def forward(
        self,                           
        x,                              # [N, T, F]
        device,
    ):
        # 1. add position embeddings to input
        N = x.size(0)
        position_inputs = torch.arange(self.num_time_steps, device=device)
        position_inputs = position_inputs.expand(N, self.num_time_steps)  # [N, T]: 
        temporal_inputs = x + F.embedding(position_inputs, self.Wp)       # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Wq, dims=([2],[0]))   # [N, T, F']
        k = torch.tensordot(temporal_inputs, self.Wk, dims=([2],[0]))   # [N, T, F']
        v = torch.tensordot(temporal_inputs, self.Wv, dims=([2],[0]))   # [N, T, F']

        # 3. Split, concat and scale
        q_ = torch.cat(torch.split(q, q.size(2)//self.n_heads, dim=2), dim=0)    # [GN, T, F'/G]
        k_ = torch.cat(torch.split(k, k.size(2)//self.n_heads, dim=2), dim=0)    # [GN, T, F'/G]
        v_ = torch.cat(torch.split(v, v.size(2)//self.n_heads, dim=2), dim=0)    # [GN, T, F'/G]

        # 4. Scaled dot-product attention
        outputs = torch.matmul(q_, k_.permute(0, 2, 1))                     # [GN, T, T]
        outputs = outputs / (k_.size(-1)**0.5)                       # [GN, T, T]

        # 5. Masked (causal) softmax to compute attention weights
        diag_val = torch.ones_like(outputs[0,:,:], device=device)                 # [T, T]  
        tril = torch.tril(diag_val)                                               # [T, T]
        masks = tril.repeat(outputs.size(0), 1, 1)                                # [GN, T, T]
        zero_vec = (-2 ** 32 + 1) * torch.ones_like(masks)                       
        outputs = torch.where(torch.eq(masks, 0), zero_vec, outputs)          
        outputs = F.softmax(outputs, dim=-1)                                        # [GN, T, T]

        # 6. Dropout, and final projection
        outputs = F.dropout(outputs, self.attn_drop, training=self.training)
        outputs = torch.matmul(outputs, v_)                              # [GN, T, F'/G]

        # 7. Split and re-concat
        split_outputs = torch.split(outputs, outputs.size(0)//self.n_heads, dim=0)
        outputs = torch.cat(split_outputs, dim=2)                        # [N, T, F']
        
        if self.use_residual:
            assert self.input_dim == self.output_dim
            outputs += temporal_inputs

        return outputs
        
    def _reset_parameters(self, ):
        glorot(self.Wq)
        glorot(self.Wk)
        glorot(self.Wv)
        glorot(self.Wp)
        