import torch
import torch.nn as nn

import math

##########################################################################
##########################################################################
##########################################################################
# https://github.com/IBM/EvolveGCN/blob/master/egcn_o.py

class EvolveGCN(nn.Module):
    def __init__(self, n_feats, n_hid, n_layers, dropout):
        super().__init__()
        self.n_feats = n_feats
        self.n_layers = n_layers
        self.n_hid = n_hid
        self.dropout = dropout

        # feats = [n_hid] + [n_hid] * n_layers
        # self.encoder = nn.Linear(n_feats, n_hid)
        
        feats = [n_feats] + [n_hid] * n_layers
        
        self.GRCU_layers = nn.ModuleList()
        
        for i in range(1,len(feats)):
            GRCU_args = Namespace({'in_feats' : feats[i-1],
                                   'out_feats': feats[i],
                                   'activation': nn.RReLU()})

            grcu_i = GRCU(GRCU_args)
            self.GRCU_layers.append(grcu_i)

    def forward(self, x_list, adj_list, device):
        # x_list = [self.encoder(x) for x in x_list]
        
        for unit in self.GRCU_layers:
            x_list = unit(adj_list, x_list)

        return x_list

##########################################################################
##########################################################################
##########################################################################

class GRCU(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = nn.RReLU()
        self.GCN_init_weights = nn.Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self, adj_list, x_list):#,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(adj_list):
            node_embs = x_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights)#,node_embs,mask_list[t])
            
            # print(node_embs.shape, GCN_weights.shape)
            # node_embs = self.activation(node_embs.matmul(GCN_weights))
            node_embs = self.activation(torch.sparse.mm(Ahat, node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq

##########################################################################
##########################################################################
##########################################################################

class mat_GRU_cell(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())

    def forward(self,prev_Q):
        z_topk = prev_Q

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        
##########################################################################
##########################################################################
##########################################################################

class mat_GRU_gate(nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = nn.Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = nn.Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = nn.Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)