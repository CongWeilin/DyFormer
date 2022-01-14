import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from torch_geometric.nn.inits import uniform, glorot
from .losses import link_forecast_loss
EPS = 1e-15

class ContrastiveWrapper(nn.Module):
    def __init__(self, num_hids, node_encode_num, edge_encode_num, edge_dist_encode_num, window_size):
        
        super(ContrastiveWrapper, self).__init__()
        self.node_encode_num      = node_encode_num
        self.edge_encode_num      = edge_encode_num
        self.edge_dist_encode_num = edge_dist_encode_num
        self.window_size          = window_size
        self.num_hids             = num_hids
        
        self.weight = nn.Parameter(torch.Tensor(self.num_hids, self.num_hids))
        self.reset_parameters()
        
    def reset_parameters(self):
        uniform(self.num_hids, self.weight)

    def get_corrupt_encode(self, node_encode_th, edge_encode_th, edge_dist_encode_th, device):
                
        node_feats_th_corrupt = torch.randint(0, self.node_encode_num*self.window_size, node_encode_th.shape).to(device)
        edge_encode_th_corrupt = torch.randint(0, self.edge_encode_num*self.window_size, edge_encode_th.shape).to(device)
        edge_dist_encode_th_corrupt = torch.randint(0, self.edge_dist_encode_num, edge_dist_encode_th.shape).to(device)
        
        node_encode_th = (node_encode_th + node_feats_th_corrupt) % (self.node_encode_num*self.window_size)
        edge_encode_th = (edge_encode_th + edge_encode_th_corrupt) % (self.edge_encode_num*self.window_size)
        edge_dist_encode_th = (edge_dist_encode_th + edge_dist_encode_th_corrupt) % (self.edge_dist_encode_num)
        
        return node_encode_th, edge_encode_th, edge_dist_encode_th
    
    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.
        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value
    
    def loss(self, pos_z, neg_z):
        summary = torch.sigmoid(torch.mean(pos_z, dim=0))
        
        r"""Computes the mutual information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss
    
class TemporalReconstruct(nn.Module):
    def __init__(self, num_hids, window_size):
        super(TemporalReconstruct, self).__init__()
        
        self.window_size          = window_size
        self.num_hids             = num_hids
        
        self.temporal_decoder = nn.ModuleList()
        for t in range(self.window_size):
            self.temporal_decoder.append(nn.Linear(self.num_hids, self.num_hids))
    
    def loss(self, x, edge_encode, neg_sample_size, neg_weight, device):
        active_time = edge_encode.shape[-1]
        
        loss = torch.tensor(0.0, device=device)
        for t in range(active_time):
            edge_label = edge_encode[:, :, t] % self.window_size
            pos_edges = np.stack(np.where(edge_label == 1))
            pos_edges_num = pos_edges.shape[1]
            neg_edges = np.stack(np.where(edge_label == 0))
            neg_edges_num = neg_edges.shape[1]
            
            if neg_edges_num > pos_edges_num * neg_sample_size:
                neg_edges_num = pos_edges_num * neg_sample_size
                select = np.random.permutation(neg_edges_num)[:neg_edges_num]
                neg_edges = neg_edges[:, select]
            temporal_x = self.temporal_decoder[t](x)
            loss += link_forecast_loss(temporal_x, pos_edges, neg_edges, neg_weight, device)
        return loss
        
class SiameseReconstruct(nn.Module):
    def __init__(self, num_hids):
        super(SiameseReconstruct, self).__init__()
        self.num_hids = num_hids
        self.decoder = nn.Linear(self.num_hids, self.num_hids)
    
    def negative_cosine_similarity(self, p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p*z).sum(dim=1).mean()
    
    def loss(self, z1, z2):
        loss = self.negative_cosine_similarity(self.decoder(z1), z2) + self.negative_cosine_similarity(self.decoder(z2), z1)
        return loss/2