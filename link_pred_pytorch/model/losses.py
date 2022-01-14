import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def link_pred_loss(
    x,                      # tensor output of shape [N, T, F]
    node_1,                 # list of shape [T, Xt]
    node_2,                 # list of shape [T, Xt]
    proximity_neg_samples,  # list of shape [T-1, num_samples]
    neg_weight,             # negative weight for negative sample
    device,
):
    """
        dot product >>> 1 x pos + n x neg
    """
    
    graph_loss = torch.tensor(0.0, device=device)
    num_time_steps_train = len(node_1)
    for t in range(num_time_steps_train):
        ## for t [T-1]
        x_t = x.permute(1, 0, 2)[t]                                 # [N, F]

        node_1_t = torch.tensor(node_1[t], device=device).long()
        node_2_t = torch.tensor(node_2[t], device=device).long()
        inputs1 = F.embedding(node_1_t, x_t)         # [Xt, F]
        inputs2 = F.embedding(node_2_t, x_t)         # [Xt, F]

        pos_score = (inputs1 * inputs2).sum(1)                      # [Xt] 

        neg_samples = F.embedding(torch.tensor(proximity_neg_samples[t], device=device), x_t)  # [# of samples, F]
        neg_score = (-1.0) * torch.matmul(inputs1, neg_samples.permute(1, 0))                  # [F, # of samples]

        pos_ent = F.binary_cross_entropy_with_logits(input=pos_score, target=torch.ones_like(pos_score), reduction='none')
        pos_ent[torch.isnan(pos_ent)] = 0
        num_pos_ent_not_nan = torch.sum(torch.isnan(pos_ent)==False)
        
        neg_ent = F.binary_cross_entropy_with_logits(input=neg_score, target=torch.ones_like(neg_score), reduction='none')
        neg_ent[torch.isnan(neg_ent)] = 0
        num_neg_ent_not_nan = torch.sum(torch.isnan(neg_ent)==False)
        
        graph_loss += torch.sum(pos_ent/num_pos_ent_not_nan) + neg_weight * torch.sum(neg_ent/num_neg_ent_not_nan)
    return graph_loss

def link_forecast_loss(x, pos_edges, neg_edges, neg_weight, device):
    pos_edges_i = torch.tensor(pos_edges[0, :]).long().to(device)
    pos_edges_j = torch.tensor(pos_edges[1, :]).long().to(device)
    pos_edges_i_x = F.embedding(pos_edges_i, x)         
    pos_edges_j_x = F.embedding(pos_edges_j, x)
    pos_score = (pos_edges_i_x * pos_edges_j_x).sum(1)  

    # print(torch.tensor(neg_edges[0, :]).long())
    neg_edges_i = torch.tensor(neg_edges[0, :]).long().to(device)
    neg_edges_j = torch.tensor(neg_edges[1, :]).long().to(device)
    neg_edges_i_x = F.embedding(neg_edges_i, x)         
    neg_edges_j_x = F.embedding(neg_edges_j, x)
    neg_score = (-1.0) * (neg_edges_i_x * neg_edges_j_x).sum(1)  

    pos_ent = F.binary_cross_entropy_with_logits(input=pos_score, target=torch.ones_like(pos_score), reduction='none')
    pos_ent[torch.isnan(pos_ent)] = 0
    num_pos_ent_not_nan = torch.sum(torch.isnan(pos_ent)==False)
    
    neg_ent = F.binary_cross_entropy_with_logits(input=neg_score, target=torch.ones_like(neg_score), reduction='none')
    neg_ent[torch.isnan(neg_ent)] = 0
    num_neg_ent_not_nan = torch.sum(torch.isnan(neg_ent)==False)

    graph_loss = torch.sum(pos_ent/num_pos_ent_not_nan) + neg_weight * torch.sum(neg_ent/num_neg_ent_not_nan)
    return graph_loss


