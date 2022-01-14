import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform, BertAttention, BertIntermediate, BertOutput
from transformers.configuration_utils import PretrainedConfig

##########################################################################
##########################################################################
##########################################################################

class GraphBert(BertPreTrainedModel):
    def __init__(self, config):
        super(GraphBert, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()
        
    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=None):

        embedding_output = self.embeddings(raw_features = raw_features, 
                                           wl_role_ids  = wl_role_ids, 
                                           init_pos_ids = init_pos_ids, 
                                           hop_dis_ids  = hop_dis_ids)
        
        encoder_outputs = self.encoder(embedding_output, residual_h=residual_h)
        
        pooled_output = self.pooler(encoder_outputs)
        return pooled_output

class GraphBertConfig(PretrainedConfig):

    def __init__(
        self,
        max_wl_role_index = 100,
        max_hop_dis_index = 100,
        max_inti_pos_index = 100,
        num_features=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.3,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs
    ):
        super(GraphBertConfig, self).__init__(**kwargs)
        self.max_wl_role_index = max_wl_role_index
        self.max_hop_dis_index = max_hop_dis_index
        self.max_inti_pos_index = max_inti_pos_index
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        
##########################################################################
##########################################################################
##########################################################################

class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from features, wl, position and hop vectors.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.raw_feature_embeddings = nn.Linear(config.num_features, config.hidden_size)
        self.wl_role_embeddings     = nn.Embedding(config.max_wl_role_index, config.hidden_size)
        self.inti_pos_embeddings    = nn.Embedding(config.max_inti_pos_index, config.hidden_size)
        self.hop_dis_embeddings     = nn.Embedding(config.max_hop_dis_index, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout =   nn.Dropout(config.hidden_dropout_prob)

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids):

        raw_feature_embeds  = self.raw_feature_embeddings(raw_features)
        role_embeddings     = self.wl_role_embeddings(wl_role_ids)
        position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        hop_embeddings      = self.hop_dis_embeddings(hop_dis_ids)

        #---- here, we use summation ----
        embeddings = raw_feature_embeds + role_embeddings + position_embeddings + hop_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

##########################################################################
##########################################################################
##########################################################################

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, residual_h=None):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
            #---- add residual ----
            if residual_h is not None:
                for index in range(hidden_states.size(1)):
                    hidden_states[:, index, :] += residual_h
        return hidden_states  
    
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states):
        
        attention_output = self.attention(hidden_states)[0]        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
