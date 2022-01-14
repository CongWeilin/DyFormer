

import json
import yaml



dataset_to_ind = {'wiki_classification':0, 'reddit_classification':1}

datasets = ['wiki_classification', 'reddit_classification']   # change it back to 6
min_time_list = [2, 2]
max_time_list = [11, 11]

seed_list = [123, 231, 321]
neg_weight_list = [0.1, 0.1]
str_head_list = [[16, 8], [16, 8]]
str_layer_list = [[256, 128], [256, 128]]
temp_head_list = [[16], [16]]
temp_layer_list = [[128], [128]]

neg_sample_size_list = [10, 10]
dmax = [5, 5]
neighbor_sampling_size_list = [0.5, 0.5]

transformer_window_size = [5, 5]



for dataset in datasets:
    ########################################
    ########################################
    ########################################
    config = {
        "min_time": min_time_list[dataset_to_ind[dataset]],
        "max_time": max_time_list[dataset_to_ind[dataset]],
        "num_hiddens": 128,
        "num_layers": 2,
        "num_heads": 8,
        "feat_drop": 0.5,
        "spatial_drop": 0.1,
        "use_residual": False,
        "neg_weight": neg_weight_list[dataset_to_ind[dataset]]
    }

    with open('./configs/GAT_%s.yaml'%dataset, 'w') as f:
        yaml.dump(config, f)
    
    with open('./configs/GAT_RNN_%s.yaml'%dataset, 'w') as f:
        yaml.dump(config, f)
        
    with open('./configs/GCN_LSTM_v1_%s.yaml'%dataset, 'w') as f:
        yaml.dump(config, f)
    
    ########################################
    ########################################
    ########################################

    config = {
        "min_time": min_time_list[dataset_to_ind[dataset]],
        "max_time": max_time_list[dataset_to_ind[dataset]],
        "structural_head_config": str_head_list[dataset_to_ind[dataset]],
        "structural_layer_config": str_layer_list[dataset_to_ind[dataset]],
        "temporal_head_config": temp_head_list[dataset_to_ind[dataset]],
        "temporal_layer_config": temp_layer_list[dataset_to_ind[dataset]],
        "spatial_drop": 0.1,
        "temporal_drop": 0.5,
        "neg_weight": neg_weight_list[dataset_to_ind[dataset]]
    }

    with open('./configs/DySAT_%s.yaml'%dataset, 'w') as f:
        yaml.dump(config, f)

    ########################################
    ########################################
    ########################################

    config = {
        "min_time": min_time_list[dataset_to_ind[dataset]],
        "max_time": max_time_list[dataset_to_ind[dataset]],
        "num_hiddens": 256,
        "num_layers": 2,
        "feat_drop": 0.5,
        "neg_weight": neg_weight_list[dataset_to_ind[dataset]],
        "neg_sample_size": neg_sample_size_list[dataset_to_ind[dataset]],
        "use_edge_feats": False,
        "linear_cls_input_size": 256
    }

    with open('./configs/EvolveGCN_O_%s.yaml'%dataset, 'w') as f:
        yaml.dump(config, f)


    ########################################
    ########################################
    ########################################
    
    config = {
        "min_time": min_time_list[dataset_to_ind[dataset]],
        "max_time": max_time_list[dataset_to_ind[dataset]],
        "num_hiddens": 256,
        "num_layers": 2,
        "neg_weight": neg_weight_list[dataset_to_ind[dataset]],
        "neg_sample_size": neg_sample_size_list[dataset_to_ind[dataset]],
        "num_heads": 8,
        "num_types": 1,
        "num_relations": 1,
        "feat_drop": 0.5,
    }

    with open('./configs/DyTransformer_compact_%s.yaml'%dataset, 'w') as f:
        yaml.dump(config, f)

    ########################################
    ########################################
    ########################################

    config = {
        "min_time": min_time_list[dataset_to_ind[dataset]],
        "max_time": max_time_list[dataset_to_ind[dataset]],
        "num_layers": 2,
        "neg_weight": neg_weight_list[dataset_to_ind[dataset]],
        "neg_sample_size": neg_sample_size_list[dataset_to_ind[dataset]],
        "num_heads": 8,
        "attn_drop": 0.1,
        "feat_drop": 0.5,
        "num_hids": 128,
        "max_dist": dmax[dataset_to_ind[dataset]],
        "memory_size":32,
        "max_neighbors": -1,
        "neighbor_sampling_size": neighbor_sampling_size_list[dataset_to_ind[dataset]],
        "window": transformer_window_size[dataset_to_ind[dataset]]
    }

    with open('./configs/DyGraphTransformer_MLR_%s.yaml'%dataset, 'w') as f:
        yaml.dump(config, f)
    with open('./configs/DyGraphTransformer_%s.yaml'%dataset, 'w') as f:
        yaml.dump(config, f)
    with open('./configs/DyGraphTransformer_two_stream_%s.yaml'%dataset, 'w') as f:
        yaml.dump(config, f)

    ########################################
    ########################################
    ########################################
    config = {
        "min_time": min_time_list[dataset_to_ind[dataset]],
        "max_time": max_time_list[dataset_to_ind[dataset]],
        "neg_weight": neg_weight_list[dataset_to_ind[dataset]],
        "hidden_size": 32,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
    }
    with open('./configs/GraphBert_%s.yaml'%dataset, 'w') as f:
        yaml.dump(config, f)
