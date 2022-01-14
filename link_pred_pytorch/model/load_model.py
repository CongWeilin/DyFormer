
def load_model(FLAGS, device):
    if FLAGS.model_name == 'DySAT':
        if FLAGS.use_edge_feats:
            from .DySAT_edge import DySAT
            model = DySAT(
                num_features=FLAGS.num_features,                        ## max num of nodes
                 num_edge_features=FLAGS.num_edge_features,
                num_time_steps=FLAGS.cur_window,                        ## number of trainina snapshots + 1
                spatial_drop=FLAGS.spatial_drop,                        ## dropout % for structural layer
                temporal_drop=FLAGS.temporal_drop,                      ## dropout % for temporal layer
                num_structural_heads_list=FLAGS.structural_head_config, ## number of attention heads for GAT: H
                num_structural_hids_list=FLAGS.structural_layer_config, ## number of hidden units for GAT: also as output embedding for GAT: F
                num_temporal_heads_list=FLAGS.temporal_head_config,     ## number of attention heads for temporal block: G
                num_temporal_hids_list=FLAGS.temporal_layer_config,     ## number of hidden units for temporal block: F' (normally = F)
            ).to(device)
        else:
            from .DySAT import DySAT
            model = DySAT(
                num_features=FLAGS.num_features,                        ## max num of nodes
                num_time_steps=FLAGS.cur_window,                        ## number of trainina snapshots + 1
                spatial_drop=FLAGS.spatial_drop,                        ## dropout % for structural layer
                temporal_drop=FLAGS.temporal_drop,                      ## dropout % for temporal layer
                num_structural_heads_list=FLAGS.structural_head_config, ## number of attention heads for GAT: H
                num_structural_hids_list=FLAGS.structural_layer_config, ## number of hidden units for GAT: also as output embedding for GAT: F
                num_temporal_heads_list=FLAGS.temporal_head_config,     ## number of attention heads for temporal block: G
                num_temporal_hids_list=FLAGS.temporal_layer_config,     ## number of hidden units for temporal block: F' (normally = F)
            ).to(device)
    elif FLAGS.model_name == 'DyGraphTransformer_two_stream': ### our proposal
        # if FLAGS.sparse_attn:
        #     from .DyGraphTransformer_two_stream_sparse import DyGraphTransformer
        # else:
        #     from .DyGraphTransformer_two_stream import DyGraphTransformer
        from .DyGraphTransformer_two_stream_sparse import DyGraphTransformer
        model = DyGraphTransformer(
            num_features      = FLAGS.num_features,                                ## max num of nodes
            num_heads         = FLAGS.num_heads,                                ## number of trainina snapshots + 1
            num_hids          = FLAGS.num_hids,                                ## dropout % for structural layer
            num_layers        = FLAGS.num_layers,                              ## dropout % for temporal layer
            attn_drop         = FLAGS.attn_drop, ## number of attention heads for GAT: H
            feat_drop         = FLAGS.feat_drop, ## number of hidden units for GAT: also as output embedding for GAT: F
            edge_encode_num        = FLAGS.edge_encode_num,     ## number of hidden units for temporal block: F' (normally = F)
            edge_dist_encode_num   = FLAGS.edge_dist_encode_num,
            window_size            = FLAGS.cur_window,
            use_unsupervised_loss  = FLAGS.unsupervised_loss, 
            neighbor_sampling_size = FLAGS.neighbor_sampling_size,
            # use_edge_encode        = FLAGS.use_edge_encode,
            # use_edge_dist_encode   = FLAGS.use_edge_dist_encode
        ).to(device)
    elif FLAGS.model_name == 'EvolveGCN_O':
        from .EvolveGCN_O import EvolveGCN
        model = EvolveGCN(n_feats=FLAGS.num_features,
                          n_hid=FLAGS.num_hiddens,
                          n_layers=FLAGS.num_layers,
                          dropout=FLAGS.feat_drop).to(device)
    elif FLAGS.model_name == 'EvolveGCN_H':
        from .EvolveGCN_H import EvolveGCN
        model = EvolveGCN(n_feats=FLAGS.num_features,
                          n_hid=FLAGS.num_hiddens,
                          n_layers=FLAGS.num_layers,
                          dropout=FLAGS.feat_drop, 
                          RNN_type='EvolveGCN_H').to(device)
    elif FLAGS.model_name == 'GCN_LSTM_v1':
        from .GCN_RNN import GCN_RNN
        RNN_type = FLAGS.model_name.split('_')[1]
        model = GCN_RNN(n_feats=FLAGS.num_features,
                          n_hid=FLAGS.num_hiddens,
                          n_layers=FLAGS.num_layers,
                          dropout=FLAGS.feat_drop, 
                          RNN_type=RNN_type).to(device)
    elif FLAGS.model_name == 'GCN_LSTM_v2':
        from .GCN_RNN import GCN_RNN_v2
        RNN_type = FLAGS.model_name.split('_')[1]
        model = GCN_RNN_v2(n_feats=FLAGS.num_features,
                          n_hid=FLAGS.num_hiddens,
                          n_layers=FLAGS.num_layers,
                          dropout=FLAGS.feat_drop, 
                          RNN_type=RNN_type).to(device)
    elif FLAGS.model_name == 'GAT':
        from .GAT import GAT
        model = GAT(num_features = FLAGS.num_features,
                    num_hiddens  = FLAGS.num_hiddens, 
                    num_layers   = FLAGS.num_layers, 
                    num_heads    = FLAGS.num_heads, 
                    spatial_drop = FLAGS.spatial_drop, 
                    feat_drop    = FLAGS.feat_drop,
                    use_residual = FLAGS.use_residual,
                    use_rnn=False).to(device)
    elif FLAGS.model_name == 'GAT_RNN':
        from .GAT import GAT
        model = GAT(num_features = FLAGS.num_features,
                    num_hiddens  = FLAGS.num_hiddens, 
                    num_layers   = FLAGS.num_layers, 
                    num_heads    = FLAGS.num_heads, 
                    spatial_drop = FLAGS.spatial_drop, 
                    feat_drop    = FLAGS.feat_drop,
                    use_residual = FLAGS.use_residual,
                    use_rnn=True).to(device)
    elif FLAGS.model_name == 'GraphBert':
        from .GraphBert import GraphBert, GraphBertConfig
        config = GraphBertConfig(
            num_features        = FLAGS.num_features, 
            hidden_size         = FLAGS.hidden_size, 
            num_attention_heads = FLAGS.num_attention_heads, 
            num_hidden_layers   = FLAGS.num_hidden_layers
        )
        model = GraphBert(config).to(device)
#     for n, p in model.named_parameters():
#         print(n, p.shape)
    return model