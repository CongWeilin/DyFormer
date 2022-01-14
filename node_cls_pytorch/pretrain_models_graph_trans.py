
from train_inits import *
from utils.dynamic_graph_transformer_utils import *

import math

def train_current_time_step(FLAGS, graphs, adjs, device, res_id=None):
    """
    Setup
    """
    FLAGS.supervised=False
    FLAGS.supervised_loss=False
    FLAGS.unsupervised_loss=True

    ###########################################################
    ###########################################################
    ###########################################################
    # Set random seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)

    # Recursively make directories if they do not exist
    if res_id is None:
        FLAGS.res_id = 'Final_%s_%s_seed_%d_time_%d'%(FLAGS.model_name, FLAGS.dataset, FLAGS.seed, FLAGS.time_step)
    else:
        FLAGS.res_id = 'Final_%s_%s_seed_%d_time_%d_resid_%s'%(FLAGS.model_name, FLAGS.dataset, FLAGS.seed, FLAGS.time_step, res_id)
        
    output_dir = create_save_path(FLAGS)
    print('Savr dir:', output_dir)

    # get the correct start and end time
    cur_window, train_start_time, eval_time = get_train_time_interval(FLAGS)
    print(cur_window, train_start_time, eval_time)

    FLAGS.cur_window        = cur_window
    FLAGS.eval_time         = eval_time
    FLAGS.train_start_time  = train_start_time

    FLAGS.edge_encode_num = 2
    FLAGS.edge_dist_encode_num = FLAGS.max_dist + 2

    ###########################################################
    ###########################################################
    ###########################################################
    graphs_train = graphs[train_start_time : eval_time]
    adjs_train   = adjs[train_start_time : eval_time]

    graphs_eval = graphs[eval_time]
    adjs_eval   = adjs[eval_time]

    print('Train graphs size')
    for graph in graphs_train:
        print("Node %d, edges %d"%(len(graph.nodes), len(graph.edges)))

    print('Eval graphs size')
    print("Node %d, edges %d"%(len(graphs_eval.nodes), len(graphs_eval.edges)))

    ###########################################################
    ###########################################################
    ###########################################################
    # Load node feats
    num_nodes_graph_eval = len(graphs_eval.nodes)

    feats = get_feats(adjs_train, num_nodes_graph_eval, 0, eval_time, FLAGS)
    FLAGS.num_features = feats[0].shape[1]

    ###########################################################
    ############ Used for DyGraphTransformer ##################
    ###########################################################
    data_dict = compute_node_edge_encoding(graphs_train, FLAGS, force_regen=FLAGS.force_regen)
    eval_edge_encode, eval_PPR, eval_edge_dist_encode = data_dict['st_%d_et_%d'%(train_start_time, eval_time)]

    print(data_dict.keys())

    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        get_evaluation_data(adjs_train[-1], adjs_eval, FLAGS.time_step, FLAGS.dataset, force_regen=FLAGS.force_regen)

    all_edge_num = len(train_edges) + len(val_edges) + len(test_edges)
    if all_edge_num > 1000:
        sample_ratio = 1000/all_edge_num
        train_edges       = train_edges[np.random.permutation(len(train_edges))[:int(sample_ratio*len(train_edges))]]
        train_edges_false = train_edges_false[np.random.permutation(len(train_edges_false))[:int(sample_ratio*len(train_edges_false))]]
        val_edges         = val_edges[np.random.permutation(len(val_edges))[:int(sample_ratio*len(val_edges))]]
        val_edges_false   = val_edges_false[np.random.permutation(len(val_edges_false))[:int(sample_ratio*len(val_edges_false))]]
        test_edges        = test_edges[np.random.permutation(len(test_edges))[:int(sample_ratio*len(test_edges))]]
        test_edges_false  = test_edges_false[np.random.permutation(len(test_edges_false))[:int(sample_ratio*len(test_edges_false))]]

    print("# train: {}, # val: {}, # test: {}\n".format(len(train_edges), len(val_edges), len(test_edges)))

    ###########################################################
    ############ Used for DyGraphTransformer ##################
    ###########################################################
    activate_eval_nodes = np.concatenate([train_edges, train_edges_false,
                                          val_edges, val_edges_false,
                                          test_edges, test_edges_false], axis=0)

    activate_eval_nodes = np.unique(activate_eval_nodes)
    num_activate_eval_nodes = len(activate_eval_nodes)
    activate_eval_nodes_splits = np.array_split(activate_eval_nodes, math.ceil(num_activate_eval_nodes/FLAGS.eval_batch_size))
    print('Eval split size', len(activate_eval_nodes_splits))
    
    ### get new indicies
    eval_all_nodes_to_new_index = dict(zip(activate_eval_nodes, np.arange(num_activate_eval_nodes)))
    train_edges_new       = translate(train_edges,       eval_all_nodes_to_new_index, shape='Nx2') # edges [Nx2] -> [2xN]
    train_edges_false_new = translate(train_edges_false, eval_all_nodes_to_new_index, shape='Nx2')
    val_edges_new         = translate(val_edges,         eval_all_nodes_to_new_index, shape='Nx2')
    val_edges_false_new   = translate(val_edges_false,   eval_all_nodes_to_new_index, shape='Nx2')
    test_edges_new        = translate(test_edges,        eval_all_nodes_to_new_index, shape='Nx2')
    test_edges_false_new  = translate(test_edges_false,  eval_all_nodes_to_new_index, shape='Nx2')

    ###########################################################
    ###########################################################
    ###########################################################
    # Setup minibatchsampler
    from utils.minibatch_pretrain import NodeMinibatchIterator
    minibatchIterator = NodeMinibatchIterator(
        graphs=graphs_train,                        # graphs (total) 
        adjs=adjs_train,                            # adjs (total)
        start_time=FLAGS.train_start_time,
        batch_size=FLAGS.batch_size,
    )

    # Setup model and optimizer
    model = load_model(FLAGS, device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)

    ###########################################################
    ###########################################################
    ###########################################################
    # Setup result accumulator variables.
    epochs_test_result = defaultdict(lambda: [])
    epochs_val_result = defaultdict(lambda: [])

    """
    Training starts
    """
    epoch_train_loss_all = []

    best_valid_result = 0
    best_valid_epoch = 0

    best_valid_model_path = os.path.join(output_dir, 'best_valid_model_{}.pt'.format(FLAGS.dataset))

    total_epoch_time = 0.0
    for epoch in range(FLAGS.num_epoches):
        model.train()
        minibatchIterator.shuffle()
        
        ### Start forward

        iter_loss = []
        epoch_time = 0.0
        
        ############################################################################
        ######################### Training #########################################
        ############################################################################
        start_t = time.time()
        
        while not minibatchIterator.end():
            # sample mini-batch
            train_start, train_end, batch_nodes = minibatchIterator.next_minibatch_feed_dict()

            # get spatial-temporal encoding
            edge_encode, PPR, edge_dist_encode = data_dict['st_%d_et_%d'%(train_start, train_end)]
            active_nodes = np.unique(batch_nodes)

            ### First view
            node_feats_np_1, edge_encode_np_1, edge_dist_encode_np_1, target_node_size_1, context_nodes_size_1, _ = get_common_neighbors(active_nodes,
                                                                                                                                         edge_encode, PPR, edge_dist_encode,
                                                                                                                                         feats[train_end-1-train_start], FLAGS, deterministic_neighbor_sampling=True)
            node_feats_th       = torch.FloatTensor(node_feats_np_1).to(device)
            edge_encode_th      = torch.LongTensor(edge_encode_np_1).to(device)
            edge_dist_encode_th = torch.LongTensor(edge_dist_encode_np_1).to(device)

            if len(active_nodes) <= 2:
                continue
            
            # forward propagation
            output_1, output_unsupervised_1 = model(node_feats_th, edge_encode_th, edge_dist_encode_th, target_node_size_1, context_nodes_size_1, device)

            ### Second view
            node_feats_np_2, edge_encode_np_2, edge_dist_encode_np_2, target_node_size_2, context_nodes_size_2, _ = get_common_neighbors(active_nodes,
                                                                                                                                         edge_encode, PPR, edge_dist_encode,
                                                                                                                                         feats[train_end-1-train_start], FLAGS, deterministic_neighbor_sampling=False)
            node_feats_th       = torch.FloatTensor(node_feats_np_2).to(device)
            edge_encode_th      = torch.LongTensor(edge_encode_np_2).to(device)
            edge_dist_encode_th = torch.LongTensor(edge_dist_encode_np_2).to(device)

            # forward propagation
            output_2, output_unsupervised_2 = model(node_feats_th, edge_encode_th, edge_dist_encode_th, target_node_size_2, context_nodes_size_2, device)

            # compute loss
            loss = torch.tensor(0.0, device=device)

            # First view
            sampled_temporal_edges = generate_temporal_edges(edge_encode_np_1, target_node_size_1, FLAGS.neg_sample_size)

            for t, (pos_edges, neg_edges) in enumerate(sampled_temporal_edges):
                loss += link_forecast_loss(output_unsupervised_1[t, :, :], pos_edges, neg_edges, FLAGS.neg_weight, device)/output_unsupervised_1.size(0)

            # Second view
            sampled_temporal_edges = generate_temporal_edges(edge_encode_np_2, target_node_size_2, FLAGS.neg_sample_size)

            for t, (pos_edges, neg_edges) in enumerate(sampled_temporal_edges):
                loss += link_forecast_loss(output_unsupervised_2[t, :, :], pos_edges, neg_edges, FLAGS.neg_weight, device)/output_unsupervised_2.size(0)

            loss += multi_view_loss(output_1[:target_node_size_1, :], output_2[:target_node_size_2, :], model.BYOL_decoder)

            iter_loss += [loss.item()]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FLAGS.max_gradient_norm)
            optimizer.step()

        epoch_train_loss = np.mean(iter_loss)
        epoch_time = time.time() - start_t 
        epoch_train_loss_all.append(epoch_train_loss)
        
        print("Train Loss at epoch {} : {}".format(epoch, epoch_train_loss))
        print("Time for epoch ", epoch_time)
        
        total_epoch_time += epoch_time

        ############################################################################
        ######################### Testing ##########################################
        ############################################################################
        # Validation in training:
        if (epoch + 1) % FLAGS.test_freq == 0:
            model.eval()                                # disable dropout in model

            emb = []
            for cur_activate_eval_nodes in activate_eval_nodes_splits:

                node_feats_np_1, edge_encode_np_1, edge_dist_encode_np_1, target_node_size_1, context_nodes_size_1, eval_all_nodes_to_new_index  = get_common_neighbors(cur_activate_eval_nodes,
                                                                                                                                                              eval_edge_encode, eval_PPR, eval_edge_dist_encode,
                                                                                                                                                              feats[-1], FLAGS, deterministic_neighbor_sampling=True)
                node_feats_th       = torch.FloatTensor(node_feats_np_1).to(device)
                edge_encode_th  = torch.LongTensor(edge_encode_np_1).to(device)
                edge_dist_encode_th = torch.LongTensor(edge_dist_encode_np_1).to(device)

                # forward
                output, _ = model(node_feats_th, edge_encode_th, edge_dist_encode_th, target_node_size_1, context_nodes_size_1, device)
                emb.append(output[:target_node_size_1, :].detach().cpu().numpy())
            emb = np.concatenate(emb, axis=0)

            # Use external classifier to get validation and test results.

            val_results, test_results, _, _ = evaluate_classifier(
                train_edges_new, train_edges_false_new,
                val_edges_new,   val_edges_false_new,
                test_edges_new,  test_edges_false_new,
                emb, emb)

            val_HAD = val_results["HAD"][0]
            test_HAD = test_results["HAD"][0]
            val_SIGMOID = val_results["SIGMOID"][0]
            test_SIGMOID = test_results["SIGMOID"][0]

            print("Epoch %d, Val AUC_HAD %.4f, Test AUC_HAD %.4f, Val AUC_SIGMOID %.4f, Test AUC_SIGMOID %.4f, "%(epoch, val_HAD, test_HAD, val_SIGMOID, test_SIGMOID))

            epochs_test_result["HAD"].append(test_HAD)
            epochs_val_result["HAD"].append(val_HAD)
            epochs_test_result["SIGMOID"].append(test_SIGMOID)
            epochs_val_result["SIGMOID"].append(val_SIGMOID)

            if val_HAD > best_valid_result:
                best_valid_result = val_HAD
                best_valid_epoch = epoch
                best_valid_epoch_emb = emb,
                torch.save(model.state_dict(), best_valid_model_path)

            if epoch - best_valid_epoch > FLAGS.patient_iters: 
                break

    """
    Done training: choose best model by validation set performance.
    """
    best_epoch = epochs_val_result["HAD"].index(max(epochs_val_result["HAD"]))
    print("Total used time is: {}\n".format(total_epoch_time))

    print("Best epoch ", best_epoch)

    val_results, test_results = epochs_val_result["HAD"][best_epoch], epochs_test_result["HAD"][best_epoch]

    print("Best epoch val results {}".format(val_results))
    print("Best epoch test results {}".format(test_results))

    """
    Get final results
    """
    result = {
        'id': FLAGS.res_id,
        'best_epoch': best_epoch,
        'best_valid_epoch_result': val_results,
        'best_test_epoch_result': test_results,
        'valid_epoch_auc': epochs_val_result["HAD"],
        'test_epoch_auc': epochs_test_result["HAD"],
        'epoch_train_loss': epoch_train_loss_all,
    }
            
    with open(os.path.join(output_dir, 'result_{}.json'.format(FLAGS.dataset)), 'w') as outfile:
        json.dump(result, outfile)
        
    result = {
        'best_valid_epoch_emb': best_valid_epoch_emb, 
        'train_edge_pos': train_edges_new,
        'train_edge_neg': train_edges_false_new,
        'valid_edge_pos': val_edges_new,
        'valid_edge_neg': val_edges_false_new,
        'test_edge_pos': test_edges_new,
        'test_edge_neg': test_edges_false_new,
    }
    with open(os.path.join(output_dir, 'result_emb_edge_{}.pkl'.format(FLAGS.dataset)), 'wb') as outfile:
        pickle.dump(result, outfile)
        
    return best_valid_model_path

###########################################################
###########################################################
###########################################################
def multi_view_loss(z1, z2, decoder):
    def similarity(p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return F.mse_loss(p,z) # -(p*z).sum(dim=1).mean()

    loss = similarity(decoder(z1), z2) + similarity(decoder(z2), z1)
    return loss/2

if __name__ == '__main__':

    """
    Run GraphTransformer (our proposal)
    """
    
    model_name = 'DyGraphTransformer'
    
    ########################################
    FLAGS = flags()
    FLAGS.model_name = model_name    
    FLAGS = update_args(FLAGS)

    ########################################
    FLAGS.max_neighbors = -1
    FLAGS.deterministic_neighbor_sampling = True

    FLAGS.two_steam_model = True
    FLAGS.window = 5

    FLAGS.num_epoches = 200
    FLAGS.GPU_ID = 0
    FLAGS.patient_iters = 20

    FLAGS.use_edge_feats    = False
    FLAGS.force_regen = False

    ########################################
    FLAGS.neighbor_sampling_size = 0.3
    FLAGS.num_heads = 8
    FLAGS.num_hids = 128
    ########################################
    """
    Setup layers and res_id
    """

    device = get_device(FLAGS)
    graphs, adjs = load_graphs(FLAGS.dataset)

    RES_ID = 'pretrain_%d_layer_model'%FLAGS.num_layers
    train_current_time_step(FLAGS, graphs, adjs, device, RES_ID)

