
from train_inits import *
from utils.dynamic_graph_transformer_utils import *

import math

def train_current_time_step(FLAGS, graphs, adjs, device, res_id=None, model_path=None):
    """
    Setup
    """

    # Set random seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    print(FLAGS)

   # Recursively make directories if they do not exist
    if res_id is None:
        FLAGS.res_id = 'Final_%s_%s_seed_%d_time_%d'%(FLAGS.model_name, FLAGS.dataset, FLAGS.seed, FLAGS.time_step)
    else:
        FLAGS.res_id = 'Final_%s_%s_seed_%d_time_%d_resid_%s'%(FLAGS.model_name, FLAGS.dataset, FLAGS.seed, FLAGS.time_step, res_id)

    log_file, output_dir = create_logger(FLAGS)
    print('Savr dir:', output_dir)
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info(vars(FLAGS))

    cur_window, train_start_time, eval_time = get_train_time_interval(FLAGS)
    print(cur_window, train_start_time, eval_time)

    FLAGS.cur_window        = cur_window
    FLAGS.eval_time         = eval_time
    FLAGS.train_start_time  = train_start_time

    FLAGS.edge_encode_num = 2
    FLAGS.edge_dist_encode_num = FLAGS.max_dist + 2

    graphs_train = graphs[train_start_time : eval_time]
    adjs_train   = adjs[train_start_time : eval_time]

    graphs_eval = graphs[eval_time]
    adjs_eval   = adjs[eval_time]

    num_nodes_graph_eval = len(graphs_eval.nodes)
    feats = get_feats(adjs_train, num_nodes_graph_eval, 0, eval_time, FLAGS)
    FLAGS.num_features = feats[0].shape[1]

    print('Train graphs size')
    for graph in graphs_train:
        print("Node %d, edges %d"%(len(graph.nodes), len(graph.edges)))

    print('Eval graphs size')
    print("Node %d, edges %d"%(len(graphs_eval.nodes), len(graphs_eval.edges)))

    ###
    data_dict = compute_node_edge_encoding(graphs_train, FLAGS, force_regen=FLAGS.force_regen)
    print(data_dict.keys())

    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        get_evaluation_data(adjs_train[-1], adjs_eval, FLAGS.time_step, FLAGS.dataset, force_regen=FLAGS.force_regen)

    print(train_edges.shape, train_edges_false.shape, val_edges.shape, val_edges_false.shape, test_edges.shape, test_edges_false.shape)

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


    print("# train: {}, # val: {}, # test: {}\n".format(len(train_edges), len(val_edges), len(test_edges)))
    logging.info("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))

    eval_edge_encode, eval_PPR, eval_edge_dist_encode = data_dict['st_%d_et_%d'%(train_start_time, eval_time)]
    # print(eval_edge_encode.shape, eval_PPR.shape, eval_edge_dist_encode.shape)

    if FLAGS.supervised:
        from utils.minibatch_sup import NodeMinibatchIterator
        minibatchIterator = NodeMinibatchIterator(
            negative_mult_training=FLAGS.neg_sample_size,      # negative sample size
            graphs=graphs_train,                        # graphs (total)
            adjs=adjs_train,                            # adjs (total)
            start_time=FLAGS.train_start_time
        )
    else: # from DySAT paper
        pass

    model = load_model(FLAGS, device)
    print(model)
    if model_path:
        print('Load parameters from ', model_path)
        model.load_state_dict(torch.load(model_path), strict=False)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)

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
    best_valid_epoch_predict_true = None

    total_epoch_time = 0.0
    for epoch in range(FLAGS.num_epoches):
        model.train()
        minibatchIterator.shuffle()
        epoch_train_loss = []
        epoch_time = 0.0
        it = 0

        ### Start forward
        t = time.time()

        iter_loss = 0
        while not minibatchIterator.end():
            if FLAGS.supervised:
                train_start, train_end, pos_edges, neg_edges = minibatchIterator.next_minibatch_feed_dict()
                edge_encode, PPR, edge_dist_encode = data_dict['st_%d_et_%d'%(train_start, train_end)]

                active_nodes = np.unique(np.concatenate([pos_edges, neg_edges], axis=1))
                node_feats_np, edge_encode_np, edge_dist_encode_np, target_node_size, context_nodes_size, all_nodes_to_new_index = get_common_neighbors(active_nodes,
                                                                                                                                                        edge_encode, PPR, edge_dist_encode,
                                                                                                                                                        feats[train_end-1-train_start], FLAGS)
                node_feats_th       = torch.FloatTensor(node_feats_np).to(device)
                edge_encode_th      = torch.LongTensor(edge_encode_np).to(device)
                edge_dist_encode_th = torch.LongTensor(edge_dist_encode_np).to(device)

                ###
                output, output_unsupervised = model(node_feats_th, edge_encode_th, edge_dist_encode_th, target_node_size, context_nodes_size, device)
                loss = torch.tensor(0.0, device=device)

                if FLAGS.supervised_loss:
                    pos_edges_new = translate(pos_edges, all_nodes_to_new_index, shape='2xN')
                    neg_edges_new = translate(neg_edges, all_nodes_to_new_index, shape='2xN')
                    loss += link_forecast_loss(output, pos_edges_new, neg_edges_new, FLAGS.neg_weight, device)

                if FLAGS.unsupervised_loss:
                    sampled_temporal_edges = generate_temporal_edges(edge_encode_np, target_node_size, FLAGS.neg_sample_size)
                    # print(output_unsupervised.shape)
                    for t, (pos_edges, neg_edges) in enumerate(sampled_temporal_edges):
                        loss += link_forecast_loss(output_unsupervised[t, :, :], pos_edges, neg_edges, FLAGS.neg_weight, device)/output_unsupervised.size(0)

            else:
                pass

            iter_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FLAGS.max_gradient_norm)
            optimizer.step()

        epoch_train_loss.append(iter_loss)
        # track current time
        epoch_time += time.time() - t
        # logging
        logging.info("Mini batch Iter: {} train_loss= {:.5f}".format(it, loss.data.item()))
        logging.info("Time for Mini batch Iter {}: {}".format(it, time.time() - t))
        it += 1

        epoch_train_loss_all.append(np.mean(epoch_train_loss))
        print("Train Loss at epoch {} : {}".format(epoch, np.mean(epoch_train_loss)))
        print("Time for epoch ", epoch_time)
        total_epoch_time += epoch_time
        logging.info("Time for epoch : {}".format(epoch_time))


        # Validation in training:
        if (epoch + 1) % FLAGS.test_freq == 0:
            model.eval()                                # disable dropout in model

            emb = []
            for cur_activate_eval_nodes in activate_eval_nodes_splits:

                node_feats_np, edge_encode_np, edge_dist_encode_np, target_node_size, context_nodes_size, eval_all_nodes_to_new_index  = get_common_neighbors(cur_activate_eval_nodes,
                                                                                                                                                              eval_edge_encode, eval_PPR, eval_edge_dist_encode,
                                                                                                                                                              feats[-1], FLAGS, deterministic_neighbor_sampling=True)
                node_feats_th       = torch.FloatTensor(node_feats_np).to(device)
                edge_encode_th  = torch.LongTensor(edge_encode_np).to(device)
                edge_dist_encode_th = torch.LongTensor(edge_dist_encode_np).to(device)

                #########
                output, _ = model(node_feats_th, edge_encode_th, edge_dist_encode_th, target_node_size, context_nodes_size, device)
                emb.append(output[:target_node_size, :].detach().cpu().numpy())
            emb = np.concatenate(emb, axis=0)

            # Use external classifier to get validation and test results.

            val_results, test_results, val_pred_true, test_pred_true = evaluate_classifier(
                train_edges_new, train_edges_false_new,
                val_edges_new,   val_edges_false_new,
                test_edges_new,  test_edges_false_new,
                emb, emb)

            val_HAD = val_results["HAD"][0]
            test_HAD = test_results["HAD"][0]
            val_SIGMOID = val_results["SIGMOID"][0]
            test_SIGMOID = test_results["SIGMOID"][0]

            print("Epoch %d, Val AUC_HAD %.4f, Test AUC_HAD %.4f, Val AUC_SIGMOID %.4f, Test AUC_SIGMOID %.4f, "%(epoch, val_HAD, test_HAD, val_SIGMOID, test_SIGMOID))
            logging.info("Epoch %d, Val AUC_HAD %.4f, Test AUC_HAD %.4f, Val AUC_SIGMOID %.4f, Test AUC_SIGMOID %.4f, "%(epoch, val_HAD, test_HAD, val_SIGMOID, test_SIGMOID))

            epochs_test_result["HAD"].append(test_HAD)
            epochs_val_result["HAD"].append(val_HAD)
            epochs_test_result["SIGMOID"].append(test_SIGMOID)
            epochs_val_result["SIGMOID"].append(val_SIGMOID)

            if val_HAD > best_valid_result:
                best_valid_result = val_HAD
                best_valid_epoch = epoch
                best_valid_epoch_predict_true = val_pred_true["HAD"],
                torch.save(model.state_dict(), best_valid_model_path)

            if epoch - best_valid_epoch > 100: # FLAGS.num_epoches/2:
                break

    """
    Done training: choose best model by validation set performance.
    """
    best_epoch = epochs_val_result["HAD"].index(max(epochs_val_result["HAD"]))
    logging.info("Total used time is: {}\n".format(total_epoch_time))
    print("Total used time is: {}\n".format(total_epoch_time))

    print("Best epoch ", best_epoch)
    logging.info("Best epoch {}".format(best_epoch))

    val_results, test_results = epochs_val_result["HAD"][best_epoch], epochs_test_result["HAD"][best_epoch]

    print("Best epoch val results {}".format(val_results))
    print("Best epoch test results {}".format(test_results))

    logging.info("Best epoch val results {}\n".format(val_results))
    logging.info("Best epoch test results {}\n".format(test_results))

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
        'epoch_train_loss': epoch_train_loss_all
    }

    with open(os.path.join(output_dir, 'result_{}.json'.format(FLAGS.dataset)), 'w') as outfile:
        json.dump(result, outfile)
    np.save(os.path.join(output_dir, 'test_pred_true.npy'), np.array(best_valid_epoch_predict_true))


def get_common_neighbors(active_nodes, edge_encode_list, PPR, edge_dist_encode, feats, FLAGS, deterministic_neighbor_sampling=None):

    if deterministic_neighbor_sampling == None:
        deterministic_neighbor_sampling = FLAGS.deterministic_neighbor_sampling

    all_nodes, all_nodes_to_new_index, target_node_size_1, context_nodes_size_1 = sample_joint_neighbors(active_nodes, PPR, 
                                                                                                     deterministic_neighbor_sampling, 
                                                                                                     FLAGS.two_steam_model,
                                                                                                     FLAGS.max_neighbors)
    # print(target_node_size_1, context_nodes_size_1)
    all_node_size = target_node_size_1 + context_nodes_size_1

    ###
    time_size = len(edge_encode_list)

    edge_encode_np_1 = np.zeros((all_node_size, all_node_size, time_size))
    for t in range(time_size):
        edge_encode_np_1[:, :, t] = 2*t
        edge_encode_tmp = edge_encode_list[t]
        edge_encode_tmp = edge_encode_tmp[all_nodes, :][:, all_nodes].tocoo()
        edge_encode_np_1[edge_encode_tmp.row, edge_encode_tmp.col, t] = 2*t+1


    ###
    edge_dist_encode_np_1 = np.ones((all_node_size, all_node_size)) * (FLAGS.max_dist+1)
    edge_dist_encode_tmp = edge_dist_encode[all_nodes, :][:, all_nodes].tocoo()
    edge_dist_encode_np_1[edge_dist_encode_tmp.row, edge_dist_encode_tmp.col] = edge_dist_encode_tmp.data


    # print(FLAGS.max_dist+1, np.max(edge_dist_encode_tmp.data), np.max(edge_dist_encode_np_1))
    ###
    node_feats_np_1 = feats[all_nodes].toarray()
    # print('>>>',node_feats_th.size(), edge_encode_th.size(), edge_dist_encode_th.size(), target_node_size_1, context_nodes_size_1)
    return node_feats_np_1, edge_encode_np_1, edge_dist_encode_np_1, target_node_size_1, context_nodes_size_1, all_nodes_to_new_index

if __name__ == '__main__':

    """
    Run GraphTransformer (our proposal)
    """

    model_name = 'DyGraphTransformer_two_stream'

    FLAGS = flags()
    FLAGS.supervised=True
    FLAGS.supervised_loss=True
    FLAGS.unsupervised_loss=False

    FLAGS.model_name = model_name

    FLAGS = update_args(FLAGS)
    FLAGS.max_neighbors = -1
    FLAGS.deterministic_neighbor_sampling = True

    if FLAGS.dataset in ['Enron_16', 'Enron_92']:

        FLAGS.two_steam_model = False
    else:
        FLAGS.two_steam_model = True


    for time_step in reversed(range(FLAGS.min_time+1, FLAGS.max_time)):
        FLAGS.time_step = time_step

        """
        Setup layers and res_id
        """
        # RES_ID = '%d_layer_model'%FLAGS.num_layers
        RES_ID = '%d_layer_model_window_%d'%(FLAGS.num_layers, FLAGS.window)

        device = get_device(FLAGS)
        # # load graphs
        graphs, adjs = load_graphs(FLAGS.dataset)

        for seed in [123, 321, 231]:
            FLAGS.seed = seed
            train_current_time_step(FLAGS, graphs, adjs, device, RES_ID)
            torch.cuda.empty_cache()
