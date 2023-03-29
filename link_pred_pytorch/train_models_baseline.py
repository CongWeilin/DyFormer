

from train_inits import *


def train_current_time_step(FLAGS, graphs, adjs, device, res_id=None):
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

    FLAGS.cur_window = cur_window
    FLAGS.eval_time         = eval_time
    FLAGS.train_start_time  = train_start_time

    graphs_train = graphs[train_start_time: eval_time]
    adjs_train = adjs[train_start_time: eval_time]
    graphs_eval = graphs[eval_time]
    adjs_eval = adjs[eval_time]

    norm_adjs_train = [normalize_graph_gcn(adj) for adj in adjs_train]

    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        get_evaluation_data(adjs_train[-1], adjs_eval, FLAGS.time_step, FLAGS.dataset, force_regen=FLAGS.force_regen)

    print("# train: {}, # val: {}, # test: {}\n".format(len(train_edges), len(val_edges), len(test_edges)))
    logging.info("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))


    # Load node feats
    num_nodes_graph_eval = len(graphs_eval.nodes)
    feats = get_feats(adjs_train, num_nodes_graph_eval, train_start_time, eval_time, FLAGS)
    FLAGS.num_features = feats[0].shape[1]

    # Normalize and convert feats to sparse tuple format
    feats_train = [preprocess_features(feat) for feat in feats]
    assert len(feats_train) == len(adjs_train)
    feats_train = [tuple_to_sparse(feature, torch.float32).to(device) for feature in feats_train]
    norm_adjs_train  = [tuple_to_sparse(adj, torch.float32).to(device) for adj in norm_adjs_train]
    print('# feas_train: %d, # adjs_train: %d'%(len(feats_train), len(adjs_train)))

    # Setup minibatchsampler
    if FLAGS.supervised:
        from utils.minibatch_sup import NodeMinibatchIterator
        minibatchIterator = NodeMinibatchIterator(
            negative_mult_training=FLAGS.neg_sample_size,      # negative sample size
            graphs=graphs_train,                        # graphs (total)
            adjs=adjs_train,                            # adjs (total)
        )
    else: # from DySAT paper
        # Load training context pairs (or compute them if necessary)
        context_pairs_train = get_context_pairs(graphs_train, eval_time,
                                                FLAGS.cur_window, FLAGS.dataset,
                                                force_regen=FLAGS.force_regen)
        assert len(context_pairs_train) == FLAGS.cur_window

        from utils.minibatch import NodeMinibatchIterator
        minibatchIterator = NodeMinibatchIterator(
            window=FLAGS.cur_window,                    # training window
            neg_sample_size=FLAGS.neg_sample_size,      # negative sample size
            graphs=graphs_train,                        # graphs (total)
            batch_size=FLAGS.batch_size,                # batch size
            context_pairs=context_pairs_train,          # 1st index: time_step; second index: dic key for specific node -> its context as walk
        )

    # Setup model and optimizer
    model = load_model(FLAGS, device)
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

        while not minibatchIterator.end():
            t = time.time()
            # sample; forward; loss
            if FLAGS.supervised:
                train_start, train_end, pos_edges, neg_edges = minibatchIterator.next_minibatch_feed_dict()
                output = model(feats_train[train_start:train_end],
                               norm_adjs_train[train_start:train_end], device)

                if isinstance(output, list):
                    assert len(output) == (train_end - train_start)
                    loss = link_forecast_loss(output[-1], pos_edges, neg_edges, FLAGS.neg_weight, device)
                else:
                    assert output.shape[1] == (train_end - train_start)
                    loss = link_forecast_loss(output[:, -1, :], pos_edges, neg_edges, FLAGS.neg_weight, device)
            else:
                output = model(feats_train, norm_adjs_train, device)
                node_1_all, node_2_all, proximity_neg_samples = minibatchIterator.next_minibatch_feed_dict()
                # print(np.array(node_1_all).shape, np.array(node_2_all).shape, np.array(proximity_neg_samples).shape)
                # assert False
                assert FLAGS.cur_window == output.shape[1] == len(feats_train) == len(norm_adjs_train)
                loss = link_pred_loss(output, node_1_all, node_2_all, proximity_neg_samples, FLAGS.neg_weight, device)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FLAGS.max_gradient_norm)
            optimizer.step()
            epoch_train_loss.append(loss.item())
            # track current time
            epoch_time += time.time() - t
            # logging
            logging.info("Mini batch Iter: {} train_loss= {:.5f}".format(it, loss.data.item()))
            logging.info("Time for Mini batch Iter {}: {}".format(it, time.time() - t))
            it += 1

        epoch_train_loss_all.append(np.mean(epoch_train_loss))
        print("Time for epoch ", epoch_time)
        total_epoch_time += epoch_time
        logging.info("Time for epoch : {}".format(epoch_time))

        # Validation in training:
        if (epoch + 1) % FLAGS.test_freq == 0:
            model.eval()                                # disable dropout in model
            output = model(feats_train, norm_adjs_train, device)
            if isinstance(output, list):
                assert FLAGS.cur_window == len(output)
                emb = output[-1].detach().cpu().numpy()
            else:
                assert FLAGS.cur_window == output.shape[1]
                emb = output.detach().cpu().numpy()[:, -1, :]
            # Use external classifier to get validation and test results.

            val_results, test_results, val_pred_true, test_pred_true = evaluate_classifier(
                train_edges, train_edges_false,
                val_edges, val_edges_false,
                test_edges, test_edges_false,
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

    return result


if __name__ == '__main__':

    #########################################################################
    #########################################################################
    #########################################################################

    FLAGS = flags()

    if FLAGS.model_name == 'DySAT':
        FLAGS.supervised = False # DySAT's loss
    else:
        FLAGS.supervised = True

    FLAGS = update_args(FLAGS)
    print(FLAGS)
    # for method_name in ['DySAT']: # ['DySAT', 'GAT', 'EvolveGCN_O']:
    for time_step in reversed(range(FLAGS.min_time, FLAGS.max_time)):

        FLAGS.time_step = time_step

        for seed in [123, 231, 321]:
            FLAGS.seed = seed

            # Set device
            device = get_device(FLAGS)

            # load graphs
            graphs, adjs = load_graphs(FLAGS.dataset)
            print('Number of training graphs:', len(graphs))
            result = train_current_time_step(FLAGS, graphs, adjs, device)
