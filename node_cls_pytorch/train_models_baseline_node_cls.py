
from train_inits_node_cls import *


def train_current_time_step(FLAGS, graphs, adjs, device, res_id=None):

    FLAGS.use_torch_linear = False

    ###########################################################
    ###########################################################
    ###########################################################
    # Set random seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)

    # Create log dir
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

    ###########################################################
    ###########################################################
    ###########################################################
    # prepare the train + test graphs
    graphs_train = graphs[train_start_time: eval_time]
    graphs_train = list(map(lambda graph: cls_process_graph(graph), graphs_train))
    adjs_train = adjs[train_start_time: eval_time]
    
    graphs_eval = graphs[eval_time-1]

    # print graph info
    print('Train graphs size:')
    for graph in graphs_train:
        print("Node %d, edges %d"%(len(graph.nodes), len(graph.edges)))

    print('Eval graphs size:')
    print("Node %d, edges %d"%(len(graphs_eval.nodes), len(graphs_eval.edges)))

    # compute norm adjs
    norm_adjs_train = [cls_normalize_graph_gcn(adj) for adj in adjs_train]
    norm_adjs_train = [cls_tuple_to_sparse(adj, torch.float32).to(device) for adj in norm_adjs_train]

    ###########################################################
    ###########################################################
    ###########################################################
    # Load node feats
    num_nodes_graph_eval = len(graphs_eval.nodes)

    feats_train          = cls_get_feats(adjs_train, num_nodes_graph_eval, 0, eval_time, FLAGS)
    feats_train          = [cls_preprocess_features(feat)[1] for feat in feats_train]
    feats_train          = [cls_tuple_to_sparse(feat, torch.float32).to(device) for feat in feats_train]

    edge_feats_train     = [cls_extract_edge_features_dense(graph) for graph in graphs_train]
    edge_feats_train     = [torch.from_numpy(edge_feat).to(torch.float32).to(device) for edge_feat in edge_feats_train]

    FLAGS.num_features      = feats_train[0].shape[1]
    FLAGS.num_edge_features = edge_feats_train[0].shape[1]    
    
    ###########################################################
    ###########################################################
    ###########################################################
    # prepare train, valid, test data
    if FLAGS.use_torch_linear:
        x_train_list, y_train_list, x_val, y_val, x_test, y_test, pos_weight = cls_get_evaluation_data(graphs_train)
    else:
        x_train_list, y_train_list, x_train, y_train, x_val, y_val, x_test, y_test, pos_weight = cls_get_evaluation_data_v2(graphs_train)

    pos_weight =  torch.tensor(pos_weight)
    print('pos_weight', pos_weight)

    ###########################################################
    ###########################################################
    ###########################################################
    # Setup minibatchsampler
    from utils.minibatch_node_cls import NodeMinibatchIterator
    minibatchIterator = NodeMinibatchIterator(x_train_list, y_train_list)

    # Setup model and optimizer
    model = load_model(FLAGS, device)
    print(model)

    linear_cls = LR(FLAGS.linear_cls_input_size).to(device)
    print(linear_cls)
    
    optimizer = optim.Adam(list(model.parameters())+list(linear_cls.parameters()), 
                           lr=FLAGS.learning_rate, 
                           weight_decay=FLAGS.weight_decay)

    ###########################################################
    ###########################################################
    ###########################################################
    # Setup result accumulator variables.
    epochs_test_result = []
    epochs_val_result  = []

    """
    Training starts
    """
    epoch_train_loss_all = []
    
    best_valid_result = 0
    best_valid_epoch  = 0

    best_valid_model_path = os.path.join(output_dir, 'best_valid_model_{}.pt'.format(FLAGS.dataset))

    total_epoch_time = 0.0
    for epoch in range(FLAGS.num_epoches):

        ############################################################################
        ######################### Training #########################################
        ############################################################################
        model.train()
        minibatchIterator.shuffle()
        epoch_train_loss = []
        epoch_time = 0.0

        it = 0

        while not minibatchIterator.end():
            t = time.time()
            
            # sample mini-batch
            cur_x_train, cur_y_train, cur_time_step = minibatchIterator.next_minibatch_feed_dict()

            # forward-propagation
            if FLAGS.use_edge_feats:
                output = model(feats_train, norm_adjs_train, edge_feats_train, device)
            else:
                output = model(feats_train, norm_adjs_train, device)
            
            if isinstance(output, list):
                assert FLAGS.cur_window == len(output)
                mini_batch_pred = linear_cls(output[cur_time_step][cur_x_train, :])
            else:
                assert FLAGS.cur_window == output.shape[1]     
                mini_batch_pred = linear_cls(output[cur_x_train, cur_time_step, :])
                
            mini_batch_y = torch.FloatTensor(cur_y_train).flatten().to(device)
            loss = F.binary_cross_entropy_with_logits(mini_batch_pred, mini_batch_y, pos_weight)
    
            all_pred_np = torch.sigmoid(mini_batch_pred).detach().cpu().numpy()
            all_targ_np = mini_batch_y.cpu().numpy()
            print('train AUC / F1', compute_auc_f1(all_pred_np, all_targ_np))

            # backward-propagation
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), FLAGS.max_gradient_norm)
            nn.utils.clip_grad_norm_(linear_cls.parameters(), FLAGS.max_gradient_norm)

            optimizer.step()
            epoch_train_loss.append(loss.item())
            
            # track current time
            epoch_time += time.time() - t
            it += 1

        epoch_train_loss_all.append(np.mean(epoch_train_loss))
        ############################################################################
        ######################### Testing ##########################################
        ############################################################################
        if (epoch + 1) % FLAGS.test_freq == 0:
            model.eval()                                # disable dropout in model
            
            if FLAGS.use_edge_feats:
                output = model(feats_train, norm_adjs_train, edge_feats_train, device)
            else:
                output = model(feats_train, norm_adjs_train, device)
                
            if isinstance(output, list):
                assert FLAGS.cur_window == len(output)
                output = output[-1]
            else:
                assert FLAGS.cur_window == output.shape[1]
                output = output[:, -1, :]
            
            # Use external classifier to get validation and test results.
            if FLAGS.use_torch_linear:
                pred = linear_cls(output)
                pred_prob = torch.sigmoid(pred).detach().cpu().numpy()
                epoch_AUC_val,  epoch_F1_val  = compute_auc_f1(pred_prob[x_val] , y_val)
                epoch_AUC_test, epoch_F1_test = compute_auc_f1(pred_prob[x_test], y_test)
            else:
                output_np = output.detach().cpu().numpy()
                result_dict = cls_evaluate_classifier(x_train, y_train, x_val, y_val, x_test, y_test, output_np)
                epoch_AUC_val  = result_dict['val_roc_score']
                epoch_AUC_test = result_dict['test_roc_score']
                epoch_F1_val   = result_dict['val_f1_score']
                epoch_F1_test  = result_dict['test_f1_score']

            print("Epoch {}, Val AUC {},  Val F1 {}".format(epoch, epoch_AUC_val, epoch_F1_val))
            print("Epoch {}, Test AUC {}, Test F1 {}".format(epoch, epoch_AUC_test, epoch_F1_test))
            
            epochs_test_result.append(epoch_AUC_test)
            epochs_val_result.append(epoch_AUC_val)
            
            if epoch_AUC_val > best_valid_result:
                best_valid_result = epoch_AUC_val
                best_valid_epoch = epoch
                torch.save(model.state_dict(), best_valid_model_path)

            if epoch - best_valid_epoch > FLAGS.patient_iters: 
                break

    """
    Done training: choose best model by validation set performance.
    """
    best_epoch = best_valid_epoch
    val_results, test_results = epochs_val_result[best_valid_epoch], epochs_test_result[best_valid_epoch] 

    print("Total used time is: {}\n".format(total_epoch_time))
    print("Best epoch ", best_valid_epoch)
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
        'valid_epoch_AUC': epochs_val_result,
        'test_epoch_AUC':  epochs_test_result,
        'epoch_train_loss': epoch_train_loss_all
    }

    with open(os.path.join(output_dir, 'result_{}.json'.format(FLAGS.dataset)), 'w') as outfile:
        json.dump(result, outfile)

    return result

if __name__ == '__main__':

    linear_cls_input_size_dict = {
        'EvolveGCN_O': 256,
        'GAT_RNN': 128,
        'DySAT': 128,
    }

    use_edge_feats = {
        'DySAT': False,
        'EvolveGCN_O': False,
        'GAT_RNN': False,
    }

    #########################################################################
    #########################################################################
    #########################################################################
    FLAGS = flags()
    
    # FLAGS.model_name = 'DySAT'
    # FLAGS.dataset = 'wiki_classification'
    # FLAGS.time_step = 12
    # FLAGS.seed = 123

    FLAGS = update_args(FLAGS)
    FLAGS.window = 5
    ########################################
    FLAGS.num_epoches = 200
    FLAGS.patient_iters = 20
    
    FLAGS.use_edge_feats = use_edge_feats[FLAGS.model_name]
    FLAGS.linear_cls_input_size = linear_cls_input_size_dict[FLAGS.model_name]
    ########################################
    graphs, adjs = cls_load_graphs(FLAGS.dataset)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RES_ID = 'node_cls'
    ########################################

    print(device, torch.cuda.is_available() )
    results = train_current_time_step(FLAGS, graphs, adjs, device, RES_ID)
        