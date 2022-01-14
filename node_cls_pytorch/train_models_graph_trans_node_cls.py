
from train_inits_node_cls import *
from utils.dynamic_graph_transformer_utils import *

def train_current_time_step(FLAGS, graphs, adjs, device, res_id=None, model_path=None):
    """
    Setup
    """
    FLAGS.unsupervised_loss = False
    FLAGS.use_torch_linear = False

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
    # prepare the train + test graphs
    graphs_train = graphs[train_start_time: eval_time]
    graphs_train = list(map(lambda graph: cls_process_graph(graph), graphs_train))
    adjs_train = adjs[train_start_time: eval_time]
    
    graphs_eval = graphs[eval_time] 

    # print graph info
    print('Train graphs size:')
    for graph in graphs_train:
        print("Node %d, edges %d"%(len(graph.nodes), len(graph.edges)))

    print('Eval graphs size:')
    print("Node %d, edges %d"%(len(graphs_eval.nodes), len(graphs_eval.edges)))

    ###########################################################
    ###########################################################
    ###########################################################
    # Load node feats
    num_nodes_graph_eval = len(graphs_eval.nodes)

    feats_train          = cls_get_feats(adjs_train, num_nodes_graph_eval, 0, eval_time, FLAGS) 

    edge_feat_encoding_list, edge_feat_ind_list = [], []
    for G in  graphs_train:
        edge_feats, edge_feat_ind = edge_feats_encoding(G)
        edge_feat_encoding_list.append(edge_feats)
        edge_feat_ind_list.append(edge_feat_ind)

    FLAGS.num_features      = feats_train[0].shape[1]
    FLAGS.num_edge_features = edge_feat_encoding_list[0].shape[1] 

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
    ############ Used for DyGraphTransformer ##################
    ###########################################################
    data_dict = compute_node_edge_encoding(graphs_train, FLAGS, force_regen=FLAGS.force_regen)
    print(data_dict.keys())

    # only need to load once
    eval_edge_encode, eval_PPR, eval_edge_dist_encode = data_dict['st_%d_et_%d'%(train_start_time, eval_time)] 
    
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

    if model_path:
        print('Load parameters from ', model_path)
        model.load_state_dict(torch.load(model_path), strict=False)
    
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
    best_valid_epoch = 0

    best_valid_model_path = os.path.join(output_dir, 'best_valid_model_{}.pt'.format(FLAGS.dataset))
    best_valid_epoch_predict_true = None
    
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
            
            # get spatial-temporal encoding
            edge_encode, PPR, edge_dist_encode = data_dict['st_%d_et_%d'%(train_start_time, train_start_time+1+cur_time_step)]

            cur_x_train_with_pad = pad_more_nodes(cur_x_train, num_all_nodes=PPR.shape[0])
            cur_x_train_size = len(cur_x_train)

            node_feats_np, edge_encode_np, edge_dist_encode_np, target_node_size, context_nodes_size, all_nodes_to_new_index = get_common_neighbors(cur_x_train_with_pad,
                                                                                                                                                    edge_encode, PPR, 
                                                                                                                                                    edge_dist_encode,
                                                                                                                                                    feats_train[cur_time_step], FLAGS)            
            node_feats_th       = torch.FloatTensor(node_feats_np).to(device)
            edge_encode_th      = torch.LongTensor(edge_encode_np).to(device)
            edge_dist_encode_th = torch.LongTensor(edge_dist_encode_np).to(device)
            
            # forward propagation
            output, _ = model(node_feats_th, edge_encode_th, edge_dist_encode_th, target_node_size, context_nodes_size, device)
            mini_batch_pred = linear_cls(output[:cur_x_train_size, :])
            
            
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
        
            model.eval()                          
    
            eval_nodes = np.concatenate([x_train, x_val, x_test])


            eval_nodes_with_pad = pad_more_nodes(eval_nodes, num_all_nodes=eval_PPR.shape[0])
            eval_nodes_size = len(eval_nodes)

            node_feats_np, edge_encode_np, edge_dist_encode_np, target_node_size, context_nodes_size, _  = get_common_neighbors(eval_nodes_with_pad, eval_edge_encode, eval_PPR, 
                                                                                                                                eval_edge_dist_encode,
                                                                                                                                feats_train[-1], FLAGS, 
                                                                                                                                deterministic_neighbor_sampling=True)

            node_feats_th       = torch.FloatTensor(node_feats_np).to(device)
            edge_encode_th      = torch.LongTensor(edge_encode_np).to(device)
            edge_dist_encode_th = torch.LongTensor(edge_dist_encode_np).to(device)

            output, _ = model(node_feats_th, edge_encode_th, edge_dist_encode_th, target_node_size, context_nodes_size, device)
            output = output[:eval_nodes_size, :]

            ### predict with torch nn.Linear
            if FLAGS.use_torch_linear:
                pred = linear_cls(output)
                pred_prob = torch.sigmoid(pred).detach().cpu().numpy()      
                epoch_AUC_val,  epoch_F1_val  = compute_auc_f1(pred_prob[:len(x_val)], y_val)
                epoch_AUC_test, epoch_F1_test = compute_auc_f1(pred_prob[len(x_val):], y_test)
            ### predict with scikit-learn LogisticRegression
            else:
                eval_inds_train_valid_test = np.arange(len(eval_nodes))
                eval_inds_x_train = eval_inds_train_valid_test[:len(x_train)]
                eval_inds_x_val   = eval_inds_train_valid_test[len(x_train): len(x_train) + len(x_val)]
                eval_inds_x_test  = eval_inds_train_valid_test[len(x_train) + len(x_val):]

                output_np = output.detach().cpu().numpy()
                result_dict = cls_evaluate_classifier(eval_inds_x_train, y_train, eval_inds_x_val, y_val, eval_inds_x_test, y_test, output_np)
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
    np.save(os.path.join(output_dir, 'test_pred_true.npy'), np.array(best_valid_epoch_predict_true))
    return result

# sometime training node is too less, then we over-sample some node to provide larger neighborhood
def pad_more_nodes(cur_ind, num_all_nodes, expect_size=800):

    if len(cur_ind) < expect_size:
        candidate = np.setdiff1d(np.arange(num_all_nodes), cur_ind)
        padding = np.random.permutation(candidate)[:expect_size - len(cur_ind)]
        cur_ind = np.concatenate([cur_ind, padding])
        
    return cur_ind

if __name__ == '__main__':

    """
    Run GraphTransformer (our proposal)
    """
    ########################################
    FLAGS = flags()
    
    FLAGS.model_name = 'DyGraphTransformer'
    # FLAGS.dataset = 'wiki_classification'
    # FLAGS.time_step = 12
    # FLAGS.seed = 123

    FLAGS = update_args(FLAGS)
    ########################################
    
    FLAGS.max_neighbors = -1
    FLAGS.deterministic_neighbor_sampling = True

    FLAGS.two_steam_model = True
    FLAGS.window = 5
    ########################################
    FLAGS.num_epoches = 200
    FLAGS.GPU_ID = 0
    FLAGS.patient_iters = 20

    FLAGS.use_edge_feats    = False
    FLAGS.neighbor_sampling_size = 0.2
    ########################################
    graphs, adjs = cls_load_graphs(FLAGS.dataset)
    device = get_device(FLAGS)
    
    ########################################
    FLAGS.num_heads = 8
    FLAGS.num_hids = 128
    FLAGS.linear_cls_input_size = 128
    ########################################
    
    if FLAGS.use_pretrain:
        FLAGS.pre_train_seed = 123
        model_path = os.path.join(
            'all_logs/DyGraphTransformer/%s'%FLAGS.dataset,
            'Final_DyGraphTransformer_%s_seed_%d_time_%d_resid_pretrain_2_layer_model'%(FLAGS.dataset, FLAGS.pre_train_seed, FLAGS.time_step),
            'best_valid_model_%s.pt'%FLAGS.dataset,
            )
        RES_ID = 'node_cls_with_pretrain'
    else:
        model_path = None
        RES_ID = 'node_cls_without_pretrain'

    FLAGS.supervised=True
    FLAGS.supervised_loss=True
    FLAGS.unsupervised_loss=False
    results = train_current_time_step(FLAGS, graphs, adjs, device, RES_ID, model_path)