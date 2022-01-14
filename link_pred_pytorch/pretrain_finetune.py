from train_inits import *
from pretrain_models_graph_trans import train_current_time_step as pretrain_train_current_time_step
from train_models_graph_trans import train_current_time_step as finetune_train_current_time_step


def get_pretrain_model_path(FLAGS, res_id, seed=None):
    if seed == None:
        seed = FLAGS.seed
        
    if res_id is None:
        FLAGS.res_id = 'Final_%s_%s_seed_%d_time_%d'%(FLAGS.model_name, FLAGS.dataset, seed, FLAGS.time_step)
    else:
        FLAGS.res_id = 'Final_%s_%s_seed_%d_time_%d_resid_%s'%(FLAGS.model_name, FLAGS.dataset, seed, FLAGS.time_step, res_id)
        
    output_dir = './all_logs/%s'%(FLAGS.model_name)
    return os.path.join(output_dir, FLAGS.res_id , 'best_valid_model_{}.pt'.format(FLAGS.dataset))

if __name__ == '__main__':

    model_name = 'DyGraphTransformer_two_stream'

    FLAGS = flags()
    FLAGS.num_epoches = 200
    FLAGS.model_name = model_name    

    FLAGS = update_args(FLAGS)
    FLAGS.max_neighbors = -1
    FLAGS.deterministic_neighbor_sampling = True
    
    if FLAGS.dataset in ['Enron_16', 'Enron_92', 'RDS_100']:

        FLAGS.two_steam_model = False
    else:
        FLAGS.two_steam_model = True

    FLAGS.num_layers = 2
    """
    Setup layers and res_id
    """
    RES_ID = 'pretrain_%d_layer_model'%FLAGS.num_layers

    device = get_device(FLAGS)
    # # load graphs
    graphs, adjs = load_graphs(FLAGS.dataset)


    FLAGS.force_regen = False

    FLAGS.supervised=False
    FLAGS.supervised_loss=False
    FLAGS.unsupervised_loss=True

    RES_ID = 'pretrain_%d_layer_model_v2'%FLAGS.num_layers
    best_valid_model_path = get_pretrain_model_path(FLAGS, RES_ID, seed=123)
    if os.path.exists(best_valid_model_path)==False:
        print(best_valid_model_path, 'does not exist')
        best_valid_model_path = pretrain_train_current_time_step(FLAGS, graphs, adjs, device, RES_ID)

    torch.cuda.empty_cache()

    FLAGS.force_regen = False

    # finetuning
    FLAGS.supervised=True
    FLAGS.supervised_loss=True
    FLAGS.unsupervised_loss=False

    RES_ID = 'finetune_%d_layer_model_downstream_v2'%FLAGS.num_layers
    finetune_train_current_time_step(FLAGS, graphs, adjs, device, RES_ID, model_path=best_valid_model_path)
    torch.cuda.empty_cache()

        
