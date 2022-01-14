from train_inits import *
from train_models_graph_trans import train_current_time_step as finetune_train_current_time_step

if __name__ == '__main__':

    model_name = 'DyGraphTransformer_two_stream'

    FLAGS = flags()

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
    
    for seed in [321]:
        FLAGS.seed = seed

        # finetuning
        FLAGS.supervised=True
        FLAGS.supervised_loss=True
        FLAGS.unsupervised_loss=False
        
        RES_ID = 'directly_train_%d_layer_model_downstream_v1'%FLAGS.num_layers
        finetune_train_current_time_step(FLAGS, graphs, adjs, device, RES_ID)
        torch.cuda.empty_cache()

        
