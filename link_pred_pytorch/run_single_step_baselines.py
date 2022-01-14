
from train_inits import *
from train_models_baseline import train_current_time_step


if __name__ == '__main__':

    FLAGS = flags()

    if FLAGS.model_name == 'DySAT':
        FLAGS.supervised = False # DySAT's loss
    else:
        FLAGS.supervised = True

    FLAGS = update_args(FLAGS)
    FLAGS.use_edge_feats = False
    FLAGS.use_node_cls = False
    FLAGS.node_cls_num = 0
    print(FLAGS)


    # Set device
    device = get_device(FLAGS)

    # load graphs
    graphs, adjs = load_graphs(FLAGS.dataset)
    print('Number of training graphs:', len(graphs))
    result = train_current_time_step(FLAGS, graphs, adjs, device)
