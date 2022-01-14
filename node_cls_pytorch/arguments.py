import argparse
import yaml
import os
from datetime import datetime

###############################################################################
###############################################################################
###############################################################################
def flags():
    FLAGS = argparse.ArgumentParser(description='Dynamic Graph Neural Network Model Arguments')

    # general setup
    FLAGS.add_argument("--seed", type=int, nargs='?', default=123, help="seed for pytorch and numpy")
    FLAGS.add_argument("--GPU_ID", type=int, nargs='?', default=0, help="GPU to use (if available)")

    # model training setup
    FLAGS.add_argument("--model_name", type=str, nargs='?', default='DySAT',help="Select model from XXXXXXX")
    FLAGS.add_argument("--dataset", type=str, nargs='?', default='Enron_16', help="Select Dataset for model to run.")
    FLAGS.add_argument("--num_epoches", type=int, nargs='?', default=200, help="# of epoches to train the model")
    FLAGS.add_argument("--batch_size", type=int, nargs='?', default=256, help="# of nodes to sample for each training epoch")
    FLAGS.add_argument("--eval_batch_size", type=int, nargs='?', default=2**32-1, help="# of nodes to sample for each training epoch")
    FLAGS.add_argument("--feature_less", type=str2bool, nargs='?', default=True, help="Use one-hot IDs instead of feature attributes")
    FLAGS.add_argument("--supervised", type=str2bool, nargs='?', default=True, help="Supervised/Un-supervised learning")
    FLAGS.add_argument("--use_contrastive", type=str2bool, nargs='?', default=False, help="Use contrastive learning learning")
    FLAGS.add_argument("--use_edge_reconstruct", type=str2bool, nargs='?', default=False, help="Use edge reconstruction learning")
    FLAGS.add_argument("--use_siamese_loss", type=str2bool, nargs='?', default=False, help="Use siamese contrastive learning")
    FLAGS.add_argument("--use_memory_net", type=str2bool, nargs='?', default=False, help="Use memory network for upgrade (True not working)")

    FLAGS.add_argument('--window', type=int, nargs='?', default=10, help='Window for temporal attention (default : -1 => full)')
    FLAGS.add_argument("--test_freq", type=int, nargs='?', default=1, help="Test frequency")

    # model saving (why csv_dir and mdl_dir?)
    FLAGS.add_argument("--save_dir", type=str, nargs='?', default="output", help="directory to save the final output node embeddings")
    FLAGS.add_argument("--force_regen", type=str2bool, nargs='?', default=False, help="force regenerate context nodes and train/valid/test split")

    # overall folder identifer string ###
    FLAGS.add_argument('--res_id', type=str, nargs='?', default=str(datetime.now().strftime("%m_%d_%Y_%H_%M_%S")), help='Unique output parent dir ID')

    # General hyper-parameters
    FLAGS.add_argument("--learning_rate", type=float, nargs='?', default=1e-3, help="adam optimizer learning rate")
    FLAGS.add_argument("--weight_decay", type=float, nargs='?', default=5e-4, help="adam optimizer regularization coef")
    FLAGS.add_argument("--max_gradient_norm", type=float, nargs='?', default=1.0, help="Clip gradients to this norm")

    # DySAT evaluate hyper-parameters
    FLAGS.add_argument("--neg_sample_size", type=int, nargs='?', default=10, help="# of negative samples for each batch sampling")
    FLAGS.add_argument("--walk_length", type=int, nargs='?', default=40, help="random walk sampling length")

    FLAGS.add_argument("--time_step", type=int, nargs='?', default=0, help="Which time step to run")
    FLAGS.add_argument("--patient_iters", type=int, nargs='?', default=20, help="Train iters before step")

    # Node classification
    FLAGS.add_argument("--use_edge_feats", type=str2bool, nargs='?', default=True)
    FLAGS.add_argument("--linear_cls_input_size", type=int, nargs='?', default=256)
    FLAGS.add_argument("--use_pretrain", type=int, nargs='?', default=1)
    
    FLAGS = FLAGS.parse_args()
    return FLAGS

###############################################################################
###############################################################################
###############################################################################

def update_args(FLAGS):
    config_file = './configs/%s_%s.yaml'%(FLAGS.model_name, FLAGS.dataset)

    # Load configs
    if os.path.exists(config_file):
        with open(config_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in data.items():
            FLAGS.__dict__[key] = value
        print('Update args from %s'%config_file)

    return FLAGS

###############################################################################
###############################################################################
###############################################################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

###############################################################################
###############################################################################
###############################################################################
