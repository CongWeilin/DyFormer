from __future__ import division
from __future__ import print_function

import os
import sys
import json
import time
import copy
import random
import logging
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import scipy
from scipy.sparse import vstack
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.preprocess import load_graphs, load_feats, get_feats
from utils.preprocess import sparse_to_tuple, tuple_to_sparse
from utils.preprocess import preprocess_features, normalize_graph_gcn, update_eval_graph
from utils.preprocess import get_context_pairs, create_data_splits, get_evaluation_data
from utils.dytrans_compact_utils import create_compact_graph, get_compact_adj_edges, align_output
from evaluate.link_prediction import evaluate_classifier

from model.losses import link_pred_loss, link_forecast_loss
from model.load_model import load_model

from arguments import flags, update_args

#####################################################################################
#####################################################################################
#####################################################################################

def create_save_path(FLAGS):
    output_dir = "./all_logs/{}/{}/{}".format(FLAGS.model_name, FLAGS.dataset, FLAGS.res_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    # Save arguments to json and txt files 
    with open(os.path.join(output_dir, 'flags_{}.json'.format(FLAGS.dataset)), 'w') as outfile:
        json.dump(vars(FLAGS), outfile)
                                                            
    return output_dir

#####################################################################################
#####################################################################################
#####################################################################################

def generate_context_pairs(graphs, adjs, FLAGS):
    for time_step in range(FLAGS.min_time, FLAGS.max_time+1):
        FLAGS.time_step = time_step
        cur_window, train_start_time, eval_time = get_train_time_interval(FLAGS)
        FLAGS.cur_window = cur_window 
        
        graphs_train = graphs[train_start_time: eval_time+1]
        adjs_train = adjs[train_start_time: eval_time+1]
        
        context_pairs_train = get_context_pairs(graphs_train, eval_time, FLAGS.cur_window+1, FLAGS.dataset, force_regen=True)
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_evaluation_data(adjs_train[-2], adjs_train[-1], FLAGS.time_step, FLAGS.dataset, force_regen=True)
        print('{} at timestep {} finished processing train & eval data with window size'.format(FLAGS.dataset, 
                                                                                                str(time_step), 
                                                                                                str(FLAGS.cur_window)))

def get_train_time_interval(FLAGS,):
    if FLAGS.window < 1 or FLAGS.window > FLAGS.time_step-1:
        window = FLAGS.time_step-1
    else:
        window = FLAGS.window
        
    train_start_time = FLAGS.time_step-1-window
    eval_start_time  = FLAGS.time_step-1

    print('Predict the %d-th graph (start from 0) using %d historical graphs'%(eval_start_time, window))
    print('Using graph {}'.format(np.arange(train_start_time, eval_start_time)))
    return window, train_start_time, eval_start_time

def get_device(FLAGS):
    if FLAGS.GPU_ID != -1:
        device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    return device


def create_logger(FLAGS):
    output_dir = "./all_logs/{}/{}/".format(FLAGS.model_name, FLAGS.res_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    # Save arguments to json and txt files 
    with open(os.path.join(output_dir, 'flags_{}.json'.format(FLAGS.dataset)), 'w') as outfile:
        json.dump(vars(FLAGS), outfile)
        
    # Set unique identifer
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    today = datetime.today()

    # Setup logging
    
    log_file = os.path.join(output_dir, '%s_%s_%s_%s_%s.log' % (FLAGS.dataset, 
                                                            str(FLAGS.time_step), 
                                                            str(today.year), 
                                                            str(today.month), 
                                                            str(today.day)))
                                                            
    return log_file, output_dir

def update_minmax_time(dataset):
    from .arguments import dataset_to_ind, min_time_list, max_time_list
    dataset_ind =  dataset_to_ind[dataset]
    return min_time_list[dataset], max_time_list[dataset]

