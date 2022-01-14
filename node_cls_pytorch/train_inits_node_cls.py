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

import numpy as np
import scipy
from scipy.sparse import vstack
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from utils.classification_preprocess import *

from evaluate.node_classification import cls_evaluate_classifier

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
def edge_feats_encoding(G):
    coords = G.edges()
    edge_feats = np.array([G[coord[0]][coord[1]]['feat'] for coord in coords])
    
    num_nodes = len(G.nodes)
    edge_feat_ind = np.zeros((num_nodes, num_nodes), np.int32)

    for i, coord in enumerate(coords):
        edge_feat_ind[coord[0], coord[1]] = i
    return edge_feats, edge_feat_ind

#####################################################################################
#####################################################################################
#####################################################################################
class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.5):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 10)
        self.fc_2 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        return self.fc_2(x).squeeze(dim=1)

#####################################################################################
#####################################################################################
#####################################################################################
def cls_get_evaluation_data(graphs):
    
    train_ids_list = []
    train_labels_list = []
    
    # prepare train and valid
    for t, graph in enumerate(graphs[:-1]):
        ids, labels = cls_extract_node_labels(graph)
        ids = ids[labels!=-1]
        labels = labels[labels!=-1]
        
        pos_select = np.where(labels==1)[0]
        neg_select = np.where(labels==0)[0]
        neg_select = np.random.permutation(neg_select)[:len(pos_select)]
        
        all_select = np.concatenate([pos_select, neg_select])
        ids = ids[all_select]
        labels = labels[all_select]
    
        train_ids_list.append(ids)
        train_labels_list.append(labels)
        
    # the last graph is for testing
    ids, labels = cls_extract_node_labels(graphs[-1])
    ids = ids[labels!=-1]
    labels = labels[labels!=-1]
    
    pos_select = np.where(labels==1)[0]
    neg_select = np.where(labels==0)[0]
    neg_select = np.random.permutation(neg_select)[:len(pos_select)]
    
    all_select = np.concatenate([pos_select, neg_select])
    ids = ids[all_select]
    labels = labels[all_select]
    
    x_test, x_valid, y_test, y_valid = train_test_split(ids, labels, stratify=labels, train_size=0.8)
    
    num_pos, num_neg = 0, 0
    for labels in train_labels_list:
        num_pos += np.sum(labels==1)
        num_neg += np.sum(labels==0)
    
    # print size
    for cur_x_train, cur_y_train in zip(train_ids_list, train_labels_list):
        print('train', cur_x_train.shape, cur_y_train.shape)
    print('valid', x_valid.shape, y_valid.shape)
    print('test', x_test.shape, y_test.shape)

    return train_ids_list, train_labels_list, x_valid, y_valid, x_test, y_test, num_neg/num_pos

###############################################
###############################################
###############################################
def cls_get_evaluation_data_v2(graphs):
    
    train_ids_list = []
    train_labels_list = []
    
    # prepare train and valid
    for t, graph in enumerate(graphs[:-1]):
        ids, labels = cls_extract_node_labels(graph)
        ids = ids[labels!=-1]
        labels = labels[labels!=-1]
        
        pos_select = np.where(labels==1)[0]
        neg_select = np.where(labels==0)[0]
        neg_select = np.random.permutation(neg_select)[:len(pos_select)]
        
        all_select = np.concatenate([pos_select, neg_select])
        ids = ids[all_select]
        labels = labels[all_select]
    
        train_ids_list.append(ids)
        train_labels_list.append(labels)
        
    # the last graph is for testing
    ids, labels = cls_extract_node_labels(graphs[-1])
    ids = ids[labels!=-1]
    labels = labels[labels!=-1]
    
    pos_select = np.where(labels==1)[0]
    neg_select = np.where(labels==0)[0]
    neg_select = np.random.permutation(neg_select)[:len(pos_select)]
    
    all_select = np.concatenate([pos_select, neg_select])
    ids = ids[all_select]
    labels = labels[all_select]
    
    # train/valid/test = 60%/20%/20%

    # split train / (val + test)
    x_train, x_test, y_train, y_test = train_test_split(ids, labels, stratify=labels, train_size=0.2)
    # split val / test
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, stratify=y_test, train_size=0.2/0.8)

    num_pos, num_neg = 0, 0
    for labels in train_labels_list:
        num_pos += np.sum(labels==1)
        num_neg += np.sum(labels==0)
    
    for cur_x_train, cur_y_train in zip(train_ids_list, train_labels_list):
        print('train', cur_x_train.shape, cur_y_train.shape)
    print('>> train', x_train.shape, y_train.shape)
    print('>> valid', x_valid.shape, y_valid.shape)
    print('>> test',  x_test.shape, y_test.shape)

    return train_ids_list, train_labels_list, x_train, y_train, x_valid, y_valid, x_test, y_test, num_neg/num_pos

#####################################################################################
#####################################################################################
#####################################################################################

def compute_auc_f1(pred_prob, target):
                
    predict_01 = np.zeros_like(pred_prob)
    predict_01[pred_prob>0.5] = 1

    not_nan = ~np.isnan(pred_prob)
    if len(np.unique(target)) == 2:
        epoch_AUC = roc_auc_score(target[not_nan], pred_prob[not_nan])
        epoch_F1 = f1_score(predict_01[not_nan], target[not_nan], average='micro')
    else:
        epoch_AUC = -1
        epoch_F1 = -1
    
    return epoch_AUC, epoch_F1    

#####################################################################################
#####################################################################################
#####################################################################################

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

def update_minmax_time(dataset):
    from .arguments import dataset_to_ind, min_time_list, max_time_list
    dataset_ind =  dataset_to_ind[dataset]
    return min_time_list[dataset], max_time_list[dataset]

