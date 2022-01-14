import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import *

# class NodeMinibatchIterator:
#     def __init__(self, train_nodes, batch_size=512):
        
#         self.train_nodes = train_nodes
#         self.num_nodes = len(train_nodes)
        
#         self.batch_size = batch_size # number of nodes
        
#         self.cur_train_step = 0
#         self.batch_num = 0
        
#         print('Initializae NodeMinibatchIterator')
        
#     def end(self, ):
#         # stop only if the mini-batch is too small
        
#         # return (self.batch_num) * self.batch_size - self.num_nodes < 10
#         return (self.batch_num) * self.batch_size >= self.num_nodes

#     def next_minibatch_feed_dict(self,):
        
#         start_idx = self.batch_num * self.batch_size
#         end_idx   = (self.batch_num +1) * self.batch_size
        
#         mini_batch = self.train_nodes[start_idx : end_idx]    
#         self.batch_num += 1
        
#         return mini_batch

#     def shuffle(self, ):
#         self.batch_num = 0
        
#         rand_perm = np.random.permutation(self.num_nodes)
#         self.train_nodes = self.train_nodes[rand_perm]
        
# class NodeMinibatchIterator:
#     def __init__(self, x_train_list, y_train_list, 
#                  x_val_list, y_val_list, 
#                  x_test, y_test, 
#                  pos_weight):
        
#         self.x_train_list = x_train_list
#         self.y_train_list = y_train_list
#         self.x_val_list   = x_val_list
#         self.y_val_list   = y_val_list
#         self.x_test       = x_test
#         self.y_test       = y_test
#         self.pos_weight   = pos_weight
        
#         self.cur_time_step = 0
#         self.num_time_steps = len(x_train_list)
        
#         print('Initializae NodeMinibatchIterator')
        
#     def end(self, ):
#         return self.cur_time_step >= self.num_time_steps

#     def next_minibatch_feed_dict(self,):
#         self.cur_time_step += 1
        
#         return self.x_train_list[self.cur_time_step-1], self.y_train_list[self.cur_time_step-1], \
#                self.x_val_list[self.cur_time_step-1], self.y_val_list[self.cur_time_step-1], \
#                self.cur_time_step-1

#     def shuffle(self, ):
#         self.cur_time_step = 0
        
class NodeMinibatchIterator:
    def __init__(self, x_train_list, y_train_list, 
                 x_val, y_val, 
                 x_test, y_test, 
                 pos_weight):
        
        self.x_train_list = x_train_list
        self.y_train_list = y_train_list
        self.x_val        = x_val
        self.y_val        = y_val
        self.x_test       = x_test
        self.y_test       = y_test
        self.pos_weight   = pos_weight
        
        self.cur_time_step = 0
        self.num_time_steps = len(x_train_list)
        
        print('Initializae NodeMinibatchIterator')
        
    def end(self, ):
        return self.cur_time_step >= self.num_time_steps

    def next_minibatch_feed_dict(self,):
        self.cur_time_step += 1
        return self.x_train_list[self.cur_time_step-1], self.y_train_list[self.cur_time_step-1], self.cur_time_step-1

    def shuffle(self, ):
        self.cur_time_step = 0
        
