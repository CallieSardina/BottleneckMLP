"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse
import pdb

import torch
import pandas as pd
import numpy as np
#import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from module import TGIB, TGIB_with_fc
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler



### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample') ##############################
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('-exp', '--exp_type', type=str, help='exp_type', default='normal')
parser.add_argument('-f', '--filename', type=str, help='filename', default='normal')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

EXP_TYPE = args.exp_type
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
# NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
FILENAME = args.filename

print('EXP TYPE', EXP_TYPE)
print('Dataset', DATA)



def eval_one_epoch(hint, tgib, sampler, val_src_l, val_dst_l, val_ts_l, val_e_idx_l, epoch):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgib = tgib.eval()
        for k in range(1, len(val_src_l)):
            tgib = tgib.eval()
            
            u_emb_fake, i_emb_fake = sampler.sample(1)
            
            pos_prob, neg_prob, info_loss = tgib(val_src_l[:k+1], val_dst_l[:k+1], i_emb_fake, val_ts_l[:k+1], val_e_idx_l[:k+1], k, epoch, training=False, num_neighbors=NUM_NEIGHBORS)               
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(1), np.zeros(1)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)


# pdb.set_trace()
### Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA)) 
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA)) 

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85])) 

src_l = g_df.u.values 
dst_l = g_df.i.values
e_idx_l = g_df.idx.values 
label_l = g_df.label.values 
ts_l = g_df.ts.values 

max_src_index = src_l.max() 
max_idx = max(src_l.max(), dst_l.max())

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values]))) 
num_total_unique_nodes = len(total_node_set)

mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values 
mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values

none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag) 

valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)


train_src_l = src_l[valid_train_flag] 
train_dst_l = dst_l[valid_train_flag] 
train_ts_l = ts_l[valid_train_flag]  
train_e_idx_l = e_idx_l[valid_train_flag]  
train_label_l = label_l[valid_train_flag]  


train_node_set = set(train_src_l).union(train_dst_l) 
assert(len(train_node_set - mask_node_set) == len(train_node_set)) 
new_node_set = total_node_set - train_node_set 

valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
valid_test_flag = ts_l > test_time

is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
nn_val_flag = valid_val_flag * is_new_node_edge 
nn_test_flag = valid_test_flag * is_new_node_edge 


val_src_l = src_l[valid_val_flag] 
val_dst_l = dst_l[valid_val_flag] 
val_ts_l = ts_l[valid_val_flag] 
val_e_idx_l = e_idx_l[valid_val_flag] 
val_label_l = label_l[valid_val_flag] 

test_src_l = src_l[valid_test_flag] 
test_dst_l = dst_l[valid_test_flag] 
test_ts_l = ts_l[valid_test_flag] 
test_e_idx_l = e_idx_l[valid_test_flag] 
test_label_l = label_l[valid_test_flag] 


nn_val_src_l = src_l[nn_val_flag] 
nn_val_dst_l = dst_l[nn_val_flag] 
nn_val_ts_l = ts_l[nn_val_flag] 
nn_val_e_idx_l = e_idx_l[nn_val_flag] 
nn_val_label_l = label_l[nn_val_flag] 

nn_test_src_l = src_l[nn_test_flag] 
nn_test_dst_l = dst_l[nn_test_flag]
nn_test_ts_l = ts_l[nn_test_flag]
nn_test_e_idx_l = e_idx_l[nn_test_flag] 
nn_test_label_l = label_l[nn_test_flag] 



adj_list = [[] for _ in range(max_idx + 1)] 
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l): 
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
    
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l): 
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
test_rand_sampler = RandEdgeSampler(src_l, dst_l)


nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)


### Model initialize
device = torch.device(f'cuda:{GPU}' if GPU >= 0 else 'cpu')
if EXP_TYPE == 'normal' or EXP_TYPE == 'noinfo':
    tgib = TGIB(train_ngh_finder, n_feat, e_feat, 64,
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
else:
    tgib = TGIB_with_fc(train_ngh_finder, n_feat, e_feat, 64,
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)


tgib.load_state_dict(torch.load(FILENAME, map_location='cpu'))  # or 'cuda' if using GPU
tgib.eval()  # Set model to evaluation mode

criterion = torch.nn.BCELoss()

tgib = tgib.to(device)

num_instance = len(train_src_l) 
num_batch = math.ceil(num_instance / BATCH_SIZE) 
idx_list = np.arange(num_instance) 
np.random.shuffle(idx_list) 

val_aps = []
print('train_src_l', train_src_l)

epoch = 10
# testing phase use all information
tgib.ngh_finder = full_ngh_finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgib, test_rand_sampler, test_src_l, 
test_dst_l, test_ts_l, test_e_idx_l,  epoch)

nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgib, nn_test_rand_sampler, nn_test_src_l, 
nn_test_dst_l, nn_test_ts_l, nn_test_e_idx_l, epoch)



print('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
print('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

print('Exp type', EXP_TYPE)


