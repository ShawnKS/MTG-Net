from tqdm import tqdm
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader, Dataset, RandomSampler, SubsetRandomSampler, Subset, SequentialSampler
)
from MTGNet import *
import pandas as pd
from search import *
import torch
import numpy as np
import copy
import random
import argparse
import pickle
from util import *
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='mimic27')
    parser.add_argument('--gpu_id',type=str,default='1')
    parser.add_argument('--ratio',type=str,default = '1')
    parser.add_argument('--layer_num',type=int,default=2)
    parser.add_argument('--seed',type=int,default = 0)
    parser.add_argument('--strategy',type=str,default = 'active')
    parser.add_argument('--dropout_rate',type=float, default = 0.5)
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--ensemble',type=int,default=2)
    parser.add_argument('--num_hidden',type=int,default=128)
    args = parser.parse_args()
    return args
args = get_args()
# hyperparam
dataset = args.dataset 
strategy = args.strategy
# strategy = 'pertask' # pertask or perstep
temperature = 0.00001
ratio = args.ratio
learning_rate = 0.1
num_layers = args.layer_num
gpu_id = args.gpu_id
ensemble_num = args.ensemble
end_num= 2
num_hidden = args.num_hidden
seed = args.seed
step = args.step
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# if theres a RuntimeWarning: invalid value encountered in true_divide, the model will select group the largest pred gain
total_pred_traj = model_training(dataset = dataset, ratio = ratio, temperature = temperature, num_layers= num_layers, num_hidden=num_hidden,ensemble_num= ensemble_num, gpu_id = gpu_id, end_num = end_num,step =step,strategy = strategy,seed= seed,dropout_rate = args.dropout_rate)