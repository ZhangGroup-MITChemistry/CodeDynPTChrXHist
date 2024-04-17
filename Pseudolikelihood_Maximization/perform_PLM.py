#!/state/partition1/llgrid/pkg/anaconda/anaconda3-2021a/bin/python

# Created by Xinqiang Ding (xqding@umich.edu)
# at 2019/06/24 23:43:52
# Heavily modified by Greg Schuette (gkks@mit.edu) at various later dates

#SBATCH --job-name=fit_sym_gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0#-8
#SBATCH --output=./slurm_output/fit_sym_%a_gpu_new.out

n_nearest_neighbors_ignored=1

import numpy as np
import numpy.random as random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import pickle
from sys import exit
import sys
import os
import time
import datetime
import copy
import math
import scipy.optimize as optimize
from collections import defaultdict
sys.path.append("./")
from support_functions import *
import argparse

## parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--num_beads_subchain", type = int) # Subchain size for PLM?
parser.add_argument("--num_beads", type = int)          # Beads per chain?
parser.add_argument("--mode", type = str)               # Only affects file name at end
parser.add_argument("--symmetry",type=str)
parser.add_argument("--gamma",type=float)
run_types=['run_e0.3','run_e0.35','run_e0.4']

args = parser.parse_args()
num_beads_subchain = args.num_beads_subchain
num_beads = args.num_beads
mode = args.mode
symmetry=args.symmetry
start_time = time.time()
gamma=args.gamma
    

## load states
states = {}
min_frames=1e10
for run_type in run_types:
    states[run_type]=[]
    states[run_type].append(np.load("./output/states/states_num_beads_{}_{}.npy".format(num_beads, run_type)))
    states[run_type]=np.vstack(states[run_type])
    if states[run_type].shape[0] < min_frames:
        min_frames=states[run_type].shape[0]
        min_frames=int(min_frames)

states1=np.concatenate((states[run_types[0]][0:min_frames,:],states[run_types[1]][0:min_frames,:],states[run_types[2]][0:min_frames,:]), axis=1)
states=states1
del states1
batch_size = 20000 
num_sites = (num_beads - n_nearest_neighbors_ignored) * (num_beads - n_nearest_neighbors_ignored - 1)//2

equivalent_indices, n_unique_Js = get_equivs(num_beads,num_beads_subchain,n_nearest_neighbors_ignored)

## Ising model
def calc_loss_and_grad(x,num_beads,n_nearest_neighbors_ignored, batch_size, states, gamma, equivalent_indices): 
    x = x.astype(np.float64)
    x = torch.from_numpy(x)
    x.requires_grad_(True)
    J_param = x[(num_beads-n_nearest_neighbors_ignored+1):]
    if torch.cuda.is_available():
        J_param=J_param.cuda()
      
    h_np,J_np = parse_h_J(x,run_types, num_beads,equivalent_indices,n_nearest_neighbors_ignored)

    # Disable so that this can be done without RAM running out...
    if torch.cuda.is_available():
        h_cuda= torch.cat((h_np[run_types[0]],h_np[run_types[1]],h_np[run_types[2]]) ).cuda() 
    else:
        h_cuda=torch.cat((h_np[run_types[0]],h_np[run_types[1]],h_np[run_types[2]]) ) 

    if torch.cuda.is_available():
        J_cuda = J_np.cuda() 
    else:
        J_cuda = J_np 
    J_mask = torch.eye(num_sites, dtype = torch.uint8, requires_grad = False, device = J_cuda.device).bool()
    J_cuda = J_cuda.masked_fill(J_mask, 0.0) 

    loss_tot = 0
    h_grad_tot = 0
    J_grad_tot = 0
    
    num_batches = states.shape[0]//batch_size

    print("calling calc_loss_and_grad", flush = True)
    for batch_idx in range(num_batches):
        sample = states[batch_idx*batch_size:(batch_idx+1)*batch_size, :]  # End-to-end batches in first dimension
        sample = torch.from_numpy(sample)
        sample = sample.double()
        if torch.cuda.is_available():
            sample = sample.cuda()        
        
        logit=torch.matmul(sample[:,0:num_sites],J_cuda.t()) + h_cuda[0:num_sites]
        p1=1/(1+torch.exp(logit))
        logit=torch.matmul(sample[:,num_sites:2*num_sites],J_cuda.t()) + h_cuda[num_sites:2*num_sites]
        p2=1/(1+torch.exp(logit))
        logit=torch.matmul(sample[:,2*num_sites:],J_cuda.t()) + h_cuda[2*num_sites:]
        p3=1/(1+torch.exp(logit)) #PROBABILITY OF SPIN = 0!!!
        p=torch.cat((p1,p2,p3),1)
        loss = -(sample*torch.log(p) + (1-sample)*torch.log(1-p))
        loss = torch.mean(loss.sum(1))
        loss_tot += loss
    
    loss = loss_tot / num_batches
    loss = loss + 3*0.5*gamma*torch.sum(J_param**2)
    
    loss.backward()
    grad=x.grad
    grad = grad.cpu().detach().numpy()
    return loss, grad

h_param = np.zeros((num_beads - 1-n_nearest_neighbors_ignored)+2)
J_param = np.zeros(n_unique_Js)
x_init = np.concatenate((h_param, J_param))
x_init = x_init.astype(np.float64)

num_eval = 0
def call_back(xk):
    global num_eval
    num_eval += 1    
    if num_eval % 50 == 0:
        with open("./output/symmetry_{}/model_{}/h_J_num_beads_{}_gamma_{:.3E}_iter_{}.pkl".format(symmetry,mode, num_beads, gamma, num_eval), 'wb') as file_handle:
            pickle.dump(xk, file_handle)
                

print("start L-BFGS optimizer")
x, f, d = optimize.fmin_l_bfgs_b(calc_loss_and_grad,
                                 x_init,
                                 args = [num_beads,n_nearest_neighbors_ignored, batch_size, states, gamma, equivalent_indices],
                                 iprint = 1)

h_np, J_np = parse_h_J(x,run_types, num_beads,equivalent_indices,n_nearest_neighbors_ignored)
for run_type in h_np:
    h_np[run_type]=h_np[run_type].cpu().detach().numpy()
J_np=J_np.cpu().detach().numpy()

with open("./output/symmetry_{}/model_{}/h_J_num_beads_{}_gamma_{:.3E}_final.pkl".format(symmetry,mode, num_beads, gamma), 'wb') as file_handle:
    pickle.dump({'h': h_np, 'J': J_np,'x':x}, file_handle)

