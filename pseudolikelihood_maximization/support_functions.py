__author__ = "Xinqiang Ding <xqding@umich.edu>" # Later modified by Greg Schuette (gkks@mit.edu)
__date__ = "2019/08/07 16:06:22"

import numpy as np
from collections import defaultdict
import torch

def get_equivs(num_beads,num_beads_subchain,n_nearest_neighbors_ignored): 

    # Get system index and loop sizes; truncate to account for ignored neighbors
    idx = []
    for i in range(num_beads):
        for j in range(i+1,num_beads):
            idx.append((i,j))
    idx = np.array(idx)
    ks = idx[:,1] - idx[:,0]
    iidx = ks > n_nearest_neighbors_ignored
    idx = idx[iidx,:]
    ks = ks[iidx]
    del iidx

    # Find all indices that (a) need a J/aren't truncated and (b) are topologically equivalent
    equivs = {}
    for i in range(len(ks)):
        for j in range(i+1,len(ks)):
            if ks[i] > ks[j]:
                k2 = ks[i]
                k1 = ks[j]
                L = idx[i,1] - idx[j,1]
            elif ks[j] > ks[i]:
                k2 = ks[j]
                k1 = ks[i]
                L = idx[j,1] - idx[i,1]
            else:
                k2 = ks[i]
                k1 = ks[i]
                L = abs(idx[i,1] - idx[j,1])

            if k2 > num_beads_subchain: # Truncate
                continue
            if L >= k2:  # Region III
                continue
            if -L >= k1: # Region III
                continue 

            if k2 not in equivs:
                equivs[k2]={}
            if k1 not in equivs[k2]:
                equivs[k2][k1]={}
            if L not in equivs[k2][k1]:
                equivs[k2][k1][L]=[]
            equivs[k2][k1][L].append((i,j))

    n_unique_Js = 0
    for k2 in equivs:
        for k1 in equivs[k2]:
            for L in equivs[k2][k1]:
                n_unique_Js += 1
                equivs[k2][k1][L] = np.array(equivs[k2][k1][L])

    return equivs, n_unique_Js


def parse_h_J(weights,run_types, num_beads, equivs,n_nearest_neighbors_ignored): 
    num_sites = (num_beads-n_nearest_neighbors_ignored)*(num_beads - n_nearest_neighbors_ignored - 1)//2

    ## parse h
    n=0
    h={}
    for k in run_types:

        if n==0:
            h_param=weights[0]+weights[2:(num_beads-n_nearest_neighbors_ignored+1)]
        elif n==1:
            h_param=weights[2:(num_beads-n_nearest_neighbors_ignored+1)]
        else:
            h_param=weights[1]+weights[2:(num_beads-n_nearest_neighbors_ignored+1)]

        h_expand = np.zeros((num_beads-n_nearest_neighbors_ignored, num_beads-n_nearest_neighbors_ignored))
        h_expand=torch.from_numpy(h_expand).double()
        for i in range(len(h_param)):
            h_expand[np.arange(0,num_beads-n_nearest_neighbors_ignored-(i+1)),np.arange(0,num_beads-n_nearest_neighbors_ignored-(i+1))+i+1] = h_param[i] 
        h[k] = h_expand[np.triu_indices(num_beads-n_nearest_neighbors_ignored, 1)]
        n+=1
    
    ## parse J
    J_param = weights[(num_beads-n_nearest_neighbors_ignored+1):] 
    print("J_param inside functions: "+format(J_param.shape))
    J_np = np.zeros((num_sites, num_sites), dtype = np.float64)
    J_np = torch.from_numpy(J_np)
    n=0
    for k2 in equivs:
        for k1 in equivs[k2]:
            for L in equivs[k2][k1]:
                J_np[(equivs[k2][k1][L][:,0],equivs[k2][k1][L][:,1])] = J_param[n]
                n+=1

    J = J_np + J_np.T
    
    return h, J
    
