#!/home/gkks/apps/anaconda3/bin/python

# Here, we ignore the two nearest neighbors' interactions (i.e. contacts are only considered for beads >= 3 indices apart)
# AND we use a hyberbolic tangent function to determine the 'contact probability' for each, centered at sigma = 1.75

# Created by Xinqiang Ding (xqding@umich.edu), later modified by Greg Schuette (gkks@mit.edu) 
# at 2019/06/24 23:43:52

import numpy as np
import pickle
import mdtraj as md
from sys import exit
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_beads", type = int)
args = parser.parse_args()
num_beads = args.num_beads
cutoff = 1.707
sigma = 0.1 # mdtraj will load the conformations in different units; this converts it back 

run_types=["run_e0.3","run_e0.35","run_e0.4"]


for run_type in run_types:
    for traj_idx in range(1):
    
        print(traj_idx)
        t = md.load("../polymer_simulation/{}/DUMP_FILE.dcd".format(run_type),
                    top = "../polymer_simulation/data/data2.psf")

        start_idx = (t.n_atoms - num_beads)//2 # Find starting index... integer value. 0 if n_atoms == num_beads
	    	      		            # floor((n_atoms - n_beads)/2) otherwise
                                            # NOTE: This implies that we are looking at the center...
        end_idx = start_idx + num_beads
        for k in range(1):
            pair_idx = []
            for i in range(start_idx, end_idx):
                for j in range(i+1, end_idx):
                    pair_idx.append((i,j))
            pair_idx = np.array(pair_idx)
            pair_idx = pair_idx[abs(pair_idx[:,0]-pair_idx[:,1]) > 1,:]
            dist = md.compute_distances(t, pair_idx, periodic = False)
        
            dist/=sigma

            if k == 0: 
                states=np.zeros(dist.shape)
            ## convert distance to states with a cutoff
            num_sites = dist.shape[1]
            dist_cutoff=cutoff
            states1=np.array(dist<=cutoff,dtype=np.float64)
            states1*=1.0
            states+=states1
            start_idx+=num_beads
            end_idx+=num_beads
    
        ## save states
        np.save("./output/states/states_num_beads_{}_{}.npy".format(num_beads, run_type), states)
 
