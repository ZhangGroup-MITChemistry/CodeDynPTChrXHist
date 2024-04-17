import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--symmetry", type = str)
parser.add_argument("--mode", type = str)
parser.add_argument("--num_beads", type = int)
parser.add_argument("--gamma", type = float)
parser.add_argument("--iteration",type=int)
parser.add_argument("--run_type",type=str)
args = parser.parse_args()

symmetry = args.symmetry
mode = args.mode
num_beads = args.num_beads
gamma = args.gamma
iteration=args.iteration
run_type=args.run_type


A=[]
A.append("#!/bin/bash")
A.append("")
A.append("# Created by Greg Schuette (gkks@mit.edu)")
A.append("")
A.append("#SBATCH --job-name=run_TRE")
A.append("#SBATCH --partition=xeon-p8")
A.append("#SBATCH --array=0")
A.append("#SBATCH --ntasks=15")
A.append("#SBATCH --nodes=1")
A.append("#SBATCH --output=./slurm_output/TRE_MCMC_{}_{}.output".format(run_type,num_beads))
A.append("")
A.append("symmetry=\"{}\"".format(symmetry))
A.append("num_beads={}".format(num_beads))
A.append("gamma={:.3E}".format(gamma))
A.append("run_type=\"{}\"".format(run_type))
A.append("mode={}".format(mode))
A.append("iteration={}".format(iteration))
A.append("")
A.append("cd ..")
A.append("")
A.append("~/opt/openmpi/bin/mpirun -np ${SLURM_NTASKS} /state/partition1/llgrid/pkg/anaconda/anaconda3-2021a/bin/python ./TRE_MCMC.py --symmetry=$symmetry --num_beads=$num_beads --gamma=$gamma --mode=$mode --iteration=$iteration --run_type=$run_type &> log/TRE_MCMC_GPU_${gamma}.log")
A.append("")

FILE=open("./TRE_Files/submit_TRE_MCMC_{}_{}_{}_{:.3E}.slm".format(run_type,num_beads,mode,gamma),'a')
for L in A:
    FILE.write(L)
    FILE.write("\n")
