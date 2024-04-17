There are four subdirectories here: 

1. `data`: Contains the topology files used for polymer simulations in LAMMPS (`data500.polymer`) and to load simulated conformations into Python with mdtraj (`data2.psf`)
2. `run_e0.3`: Contains the files needed to run, process LAMMPS simulations with lenndard jones potential having $\varepsilon_\text{LJ} = 0.3$
3. `run_e0.35`: Contains the files needed to run, process LAMMPS simulations with lenndard jones potential having $\varepsilon_\text{LJ} = 0.35$
4. `run_e0.4`: Contains the files needed to run, process LAMMPS simulations with lenndard jones potential having $\varepsilon_\text{LJ} = 0.4$

Each `run_e0.x` directory contains the LAMMPS in file (`in.chromosome`), a sample SLURM job submission script (`job.pbs`), and a Python script to extract the replicas associated with temperature $T=1.0$ from the dump file outputted by LAMMPS (`find_fixed_T_traj.py`).
