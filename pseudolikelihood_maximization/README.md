Though there are several files here, you should only need to run `perform_plm_and_get_stats.slm`. This will:
1. Fetch the LAMMPS dump file for each of the polymer simulations and convert them to binary states for use in PLM
2. Run the pseudolikelihood maximization procedure
3. Generate statistics for the PLM-generated model (absent alpha)
