## Thalamus in rsFC (2023)

This repository contains the data and code used in the following manuscript:

Cabrera-Álvarez et al. (2023) Modeling the role of thalamus in resting state functional connectivity: nature or structure





----

The simulations were performed with an in-house implementation of the Jansen-Rit (1995) neural mass model included in "jansen_rit_david_mine.py".

DATA contains the functional connectivity and structural connectivity matrices derived from MEG and dwMRI data.



Rx folders contain the scripts used to perform simulations, statistical analysis and create figures in each Results section. Parameter space explorations were performed in a HPC with MPI. Therefore, R1, R2, R4.1, R4.2, and R4.supp contain sets of three scripts: 

- Bash file - To interact with the HPC

- main_mpi - Distributes the sets of simulations across cores

- parallel - Function to be executed per parameter set



The simulations performed in R3 and R4.3 used the auxiliary function simulate() in functions.py.



Some of these scripts require the use of the Toolbox located at: https://github.com/jescab01/toolbox.
