#!/bin/bash -l

#SBATCH -J mypyjob
#SBATCH -A 2017-85
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1
# reserves a node with a k80 GPU
#SBATCH --gres=gpu:K80:2
#SBATCH -e error_file.e
#SBATCH -o output_file.o
#SBATCH --mail-type=ALL

# load other modules
module add cudnn/5.1-cuda-8.0
# load the anaconda module
module load anaconda/py35/4.2.0
# if you need the custom conda environment:
source activate tensorflow
# source activate my_tensorflow1.1

# install python dependence
# conda install -c anaconda scipy=0.19.0

# load mpirun
module load i-compilers/17.0.1
module load intelmpi/17.0.1

# Run the executable
mpirun -np 1 python ./inception_FCN.py > my_output_file

# to deactivate the Anaconda environment
source deactivate
