#!/bin/bash
#SBATCH -J magnetic_rotor
#SBATCH -o magnetic_rotor.out
#SBATCH -e magnetic_rotor.err
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --account=pn67xi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --export=NONE
#SBATCH --get-user-env


module purge
module load stack/24.1.0
module load intel-toolkit/2024.1.0
module load mpi_settings/2.0
module load intel/2024.1.0
module load intel-mpi/2021.12.0
module load intel-mkl/2024.1.0 
module load intel-toolkit/2024.1.0
# 1) stack/24.1.0{arch=auto}          4) intel-mpi/2021.12.0        7) intel-ccl/2021.12.0  10) intel-tbb/2021.12.0  13) intel-ippcp/2021.11.0  16) intel-toolkit/2024.1.0
# 2) mpi_settings/2.0{mode=default}   5) intel-mkl/2024.1.0         8) intel-dnn/2024.1.0   11) intel-ipp/2021.11.0  14) intel-dpl/2022.5.0
# 3) intel/2024.1.0                   6) intel-inspector/2024.1.0   9) intel-itac/2022.1.0  12) intel-dal/2024.2.0   15) intel-dpct/2024.1.0


source /hppfs/work/pn67xi/di75rur/grace-pvt/env/supermuc-ng2.sh

module list 

env | sort > job_env.txt
module list > job_modules.txt

export FI_PSM3_UUID=$(uuidgen)
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu # bind to GPUs, not tiles (2 tiles per GPU)

export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_RDMA=1
export I_MPI_OFFLOAD_IPC=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

#export OMP_NUM_THREADS=28
#export KOKKOS_NUM_THREADS=28
export KOKKOS_NUM_THREADS=$OMP_NUM_THREADS
export ZE_ENABLE_TRACING=1
export ZE_DEBUG=1

echo "Submitting from ${SLURM_SUBMIT_DIR}"


cd $SLURM_SUBMIT_DIR


export GRACE_DIR=/hppfs/work/pn67xi/di75rur/grace-pvt

echo "GRACE DIR is: ${GRACE_DIR}"

echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "GPUs:"
sycl-ls

#mpirun -np $SLURM_NTASKS ${GRACE_DIR}/build/grace --grace-parfile ${GRACE_DIR}/configs/magnetic_rotor.yaml
mpirun -np $SLURM_NTASKS ${GRACE_DIR}/build_debug/grace --grace-parfile ${GRACE_DIR}/configs/magnetic_rotor.yaml
