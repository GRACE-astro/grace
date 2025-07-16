cmake -B . -S.. -DCMAKE_CXX_COMPILER=mpiicpx -DCMAKE_C_COMPILER=mpiicx -DGRACE_ENABLE_SYCL=ON  -DGRACE_NSPACEDIM=3  -DGRACE_ENABLE_GRMHD=ON -DGRACE_ENABLE_B_FIELD_GLM=ON -DGRACE_DO_MHD=ON -DGRACE_CARTESIAN_COORDINATES=ON -DCMAKE_CXX_FLAGS="--gcc-toolchain=/dss/lrzsys/sys/spack/release/24.1.0/opt/x86_64/gcc/12.3.0-gcc-hsokiku -std=c++20" -DCMAKE_C_FLAGS="--gcc-toolchain=/dss/lrzsys/sys/spack/release/24.1.0/opt/x86_64/gcc/12.3.0-gcc-hsokiku"


cmake -B . -S.. -DCMAKE_CXX_COMPILER=mpiicpx -DCMAKE_C_COMPILER=mpiicx -DGRACE_ENABLE_SYCL=ON  -DGRACE_NSPACEDIM=3  -DGRACE_ENABLE_GRMHD=ON -DGRACE_ENABLE_B_FIELD_GLM=ON -DGRACE_DO_MHD=ON -DGRACE_CARTESIAN_COORDINATES=ON -DCMAKE_CXX_FLAGS="--gcc-toolchain=/dss/lrzsys/sys/spack/release/24.1.0/opt/x86_64/gcc/12.3.0-gcc-hsokiku -std=c++20" -DCMAKE_C_FLAGS="--gcc-toolchain=/dss/lrzsys/sys/spack/release/24.1.0/opt/x86_64/gcc/12.3.0-gcc-hsokiku"


# test dpcpp compilation of sth
mpiicpx -std=c++20 --gcc-toolchain=/dss/lrzsys/sys/spack/release/24.1.0/opt/x86_64/gcc/12.3.0-gcc-hsokiku/ ../test_check_version/test2.cpp -o your_program

# test the experimental support
mpiicpx -std=c++20 --gcc-toolchain=/dss/lrzsys/sys/spack/release/24.1.0/opt/x86_64/gcc/12.3.0-gcc-hsokiku/ ../test_check_version/test2.cpp -o your_program -fsycl

