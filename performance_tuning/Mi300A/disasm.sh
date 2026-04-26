#!/bin/bash
LLC=/mpcdf/soft/RHEL_9/packages/x86_64/rocm/6.3.4/llvm/bin/llc

for bc in $(find $BUILD -name '*-hip-amdgcn-amd-amdhsa-gfx942.bc' ! -name '*.tmp.bc'); do
    $LLC -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 "$bc" -o "${bc%.bc}.s"
done

for s in $(find $BUILD -name '*-hip-amdgcn-amd-amdhsa-gfx942.s'); do
    awk -v f="$s" '/\.amdhsa_kernel/                     {name=$2; v=""; sc=""}
    	           /\.amdhsa_private_segment_fixed_size/ {sc=$2}
                   /\.amdhsa_next_free_vgpr/           	 {v=$2}
                   /\.end_amdhsa_kernel/                 {print sc, v, f, name}' "$s"
done | sort -n > all_kernels.log

# Everything that spills, worst first
awk '$1+0 > 0' all_kernels.log | sort -n | tail -30 > spillers.log

# Everything at max VGPR (512) but scratch=0 — tight but fits
awk '$1+0 == 0 && $2+0 >= 512' all_kernels.log > max_vgpr_no_spill.log

# Anything with VGPR > 168 is below 3 waves/SIMD occupancy
awk '$2+0 > 168' all_kernels.log | sort -k2 -n | tail -30 > low_occupancy.log
