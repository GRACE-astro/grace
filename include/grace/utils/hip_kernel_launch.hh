/**
 * @file hip_kernel_launch.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-10-08
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 *                                    
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *   
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *   
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */

#ifndef GRACE_UTILS_HIP_KERNEL_LAUNCH_HH
#define GRACE_UTILS_HIP_KERNEL_LAUNCH_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>



#include <tuple>
#include <array>
#include <utility>

// Use preprocessor to check if HIP or CUDA is enabled
#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#define GRACE_KERNEL_LAUNCH(kernel, blocks, threads, sharedMem, stream, ...) \
    hipLaunchKernelGGL(kernel, blocks, threads, sharedMem, stream, __VA_ARGS__)
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#define GRACE_KERNEL_LAUNCH(kernel, blocks, threads, sharedMem, stream, ...) \
    kernel<<<blocks, threads, sharedMem, stream>>>(__VA_ARGS__)
#else
#error "Unsupported platform. Must compile with HIP or CUDA."
#endif

namespace grace { 

template< size_t rank, typename index_t >
struct MDRange
{
    std::array<index_t, rank> _ranges, _offsets ; 

    // Function to unpack global thread index into multi-dimensional indices as a tuple
    template <std::size_t... I>
    auto unpack_impl(dim3 const& block_idx, dim3 const& thread_idx, dim3 const& block_dim, std::index_sequence<I...>) const {
        // Compute the global linear index (1D index) from block and thread indices
        index_t globidx = block_idx.x * block_dim.x + thread_idx.x
                        + (block_idx.y * block_dim.y + thread_idx.y) * block_dim.x
                        + (block_idx.z * block_dim.z + thread_idx.z) * (block_dim.x * block_dim.y);

        // Unpack the global index into a tuple of multi-dimensional indices
        return std::make_tuple(((globidx % _ranges[I]) + _offsets[I], globidx /= _ranges[I])...);
    }

    // Main unpack function that calls the implementation
    auto unpack(dim3 const& block_idx, dim3 const& thread_idx, dim3 const& block_dim) const {
        return unpack_impl(block_idx, thread_idx, block_dim, std::make_index_sequence<rank>{});
    }
} ;

namespace detail {

template< size_t rank, typename index_t, typename Ft >
__global__ void grace_kernel_impl(MDRange<rank,index_t> const& _range, Ft _func)
{   
    auto ijk = _range.unpack(blockIdx,threadIdx,blockDim) ; 
    std::apply(_func, ijk) ; 
}

}

template<size_t rank, typename index_t, typename Ft>
void launch_grace_kernel(const MDRange<rank, index_t>& _range, Ft lambda, dim3 numBlocks, dim3 threadsPerBlock) {

    GRACE_KERNEL_LAUNCH(detail::grace_kernel_impl<rank,index_t,Ft>,numBlocks,threadsPerBlock,_range,lambda) ; 
}

}



#endif 
