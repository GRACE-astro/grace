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
#include <grace/utils/device_stream.hh>


#include <tuple>
#include <array>
#include <utility>

// Use preprocessor to check if HIP or CUDA is enabled
#ifdef GRACE_ENABLE_HIP
#include <hip/hip_runtime.h>
#define GRACE_KERNEL_LAUNCH(kernel, blocks, threads, sharedMem, stream, ...) \
    hipLaunchKernelGGL((kernel), blocks, threads, sharedMem, stream, __VA_ARGS__)
#elif defined(GRACE_ENABLE_CUDA)
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
    index_t tot_iterations                      ;

    MDRange(std::array<index_t,rank> const& _lbnd, std::array<index_t,rank> const& _ubnd)
     : _offsets(_lbnd) 
    {
        tot_iterations = 1; 
        for( int id=0; id<rank; ++id) {
            _ranges[id] = ( _ubnd[id] - _lbnd[id] ) ; 
            tot_iterations *= _ranges[id] ; 
        }
    } 
    
    index_t GRACE_HOST_DEVICE compress(dim3 const& block_idx, dim3 const& thread_idx, dim3 const& block_dim)
    {
        // Compute the global linear index (1D index) from block and thread indices
        return block_idx.x * block_dim.x + thread_idx.x
            + (block_idx.y * block_dim.y + thread_idx.y) * block_dim.x
            + (block_idx.z * block_dim.z + thread_idx.z) * (block_dim.x * block_dim.y) ;
    }

    // Function to unpack global thread index into multi-dimensional indices as a tuple
    template <std::size_t... I>
    auto GRACE_HOST_DEVICE unpack_impl(index_t globidx, std::index_sequence<I...>) const {
        // Unpack the global index into a tuple of multi-dimensional indices
        //return std::make_tuple(((globidx % _ranges[I]) + _offsets[I], globidx /= _ranges[I])...);
        // Row-Major
        std::array<index_t, rank> __globIdx ;
        __globIdx[0] = globidx;
        #pragma unroll 
        for( int ii=1; ii<rank; ++ii) {
            __globIdx[ii] = __globIdx[ii-1]/_ranges[rank-1-ii] ;
        }
        return  std::make_tuple(((__globIdx[I] % _ranges[rank - 1 - I]) + _offsets[rank - 1 - I])...);
    }

    // Main unpack function that calls the implementation
    auto GRACE_HOST_DEVICE unpack(index_t globidx) const {
        return unpack_impl(globidx, std::make_index_sequence<rank>{});
    }
} ;

namespace detail {

template< typename Rt, typename Ft >
__global__ void grace_kernel_impl(Rt _range, Ft _func)
{   
     
    auto iglob = _range.compress(blockIdx,threadIdx,blockDim) ;
    auto ijk = _range.unpack(iglob) ;
    if ( iglob < _range.tot_iterations )
        std::apply(_func, ijk) ; 
}

}

template<size_t rank, typename index_t, typename Ft>
void launch_grace_kernel(
    const MDRange<rank, index_t>& _range, Ft lambda, 
    dim3 numBlocks, dim3 threadsPerBlock, 
    int sharedMem, grace::device_stream_t& stream) {
    // Alias the template specialization
    
    hipLaunchKernelGGL(detail::grace_kernel_impl,numBlocks,threadsPerBlock,sharedMem,stream._stream,_range,lambda) ; 
}

}



#endif 
