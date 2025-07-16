/**
 * @file device.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-26
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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

#include <grace_config.h>

#include <grace/errors/assert.hh>
#include <grace/system/print.hh>
#include <grace/utils/device.h> 

#if defined(GRACE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#elif defined(GRACE_ENABLE_CUDA)
#include <cuda.h>
#elif defined(GRACE_ENABLE_SYCL)
#include <sycl/sycl.hpp>        
#endif 
#include <cstring>

namespace grace {

void host_malloc(void** ptr, size_t nbytes)
{
    
}
void device_malloc(void** ptr, size_t nbytes) 
{
    #if defined(GRACE_ENABLE_HIP)
    auto ret = hipMalloc(ptr, nbytes) ; 
    ASSERT(ret == hipSuccess, "Call to malloc failed");
    #elif defined(GRACE_ENABLE_CUDA)
    auto ret = cudaMalloc(ptr,nbytes) ; 
    ASSERT(ret == cudaSuccess, "Call to malloc failed");
    #elif defined(GRACE_ENABLE_SYCL)
    sycl::queue q(sycl::gpu_selector{}); // not sure if we can generalize this meaningfully
    void* tmp = sycl::malloc_device(nbytes, q);
    ASSERT(tmp != nullptr, "Call to malloc_device failed");
    *ptr = tmp;
    #else 
    GRACE_ERROR("Malloc device does not work with no device.") ; 
    #endif 
}

void memcpy_host_to_device(void* dest, void* src, size_t nbytes)
{
    #if defined(GRACE_ENABLE_HIP)
    hipError_t ret = hipMemcpyHtoD(dest,src,nbytes); 
    ASSERT(ret == hipSuccess, 
        "Call to memcpy failed (host to device) "
        "with code " << ret << '.');
    #elif defined(GRACE_ENABLE_CUDA)
    auto ret = cudaMemcpy(dest,src,nbytes,cudaMemcpyHostToDevice);
    ASSERT(ret == cudaSuccess, "Call to memcpy failed (host to device)");
    #elif defined(GRACE_ENABLE_SYCL)
    sycl::queue q(sycl::gpu_selector{});
    try { // not sure if this is a good idea - handling exceptions incurs cost... 
        auto event = q.memcpy(dest, src, nbytes);
        event.wait();  
    } catch (const sycl::exception& e) {
        ASSERT(false, std::string("Call to memcpy failed (host to device): ") + e.what());
    }
    #else 
    memcpy(dest,src,nbytes);
    #endif 
}

void memcpy_device_to_host(void* dest, void* src, size_t nbytes)
{
    #if defined(GRACE_ENABLE_HIP)
    auto ret = hipMemcpy(dest,src,nbytes,hipMemcpyDeviceToHost); 
    ASSERT(ret == hipSuccess, "Call to memcpy failed (device to host)");
    #elif defined(GRACE_ENABLE_CUDA)
    auto ret = cudaMemcpy(dest,src,nbytes,cudaMemcpyDeviceToHost);
    ASSERT(ret == cudaSuccess, "Call to memcpy failed (host to device)");
    #elif defined(GRACE_ENABLE_SYCL)
    sycl::queue q(sycl::gpu_selector{});
    try { // not sure if this is a good idea - handling exceptions incurs cost... 
        auto event = q.memcpy(dest, src, nbytes); // same syntax (dest first, src second)
        event.wait();  
    } catch (const sycl::exception& e) {
        ASSERT(false, std::string("Call to memcpy failed (device to device): ") + e.what());
    }
    #else 
    memcpy(dest,src,nbytes);
    #endif 
}

void device_free(void* ptr) noexcept
{
    #if defined(GRACE_ENABLE_HIP)
    auto ret = hipFree(ptr); 
    ASSERT(ret == hipSuccess, "Call to free failed");
    #elif defined(GRACE_ENABLE_CUDA)
    auto ret = cudaFree(ptr);
    ASSERT(ret == cudaSuccess, "Call to free failed");
    #elif defined(GRACE_ENABLE_SYCL)
    sycl::queue q(sycl::gpu_selector{});
    try {
    sycl::free(ptr, q);
    } catch (const sycl::exception& e) {
        ASSERT(false, std::string("Call to free failed: ") + e.what());
    }
    #else 
    free(ptr);
    #endif 
}

}