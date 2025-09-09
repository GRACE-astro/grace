/**
 * @file device.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-11
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference 
 * methods to simulate relativistic astrophysical systems and plasma
 * dynamics.
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

#ifndef GRACE_UTILS_DEVICE_H
#define GRACE_UTILS_DEVICE_H

#if defined(GRACE_ENABLE_CUDA) or defined (GRACE_ENABLE_HIP)
#define GRACE_DEVICE __device__ 
#define GRACE_HOST   __host__ 
#define GRACE_HOST_DEVICE __host__ __device__
#ifndef GRACE_ALLOW_DEVICE_CONDITIONALS
#define DEVICE_CONDITIONAL(cond,a,b) ((static_cast<bool>(cond)) * a + (1-static_cast<bool>(cond)) * b) 
#else 
#define DEVICE_CONDITIONAL(cond,a,b) ((cond) ? a : b)
#endif
#else 
#define GRACE_DEVICE 
#define GRACE_HOST 
#define GRACE_HOST_DEVICE 
#define DEVICE_CONDITIONAL(cond,a,b) ((cond) ? a : b)
#endif 

#ifdef GRACE_ENABLE_CUDA
    #include <cuda_runtime.h>
    #define CUDA_CALL(call)                                                               \
    do {                                                                                  \
        cudaError_t err = call;                                                           \
        if (err != cudaSuccess) {                                                         \
            ERROR("CUDA call returned error code " << cudaGetErrorString(err) << ".") ;   \
        }                                                                                 \
    } while (0)
    using device_err_t = cudaError_t ;
    #define DEVICE_SUCCESS cudaSuccess 
    #define DEVICE_NOT_READY cudaErrorNotReady
    using stream_t = cudaStream_t;
    #define STREAM_CREATE(stream) CUDA_CALL(cudaStreamCreate(&(stream)))
    #define STREAM_DESTROY(stream) CUDA_CALL(cudaStreamDestroy(stream))
    #define STREAM_SYNCHRONIZE(stream) CUDA_CALL(cudaStreamSynchronize(stream))
    using event_t = cudaEvent_t;
    #define EVENT_CREATE(event) CUDA_CALL(cudaEventCreate(&(event)))
    #define EVENT_DESTROY(event) CUDA_CALL(cudaEventDestroy(event))
    #define EVENT_RECORD(event, stream) CUDA_CALL(cudaEventRecord(event, stream))
    #define EVENT_SYNCHRONIZE(event) CUDA_CALL(cudaEventSynchronize(event))
    #define EVENT_ELAPSED_TIME(ms, start, stop) CUDA_CALL(cudaEventElapsedTime(ms, start, stop))
    #define EVENT_QUERY(event) cudaEventQuery(event)
#elif defined(GRACE_ENABLE_HIP)
    #include <hip/hip_runtime.h>
    #define HIP_CALL(call)                                                              \
    do {                                                                                \
        hipError_t err = call;                                                          \
        if (err != hipSuccess) {                                                        \
            ERROR("HIP call returned error code " << hipGetErrorString(err) << ".") ;   \
        }                                                                               \
    } while (0)
    using device_err_t  = hipError_t ; 
    #define DEVICE_SUCCESS hipSuccess
    #define DEVICE_NOT_READY hipErrorNotReady
    using stream_t = hipStream_t;
    #define STREAM_CREATE(stream) HIP_CALL(hipStreamCreate(&(stream)))
    #define STREAM_DESTROY(stream) HIP_CALL(hipStreamDestroy(stream))
    #define STREAM_SYNCHRONIZE(stream) HIP_CALL(hipStreamSynchronize(stream))
    using event_t = hipEvent_t;
    #define EVENT_CREATE(event) HIP_CALL(hipEventCreate(&(event)))
    #define EVENT_DESTROY(event) HIP_CALL(hipEventDestroy(event))
    #define EVENT_RECORD(event, stream) HIP_CALL(hipEventRecord(event, stream))
    #define EVENT_SYNCHRONIZE(event) HIP_CALL(hipEventSynchronize(event))
    #define EVENT_ELAPSED_TIME(ms, start, stop) HIP_CALL(hipEventElapsedTime(ms, start, stop))
    #define EVENT_QUERY(event) hipEventQuery(event)
#else 
    using stream_t = char ;
    #define STREAM_CREATE(stream) 
    #define STREAM_DESTROY(stream) 
    #define STREAM_SYNCHRONIZE(stream) 
    using event_t = char ;
    #define EVENT_CREATE(event) 
    #define EVENT_DESTROY(event) 
    #define EVENT_RECORD(event, stream) 
    #define EVENT_SYNCHRONIZE(event) 
    #define EVENT_ELAPSED_TIME(ms, start, stop) 
#endif 

#endif 