/**
 * @file device.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-11
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference 
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

#include <thunder_config.h>

#ifndef THUNDER_UTILS_DEVICE_H
#define THUNDER_UTILS_DEVICE_H

#if defined(THUNDER_ENABLE_CUDA) or defined (THUNDER_ENABLE_HIP)
#define THUNDER_DEVICE __device__ 
#define THUNDER_HOST   __host__ 
#define THUNDER_HOST_DEVICE __host__ __device__ 
#else 
#define THUNDER_DEVICE 
#define THUNDER_HOST 
#define THUNDER_HOST_DEVICE 
#endif 

namespace thunder {
void device_malloc(void** ptr, size_t nbyte);

void memcpy_device_to_host(void* dest, void* src, size_t nbyte);

void memcpy_host_to_device(void* dest, void* src, size_t nbyte);

void device_free(void* ptr) noexcept; 
}
#endif 