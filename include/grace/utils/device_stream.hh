/**
 * @file device_stream.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-10-07
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
#ifndef GRACE_UTILS_DEVICE_STREAM_HH
#define GRACE_UTILS_DEVICE_STREAM_HH


#include <grace_config.h>

#include <grace/errors/error.hh>
#include <grace/utils/device.h>


namespace grace {

struct device_stream_t {


    stream_t _stream ; //!< The stream_t object 

    /**
     * @brief Default ctor
     */
    device_stream_t() 
     : _stream()
    {
        STREAM_CREATE(_stream) ; 
        
    }

    /* Move & Copy */
    /* Ctor        */
    device_stream_t( const device_stream_t& ) = delete  ;  
    device_stream_t( device_stream_t&&)       = default ;
    /* Assignment  */
    device_stream_t& operator= (const device_stream_t&) = delete ; 
    device_stream_t& operator= (device_stream_t&&)      = default ;

    /**
     * @brief Dtor
     */
    ~device_stream_t()
    {
        STREAM_DESTROY(_stream) ; 
    }

    /**
     * @brief Synchronize stream
     * 
     */
    void fence()
    {
        STREAM_SYNCHRONIZE(_stream) ; 
    }

    /**
     * @brief Implicit cast to hipStream_t 
     * @note for SYCL it returns a dummy_stream with default SYCL queue as a member 
     * @return hipStream_t The underlying stream
     */
    operator stream_t() const {
            return _stream;
    }
    
    /*
    * @brief Conversion operator from dummy SYCL stream to SYCL queue 
    * @note necessary to initialize Kokkos execution spaces 
    * @return sycl_queue SYCL queue
    */
    #ifdef GRACE_ENABLE_SYCL
    operator sycl::queue&() {
        return _stream.q;
    }

    operator const sycl::queue&() const {
        return _stream.q;
    }
    #endif 


} ;


}

#endif /* GRACE_UTILS_DEVICE_STREAM_HH */