/**
 * @file device_stream_pool.hh
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
#ifndef GRACE_UTILS_DEVICE_STREAM_POOL_HH
#define GRACE_UTILS_DEVICE_STREAM_POOL_HH

#include <grace_config.h>

#include <grace/utils/device_stream.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/config/config_parser.hh>
#include <grace/errors/assert.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>


namespace grace {
//**************************************************************************************************
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief A pool of device streams (hipStream_t or cudaStream_t depending on the backend).
 * 
 * The pool is initialized at startup and the streams remain active for the whole duration of the 
 * application. Work can be scheduled on the streams in a round-robin fashion and kernels can be 
 * synchronized using events (see <code>device_event_t</code>).
 * 
 */
class device_stream_pool_impl_t 
{
    //**************************************************************************************************
    using stream_t = grace::device_stream_t ; 
    using pool_t = std::vector<grace::device_stream_t> ;  
    using size_type = pool_t::size_type ; 
    //**************************************************************************************************
 public:
    //**************************************************************************************************
    stream_t& next() 
    {
        auto& stream = _pool[_current] ; 
        _current = (_current + 1) % _pool.size() ;
        return stream ; 
    }
    //**************************************************************************************************
    stream_t& operator[] (size_type const& idx) 
    {
        _current = ( idx+1 ) % _pool.size() ; 
        return _pool[idx] ; 
    }
    //**************************************************************************************************
    stream_t& at(size_type const& idx) 
    {
        _current = ( idx+1 ) % _pool.size() ; 
        return _pool[idx] ; 
    }
    //**************************************************************************************************
    size_type size() const 
    {
        return _pool.size() ; 
    }
    //**************************************************************************************************
 protected:
    //**************************************************************************************************
    static constexpr unsigned long longevity = unique_objects_lifetimes::DEVICE_RESOURCES ; 
    //**************************************************************************************************
    device_stream_pool_impl_t()
     : _current(0)
    {
        auto const n_streams = grace::get_param<std::size_t>("system", "n_device_streams") ; 
        ASSERT( n_streams > 0, "Number of active streams must be positive." ) ;
        _pool.resize(n_streams) ; 
    }
    //**************************************************************************************************
    ~device_stream_pool_impl_t() = default ;
    //**************************************************************************************************
    

    //**************************************************************************************************
    pool_t _pool ; 
    size_type _current ; 
    //**************************************************************************************************
    friend class utils::singleton_holder<device_stream_pool_impl_t> ;
    friend class memory::new_delete_creator<device_stream_pool_impl_t, memory::new_delete_allocator> ; 
    //**************************************************************************************************
} ; 
//**************************************************************************************************
/**
 * @brief 
 * 
 */
using device_stream_pool = utils::singleton_holder<device_stream_pool_impl_t> ; 
//**************************************************************************************************
}
//**************************************************************************************************
#endif /* GRACE_UTILS_DEVICE_STREAM_POOL_HH */