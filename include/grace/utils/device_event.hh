/**
 * @file device_event.hh
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

#ifndef GRACE_UTILS_DEVICE_EVENT_HH
#define GRACE_UTILS_DEVICE_EVENT_HH


#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/device_stream.hh>



namespace grace {

/**
 * @brief Class representing a device event for timing and synchronization in CUDA/HIP.
 */
struct device_event_t {

    //! Type of event (cudaEvent_t or hipEvent_t)
    event_t _event;

    /**
     * @brief Default constructor. Creates a new device event.
     */
    device_event_t() : _event() {
        EVENT_CREATE(_event);
    }

    /**
     * @brief Destructor. Destroys the device event.
     */
    ~device_event_t() {
        EVENT_DESTROY(_event);
    }

    //! Move-only class
    device_event_t(const device_event_t&) = delete;
    device_event_t(device_event_t&&) = default;
    device_event_t& operator=(const device_event_t&) = delete;
    device_event_t& operator=(device_event_t&&) = default;

    /**
     * @brief Records the event on a given stream.
     * 
     * The event is placed in the stream's execution queue. When it reaches this point, the event is recorded.
     * 
     * @param stream The device stream on which to record the event.
     */
    void record(const device_stream_t& stream) {
        EVENT_RECORD(_event, stream._stream);
    }

    /**
     * @brief Synchronizes the event. Blocks the host until the event is reached.
     */
    void synchronize() {
        EVENT_SYNCHRONIZE(_event);
    }

    /**
     * @brief Measures the elapsed time between two events.
     * 
     * @param start The start event.
     * @param stop The stop event.
     * @return The elapsed time in milliseconds.
     */
    static float elapsed_time(const device_event_t& start, const device_event_t& stop) {
        float time_ms;
        EVENT_ELAPSED_TIME(&time_ms, start._event, stop._event);
        return time_ms;
    }

    /**
    * @brief Query event to see if work is complete
    * @return DEVICE_SUCCESS if work is complete or DEVICE_NOT_READY otherwise
     */
    device_err_t query() const {
        return EVENT_QUERY(_event) ; 
    }

    void reset() {
        EVENT_DESTROY(_event);
        EVENT_CREATE(_event);
    }

};

}

#endif 