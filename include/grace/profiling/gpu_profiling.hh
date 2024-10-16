/**
 * @file gpu_profiling.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-11
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
#ifndef GRACE_PROFILING_GPU_PROFILING_HH
#define GRACE_PROFILING_GPU_PROFILING_HH

#include <grace_config.h>

#include <grace/utils/device/device.h>
#include <grace/utils/inline.h>

#ifndef ROCPROFILER_PLUGIN_SETUP
#include <grace/errors/assert.hh>
#else 
#define ASSERT(cond,msg) \
assert(cond)
#endif 

#ifdef GRACE_ENABLE_HIP
#include <hip/hip_runtime.h>
#include <rocprofiler/v2/rocprofiler.h>
#include <amd_comgr/amd_comgr.h>
#endif 

#include <functional>
#include <unordered_map>

#ifdef ROCPROFILER_PLUGIN_SETUP
extern std::unordered_map<size_t,std::string> counter_names;
#endif 

#ifdef GRACE_ENABLE_HIP

namespace detail {
#if 0
static inline std::string rocmprofiler_cxx_demangle(const std::string& symbol) {
  amd_comgr_data_t mangled_data;
  amd_comgr_create_data(AMD_COMGR_DATA_KIND_BYTES, &mangled_data);
  amd_comgr_set_data(mangled_data, symbol.size(), symbol.data());

  amd_comgr_data_t demangled_data;
  amd_comgr_demangle_symbol_name(mangled_data, &demangled_data);

  size_t demangled_size = 0;
  amd_comgr_get_data(demangled_data, &demangled_size, nullptr);

  std::string demangled_str;
  demangled_str.resize(demangled_size);
  // amd_comgr_(get_data(demangled_data, &demangled_size, demangled_str.data())); // TODO: uncomment

  amd_comgr_release_data(mangled_data);
  amd_comgr_release_data(demangled_data);
  return demangled_str;
}
#endif
static inline std::string rocmprofiler_cxx_demangle(const std::string& symbol) {
    amd_comgr_data_t mangled_data;
    amd_comgr_status_t status;
    size_t demangled_size = 0;

    // Create and set the mangled data
    status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BYTES, &mangled_data);
    ASSERT(status == AMD_COMGR_STATUS_SUCCESS, "Failed to create AMD COMGR data for mangled name.");

    status = amd_comgr_set_data(mangled_data, symbol.size(), symbol.data());
    ASSERT(status == AMD_COMGR_STATUS_SUCCESS, "Failed to set AMD COMGR data for mangled name.");

    // Demangle the symbol name
    amd_comgr_data_t demangled_data;
    status = amd_comgr_demangle_symbol_name(mangled_data, &demangled_data);
    ASSERT(status == AMD_COMGR_STATUS_SUCCESS, "Failed to demangle symbol name.");

    // Get the size of the demangled data
    status = amd_comgr_get_data(demangled_data, &demangled_size, nullptr);
    ASSERT(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get size of demangled data.");

    // Resize the string and retrieve the demangled data
    std::string demangled_str(demangled_size, '\0');
    status = amd_comgr_get_data(demangled_data, &demangled_size, &demangled_str[0]);
    ASSERT(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get demangled data.");

    // Clean up
    amd_comgr_release_data(mangled_data);
    amd_comgr_release_data(demangled_data);

    return demangled_str;
}
}
/**
 * @brief Rocm profiling context.
 * Describes an active profiling region
 * through its session and buffer ids.
 */
struct rocm_profiling_context_t {
    rocprofiler_session_id_t _sid ; 
    rocprofiler_buffer_id_t  _bid ; 
} ; 
/**
 * @brief Initialize rocm profiling session
 *        to collect hardware counters.
 * @param[inout] context Context where this session is placed.
 * @param[in] counters   Requested counters.
 * See the available profiling parameters for the possible counters 
 * that can be requested. See profiling_runtime.hh for where this is called.
 */
void rocm_initiate_profiling_session( rocm_profiling_context_t& context, std::unordered_map<size_t,std::string>const& counters ) ;
/**
 * @brief Finalize rocm profiling session and write data.
 * 
 * @param context Context of the session to be terminated.
 * This flushes the buffer and writes data to disk.
 */
void rocm_terminate_profiling_session(rocm_profiling_context_t& context) ;
/**
 * @brief Flush profiling buffer and write results to disk.
 * @param record Record of this session.
 * @param session_id Session id.
 * @param buffer_id  Buffer id.
 */
extern "C" void flush_profiler_record( const rocprofiler_record_profiler_t* record
                                     , rocprofiler_session_id_t session_id 
                                     , rocprofiler_buffer_id_t buffer_id  ) ; 
/**
 * @brief Write profiling buffer records to disk.
 * @param begin_records First record of this session.
 * @param begin_records Last record of this session.
 * @param session_id Session id.
 * @param buffer_id  Buffer id.
 * This is the callback provided when the session is created.
 */
extern "C" void write_buffer_records( const rocprofiler_record_header_t* begin_records 
                                    , const rocprofiler_record_header_t* end_records 
                                    , rocprofiler_session_id_t session_id 
                                    , rocprofiler_buffer_id_t buffer_id ) ; 
void write_profile_data( const rocprofiler_record_profiler_t* profiler_record 
                       , const std::string& outfname, int rank, const std::string& kernel_name
                       , rocprofiler_session_id_t session_id) ; 
#endif

#endif /* GRACE_PROFILING_GPU_PROFILING_HH */