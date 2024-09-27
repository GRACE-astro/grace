/**
 * @file gpu_profiling.cpp
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

#include <grace_config.h>

#include <grace/profiling/gpu_profiling.hh>
#include <grace/profiling/profiling_runtime.hh>
#include <grace/system/grace_system.hh>
#include <grace/errors/error.hh>
#include <grace/errors/assert.hh>
#ifdef GRACE_ENABLE_HIP
#include <hip/hip_runtime.h>
#include <rocprofiler/v2/rocprofiler.h>
#endif 

#include <string>
#include <filesystem>
#include <iostream>
#include <iomanip>

#ifdef GRACE_ENABLE_HIP
#ifndef CHECK_ROCPROFILER
#define CHECK_ROCPROFILER(call)                                                                     \
  do {                                                                                              \
  rocprofiler_status_t errcode = (call) ;                                                           \
    if (errcode != ROCPROFILER_STATUS_SUCCESS)                                                      \
        ERROR("ROCProfiler API call error! Error code " << errcode );                               \
  } while (false)
#endif 

void rocm_initiate_profiling_session( rocm_profiling_context_t& context, std::unordered_map<size_t,std::string> const& _counters ) 
{
    std::vector<const char*> counters ; 
    for( auto const& x: _counters) counters.push_back(x.second.c_str()) ; 
    CHECK_ROCPROFILER(rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &context._sid) ) ;
    CHECK_ROCPROFILER(rocprofiler_create_buffer(
          context._sid
        , []( const rocprofiler_record_header_t* record, const rocprofiler_record_header_t* end_records
            , rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id ) 
        {
            write_buffer_records(record,end_records,session_id,buffer_id) ; 
        }
        , 0x9999, &context._bid ) 
    ) ; 

    rocprofiler_filter_id_t filter_id ; 
    [[maybe_unused]] rocprofiler_filter_property_t property = {} ; 
    CHECK_ROCPROFILER(
        rocprofiler_create_filter( context._sid, ROCPROFILER_COUNTERS_COLLECTION
                                 , rocprofiler_filter_data_t{ .counters_names = &counters[0] }
                                 , counters.size() 
                                 , &filter_id, property) 
    ) ; 
    CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(context._sid,filter_id,context._bid)) ;
    CHECK_ROCPROFILER(rocprofiler_start_session(context._sid));
}

void rocm_terminate_profiling_session(rocm_profiling_context_t& context) {
    CHECK_ROCPROFILER(rocprofiler_terminate_session(context._sid)) ; 
    CHECK_ROCPROFILER(rocprofiler_flush_data(context._sid, context._bid)) ; 
    CHECK_ROCPROFILER(rocprofiler_destroy_session(context._sid)) ; 
}

extern "C" void write_buffer_records( const rocprofiler_record_header_t* begin_records, const rocprofiler_record_header_t* end_records
                                    , rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id ) 
{
    while( begin_records < end_records ) {
        if( begin_records == nullptr ) return ; 
        const rocprofiler_record_profiler_t* profiler_record =
            reinterpret_cast<const rocprofiler_record_profiler_t*>(begin_records);
        flush_profiler_record(profiler_record, session_id, buffer_id);
        rocprofiler_next_record(begin_records,&begin_records,session_id,buffer_id) ; 
    }
}
extern "C" void flush_profiler_record( const rocprofiler_record_profiler_t* profiler_record 
                                     , rocprofiler_session_id_t session_id 
                                     , rocprofiler_buffer_id_t buffer_id )
{ 
    std::filesystem::path basepath = grace::profiling_runtime::get().out_basepath() ; 
    std::string const section_name = grace::profiling_runtime::get().top_gpu_region_name() ; 
    static constexpr unsigned int width = 20 ; 
    auto const iter = grace::get_iteration() ; 
    auto const rank = parallel::mpi_comm_rank() ; 
    size_t name_length = 0;
    CHECK_ROCPROFILER(rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME,
                                                        profiler_record->kernel_id, &name_length));
    // Taken from rocprofiler: The size hasn't changed in  recent past
    static const uint32_t lds_block_size = 128 * 4;
    const char* kernel_name_c = "";
    if (name_length > 1) {
        kernel_name_c = static_cast<const char*>(malloc(name_length * sizeof(char)));
        ASSERT(kernel_name_c != nullptr, "Failed to allocate buffer for kernel name.") ;
        CHECK_ROCPROFILER(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME,
                                                        profiler_record->kernel_id, &kernel_name_c));
    }
    std::string kernel_name = detail::rocmprofiler_cxx_demangle(kernel_name_c);
    // 
    std::string const pfname = section_name + std::string("_gpu_counters_") + std::to_string(rank) + std::string(".dat") ;
    std::filesystem::path outfname = 
        basepath / pfname ; 
    
    write_profile_data(profiler_record, outfname.string(), rank, kernel_name, session_id) ; 
    if( name_length > 1 ) {
        free(const_cast<char*>(kernel_name_c)) ; 
    }
}
void write_profile_data( const rocprofiler_record_profiler_t* profiler_record 
                       , const std::string& outfname, int rank, const std::string& kernel_name
                       , rocprofiler_session_id_t session_id ) 
{
    auto counter_names = grace::profiling_runtime::get().active_hardware_counters(); 
    std::ofstream output_file{outfname, std::ios::app};
    if (!output_file.is_open()) {
        ERROR("Failed to open file for profilers output.") ; 
        return;
    }
    static const uint32_t lds_block_size = 128 * 4;
    std::stringstream ss;
    ss << "iteration(" << grace::get_iteration() << ")\n"
    << "kernel-name(\"" << kernel_name << "\"),\n"
    << "dispatch[" << profiler_record->header.id.handle << "],\n"
    << "gpu_id(" << profiler_record->gpu_id.handle << "),\n"
    << "queue_id(" << profiler_record->queue_id.handle << "),\n"
    << "queue_index(" << profiler_record->queue_idx.value << "),\n"
    << "tid(" << profiler_record->thread_id.value << "),\n"
    << "Kernel Properties:\n"
    << "    grd(" << profiler_record->kernel_properties.grid_size << "),\n"
    << "    wgr(" << profiler_record->kernel_properties.workgroup_size << "),\n"
    << "    lds(" << ((profiler_record->kernel_properties.lds_size + (lds_block_size - 1)) & ~(lds_block_size - 1)) << "),\n"
    << "    scr(" << profiler_record->kernel_properties.scratch_size << "),\n"
    << "    arch_vgpr(" << profiler_record->kernel_properties.arch_vgpr_count << "),\n"
    << "    accum_vgpr(" << profiler_record->kernel_properties.accum_vgpr_count << "),\n"
    << "    sgpr(" << profiler_record->kernel_properties.sgpr_count << "),\n"
    << "    wave_size(" << profiler_record->kernel_properties.wave_size << "),\n"
    << "    sig(" << profiler_record->kernel_properties.signal_handle << "),\n"
    << "    obj(" << profiler_record->kernel_id.handle << "),\n"
    << "Timestamps (in nanoseconds):\n"
    << "    time-begin(" << profiler_record->timestamps.begin.value << "),\n"
    << "    time-end(" << profiler_record->timestamps.end.value << "),\n"
    << "    duration(" << profiler_record->timestamps.end.value - profiler_record->timestamps.begin.value << ")";

    output_file << ss.str() << std::endl;

    if (profiler_record->counters) {
        output_file << "Counters:\n" ; 
        for (uint64_t i = 0; i < profiler_record->counters_count.value; ++i) {
            if (profiler_record->counters[i].counter_handler.handle > 0) {
                std::string const counter_name = 
                    counter_names[profiler_record->counters[i].counter_handler.handle] ; 
                ASSERT(counter_name != "", "Counter name returned empty string in rocprofiler, this should never happen.") ; 
                GRACE_TRACE( "Writing profiling counter data: counter id {}, name {}"
                           , profiler_record->counters[i].counter_handler.handle, counter_name ) ; 
                output_file << "    " << counter_name << " (" << profiler_record->counters[i].value.value << ")" << std::endl;
            }
        }
    }
    output_file << '\n'; 
}
#undef CHECK_ROCPROFILER
#endif
