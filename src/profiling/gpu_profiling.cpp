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

#include <grace/profiling/gpu_profiling.hh>
#include <grace/profiling/profiling_runtime.hh>
#include <grace/system/grace_system.hh>
#include <grace/errors/error.hh>
#ifdef GRACE_ENABLE_HIP
#include <hip/hip_runtime.h>
#include <rocprofiler/v2/rocprofiler.h>
#endif 

#include <string>
#include <filesystem>
#include <iostream>
#include <iomanip>
#define GRACE_ENABLE_HIP

#ifdef GRACE_ENABLE_HIP
#ifndef CHECK_ROCPROFILER
#define CHECK_ROCPROFILER(call)                                                                    \
  do {                                                                                             \
    if ((call) != ROCPROFILER_STATUS_SUCCESS)                                                      \
        ERROR("ROCProfiler API call error!");                                    \
  } while (false)
#endif 

void rocm_initiate_profiling_session( rocm_profiling_context_t& context, std::vector<const char*> counters ) 
{

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
                                 , rocprofiler_filter_data_t{ .counters_names = counters.data() }
                                 , counters.size() 
                                 , &filter_id, property) 
    ) ; 
    CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(context._sid,filter_id,context._bid)) ;
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
        CHECK_ROCPROFILER(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME,
                                                        profiler_record->kernel_id, &kernel_name_c));
    }
    std::string kernel_name = detail::rocmprofiler_cxx_demangle(kernel_name_c);
    // 
    std::string const pfname = section_name + std::string("_gpu_counters_") + std::to_string(rank) + std::string(".dat") ;
    std::filesystem::path outfname = 
        basepath / pfname ; 
    
    std::ofstream output_file{outfname.string(), std::ios::app} ;
    output_file << std::string("Rank[") << std::to_string(rank) << "], " ; 
    output_file << std::string("dispatch[") << std::to_string(profiler_record->header.id.handle)
              << "], " << std::string("gpu_id(") << std::to_string(profiler_record->gpu_id.handle)
              << "), " << std::string("queue_id(")
              << std::to_string(profiler_record->queue_id.handle) << "), "
              << std::string("queue_index(") << std::to_string(profiler_record->queue_idx.value)
              << "), " << std::string("tid(") << std::to_string(profiler_record->thread_id.value) << ")";
    output_file << ", " << std::string("grd(")
                << std::to_string(profiler_record->kernel_properties.grid_size) << "), "
                << std::string("wgr(")
                << std::to_string(profiler_record->kernel_properties.workgroup_size) << "), "
                << std::string("lds(")
                << std::to_string(
                        ((profiler_record->kernel_properties.lds_size + (lds_block_size - 1)) &
                        ~(lds_block_size - 1)))
                << "), " << std::string("scr(")
                << std::to_string(profiler_record->kernel_properties.scratch_size) << "), "
                << std::string("arch_vgpr(")
                << std::to_string(profiler_record->kernel_properties.arch_vgpr_count) << "), "
                << std::string("accum_vgpr(")
                << std::to_string(profiler_record->kernel_properties.accum_vgpr_count) << "), "
                << std::string("sgpr(")
                << std::to_string(profiler_record->kernel_properties.sgpr_count) << "), "
                << std::string("wave_size(")
                << std::to_string(profiler_record->kernel_properties.wave_size) << "), "
                << std::string("sig(")
                << std::to_string(profiler_record->kernel_properties.signal_handle);
    output_file << "), " << std::string("obj(") << std::to_string(profiler_record->kernel_id.handle)
                << "), " << std::string("kernel-name(\"") << kernel_name << "\")"
                << std::string(", time(") << std::to_string(profiler_record->timestamps.begin.value)
                << ") ";
    output_file << std::endl;
    if (profiler_record->counters) {
        for (uint64_t i = 0; i < profiler_record->counters_count.value; i++) {
        if (profiler_record->counters[i].counter_handler.handle > 0) {
            size_t counter_name_length = 0;
            CHECK_ROCPROFILER(rocprofiler_query_counter_info_size(
                session_id, ROCPROFILER_COUNTER_NAME, profiler_record->counters[i].counter_handler,
                &counter_name_length));
            if (counter_name_length > 1) {
            const char* name_c = static_cast<const char*>(malloc(name_length * sizeof(char)));
            CHECK_ROCPROFILER(rocprofiler_query_counter_info(
                session_id, ROCPROFILER_COUNTER_NAME, profiler_record->counters[i].counter_handler,
                &name_c));
            output_file << ", " << name_c << " ("
                        << std::to_string(profiler_record->counters[i].value.value) << ")"
                        << std::endl;
            }
        }
        }
    }

}
#undef CHECK_ROCPROFILER
#endif
