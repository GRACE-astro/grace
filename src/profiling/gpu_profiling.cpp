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
#include <string>
#include <filesystem>
#include <iostream>
#include <iomanip>

#ifdef GRACE_ENABLE_HIP
#define ROCTRACER_CALL(ret,call) \
ASSERT( (ret = call) == ROCTRACER_STATUS_SUCCESS \
      , "Call to roctracer failed with error code " << err )
#endif 

#ifdef GRACE_ENABLE_HIP
extern "C" void roctracer_activity_callback(const char* begin, const char* end, void* arg) {
    roctracer_status_t err ;
    std::filesystem::path basepath = grace::profiling_runtime::get().out_basepath() ; 
    static constexpr unsigned int width = 20 ; 
    auto const iter = grace::get_iteration() ; 
    const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
    while (record != reinterpret_cast<const roctracer_record_t*>(end)) {
        if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
            std::filesystem::path outfname = basepath / std::string(record->name) + "_gpu_timers.dat" ; 
            if( not std::filesystem::exists(outfname) ) {
                std::ofstream outfile{outfname.string(), std::ios::app} ; 
                outfile << std::left  << std::setw(width) << "Iteration"
                        << std::left  << std::setw(width) << "Begin [ns]"
                        << std::left  << std::setw(width) << "End [ns]"
                        << std::left  << std::setw(width) << "Duration [ns]"
                        << std::left  << std::setw(width) << "VALU"
                        << std::left  << std::setw(width) << "SALU"
                        << std::left  << std::setw(width) << "Load"
                        << std::left  << std::setw(width) << "Store"
                        << std::left  << std::setw(width) << "Tot instructions"
                        << std::left  << std::setw(width) << "Bytes read"
                        << std::left  << std::setw(width) << "Bytes written\n" ; 
            } 
            std::ofstream outfile{outfname.string(), std::ios::app} ; 
            outfile << std::fixed << std::setprecision(15) ; 
            outfile << std::left  << std::setw(width) << iter
                    << std::left  << std::setw(width) << record->begin_ns 
                    << std::left  << std::setw(width) << record->end_ns  
                    << std::left  << std::setw(width) << record->end_ns - record->begin_ns ;
            outfile << "Kernel Execution: " << record->name << std::endl;
            outfile << "Start Time: " << record->begin_ns << " ns" << std::endl;
            outfile << "End Time: " << record->end_ns << " ns" << std::endl;
            if (record->data.instruction_count != 0) {
                outfile << std::left  << std::setw(width) << record->data.instruction_count.valu
                        << std::left  << std::setw(width) << record->data.instruction_count.salu
                        << std::left  << std::setw(width) << record->data.instruction_count.load
                        << std::left  << std::setw(width) << record->data.instruction_count.store
                        << std::left  << std::setw(width) << record->data.instruction_count ; 
            } 
            if (record->data.mem_bytes != 0) {
                outfile << std::left  << std::setw(width) << record->data.mem_bytes.read
                        << std::left  << std::setw(width) << record->data.mem_bytes.write ; 
            }
            outfile << '\n' ; 
        }
        ROCTRACER_CALL(err,roctracer_next_record(record, &record));
    }
}
#endif
