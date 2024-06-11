/**
 * @file profiling_runtime.hh
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
#ifndef GRACE_PROFILING_PROFILING_RUNTIME_HH
#define GRACE_PROFILING_PROFILING_RUNTIME_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/profiling/profiling.hh>
#include <grace/system/runtime_functions.hh>
#include <grace/config/config_parser.hh>

#ifdef GRACE_ENABLE_HIP
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_ext.h>
#endif 

#include <stack>
#include <string>
#include <chrono>
#include <filesystem>
#include <vector>
#include <iostream>
#include <iomanip>

namespace grace {
//*********************************************************************************************************************
/**
 * \defgroup profiling Profiling Utilities
 * 
 */
//*********************************************************************************************************************
//*********************************************************************************************************************
/**
 * @brief Class in charge of initializing / finalizing profiling runtimes.
 * \ingroup profiling
 * Provides an interface which is then wrapped by a singleton_holder.
 */
class profiling_runtime_impl_t
{
 private:
    //*********************************************************************************************************************
    //! Timers for host code sections
    std::stack<std::pair<std::string,std::chrono::high_resolution_clock::time_point>> _host_timers ; 
    //! Durations of timers for host code sections (in microseconds)
    std::unordered_map<std::string, std::pair<long,long long>> _host_timers_results ;
    //! Base path for profiling output
    std::filesystem::path _base_outpath ; 
    //*********************************************************************************************************************
 public:
    //*********************************************************************************************************************
    /**
     * @brief Get the output path for timers.
     * 
     * @return std::string The output path.
     */
    std::string GRACE_ALWAYS_INLINE 
    out_basepath() const {
        return _base_outpath ; 
    }
    //*********************************************************************************************************************
    /**
     * @brief Initiate a device profiling region.
     * 
     * @param name Name of the profiling region.
     * When the backend is HIP, this translates to a roctracer call.
     * Ensure that roctracer is available on your system or deactivate
     * profiling altogether.
     */
    void push_device_region(std::string const& name) {
        #ifdef GRACE_ENABLE_HIP
        roctracer_start() ; 
        #endif 
    }
    //*********************************************************************************************************************
    /**
     * @brief End the last device profiling region.
     * 
     * When the backend is HIP, this translates to a roctracer call.
     * Ensure that roctracer is available on your system or deactivate
     * profiling altogether. 
     */
    void pop_device_region() const {
        #ifdef GRACE_ENABLE_HIP
        roctracer_pop() ; 
        #endif 
    }
    //*********************************************************************************************************************
    /**
     * @brief Initiate a host profiling region.
     * 
     * @param name Name of the profiling region.
     * This will output timing information to a file.
     */
    void push_host_region(std::string const& name) {
        _host_timers.push({name, std::chrono::high_resolution_clock::now()}) ; 
    }
    //*********************************************************************************************************************
    /**
     * @brief End the last host profiling region.
     * 
     */
    void pop_host_region() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto start_time = timers.top().second;
        auto label = _host_timers.top().first;
        _host_timers.pop();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        _host_timers_results[label].push_back({grace::get_iteration(), elapsed_time}); 
        write_host_timers() ;
    } 
    //*********************************************************************************************************************
 private:
    //*********************************************************************************************************************
    /**
     * @brief (Never) construct a new profiling runtime.
     * 
     */
    profiling_runtime_impl_t() {
        #ifdef GRACE_ENABLE_HIP
        roctracer_properties_t properties{};
        properties.buffer_size = 0x1000;  // Example buffer size
        properties.counters |= ROCTRACER_COUNTER_INSTRUCTION_COUNT;
        properties.counters |= ROCTRACER_COUNTER_MEM_BYTES;
        roctracer_open_pool(&properties);
        roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, roctracer_activity_callback, nullptr);
        #endif 
        auto& params = grace::config_parser::get() ; 
        _base_outpath = 
            static_cast<std::filesystem::path>(params["profiling"]["output_base_directory"].as<std::string>()) ; 
        
    }
    //*********************************************************************************************************************
    /**
     * @brief (Never) destroy the profiling object.
     * 
     */
    ~profiling_runtime_impl_t() {
        #ifdef GRACE_ENABLE_HIP
        roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
        roctracer_close_pool();
        #endif 
    }
    //*********************************************************************************************************************
    
    //*********************************************************************************************************************
    friend class utils::singleton_holder<profiling_runtime_impl_t,memory::default_create> ;           //!< Give access
    friend class memory::new_delete_creator<profiling_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access
    //*********************************************************************************************************************
    static constexpr size_t longevity = GRACE_PROFILING_RUNTIME ; //!< Longevity
    //*********************************************************************************************************************
    /**
     * @brief Write host timers to files.
     * 
     */
    void write_host_timers() {
        for( const auto& entry: _host_timers_results) {
            std::filesystem::path outf = 
                _base_outpath / (entry.first + "_host_timers.dat") ; 
            if( not std::filesystem::exists(outf) ) {
                std::ofstream outfile { _base_outpath.string() };
                outfile << std::left  << std::setw(20) << "Iteration"
                        << std::left  << std::setw(20) << "Time [mus]\n" ;
            }
            std::ofstream outfile { _base_outpath.string(), std::ios::app} ; 
            outfile << std::fixed << std::setprecision(15) ; 
            outfile << std::left  << std::setw(20) << entry.second.first 
                    << std::left  << std::setw(20) << entry.second.second << '\n'; 
        }
    }
    //*********************************************************************************************************************
} ; 
//*********************************************************************************************************************
/**
 * @brief Singleton in charge of initializing / finalizing profiling environment.
 * \ingroup profiling
 * Profilers in GRACE are implemented as three LIFO queues. The first queue holds host profiling timers 
 * that are simply written to a plain text file when the execution section ends. The second queue regards 
 * device performance counters and its implementation is backend dependent. For HIP backends, this is implemented 
 * using <code>roctracer</code>. This queue is used specifically to collect information about device kernels that 
 * is not easy to obtain with native Kokkos tools (without something like TAU or VTune), such as instruction counts 
 * and/or memory events. Note that the roctracer library and rocprof have to be present on the system and properly 
 * configured for this to produce meaningful output. The final profiling queue consists of <code>Kokkos::Profiling</code> 
 * regions which can be used with the quite flexible Kokkos Tools ecosystem to provide basic timings and memory 
 * information as well as more detailed profiling data when coupled to third party tools. The way to open / close a profiling
 * region in GRACE is through the macros provided in grace/profiling/profiling.hh.
 * NB: \b All profiling-related calls, including those for GPU profiling, need to happen on Host. In other words, it is 
 * illegal to call PUSH/POP from device code.
 */
using profiling_runtime = utils::singleton_holder<profiling_runtime_impl_t, memory::default_create> ; 
//*********************************************************************************************************************
//*********************************************************************************************************************
} /* namespace grace */


#endif /* GRACE_PROFILING_PROFILING_RUNTIME_HH */