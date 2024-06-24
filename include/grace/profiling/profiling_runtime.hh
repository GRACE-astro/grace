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
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/runtime_functions.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/error.hh>
#include <grace/system/print.hh>
#ifdef GRACE_ENABLE_HIP
#include <hip/hip_runtime.h>
#include <rocprofiler/v2/rocprofiler.h>
#endif 

#include <stack>
#include <string>
#include <chrono>
#include <filesystem>
#include <vector>
#include <iostream>
#include <iomanip>
#include <functional>
#include <unordered_map>

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
    //! Timers for host code sections.
    std::stack<std::pair<std::string,std::chrono::high_resolution_clock::time_point>> _host_timers ; 
    //! Base path for profiling output.
    std::filesystem::path _base_outpath ; 
    #ifdef GRACE_ENABLE_HIP
    //! GPU profiling regions LIFO queue.
    std::stack< rocm_profiling_context_t > _gpu_profiling_active_regions  ;
    //! Names of GPU profiling regions currently in the queue.
    std::stack< std::string >  _gpu_profiling_active_regions_names        ;
    //! Hardware or derived counters requested for sampling.
    std::unordered_map<size_t,std::string> _gpu_profiling_active_counters ;
    #endif 
    bool _do_gpu_profiling ; 
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
    std::unordered_map<size_t,std::string> GRACE_ALWAYS_INLINE 
    active_hardware_counters() {
        return _gpu_profiling_active_counters ; 
    }
    std::string GRACE_ALWAYS_INLINE 
    top_gpu_region_name() const {
        return _gpu_profiling_active_regions_names.top() ; 
    }
    //*********************************************************************************************************************
    /**
     * @brief Initiate a device profiling region.
     * 
     * @param name Name of the profiling region.
     * When the backend is HIP, this translates to a rocprofiler call.
     * Ensure that rocprofiler is available on your system or deactivate
     * profiling altogether.
     */
    void push_device_region(std::string const& name) {
        if(_do_gpu_profiling){
        #ifdef GRACE_ENABLE_HIP
            _gpu_profiling_active_regions.emplace(
                rocprofiler_session_id_t{}, rocprofiler_buffer_id_t{}
            ); 
            _gpu_profiling_active_regions_names.emplace(  name  ) ; 
            // Create new session and start collecting kernel data
            rocm_initiate_profiling_session(_gpu_profiling_active_regions.top(), _gpu_profiling_active_counters) ;
            GRACE_TRACE("Entering profiling region (sid {})", 
            static_cast<int>(_gpu_profiling_active_regions.top()._sid.handle)) ;  
        #endif 
        }
    }
    //*********************************************************************************************************************
    /**
     * @brief End the last device profiling region.
     * 
     * When the backend is HIP, this translates to a rocprofiler call.
     * Ensure that rocprofiler is available on your system or deactivate
     * profiling altogether. 
     */
    void pop_device_region() {
        if(_do_gpu_profiling){
        #ifdef GRACE_ENABLE_HIP
            rocm_terminate_profiling_session(_gpu_profiling_active_regions.top());
            GRACE_TRACE("Exiting profiling region (sid {})", 
            static_cast<int>(_gpu_profiling_active_regions.top()._sid.handle)) ; 
            _gpu_profiling_active_regions.pop() ; 
            _gpu_profiling_active_regions_names.pop() ; 
        #endif 
        }
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
        auto start_time = _host_timers.top().second;
        auto label = _host_timers.top().first;
        _host_timers.pop();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        write_host_timer(label, grace::get_iteration(), elapsed_time) ;
    } 
    //*********************************************************************************************************************
 private:
    //*********************************************************************************************************************
    /**
     * @brief (Never) construct a new profiling runtime.
     * 
     */
    profiling_runtime_impl_t() {
        auto hasher = std::hash<std::string>{} ; 
        auto const counters 
            = grace::get_param<std::vector<std::string>>("profiling","enabled_hardware_counters") ;
        for( auto const& x: counters) 
            _gpu_profiling_active_counters.emplace(std::make_pair(hasher(x),x)) ;  
        _base_outpath = 
            static_cast<std::filesystem::path>(
                grace::get_param<std::string>("profiling","output_base_directory") 
            ) ;
        _do_gpu_profiling = grace::get_param<bool>("profiling", "do_gpu_profiling") ;  
        if (!std::filesystem::exists(_base_outpath)) {
            // Create the directory if it doesn't exist
            if (!std::filesystem::create_directory(_base_outpath)) {
                ERROR("Failed to create directory.") ;
            }
        }
        if(_do_gpu_profiling){
        #ifdef GRACE_ENABLE_HIP
            std::stringstream os ; 
            os << "GPU profiling enabled on HIP gpu (rocprofiler)\n" ;
            os << "   requested counters: [ " ; 
            for( auto const& x: _gpu_profiling_active_counters)
                os << x.second << " " ; 
            os << "]\n" ; 
            GRACE_INFO(os.str()) ; 
            rocprofiler_initialize();
        #endif    
        }      
    }
    //*********************************************************************************************************************
    /**
     * @brief (Never) destroy the profiling object.
     * 
     */
    ~profiling_runtime_impl_t() {
        if(_do_gpu_profiling){
        #ifdef GRACE_ENABLE_HIP
            rocprofiler_finalize();
        #endif 
        }
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
    void write_host_timer(std::string const& name, size_t iter, double time) {
        if( parallel::mpi_comm_rank() == 0 ) {
            GRACE_TRACE("Writing host profiling data at iteration {}", iter) ; 
            std::filesystem::path outf = 
                _base_outpath / (name + "_host_timers.dat") ; 
            if( not std::filesystem::exists(outf) ) {
                std::ofstream outfile { outf.string() };
                outfile << std::left  << std::setw(20) << "Iteration"
                        << std::left  << std::setw(20) << "Time [mus]\n" ;
            }
            std::ofstream outfile { outf.string(), std::ios::app} ; 
            outfile << std::fixed << std::setprecision(15) ; 
            outfile << std::left  << std::setw(20) << iter 
                    << std::left  << std::setw(20) << time << '\n'; 
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