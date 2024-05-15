/**
 * @file thunder_runtime.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-12
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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
#ifndef INCLUDE_THUNDER_SYSTEM_THUNDER_RUNTIME
#define INCLUDE_THUNDER_SYSTEM_THUNDER_RUNTIME

#include <thunder_config.h>

#include <thunder/utils/inline.h>
#include <thunder/utils/singleton_holder.hh> 
#include <thunder/utils/creation_policies.hh>
#include <thunder/utils/lifetime_tracker.hh> 

#include <thunder/system/runtime_functions.hh>

#include <thunder/config/config_parser.hh>

#include <thunder/parallel/mpi_wrappers.hh>

#include <thunder/data_structures/variable_indices.hh>

#include <spdlog/stopwatch.h>

#include <string>
#include <vector> 
#include <iostream>
#include <algorithm> 
#include <filesystem> 

namespace thunder {

class thunder_runtime_impl_t 
{
 private:
    /* Volume output */
    std::vector<std::string> _cell_volume_output_scalar_vars ;
    std::vector<std::string> _cell_volume_output_vector_vars ;
    std::vector<std::string> _cell_volume_output_scalar_aux ;
    std::vector<std::string> _cell_volume_output_vector_aux ;
    /* Surface output */
    std::vector<std::string> _cell_surface_output_scalar_vars ;
    std::vector<std::string> _cell_surface_output_vector_vars ;
    std::vector<std::string> _cell_surface_output_scalar_aux ;
    std::vector<std::string> _cell_surface_output_vector_aux ;
    /* Output parameters */ 
    bool   _volume_output        ;
    bool   _surface_output       ; 
    size_t _volume_output_every  ; 
    size_t _surface_output_every ; 
    std::filesystem::path _volume_io_basepath ;
    std::filesystem::path _surface_io_basepath ;
    std::string _volume_io_basename  ; 
    std::string _surface_io_basename ;
    /* iteration count */ 
    size_t _iter ; 
    /* current simulation time */
    double _time, _dt ; 
    /* total runtime clock */
    spdlog::stopwatch _runtime ; 
 public: 
    
    size_t THUNDER_ALWAYS_INLINE 
    iteration() const { return _iter ; }

    void THUNDER_ALWAYS_INLINE  
    increment_iteration() {
        _iter++ ; 
    }

    double THUNDER_ALWAYS_INLINE
    time() const {
        return _time ;
    }

    double THUNDER_ALWAYS_INLINE 
    timestep() const {
        return _dt ; 
    }

    void THUNDER_ALWAYS_INLINE 
    increment_time() {
        _time += _dt ; 
    }

    double THUNDER_ALWAYS_INLINE 
    timestep_size() const {
        return _dt ; 
    }

    void THUNDER_ALWAYS_INLINE 
    set_timestep(double const& _new_dt ) {
        _dt = _new_dt ; 
    }

    size_t THUNDER_ALWAYS_INLINE 
    volume_output_every()  const { return _volume_output_every ; }

    size_t THUNDER_ALWAYS_INLINE 
    surface_output_every() const { return _surface_output_every ; }

    std::string THUNDER_ALWAYS_INLINE
    volume_io_basepath() const { return _volume_io_basepath ; }

    std::string THUNDER_ALWAYS_INLINE
    surface_io_basepath() const { return _surface_io_basepath ; }

    std::string THUNDER_ALWAYS_INLINE
    volume_io_basename() const { return _volume_io_basename ; }

    std::string THUNDER_ALWAYS_INLINE
    surface_io_basename() const { return _surface_io_basename ; }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_volume_output_scalar_vars() const {
        return _cell_volume_output_scalar_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_volume_output_vector_vars() const {
        return _cell_volume_output_vector_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_volume_output_scalar_aux() const {
        return _cell_volume_output_scalar_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_volume_output_vector_aux() const {
        return _cell_volume_output_vector_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_surface_output_scalar_vars() const {
        return _cell_surface_output_scalar_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_surface_output_vector_vars() const {
        return _cell_surface_output_vector_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_surface_output_scalar_aux() const {
        return _cell_surface_output_scalar_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_surface_output_vector_aux() const {
        return _cell_surface_output_vector_aux; 
    }

    double THUNDER_ALWAYS_INLINE 
    elapsed() {
        return _runtime.elapsed().count() ; 
    }

 private:

    thunder_runtime_impl_t() {
        auto& params = thunder::config_parser::get() ; 
        /* 
         * parse IO section of parfile and sort variables into aux and state 
         * and into scalars and vectors.
        */
        _surface_output = params["IO"]["surface_output"].as<bool>() ; 
        _volume_output = params["IO"]["volume_output"].as<bool>() ; 
        _surface_output_every = params["IO"]["surface_output_every"].as<size_t>() ; 
        _volume_output_every = params["IO"]["volume_output_every"].as<size_t>() ; 
        _volume_io_basename  = params["IO"]["volume_output_base_filename"].as<std::string>(); 
        _surface_io_basename  = params["IO"]["surface_output_base_filename"].as<std::string>();
        _volume_io_basepath  = 
            std::filesystem::path(params["IO"]["volume_output_base_directory"].as<std::string>()); 
        _surface_io_basepath  = 
            std::filesystem::path(params["IO"]["surface_output_base_directory"].as<std::string>()); 

        if( not std::filesystem::exists( _volume_io_basepath ) ){
            std::filesystem::create_directory(_volume_io_basepath) ; 
        }

        if( not std::filesystem::exists( _surface_io_basepath ) ){
            std::filesystem::create_directory(_surface_io_basepath) ; 
        }

        auto out_cell_vars_volume = 
            params["IO"]["volume_output_cell_variables"].as<std::vector<std::string>>() ; 
        auto out_cell_vars_surface = 
            params["IO"]["surface_output_cell_variables"].as<std::vector<std::string>>() ; 
        auto& vnames = thunder::variables::detail::_varnames ; 
        auto& vprops = thunder::variables::detail::_varprops ; 
        auto& auxnames = thunder::variables::detail::_auxnames ;
        auto& auxprops = thunder::variables::detail::_auxprops ;  
        for( auto const& x: out_cell_vars_volume ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                if( vprops[x].is_vector ){
                     _cell_volume_output_vector_vars.push_back(vprops[x].name) ; 
                } else {
                    _cell_volume_output_scalar_vars.push_back(x) ;
                }
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                if( auxprops[x].is_vector ){
                     _cell_volume_output_vector_aux.push_back(auxprops[x].name) ; 
                } else {
                    _cell_volume_output_scalar_aux.push_back(x) ;
                } 
            } else { 
                /* WARN(1, "variable " << x " not found.") ; */
            }
        } 

        for( auto const& x: out_cell_vars_surface ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                if( vprops[x].is_vector ){
                     _cell_surface_output_vector_vars.push_back(vprops[x].name) ; 
                } else {
                    _cell_surface_output_scalar_vars.push_back(x) ;
                }
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                if( auxprops[x].is_vector ){
                     _cell_surface_output_vector_aux.push_back(auxprops[x].name) ; 
                } else {
                    _cell_surface_output_scalar_aux.push_back(x) ;
                } 
            } else { 
                /* WARN(1, "variable " << x " not found.") ; */
            }
        } 
        /****************************/
        /* Set iteration count to 0 */ 
        _iter = 0UL ; 
        /* Set time to 0            */
        _time = 0.0 ; 
        _dt   = 0.0 ;  
        /****************************/
        if( parallel::mpi_comm_rank() == thunder::master_rank() ) 
        {
            if ( _volume_output ) 
            {
                std::cout << "Volume output requested every " << _volume_output_every << " iterations\n" ; 
                std::cout << "Variables registered for volume (co-dimension 0) output:\n" ; 
                std::cout << "Scalars: \n";
                for(auto const& x: _cell_volume_output_scalar_vars){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Vectors: \n";
                for(auto const& x: _cell_volume_output_vector_vars){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Auxiliaries: \n";
                std::cout << "Scalars: \n";
                for(auto const& x: _cell_volume_output_scalar_aux){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Vectors: \n";
                for(auto const& x: _cell_volume_output_vector_aux){
                    std::cout << x << std::endl ; 
                }      
            }
            if ( _surface_output )
            {
                std::cout << "Surface output requested every " << _surface_output_every << " iterations\n" ; 
                std::cout << "Variables registered for surface (co-dimension 1) output: \n" ; 
                std::cout << "Scalars: \n";
                for(auto const& x: _cell_surface_output_scalar_vars){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Vectors: \n";
                for(auto const& x: _cell_surface_output_vector_vars){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Auxiliaries: \n";
                std::cout << "Scalars: \n";
                for(auto const& x: _cell_surface_output_scalar_aux){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Vectors: \n";
                for(auto const& x: _cell_surface_output_vector_aux){
                    std::cout << x << std::endl ; 
                }    
            }
        }
    }
    ~thunder_runtime_impl_t() {} 

    friend class utils::singleton_holder<thunder_runtime_impl_t,memory::default_create> ; 
    friend class memory::new_delete_creator<thunder_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access

    static constexpr size_t longevity = THUNDER_RUNTIME ; 

} ; 

using runtime = utils::singleton_holder<thunder_runtime_impl_t,memory::default_create> ;

} /* namespace thunder */

#endif /* INCLUDE_THUNDER_SYSTEM_THUNDER_RUNTIME */
