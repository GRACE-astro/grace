/**
 * @file grace_runtime.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-12
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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
#ifndef INCLUDE_GRACE_SYSTEM_GRACE_RUNTIME
#define INCLUDE_GRACE_SYSTEM_GRACE_RUNTIME

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/singleton_holder.hh> 
#include <grace/utils/creation_policies.hh>
#include <grace/utils/lifetime_tracker.hh> 

#include <grace/system/runtime_functions.hh>

#include <grace/config/config_parser.hh>

#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/print.hh>
#include <grace/data_structures/variable_indices.hh>

#include <spdlog/stopwatch.h>

#include <string>
#include <vector> 
#include <iostream>
#include <algorithm> 
#include <unordered_set>
#include <filesystem> 

namespace grace {

template <typename Container>
static void print_list(const std::string& title, const Container& c)
{
    std::cout << "    " << title << ":\n";

    if (c.empty()) {
        std::cout << "      (none)\n";
        return;
    }

    for (const auto& x : c) {
        std::cout << "      - " << x << '\n';
    }
}

template <typename S, typename V, typename AS, typename AV>
void print_variable_group(const S& scalars,
                          const V& vectors,
                          const AS& aux_scalars,
                          const AV& aux_vectors)
{
    print_list("Scalars", scalars);
    print_list("Vectors", vectors);

    std::cout << "    Auxiliaries:\n";
    print_list("Scalars", aux_scalars);
    print_list("Vectors", aux_vectors);
}

enum terminate : uint8_t {ITERATION, TIME, WALLTIME} ; 

class grace_runtime_impl_t 
{
 private:
    /* Volume output */
    std::vector<std::string> _cell_volume_output_scalar_vars ;
    std::vector<std::string> _cell_volume_output_vector_vars ;
    std::vector<std::string> _cell_volume_output_scalar_aux ;
    std::vector<std::string> _cell_volume_output_vector_aux ;
    /* Surface output */
    std::vector<std::string> _cell_plane_surface_output_scalar_vars ;
    std::vector<std::string> _cell_plane_surface_output_vector_vars ;
    std::vector<std::string> _cell_plane_surface_output_scalar_aux ;
    std::vector<std::string> _cell_plane_surface_output_vector_aux ;
    /* Sphere surface output */
    std::vector<std::string> _cell_sphere_surface_output_scalar_vars ;
    std::vector<std::string> _cell_sphere_surface_output_vector_vars ;
    std::vector<std::string> _cell_sphere_surface_output_scalar_aux ;
    std::vector<std::string> _cell_sphere_surface_output_vector_aux ;
    /* Scalar output         */
    std::vector<std::string> _scalar_output_minmax_vars   ; 
    std::vector<std::string> _scalar_output_minmax_aux    ;
    std::vector<std::string> _scalar_output_norm2_vars    ; 
    std::vector<std::string> _scalar_output_norm2_aux     ;
    std::vector<std::string> _scalar_output_integral_vars ; 
    std::vector<std::string> _scalar_output_integral_aux  ;
    /* Info output         */
    std::vector<std::string> _info_output_max_vars   ; 
    std::vector<std::string> _info_output_max_aux    ;
    std::vector<std::string> _info_output_min_vars   ; 
    std::vector<std::string> _info_output_min_aux    ;
    std::vector<std::string> _info_output_norm2_vars ; 
    std::vector<std::string> _info_output_norm2_aux  ;
    /* Reduction variable lists */
    std::vector<std::string> _minmax_reduction_vars   ;
    std::vector<std::string> _minmax_reduction_aux    ;
    std::vector<std::string> _norm2_reduction_vars    ;
    std::vector<std::string> _norm2_reduction_aux     ;
    std::vector<std::string> _integral_reduction_vars ;
    std::vector<std::string> _integral_reduction_aux  ;
    /* Output planes */
    std::vector<double> _output_planes_origins ; 
    /* Output spheres */
    int _n_output_spheres ; 
    std::vector<std::string>          _output_spheres_names   ; 
    int _volume_output_every  ; 
    int _plane_surface_output_every ; 
    int _sphere_surface_output_every ; 
    int _scalar_output_every         ; 
    int _info_output_every           ;
    std::filesystem::path _volume_io_basepath ;
    std::filesystem::path _surface_io_basepath ;
    std::filesystem::path _sphere_io_basepath ; 
    std::filesystem::path _scalar_io_basepath ;
    std::string _volume_io_basename  ; 
    std::string _surface_io_basename ;
    
    std::string _scalar_io_basename ;
    /* iteration count */ 
    size_t _iter ; 
    /* current simulation time */
    double _time, _dt, _initial_time, _initial_wtime ; 
    /* total walltime clock */
    spdlog::stopwatch _walltime ; 
    /* termination condition */
    terminate _term_cnd ; 
    /* max iter */
    size_t _max_iter ; 
    /* max time */
    double _max_time ; 
    /* max walltime */
    double _max_walltime ; 
 public: 
    
    terminate GRACE_ALWAYS_INLINE
    termination_condition() const {return _term_cnd; }

    size_t GRACE_ALWAYS_INLINE
    max_iteration() const {return _max_iter; }

    double GRACE_ALWAYS_INLINE
    max_walltime() const {return _max_walltime;} 

    double GRACE_ALWAYS_INLINE
    max_time() const {return _max_time;} 

    size_t GRACE_ALWAYS_INLINE 
    iteration() const { return _iter ; }

    void GRACE_ALWAYS_INLINE  
    increment_iteration() {
        _iter++ ; 
    }

    double GRACE_ALWAYS_INLINE
    time() const {
        return _time ;
    }

    double GRACE_ALWAYS_INLINE
    initial_time() const {
        return _initial_time ;
    }

    double GRACE_ALWAYS_INLINE 
    timestep() const {
        return _dt ; 
    }

    void GRACE_ALWAYS_INLINE 
    increment_time() {
        _time += _dt ; 
    }

    double GRACE_ALWAYS_INLINE 
    timestep_size() const {
        return _dt ; 
    }

    void GRACE_ALWAYS_INLINE 
    set_timestep(double const& _new_dt ) {
        _dt = _new_dt ; 
    }

    void GRACE_ALWAYS_INLINE 
    set_simulation_time(double const& _new_t ) {
        _time = _new_t ; 
    }

    void GRACE_ALWAYS_INLINE 
    set_initial_simulation_time(double const& _new_t ) {
        _initial_time = _new_t ; 
    }

    void GRACE_ALWAYS_INLINE 
    set_iteration(size_t const& _new_it ) {
        _iter = _new_it ; 
    }

    int GRACE_ALWAYS_INLINE 
    volume_output_every()  const { return _volume_output_every ; }

    int GRACE_ALWAYS_INLINE 
    plane_surface_output_every() const { return _plane_surface_output_every ; }

    int GRACE_ALWAYS_INLINE 
    sphere_surface_output_every() const { return _sphere_surface_output_every ; }

    int GRACE_ALWAYS_INLINE 
    scalar_output_every() const { return _scalar_output_every ; }

    int GRACE_ALWAYS_INLINE 
    info_output_every() const { return _info_output_every ; }

    std::string GRACE_ALWAYS_INLINE
    volume_io_basepath() const { return _volume_io_basepath ; }

    std::string GRACE_ALWAYS_INLINE
    surface_io_basepath() const { return _surface_io_basepath ; }

    std::string GRACE_ALWAYS_INLINE
    sphere_io_basepath() const { return _sphere_io_basepath ; }

    std::string GRACE_ALWAYS_INLINE
    scalar_io_basepath() const { return _scalar_io_basepath ; }

    std::string GRACE_ALWAYS_INLINE
    volume_io_basename() const { return _volume_io_basename ; }

    std::string GRACE_ALWAYS_INLINE
    surface_io_basename() const { return _surface_io_basename ; }

    std::string GRACE_ALWAYS_INLINE
    scalar_io_basename() const { return _scalar_io_basename ; }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_volume_output_scalar_vars() const {
        return _cell_volume_output_scalar_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_volume_output_vector_vars() const {
        return _cell_volume_output_vector_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_volume_output_scalar_aux() const {
        return _cell_volume_output_scalar_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_volume_output_vector_aux() const {
        return _cell_volume_output_vector_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_plane_surface_output_scalar_vars() const {
        return _cell_plane_surface_output_scalar_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_plane_surface_output_vector_vars() const {
        return _cell_plane_surface_output_vector_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_plane_surface_output_scalar_aux() const {
        return _cell_plane_surface_output_scalar_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_plane_surface_output_vector_aux() const {
        return _cell_plane_surface_output_vector_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_scalar_vars() const {
        return _cell_sphere_surface_output_scalar_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_vector_vars() const {
        return _cell_sphere_surface_output_vector_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_scalar_aux() const {
        return _cell_sphere_surface_output_scalar_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_vector_aux() const {
        return _cell_sphere_surface_output_vector_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    scalar_output_minmax_vars() const {
        return _scalar_output_minmax_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    scalar_output_minmax_aux() const {
        return _scalar_output_minmax_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    scalar_output_norm2_vars() const {
        return _scalar_output_norm2_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    scalar_output_norm2_aux() const {
        return _scalar_output_norm2_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    scalar_output_integral_vars() const {
        return _scalar_output_integral_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    scalar_output_integral_aux() const {
        return _scalar_output_integral_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    info_output_max_vars() const {
        return _info_output_max_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    info_output_max_aux() const {
        return _info_output_max_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    info_output_min_vars() const {
        return _info_output_min_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    info_output_min_aux() const {
        return _info_output_min_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    info_output_norm2_vars() const {
        return _info_output_norm2_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    info_output_norm2_aux() const {
        return _info_output_norm2_aux; 
    }
    
    decltype(auto) GRACE_ALWAYS_INLINE 
    minmax_reduction_vars() const {
        return _minmax_reduction_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    minmax_reduction_aux() const {
        return _minmax_reduction_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    norm2_reduction_vars() const {
        return _norm2_reduction_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    norm2_reduction_aux() const {
        return _norm2_reduction_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    integral_reduction_vars() const {
        return _integral_reduction_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    integral_reduction_aux() const {
        return _integral_reduction_aux; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_plane_surface_output_origins() const {
        return _output_planes_origins ;  
    }

    int GRACE_ALWAYS_INLINE 
    n_surface_output_spheres() const {
        return _n_output_spheres ; 
    }


    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_names() const {
        return _output_spheres_names ;
    }

    void GRACE_ALWAYS_INLINE 
    start_walltime_clock() {
        _initial_wtime = _walltime.elapsed().count() ; 
    }

    double GRACE_ALWAYS_INLINE 
    elapsed() {
        return _walltime.elapsed().count() - _initial_wtime; 
    }
    
    double GRACE_ALWAYS_INLINE 
    total_elapsed() {
        return _walltime.elapsed().count(); 
    }

 private:

    grace_runtime_impl_t() {
        auto& params = grace::config_parser::get() ; 
        /* 
         * parse IO section of parfile and sort variables into aux and state 
         * and into scalars and vectors.
        */
        /* Output frequencies              */
        _sphere_surface_output_every = grace::get_param<int>("IO", "sphere_surface_output_every") ; 
        _plane_surface_output_every = grace::get_param<int>("IO", "plane_surface_output_every") ; 
        _volume_output_every = grace::get_param<int>("IO", "volume_output_every") ; 
        _scalar_output_every = grace::get_param<int>("IO", "scalar_output_every") ; 
        _info_output_every = grace::get_param<int>("IO", "info_output_every") ; 
        /* Output filenames and directories */
        _volume_io_basename  = grace::get_param<std::string>("IO", "volume_output_base_filename"); 
        _surface_io_basename  = grace::get_param<std::string>("IO", "surface_output_base_filename");
        _scalar_io_basename  = grace::get_param<std::string>("IO", "scalar_output_base_filename");
        _volume_io_basepath  = 
            std::filesystem::path(grace::get_param<std::string>("IO", "volume_output_base_directory")); 
        _surface_io_basepath  = 
            std::filesystem::path(grace::get_param<std::string>("IO", "surface_output_base_directory")); 
        _scalar_io_basepath  = 
            std::filesystem::path(grace::get_param<std::string>("IO", "scalar_output_base_directory")); 
        _sphere_io_basepath = 
            std::filesystem::path(grace::get_param<std::string>("IO","sphere_surface_output_base_directory")) ; 
        /* Create output directories if they don't exist */
        if( not std::filesystem::exists( _volume_io_basepath ) and (parallel::mpi_comm_rank() == 0)){
            std::filesystem::create_directory(_volume_io_basepath) ; 
        }

        if( not std::filesystem::exists( _surface_io_basepath) and (parallel::mpi_comm_rank() == 0)) {
            std::filesystem::create_directory(_surface_io_basepath) ; 
        }

        if( not std::filesystem::exists( _sphere_io_basepath) and (parallel::mpi_comm_rank() == 0)) {
            std::filesystem::create_directory(_sphere_io_basepath) ; 
        }

        if( not std::filesystem::exists( _scalar_io_basepath) and (parallel::mpi_comm_rank() == 0)) {
            std::filesystem::create_directory(_scalar_io_basepath) ; 
        }
        /* Set output planes and spheres properties      */
        _output_planes_origins.resize(3) ;
        _output_planes_origins[0] = grace::get_param<double>("IO","xy_plane_offset") ;

        _output_planes_origins[1] = grace::get_param<double>("IO","xz_plane_offset") ; 

        _output_planes_origins[2] = grace::get_param<double>("IO","yz_plane_offset") ; 
        
        _n_output_spheres = grace::get_param<int>("IO", "n_output_spheres") ;
        _output_spheres_names = grace::get_param<std::vector<std::string>>("IO", "output_sphere_names") ; 
        /* Volume and surface output variables */
        #ifdef GRACE_ENABLE_COWLING_METRIC
        const std::vector<std::string> metric_vars = {
            "gamma[0,0]", "gamma[0,1]", 
            "gamma[0,2]", "gamma[1,1]", 
            "gamma[1,2]", "gamma[2,2]", 
            "ext_curv[0,0]", "ext_curv[0,1]", 
            "ext_curv[0,2]", "ext_curv[1,1]", 
            "ext_curv[1,2]", "ext_curv[2,2]", 
            "beta[0]",
            "alp"
        } ; 
        const std::vector<std::string> metric_aux = {};
        #endif 
        #ifdef GRACE_ENABLE_Z4C_METRIC
        const std::vector<std::string> metric_vars = {
            "gamma_tilde[0,0]", "gamma_tilde[0,1]", 
            "gamma_tilde[0,2]", "gamma_tilde[1,1]", 
            "gamma_tilde[1,2]", "gamma_tilde[2,2]", 
            "A_tilde[0,0]", "A_tilde[0,1]",
            "A_tilde[0,2]", "A_tilde[1,1]",
            "A_tilde[1,2]", "A_tilde[2,2]",
            "beta[0]",
            "alp",
            "z4c_Khat",
            "conf_fact",
            "z4c_Gamma[0]",
            "z4c_theta",
            "z4c_Bdriver[0]"
        } ;  
        const std::vector<std::string> metric_aux = {
            "z4c_H",
            "z4c_M[0]"
        };
        #endif
        const std::vector<std::string> hydro_aux  = {
            "rho", "press", "eps", "ye", "temperature",
            "Bvec[0]", "zvec[0]", "Bdiv", "c2p_err"
        };
        auto out_cell_vars_volume = get_param<std::vector<std::string>>("IO","volume_output_cell_variables") ; 
        auto out_cell_vars_plane_surface = get_param<std::vector<std::string>>("IO","plane_surface_output_cell_variables") ; 
        auto out_cell_vars_sphere_surface = get_param<std::vector<std::string>>("IO","sphere_surface_output_cell_variables") ; 

        auto& vnames = grace::variables::detail::_varnames ; 

        auto& vprops = grace::variables::detail::_varprops ; 

        auto& auxnames = grace::variables::detail::_auxnames ;

        auto& auxprops = grace::variables::detail::_auxprops ; 

        for( auto const& x: out_cell_vars_volume ) {
            if ( x == "metric" ) {
                for( auto const& vn: metric_vars ) {
                    if ( vprops[vn].is_vector ) {
                        _cell_volume_output_vector_vars.push_back(vprops[vn].name) ;
                    } else {
                        _cell_volume_output_scalar_vars.push_back(vn) ; 
                    }
                }
                for( auto const& vn: metric_aux ) {
                    if ( auxprops[vn].is_vector ) {
                        _cell_volume_output_vector_aux.push_back(auxprops[vn].name) ;
                    } else {
                        _cell_volume_output_scalar_aux.push_back(vn) ; 
                    }
                }
            } else if ( x == "hydro" ) {
                for( auto const& vn: hydro_aux ) {
                    if ( auxprops[vn].is_vector ) {
                        _cell_volume_output_vector_aux.push_back(auxprops[vn].name) ;
                    } else {
                        _cell_volume_output_scalar_aux.push_back(vn) ; 
                    }
                }
            } else {
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
                    GRACE_WARN("Variable {} not found (requested for volume output).", x) ; 
                }
            }
        } 

        for( auto const& x: out_cell_vars_plane_surface ) {
            if ( x == "metric" ) {
                for( auto const& vn: metric_vars ) {
                    if ( vprops[vn].is_vector ) {
                        _cell_plane_surface_output_vector_vars.push_back(vprops[vn].name) ;
                    } else {
                        _cell_plane_surface_output_scalar_vars.push_back(vn) ; 
                    }
                }
                for( auto const& vn: metric_aux ) {
                    if ( auxprops[vn].is_vector ) {
                        _cell_plane_surface_output_vector_aux.push_back(auxprops[vn].name) ;
                    } else {
                        _cell_plane_surface_output_scalar_aux.push_back(vn) ; 
                    }
                }
            } else if ( x == "hydro" ) {
                for( auto const& vn: hydro_aux ) {
                    if ( auxprops[vn].is_vector ) {
                        _cell_plane_surface_output_vector_aux.push_back(auxprops[vn].name) ;
                    } else {
                        _cell_plane_surface_output_scalar_aux.push_back(vn) ; 
                    }
                }
            } else {
                if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                    if( vprops[x].is_vector ){
                        _cell_plane_surface_output_vector_vars.push_back(vprops[x].name) ; 
                    } else {
                        _cell_plane_surface_output_scalar_vars.push_back(x) ;
                    }
                } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                    if( auxprops[x].is_vector ){
                        _cell_plane_surface_output_vector_aux.push_back(auxprops[x].name) ; 
                    } else {
                        _cell_plane_surface_output_scalar_aux.push_back(x) ;
                    } 
                } else { 
                    GRACE_WARN("Variable {} not found (requested for plane surface output).", x) ; 
                }
            }
        } 

        for( auto const& x: out_cell_vars_sphere_surface ) {
            if ( x == "metric" ) {
                for( auto const& vn: metric_vars ) {
                    if ( vprops[vn].is_vector ) {
                        _cell_sphere_surface_output_vector_vars.push_back(vprops[vn].name) ;
                    } else {
                        _cell_sphere_surface_output_scalar_vars.push_back(vn) ; 
                    }
                }
                for( auto const& vn: metric_aux ) {
                    if ( auxprops[vn].is_vector ) {
                        _cell_sphere_surface_output_vector_aux.push_back(auxprops[vn].name) ;
                    } else {
                        _cell_sphere_surface_output_scalar_aux.push_back(vn) ; 
                    }
                }
            } else if ( x == "hydro" ) {
                for( auto const& vn: hydro_aux ) {
                    if ( auxprops[vn].is_vector ) {
                        _cell_sphere_surface_output_vector_aux.push_back(auxprops[vn].name) ;
                    } else {
                        _cell_sphere_surface_output_scalar_aux.push_back(vn) ; 
                    }
                }
            } else {
                if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                    if( vprops[x].is_vector ){
                        _cell_sphere_surface_output_vector_vars.push_back(vprops[x].name) ; 
                    } else {
                        _cell_sphere_surface_output_scalar_vars.push_back(x) ;
                    }
                } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                    if( auxprops[x].is_vector ){
                        _cell_sphere_surface_output_vector_aux.push_back(auxprops[x].name) ; 
                    } else {
                        _cell_sphere_surface_output_scalar_aux.push_back(x) ;
                    } 
                } else { 
                    GRACE_WARN("Variable {} not found (requested for sphere surface output).", x) ; 
                }
            }
        } 
        auto dedup = [](std::vector<std::string>& v)
        {
            std::unordered_set<std::string> seen;
            seen.reserve(v.size());

            auto it = std::remove_if(v.begin(), v.end(),
                [&seen](const std::string& s)
                {
                    return !seen.insert(s).second; // true if already seen
                });

            v.erase(it, v.end());
        };
        dedup(_cell_volume_output_vector_vars) ; 
        dedup(_cell_volume_output_scalar_vars) ; 
        dedup(_cell_volume_output_vector_aux) ; 
        dedup(_cell_volume_output_scalar_aux) ; 

        dedup(_cell_plane_surface_output_vector_vars) ; 
        dedup(_cell_plane_surface_output_scalar_vars) ; 
        dedup(_cell_plane_surface_output_vector_aux) ; 
        dedup(_cell_plane_surface_output_scalar_aux) ; 

        dedup(_cell_sphere_surface_output_vector_vars) ; 
        dedup(_cell_sphere_surface_output_scalar_vars) ; 
        dedup(_cell_sphere_surface_output_vector_aux) ; 
        dedup(_cell_sphere_surface_output_scalar_aux) ; 
        
        /* Scalar output variables */
        auto out_minmax = 
            params["IO"]["scalar_output_minmax"].as<std::vector<std::string>>() ;
        auto out_norm2 = 
            params["IO"]["scalar_output_norm2"].as<std::vector<std::string>>() ;
        auto out_integral = 
            params["IO"]["scalar_output_integral"].as<std::vector<std::string>>() ;
        for( auto const& x: out_minmax ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                _scalar_output_minmax_vars.push_back(x) ;
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                _scalar_output_minmax_aux.push_back(x) ;
            } else { 
                GRACE_WARN("Variable {} not found (requested for scalar minmax output).", x) ; 
            }
        } 
        for( auto const& x: out_norm2 ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                _scalar_output_norm2_vars.push_back(x) ;
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                _scalar_output_norm2_aux.push_back(x) ;
            } else { 
                GRACE_WARN("Variable {} not found (requested for scalar norm2 output).", x) ; 
            }
        } 
        for( auto const& x: out_integral ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                _scalar_output_integral_vars.push_back(x) ;
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                _scalar_output_integral_aux.push_back(x) ;
            } else { 
                GRACE_WARN("Variable {} not found (requested for scalar integral output).", x) ; 
            }
        } 
        

        /* Info output variables */
        auto out_info_max = 
            params["IO"]["info_output_max_reductions"].as<std::vector<std::string>>() ;
        auto out_info_min = 
            params["IO"]["info_output_min_reductions"].as<std::vector<std::string>>() ;
        auto out_info_norm2 = 
            params["IO"]["info_output_norm2_reductions"].as<std::vector<std::string>>() ;
        for( auto const& x: out_info_max ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                _info_output_max_vars.push_back(x) ;
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                _info_output_max_aux.push_back(x) ;
            } else { 
                GRACE_WARN("Variable {} not found (requested for info minmax output).", x) ; 
            }
        } 
        for( auto const& x: out_info_min ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                _info_output_min_vars.push_back(x) ;
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                _info_output_min_aux.push_back(x) ;
            } else { 
                GRACE_WARN("Variable {} not found (requested for info norm2 output).", x) ; 
            }
        } 
        for( auto const& x: out_info_norm2 ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                _info_output_norm2_vars.push_back(x) ;
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                _info_output_norm2_aux.push_back(x) ;
            } else { 
                GRACE_WARN("Variable {} not found (requested for info integral output).", x) ; 
            }
        } 
        /***************************************************************/
        /* Now we create a vector containing all unique variable names */
        /* requested for reductions.                                   */
        /***************************************************************/
        /* Minmax */
        for( auto const& x: _info_output_max_vars ) {
            if(std::find( _minmax_reduction_vars.begin()
                        , _minmax_reduction_vars.end(), x ) == _minmax_reduction_vars.end()) {
                _minmax_reduction_vars.push_back(x) ; 
            }
        }
        for( auto const& x: _info_output_min_vars ) {
            if(std::find( _minmax_reduction_vars.begin()
                        , _minmax_reduction_vars.end(), x ) == _minmax_reduction_vars.end()) {
                _minmax_reduction_vars.push_back(x) ; 
            }
        }
        for( auto const& x: _scalar_output_minmax_vars   ) {
            if(std::find( _minmax_reduction_vars.begin()
                        , _minmax_reduction_vars.end(), x ) == _minmax_reduction_vars.end()) {
                _minmax_reduction_vars.push_back(x) ; 
            }
        }
        /***************************************************************/
        for( auto const& x: _info_output_max_aux ) {
            if(std::find( _minmax_reduction_aux.begin()
                        , _minmax_reduction_aux.end(), x ) == _minmax_reduction_aux.end()) {
                _minmax_reduction_aux.push_back(x) ; 
            }
        }
        for( auto const& x: _info_output_min_aux ) {
            if(std::find( _minmax_reduction_aux.begin()
                        , _minmax_reduction_aux.end(), x ) == _minmax_reduction_aux.end()) {
                _minmax_reduction_aux.push_back(x) ; 
            }
        }
        for( auto const& x: _scalar_output_minmax_aux   ) {
            if(std::find( _minmax_reduction_aux.begin()
                        , _minmax_reduction_aux.end(), x ) == _minmax_reduction_aux.end()) {
                _minmax_reduction_aux.push_back(x) ; 
            }
        }
        /***************************************************************/
        /* Norm 2 */
        for( auto const& x: _info_output_norm2_vars ) {
            if(std::find( _norm2_reduction_vars.begin()
                        , _norm2_reduction_vars.end(), x ) == _norm2_reduction_vars.end()) {
                _norm2_reduction_vars.push_back(x) ; 
            }
        }
        for( auto const& x: _scalar_output_norm2_vars ) {
            if(std::find( _norm2_reduction_vars.begin()
                        , _norm2_reduction_vars.end(), x ) == _norm2_reduction_vars.end()) {
                _norm2_reduction_vars.push_back(x) ; 
            }
        }
        /***************************************************************/
        for( auto const& x: _info_output_norm2_aux ) {
            if(std::find( _norm2_reduction_aux.begin()
                        , _norm2_reduction_aux.end(), x ) == _norm2_reduction_aux.end()) {
                _norm2_reduction_aux.push_back(x) ; 
            }
        }
        for( auto const& x: _scalar_output_norm2_aux ) {
            if(std::find( _norm2_reduction_aux.begin()
                        , _norm2_reduction_aux.end(), x ) == _norm2_reduction_aux.end()) {
                _norm2_reduction_aux.push_back(x) ; 
            }
        }
        /***************************************************************/
        /* Integral */
        for( auto const& x: _scalar_output_integral_vars   ) {
            if(std::find( _integral_reduction_vars.begin()
                        , _integral_reduction_vars.end(), x ) == _integral_reduction_vars.end()) {
                _integral_reduction_vars.push_back(x) ; 
            }
        }
        /***************************************************************/
        for( auto const& x: _scalar_output_integral_aux   ) {
            if(std::find( _integral_reduction_aux.begin()
                        , _integral_reduction_aux.end(), x ) == _integral_reduction_aux.end()) {
                _integral_reduction_aux.push_back(x) ; 
            }
        }
        /***************************************************************/
        /****************************/
        /* Set iteration count to 0 */ 
        _iter = 0UL ; 
        /* Set time to 0            */
        _time = 0.0 ; 
        _initial_time = 0.0 ; 
        _dt   = 0.0 ;  
        _initial_wtime = 0.0 ; 
        /****************************/
        if( parallel::mpi_comm_rank() == grace::master_rank() ) 
        {
            
            std::cout << "\n========================================\n";
            std::cout << " Output Configuration\n";
            std::cout << "========================================\n\n";

            /* ---------------- Volume ---------------- */

            std::cout << "[Volume Output | co-dimension 0]\n";
            std::cout << "  Interval: every "
                    << _volume_output_every
                    << " iterations\n";

            print_variable_group(
                _cell_volume_output_scalar_vars,
                _cell_volume_output_vector_vars,
                _cell_volume_output_scalar_aux,
                _cell_volume_output_vector_aux
            );

            std::cout << "\n";

            /* ---------------- Surfaces ---------------- */

            std::cout << "[Surface Output | co-dimension 1]\n";

            std::cout << "  Plane interval:  every "
                    << _plane_surface_output_every
                    << " iterations\n";

            std::cout << "  Sphere interval: every "
                    << _sphere_surface_output_every
                    << " iterations\n\n";

            /* ---- Plane ---- */

            std::cout << "  > Plane Surface\n";

            print_variable_group(
                _cell_plane_surface_output_scalar_vars,
                _cell_plane_surface_output_vector_vars,
                _cell_plane_surface_output_scalar_aux,
                _cell_plane_surface_output_vector_aux
            );

            std::cout << "\n";

            /* ---- Sphere ---- */

            std::cout << "  > Sphere Surface\n";

            print_variable_group(
                _cell_sphere_surface_output_scalar_vars,
                _cell_sphere_surface_output_vector_vars,
                _cell_sphere_surface_output_scalar_aux,
                _cell_sphere_surface_output_vector_aux
            );

            std::cout << "\n========================================\n\n";
        }

        auto const term_cnd = grace::get_param<std::string>("evolution", "termination_condition") ; 
        if ( term_cnd == "time" ) {
            _term_cnd = terminate::TIME ; 
        } else if ( term_cnd == "iteration" ) {
            _term_cnd = terminate::ITERATION ; 
        } else if ( term_cnd == "walltime" ) {
            _term_cnd = terminate::WALLTIME ; 
        } else {
            ERROR("Unrecognized termination condition, please choose one of 'time', 'iteration' and 'walltime'."); 
        }

        _max_time = grace::get_param<double>("evolution","final_time") ; 
        _max_walltime = grace::get_param<double>("evolution","final_walltime") ; 
        _max_iter = grace::get_param<size_t>("evolution","final_iteration") ; 

    }
    ~grace_runtime_impl_t() {} 

    friend class utils::singleton_holder<grace_runtime_impl_t,memory::default_create> ; 
    friend class memory::new_delete_creator<grace_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access

    static constexpr size_t longevity = GRACE_RUNTIME ; 

} ; 

using runtime = utils::singleton_holder<grace_runtime_impl_t,memory::default_create> ;

} /* namespace grace */

#endif /* INCLUDE_GRACE_SYSTEM_GRACE_RUNTIME */
