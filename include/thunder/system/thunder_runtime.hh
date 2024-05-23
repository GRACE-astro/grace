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
#include <thunder/system/print.hh>
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
    int _n_output_planes ; 
    std::vector<std::array<double,3>> _output_planes_origins ; 
    std::vector<std::array<double,3>> _output_planes_normals ; 
    std::vector<std::string>          _output_planes_names   ; 
    /* Output spheres */
    int _n_output_spheres ; 
    std::vector<std::array<double,3>> _output_spheres_centers  ; 
    std::vector<double>               _output_spheres_radii    ; 
    std::vector<std::string>          _output_spheres_names    ; 
    std::vector<std::string>          _output_spheres_tracking ;
    /* Output parameters */ 
    bool   _volume_output        ;
    bool   _surface_output       ; 
    int _volume_output_every  ; 
    int _plane_surface_output_every ; 
    int _sphere_surface_output_every ; 
    int _scalar_output_every         ; 
    int _info_output_every           ;
    std::filesystem::path _volume_io_basepath ;
    std::filesystem::path _surface_io_basepath ;
    std::filesystem::path _scalar_io_basepath ;
    std::string _volume_io_basename  ; 
    std::string _surface_io_basename ;
    std::string _scalar_io_basename ;
    /* iteration count */ 
    size_t _iter ; 
    /* current simulation time */
    double _time, _dt ; 
    /* total walltime clock */
    spdlog::stopwatch _walltime ; 
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

    int THUNDER_ALWAYS_INLINE 
    volume_output_every()  const { return _volume_output_every ; }

    int THUNDER_ALWAYS_INLINE 
    plane_surface_output_every() const { return _plane_surface_output_every ; }

    int THUNDER_ALWAYS_INLINE 
    sphere_surface_output_every() const { return _sphere_surface_output_every ; }

    int THUNDER_ALWAYS_INLINE 
    scalar_output_every() const { return _scalar_output_every ; }

    int THUNDER_ALWAYS_INLINE 
    info_output_every() const { return _info_output_every ; }

    std::string THUNDER_ALWAYS_INLINE
    volume_io_basepath() const { return _volume_io_basepath ; }

    std::string THUNDER_ALWAYS_INLINE
    surface_io_basepath() const { return _surface_io_basepath ; }

    std::string THUNDER_ALWAYS_INLINE
    scalar_io_basepath() const { return _scalar_io_basepath ; }

    std::string THUNDER_ALWAYS_INLINE
    volume_io_basename() const { return _volume_io_basename ; }

    std::string THUNDER_ALWAYS_INLINE
    surface_io_basename() const { return _surface_io_basename ; }

    std::string THUNDER_ALWAYS_INLINE
    scalar_io_basename() const { return _scalar_io_basename ; }

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
    cell_plane_surface_output_scalar_vars() const {
        return _cell_plane_surface_output_scalar_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_plane_surface_output_vector_vars() const {
        return _cell_plane_surface_output_vector_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_plane_surface_output_scalar_aux() const {
        return _cell_plane_surface_output_scalar_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_plane_surface_output_vector_aux() const {
        return _cell_plane_surface_output_vector_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_sphere_surface_output_scalar_vars() const {
        return _cell_sphere_surface_output_scalar_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_sphere_surface_output_vector_vars() const {
        return _cell_sphere_surface_output_vector_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_sphere_surface_output_scalar_aux() const {
        return _cell_sphere_surface_output_scalar_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_sphere_surface_output_vector_aux() const {
        return _cell_sphere_surface_output_vector_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    scalar_output_minmax_vars() const {
        return _scalar_output_minmax_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    scalar_output_minmax_aux() const {
        return _scalar_output_minmax_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    scalar_output_norm2_vars() const {
        return _scalar_output_norm2_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    scalar_output_norm2_aux() const {
        return _scalar_output_norm2_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    scalar_output_integral_vars() const {
        return _scalar_output_integral_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    scalar_output_integral_aux() const {
        return _scalar_output_integral_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    info_output_max_vars() const {
        return _info_output_max_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    info_output_max_aux() const {
        return _info_output_max_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    info_output_min_vars() const {
        return _info_output_min_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    info_output_min_aux() const {
        return _info_output_min_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    info_output_norm2_vars() const {
        return _info_output_norm2_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    info_output_norm2_aux() const {
        return _info_output_norm2_aux; 
    }
    
    decltype(auto) THUNDER_ALWAYS_INLINE 
    minmax_reduction_vars() const {
        return _minmax_reduction_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    minmax_reduction_aux() const {
        return _minmax_reduction_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    norm2_reduction_vars() const {
        return _norm2_reduction_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    norm2_reduction_aux() const {
        return _norm2_reduction_aux; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    integral_reduction_vars() const {
        return _integral_reduction_vars; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    integral_reduction_aux() const {
        return _integral_reduction_aux; 
    }

    int THUNDER_ALWAYS_INLINE 
    n_surface_output_planes() const {
        return _n_output_planes ; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_plane_surface_output_origins() const {
        return _output_planes_origins ;  
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_plane_surface_output_normals() const {
        return _output_planes_normals ; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_plane_surface_output_names() const {
        return _output_planes_names ; 
    }

    int THUNDER_ALWAYS_INLINE 
    n_surface_output_spheres() const {
        return _n_output_spheres ; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_sphere_surface_output_centers() const {
        return _output_spheres_centers ; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_sphere_surface_output_radii() const {
        return _output_spheres_radii ; 
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_sphere_surface_output_names() const {
        return _output_spheres_names ;
    }

    decltype(auto) THUNDER_ALWAYS_INLINE 
    cell_sphere_surface_output_tracking() const {
        return _output_spheres_tracking ;
    }

    void THUNDER_ALWAYS_INLINE 
    set_output_sphere_center(int isphere, std::array<double,3>const& new_center)
    {
        for( int ii=0; ii<3; ++ii)
            _output_spheres_centers[isphere][ii] = new_center[ii] ;
    }

    double THUNDER_ALWAYS_INLINE 
    elapsed() {
        return _walltime.elapsed().count() ; 
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
        /* Output frequencies              */
        _sphere_surface_output_every = params["IO"]["sphere_surface_output_every"].as<int>() ; 
        _plane_surface_output_every = params["IO"]["plane_surface_output_every"].as<int>() ; 
        _volume_output_every = params["IO"]["volume_output_every"].as<int>() ; 
        _scalar_output_every = params["IO"]["scalar_output_every"].as<int>() ; 
        _info_output_every = params["IO"]["info_output_every"].as<int>() ; 
        /* Output filenames and directories */
        _volume_io_basename  = params["IO"]["volume_output_base_filename"].as<std::string>(); 
        _surface_io_basename  = params["IO"]["surface_output_base_filename"].as<std::string>();
        _scalar_io_basename  = params["IO"]["scalar_output_base_filename"].as<std::string>();
        _volume_io_basepath  = 
            std::filesystem::path(params["IO"]["volume_output_base_directory"].as<std::string>()); 
        _surface_io_basepath  = 
            std::filesystem::path(params["IO"]["surface_output_base_directory"].as<std::string>()); 
        _scalar_io_basepath  = 
            std::filesystem::path(params["IO"]["scalar_output_base_directory"].as<std::string>()); 
        /* Create output directories if they don't exist */
        if( not std::filesystem::exists( _volume_io_basepath ) ){
            std::filesystem::create_directory(_volume_io_basepath) ; 
        }

        if( not std::filesystem::exists( _surface_io_basepath ) ){
            std::filesystem::create_directory(_surface_io_basepath) ; 
        }

        if( not std::filesystem::exists( _scalar_io_basepath ) ){
            std::filesystem::create_directory(_scalar_io_basepath) ; 
        }
        /* Set output planes and spheres properties      */
        _n_output_planes = params["IO"]["n_output_planes"].as<int>() ;
        #define READ_IO_PARAM(s,t) params["IO"][s].as<t>()  
        #define AS_TYPE(t) t
        _output_planes_origins.resize(_n_output_planes) ;
        _output_planes_normals.resize(_n_output_planes) ;
        _output_planes_names.resize(_n_output_planes)   ; 
        for (int iplane=0; iplane < _n_output_planes; ++iplane) {
            std::ostringstream oss_x,oss_y,oss_z;
            oss_x << "output_plane_x_origin_" << iplane;
            oss_y << "output_plane_y_origin_" << iplane;
            oss_z << "output_plane_z_origin_" << iplane;
            _output_planes_origins[iplane] = {
                READ_IO_PARAM(oss_x.str(), AS_TYPE(double)),
                READ_IO_PARAM(oss_y.str(), AS_TYPE(double)),
                READ_IO_PARAM(oss_z.str(), AS_TYPE(double))
            } ; 
            oss_x.str("");  // Reset content to empty string
            oss_x.clear();
            oss_y.str("");  // Reset content to empty string
            oss_y.clear();
            oss_z.str("");  // Reset content to empty string
            oss_z.clear();
            oss_x << "output_plane_x_normal_" << iplane;
            oss_y << "output_plane_y_normal_" << iplane;
            oss_z << "output_plane_z_normal_" << iplane;
            _output_planes_normals[iplane] = {
                READ_IO_PARAM(oss_x.str(), AS_TYPE(double)),
                READ_IO_PARAM(oss_y.str(), AS_TYPE(double)),
                READ_IO_PARAM(oss_z.str(), AS_TYPE(double))
            } ; 
            oss_x.str("");  // Reset content to empty string
            oss_x.clear();
            oss_x << "output_plane_name_" << iplane;
            _output_planes_names[iplane] = READ_IO_PARAM(oss_x.str(), AS_TYPE(std::string)) ; 
        }

        _n_output_spheres = params["IO"]["n_output_spheres"].as<int>() ;
        _output_spheres_centers.resize(_n_output_spheres)  ;
        _output_spheres_radii.resize(_n_output_spheres)    ;
        _output_spheres_names.resize(_n_output_spheres)    ;
        _output_spheres_tracking.resize(_n_output_spheres) ;
        for (int isphere=0; isphere < _n_output_spheres; ++isphere) {
            std::ostringstream oss_x,oss_y,oss_z;
            oss_x << "output_sphere_x_center_" << isphere;
            oss_y << "output_sphere_y_center_" << isphere;
            oss_z << "output_sphere_z_center_" << isphere;
            _output_spheres_centers[isphere] = {
                READ_IO_PARAM(oss_x.str(), AS_TYPE(double)),
                READ_IO_PARAM(oss_y.str(), AS_TYPE(double)),
                READ_IO_PARAM(oss_z.str(), AS_TYPE(double))
            } ; 
            oss_x.str("");  // Reset content to empty string
            oss_x.clear();
            oss_x << "output_sphere_radius_" << isphere;
            _output_spheres_radii[isphere]    = READ_IO_PARAM(oss_x.str(), AS_TYPE(double)) ; 
            oss_x.str("");  // Reset content to empty string
            oss_x.clear();
            oss_x << "output_sphere_name_" << isphere;
            _output_spheres_names[isphere]    = READ_IO_PARAM(oss_x.str(), AS_TYPE(std::string)) ; 
            oss_x.str("");  // Reset content to empty string
            oss_x.clear();
            oss_x << "output_sphere_tracking_" << isphere;
            _output_spheres_tracking[isphere] = READ_IO_PARAM(oss_x.str(), AS_TYPE(std::string)) ; 
        }
        #undef READ_IO_PARAM
        /* Volume and surface output variables */
        auto out_cell_vars_volume = 
            params["IO"]["volume_output_cell_variables"].as<std::vector<std::string>>() ; 
        auto out_cell_vars_plane_surface = 
            params["IO"]["plane_surface_output_cell_variables"].as<std::vector<std::string>>() ; 
        auto out_cell_vars_sphere_surface = 
            params["IO"]["sphere_surface_output_cell_variables"].as<std::vector<std::string>>() ; 
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
                THUNDER_WARN("Variable {} not found (requested for volume output).", x) ; 
            }
        } 

        for( auto const& x: out_cell_vars_plane_surface ) {
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
                THUNDER_WARN("Variable {} not found (requested for plane surface output).", x) ; 
            }
        } 

        for( auto const& x: out_cell_vars_sphere_surface ) {
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
                THUNDER_WARN("Variable {} not found (requested for sphere surface output).", x) ; 
            }
        } 
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
                THUNDER_WARN("Variable {} not found (requested for scalar minmax output).", x) ; 
            }
        } 
        for( auto const& x: out_norm2 ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                _scalar_output_norm2_vars.push_back(x) ;
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                _scalar_output_norm2_aux.push_back(x) ;
            } else { 
                THUNDER_WARN("Variable {} not found (requested for scalar norm2 output).", x) ; 
            }
        } 
        for( auto const& x: out_integral ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                _scalar_output_integral_vars.push_back(x) ;
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                _scalar_output_integral_aux.push_back(x) ;
            } else { 
                THUNDER_WARN("Variable {} not found (requested for scalar integral output).", x) ; 
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
                THUNDER_WARN("Variable {} not found (requested for info minmax output).", x) ; 
            }
        } 
        for( auto const& x: out_info_min ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                _info_output_min_vars.push_back(x) ;
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                _info_output_min_aux.push_back(x) ;
            } else { 
                THUNDER_WARN("Variable {} not found (requested for info norm2 output).", x) ; 
            }
        } 
        for( auto const& x: out_info_norm2 ) {
            if(std::find(vnames.begin(), vnames.end(), x) != vnames.end()) {
                _info_output_norm2_vars.push_back(x) ;
            } else if (std::find(auxnames.begin(), auxnames.end(), x) != auxnames.end()) {
                _info_output_norm2_aux.push_back(x) ;
            } else { 
                THUNDER_WARN("Variable {} not found (requested for info integral output).", x) ; 
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
                std::cout << "Plane surface output requested every " << _plane_surface_output_every << " iterations\n" ; 
                std::cout << "Sphere surface output requested every " << _sphere_surface_output_every << " iterations\n" ; 
                std::cout << "Variables registered for plane surface (co-dimension 1) output: \n" ; 
                std::cout << "Scalars: \n";
                for(auto const& x: _cell_plane_surface_output_scalar_vars){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Vectors: \n";
                for(auto const& x: _cell_plane_surface_output_vector_vars){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Auxiliaries: \n";
                std::cout << "Scalars: \n";
                for(auto const& x: _cell_plane_surface_output_scalar_aux){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Vectors: \n";
                for(auto const& x: _cell_plane_surface_output_vector_aux){
                    std::cout << x << std::endl ; 
                }  
                std::cout << "Variables registered for sphere surface (co-dimension 1) output: \n" ; 
                std::cout << "Scalars: \n";
                for(auto const& x: _cell_sphere_surface_output_scalar_vars){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Vectors: \n";
                for(auto const& x: _cell_sphere_surface_output_vector_vars){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Auxiliaries: \n";
                std::cout << "Scalars: \n";
                for(auto const& x: _cell_sphere_surface_output_scalar_aux){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Vectors: \n";
                for(auto const& x: _cell_sphere_surface_output_vector_aux){
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
