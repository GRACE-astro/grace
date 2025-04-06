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
#include <grace/data_structures/variable_utils.hh>

#include <spdlog/stopwatch.h>

#include <string>
#include <vector> 
#include <set>
#include <iostream>
#include <algorithm> 
#include <filesystem> 

namespace grace {

class grace_runtime_impl_t 
{
 private:
    /* Volume output */
    std::set<std::string> _cell_volume_output_scalar_vars ;
    std::set<std::string> _corner_volume_output_scalar_vars ;
    std::set<std::string> _cell_volume_output_vector_vars ;
    std::set<std::string> _corner_volume_output_vector_vars ;
    std::set<std::string> _cell_volume_output_tensor_vars ;
    std::set<std::string> _corner_volume_output_tensor_vars ;
    std::set<std::string> _cell_volume_output_symm_tensor_vars ;
    std::set<std::string> _corner_volume_output_symm_tensor_vars ;
    /* Surface output */
    std::set<std::string> _cell_plane_surface_output_scalar_vars ;
    std::set<std::string> _corner_plane_surface_output_scalar_vars ;
    std::set<std::string> _cell_plane_surface_output_vector_vars ;
    std::set<std::string> _corner_plane_surface_output_vector_vars ;
    std::set<std::string> _cell_plane_surface_output_tensor_vars ;
    std::set<std::string> _corner_plane_surface_output_tensor_vars ;
    std::set<std::string> _cell_plane_surface_output_symm_tensor_vars ;
    std::set<std::string> _corner_plane_surface_output_symm_tensor_vars ;
    /* Sphere surface output */
    std::set<std::string> _cell_sphere_surface_output_scalar_vars ;
    std::set<std::string> _corner_sphere_surface_output_scalar_vars ;
    std::set<std::string> _cell_sphere_surface_output_vector_vars ;
    std::set<std::string> _corner_sphere_surface_output_vector_vars ;
    std::set<std::string> _cell_sphere_surface_output_tensor_vars ;
    std::set<std::string> _corner_sphere_surface_output_tensor_vars ;
    std::set<std::string> _cell_sphere_surface_output_symm_tensor_vars ;
    std::set<std::string> _corner_sphere_surface_output_symm_tensor_vars ;
    // /* Sphere surface multipoles  */
    std::set<std::string> _cell_sphere_surface_multipole_output_scalar_vars ;
    std::set<std::string> _corner_sphere_surface_multipole_output_scalar_vars ;
    std::set<std::string> _cell_sphere_surface_multipole_output_vector_vars ;
    std::set<std::string> _corner_sphere_surface_multipole_output_vector_vars ;
    std::set<std::string> _cell_sphere_surface_multipole_output_tensor_vars ;
    std::set<std::string> _corner_sphere_surface_multipole_output_tensor_vars ;
    std::set<std::string> _cell_sphere_surface_multipole_output_symm_tensor_vars ;
    std::set<std::string> _corner_sphere_surface_multipole_output_symm_tensor_vars ;
    // /* Sphere surface reduction  */
    // std::set<std::string> _cell_sphere_surface_scalar_output_scalar_vars ;
    // std::set<std::string> _corner_sphere_surface_scalar_output_scalar_vars ;
    // std::set<std::string> _cell_sphere_surface_scalar_output_vector_vars ;
    // std::set<std::string> _corner_sphere_surface_scalar_output_vector_vars ;
    // std::set<std::string> _cell_sphere_surface_scalar_output_tensor_vars ;
    // std::set<std::string> _corner_sphere_surface_scalar_output_tensor_vars ;
    // std::set<std::string> _cell_sphere_surfaces_scalar_output_symm_tensor_vars ;
    // std::set<std::string> _corner_sphere_surface_scalar_output_symm_tensor_vars ;
    /* Scalar output         */
    std::set<std::string> _scalar_output_minmax_vars   ; 
    std::set<std::string> _scalar_output_norm2_vars    ; 
    std::set<std::string> _scalar_output_integral_vars ; 
    /* Info output         */
    std::set<std::string> _info_output_max_vars   ; 
    std::set<std::string> _info_output_min_vars   ; 
    std::set<std::string> _info_output_norm2_vars ; 
    /* Reduction variable lists */
    std::set<std::string> _minmax_reduction_vars   ;
    std::set<std::string> _norm2_reduction_vars    ;
    std::set<std::string> _integral_reduction_vars ;

    /* Multipole computations and sum reduction on the sphere*/
    std::set<std::string> _sphere_integral_reduction_vars ;
    std::set<std::string> _sphere_multipole_reduction_vars ;

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
    std::vector<std::string>          _output_spheres_types    ; 
    std::vector<std::string>          _output_spheres_tracking ;
    /* Output spheres resolution and reduction options */
    int _nside_output_spheres ; // for spheres in the HEALPIX grid setup
    int _multipole_max_degree;    
    int _ntheta_output_spheres ; // for spheres in the UNIFORM-MIDPOINT grid setup 
    int _nphi_output_spheres ; // for spheres in the UNIFORM-MIDPOINT grid setup

    /* Output parameters */ 
    bool   _volume_output        ;
    bool   _surface_output       ; 
    int _volume_output_every  ; 
    int _plane_surface_output_every ; 
    int _sphere_surface_output_every ; 
    int _sphere_surface_scalar_output_every ; 
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

    int GRACE_ALWAYS_INLINE 
    volume_output_every()  const { return _volume_output_every ; }

    int GRACE_ALWAYS_INLINE 
    plane_surface_output_every() const { return _plane_surface_output_every ; }

    int GRACE_ALWAYS_INLINE 
    sphere_surface_output_every() const { return _sphere_surface_output_every ; }

    int GRACE_ALWAYS_INLINE 
    sphere_surface_scalar_output_every() const { return _sphere_surface_scalar_output_every ; }

    int GRACE_ALWAYS_INLINE 
    scalar_output_every() const { return _scalar_output_every ; }

    int GRACE_ALWAYS_INLINE 
    info_output_every() const { return _info_output_every ; }

    std::string GRACE_ALWAYS_INLINE
    volume_io_basepath() const { return _volume_io_basepath ; }

    std::string GRACE_ALWAYS_INLINE
    surface_io_basepath() const { return _surface_io_basepath ; }

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
    corner_volume_output_scalar_vars() const {
        return _corner_volume_output_scalar_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_volume_output_vector_vars() const {
        return _cell_volume_output_vector_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    corner_volume_output_vector_vars() const {
        return _corner_volume_output_vector_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_volume_output_tensor_vars() const {
        return _cell_volume_output_tensor_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    corner_volume_output_tensor_vars() const {
        return _corner_volume_output_tensor_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_volume_output_symm_tensor_vars() const {
        return _cell_volume_output_symm_tensor_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    corner_volume_output_symm_tensor_vars() const {
        return _corner_volume_output_symm_tensor_vars; 
    }


    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_plane_surface_output_scalar_vars() const {
        return _cell_plane_surface_output_scalar_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_plane_surface_output_vector_vars() const {
        return _cell_plane_surface_output_vector_vars; 
    }

    /** 2D sphere surface vars  */

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_scalar_vars() const {
        return _cell_sphere_surface_output_scalar_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_vector_vars() const {
        return _cell_sphere_surface_output_vector_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_tensor_vars() const {
        return _cell_sphere_surface_output_tensor_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    corner_sphere_surface_output_scalar_vars() const {
        return _corner_sphere_surface_output_scalar_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    corner_sphere_surface_output_vector_vars() const {
        return _corner_sphere_surface_output_vector_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    corner_sphere_surface_output_tensor_vars() const {
        return _corner_sphere_surface_output_tensor_vars; 
    }

    /** Multipole vars */

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_multipole_output_scalar_vars() const {
        return _cell_sphere_surface_multipole_output_scalar_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_multipole_output_vector_vars() const {
        return _cell_sphere_surface_multipole_output_vector_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_multipole_output_tensor_vars() const {
        return _cell_sphere_surface_multipole_output_tensor_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    corner_sphere_surface_multipole_output_scalar_vars() const {
        return _corner_sphere_surface_multipole_output_scalar_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    corner_sphere_surface_multipole_output_vector_vars() const {
        return _corner_sphere_surface_multipole_output_vector_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    corner_sphere_surface_multipole_output_tensor_vars() const {
        return _corner_sphere_surface_multipole_output_tensor_vars; 
    }

    /** 3D reductions  */

    decltype(auto) GRACE_ALWAYS_INLINE 
    scalar_output_minmax_vars() const {
        return _scalar_output_minmax_vars; 
    }


    decltype(auto) GRACE_ALWAYS_INLINE 
    scalar_output_norm2_vars() const {
        return _scalar_output_norm2_vars; 
    }


    decltype(auto) GRACE_ALWAYS_INLINE 
    scalar_output_integral_vars() const {
        return _scalar_output_integral_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    info_output_max_vars() const {
        return _info_output_max_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    info_output_min_vars() const {
        return _info_output_min_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    info_output_norm2_vars() const {
        return _info_output_norm2_vars; 
    }

    
    decltype(auto) GRACE_ALWAYS_INLINE 
    minmax_reduction_vars() const {
        return _minmax_reduction_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    norm2_reduction_vars() const {
        return _norm2_reduction_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    integral_reduction_vars() const {
        return _integral_reduction_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    sphere_integral_reduction_vars() const {
        return _sphere_integral_reduction_vars; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    sphere_multipole_reduction_vars() const {
        return _sphere_multipole_reduction_vars; 
    }
    

    int GRACE_ALWAYS_INLINE 
    n_surface_output_planes() const {
        return _n_output_planes ; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_plane_surface_output_origins() const {
        return _output_planes_origins ;  
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_plane_surface_output_normals() const {
        return _output_planes_normals ; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_plane_surface_output_names() const {
        return _output_planes_names ; 
    }

    int GRACE_ALWAYS_INLINE 
    n_surface_output_spheres() const {
        return _n_output_spheres ; 
    }

    int GRACE_ALWAYS_INLINE 
    nside_surface_output_spheres() const {
        return _nside_output_spheres ; 
    }

    int GRACE_ALWAYS_INLINE 
    ntheta_surface_output_spheres() const {
        return _ntheta_output_spheres ; 
    }
    int GRACE_ALWAYS_INLINE 
    nphi_surface_output_spheres() const {
        return _nphi_output_spheres ; 
    }

    int GRACE_ALWAYS_INLINE 
    max_degree_multipoles_surface_output_spheres() const {
        return _multipole_max_degree ; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_centers() const {
        return _output_spheres_centers ; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_radii() const {
        return _output_spheres_radii ; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_names() const {
        return _output_spheres_names ;
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_types() const {
        return _output_spheres_types ;
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    cell_sphere_surface_output_tracking() const {
        return _output_spheres_tracking ;
    }

    void GRACE_ALWAYS_INLINE 
    set_output_sphere_center(int isphere, std::array<double,3>const& new_center)
    {
        for( int ii=0; ii<3; ++ii)
            _output_spheres_centers[isphere][ii] = new_center[ii] ;
    }

    double GRACE_ALWAYS_INLINE 
    elapsed() {
        return _walltime.elapsed().count() ; 
    }

 private:

    grace_runtime_impl_t() {
        auto& params = grace::config_parser::get() ; 
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
        _sphere_surface_scalar_output_every = params["IO"]["sphere_surface_scalar_output_every"].as<int>() ; 


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
        _nside_output_spheres = params["IO"]["sphere_surface_output_nside"].as<int>() ;
        _ntheta_output_spheres = params["IO"]["sphere_surface_output_ntheta"].as<int>() ;
        _nphi_output_spheres = params["IO"]["sphere_surface_output_nphi"].as<int>() ;
        _sphere_surface_scalar_output_every = params["IO"]["sphere_surface_scalar_output_every"].as<int>() ;
        _multipole_max_degree = params["IO"]["sphere_surface_multipoles_max_degree"].as<int>() ;

        _output_spheres_centers.resize(_n_output_spheres)  ;
        _output_spheres_radii.resize(_n_output_spheres)    ;
        _output_spheres_names.resize(_n_output_spheres)    ;
        _output_spheres_types.resize(_n_output_spheres)    ;
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
            oss_x << "output_sphere_type_" << isphere;
            _output_spheres_types[isphere]    = READ_IO_PARAM(oss_x.str(), AS_TYPE(std::string)) ; 
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
        auto out_cell_vars_multipoles = 
            params["IO"]["sphere_surface_multipoles_cell_variables"].as<std::vector<std::string>>() ; 

        //auto out_cell_vars_sphere_surface_reductions_multipoles = 
        //     params["IO"]["sphere_surface_multipoles_cell_variables"].as<std::vector<std::string>>() ; 

        auto const add_to_scalar_vector_or_tensor_list = 
            [&] ( 
                std::vector<std::string> const& in_vars, 
                std::set<std::string>& svars, std::set<std::string>& ssvars,
                std::set<std::string>& vvars, std::set<std::string>& svvars,
                std::set<std::string>& tvars, std::set<std::string>& stvars )
        {
            for( auto const& x: in_vars ) {
                int err ; 
                auto const& props = variables::get_variable_properties(x,err) ; 
                if ( err < 0 ) {
                    GRACE_WARN("Variable {} requested for output is not registered.", x) ; 
                } else {
                    if( props.is_vector ) {
                        if ( props.staggering == var_staggering_t::CELL_CENTER ) {
                            vvars.insert(props.name) ; 
                        } else if ( props.staggering == var_staggering_t::CORNER ) {
                            svvars.insert(props.name) ;
                        } else {
                            GRACE_WARN("Variable {}'s staggering is not supported for output.", x) ;
                        }
                        
                    } else if ( props.is_tensor ) {
                        if ( props.staggering == var_staggering_t::CELL_CENTER ) {
                            tvars.insert(props.name) ; 
                        } else if ( props.staggering == var_staggering_t::CORNER ) {
                            stvars.insert(props.name) ;
                        } else {
                            GRACE_WARN("Variable {}'s staggering is not supported for output.", x) ;
                        }
                    } else {
                        if ( props.staggering == var_staggering_t::CELL_CENTER ) {
                            svars.insert(x) ; 
                        } else if ( props.staggering == var_staggering_t::CORNER ) {
                            ssvars.insert(x) ;
                        } else {
                            GRACE_WARN("Variable {}'s staggering is not supported for output.", x) ;
                        }
                    }
                }
            } 
        } ;

        add_to_scalar_vector_or_tensor_list(out_cell_vars_volume,
                                           _cell_volume_output_scalar_vars, _corner_volume_output_scalar_vars,
                                           _cell_volume_output_vector_vars, _corner_volume_output_vector_vars,
                                           _cell_volume_output_symm_tensor_vars, _corner_volume_output_symm_tensor_vars) ; 
        add_to_scalar_vector_or_tensor_list(out_cell_vars_plane_surface,
                                    _cell_plane_surface_output_scalar_vars, _corner_plane_surface_output_scalar_vars,
                                    _cell_plane_surface_output_vector_vars, _corner_plane_surface_output_vector_vars,
                                    _cell_plane_surface_output_symm_tensor_vars, _corner_plane_surface_output_symm_tensor_vars) ; 
        add_to_scalar_vector_or_tensor_list(out_cell_vars_sphere_surface,
                                    _cell_sphere_surface_output_scalar_vars, _corner_sphere_surface_output_scalar_vars,
                                    _cell_sphere_surface_output_vector_vars, _corner_sphere_surface_output_vector_vars,
                                    _cell_sphere_surface_output_symm_tensor_vars, _corner_sphere_surface_output_symm_tensor_vars) ; 
        add_to_scalar_vector_or_tensor_list(out_cell_vars_multipoles,
                                    _cell_sphere_surface_multipole_output_scalar_vars, _corner_sphere_surface_multipole_output_scalar_vars,
                                    _cell_sphere_surface_multipole_output_vector_vars, _corner_sphere_surface_multipole_output_vector_vars,
                                    _cell_sphere_surface_multipole_output_symm_tensor_vars, _corner_sphere_surface_multipole_output_symm_tensor_vars) ; 
        // add_to_scalar_vector_or_tensor_list(out_cell_vars_sphere_surface,
        //                             _cell_sphere_surface_output_scalar_vars, _corner_sphere_surface_output_scalar_vars,
        //                             _cell_sphere_surface_output_vector_vars, _corner_sphere_surface_output_vector_vars,
        //                             _cell_sphere_surface_output_symm_tensor_vars, _corner_sphere_surface_output_symm_tensor_vars) ; 

        /* Define helper lambda */
        auto const check_vars_exist_and_insert = 
            [&] (std::vector<std::string> const& vlist, std::set<std::string>& olist, std::set<std::string>& olist2 )
        {
            for( auto const& x: vlist ) {
                if ( variables::var_exists(x) ) {
                    olist.insert(x) ; 
                    olist2.insert(x) ; 
                } else {
                    GRACE_WARN("Variable {} requested for output is not registered.", x ) ; 
                }
            }
        } ; 
        /* Scalar output variables */
        auto out_minmax = 
            params["IO"]["scalar_output_minmax"].as<std::vector<std::string>>() ;
        auto out_norm2 = 
            params["IO"]["scalar_output_norm2"].as<std::vector<std::string>>() ;
        auto out_integral = 
            params["IO"]["scalar_output_integral"].as<std::vector<std::string>>() ;
        check_vars_exist_and_insert(out_minmax,_scalar_output_minmax_vars,_minmax_reduction_vars) ; 
        check_vars_exist_and_insert(out_norm2,_scalar_output_norm2_vars,_norm2_reduction_vars) ; 
        check_vars_exist_and_insert(out_integral,_scalar_output_integral_vars,_integral_reduction_vars) ; 
        /* Info output variables */
        auto out_info_max = 
            params["IO"]["info_output_max_reductions"].as<std::vector<std::string>>() ;
        auto out_info_min = 
            params["IO"]["info_output_min_reductions"].as<std::vector<std::string>>() ;
        auto out_info_norm2 = 
            params["IO"]["info_output_norm2_reductions"].as<std::vector<std::string>>() ;
        check_vars_exist_and_insert(out_info_max,_info_output_max_vars,_minmax_reduction_vars) ; 
        check_vars_exist_and_insert(out_info_min,_info_output_min_vars,_minmax_reduction_vars) ; 
        check_vars_exist_and_insert(out_info_norm2,_info_output_norm2_vars,_norm2_reduction_vars) ; 
        /* Sphere reductions (sum and multipoles) */
        auto out_sphere_integral_reduction_vars =  
            params["IO"]["sphere_surface_integrals_cell_variables"].as<std::vector<std::string>>() ;
        // auto out_multipole_reduction_vars =
        //     params["IO"]["sphere_surface_multipoles_cell_variables"].as<std::vector<std::string>>() ;
        
        // we can reuse the lambda above thanks to the properties of std::set
        // _sphere_multipole_reduction_vars=
        //_sphere_integral_reduction_vars
        //check_vars_exist_and_insert(out_integral_reduction_vars, _sphere_integral_reduction_vars,   _sphere_integral_reduction_vars);
        // check_vars_exist_and_insert(out_multipole_reduction_vars, _sphere_multipole_reduction_vars, _sphere_multipole_reduction_vars );

        /****************************/
        /* Set iteration count to 0 */ 
        _iter = 0UL ; 
        /* Set time to 0            */
        _time = 0.0 ; 
        _dt   = 0.0 ;  
        /****************************/
        if( parallel::mpi_comm_rank() == grace::master_rank() ) 
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
                std::cout << "Variables registered for sphere surface (co-dimension 1) output: \n" ; 
                std::cout << "Scalars: \n";
                for(auto const& x: _cell_sphere_surface_output_scalar_vars){
                    std::cout << x << std::endl ; 
                }
                std::cout << "Vectors: \n";
                for(auto const& x: _cell_sphere_surface_output_vector_vars){
                    std::cout << x << std::endl ; 
                }  
            }
        }
    }
    ~grace_runtime_impl_t() {} 

    friend class utils::singleton_holder<grace_runtime_impl_t,memory::default_create> ; 
    friend class memory::new_delete_creator<grace_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access

    static constexpr size_t longevity = GRACE_RUNTIME ; 

} ; 

using runtime = utils::singleton_holder<grace_runtime_impl_t,memory::default_create> ;

} /* namespace grace */

#endif /* INCLUDE_GRACE_SYSTEM_GRACE_RUNTIME */
