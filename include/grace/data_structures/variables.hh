/**
 * @file variables.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-07
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

#ifndef GRACE_DATA_STRUCTURES_VARIABLES_HH
#define GRACE_DATA_STRUCTURES_VARIABLES_HH

#include <Kokkos_Core.hpp>

#include <grace_config.h>

#include<code_modules.h>
#include<grace/data_structures/variable_properties.hh>
#include<grace/data_structures/variable_indices.hh>
#include<grace/amr/amr_functions.hh>
#include<grace/utils/inline.h>
#include<grace/utils/singleton_holder.hh> 
#include<grace/utils/creation_policies.hh>
#include<grace/utils/lifetime_tracker.hh>

namespace grace { 
//*****************************************************************************************************
int get_variable_index(std::string const& name, bool is_aux=false) ;
//*****************************************************************************************************
/**
 * @brief Implementation of the variable list type.
 * 
 * \ingroup variables
 * 
 * This class is wrapped with a <code>singleton_holder</code>
 * and is used as a utility to collect all variables and coordinates
 * and ensure they have the appropriate lifetime during GRACE's execution.
 */
class variable_list_impl_t
{

public:
    //*****************************************************************************************************
    /**
     * @brief Get quadrant coordinates.
     * 
     * @return Quadrant coordinates. 
     */
    GRACE_ALWAYS_INLINE scalar_array_t<GRACE_NSPACEDIM>&
    getcoords() { return _coords ; } 
    //*****************************************************************************************************
    /**
     * @brief Get quadrant coordinates.
     * 
     * @return Quadrant coordinates. 
     */
    GRACE_ALWAYS_INLINE staggered_coordinate_arrays_t&
    getstaggeredcoords() { return _staggered_coords ; } 
    //*****************************************************************************************************
    /**
     * @brief Get inverse spacing of cell coordinates.
     * 
     * @return Spacing of cell coordinates  
     */
    GRACE_ALWAYS_INLINE cell_vol_array_t<GRACE_NSPACEDIM>&
    getvolumes() { return _cell_volumes ; }
    //*****************************************************************************************************
    /**
     * @brief Get inverse spacing of cell coordinates.
     * 
     * @return Spacing of cell coordinates  
     */
    GRACE_ALWAYS_INLINE scalar_array_t<GRACE_NSPACEDIM>&
    getinvspacings() { return _coords_ispacing ; }  
    //*****************************************************************************************************
    /**
     * @brief Get spacing of cell coordinates.
     * 
     * @return Spacing of cell coordinates  
     */
    GRACE_ALWAYS_INLINE scalar_array_t<GRACE_NSPACEDIM>&
    getspacings() { return _coords_spacing ; } 
    //*****************************************************************************************************
    /**
     * @brief Get the auxiliary variables.
     * 
     * @return The auxiliary variables. 
     */
    GRACE_ALWAYS_INLINE var_array_t&  
    getaux() { return _aux ; }
    //*****************************************************************************************************
    /**
     * @brief Get the staggered auxiliary variables.
     * 
     * @return The staggered auxiliary variables. 
     */
    GRACE_ALWAYS_INLINE staggered_variable_arrays_t&  
    getstaggeredaux() { return _staggered_aux  ; }
    //*****************************************************************************************************
    /**
     * @brief Get state vector
     * 
     * @return The state vector, containing all evolved variables
     *         on all local cells.  
     */
    GRACE_ALWAYS_INLINE var_array_t&  
    getstate() { return _state ; }
    //*****************************************************************************************************
    /**
     * @brief Get staggered state vector
     * 
     * @return The staggered state vector, containing all evolved staggered variables
     *         on all local cells.  
     */
    GRACE_ALWAYS_INLINE staggered_variable_arrays_t&  
    getstaggeredstate() { return _staggered_vars ; }
    //*****************************************************************************************************
    /**
     * @brief Get staggered state scratch vector
     * 
     * @return The staggered state scratch vector, containing all evolved staggered variables
     *         on all local cells.  
     */
    GRACE_ALWAYS_INLINE staggered_variable_arrays_t&  
    getstaggeredscratch() { return _staggered_vars_p ; }
    //*****************************************************************************************************
    /**
     * @brief Get the scratch state vector 
     * 
     * @return The scratch state vector, used during time 
     *         evolution to hold the previous state. 
     */
    GRACE_ALWAYS_INLINE var_array_t& 
    getscratch() { return _state_p ; }
    //*****************************************************************************************************
    /**
     * @brief Get the staging buffer for time-stepping. 
     * 
     * @return The staging buffer, it consists in a set of var_arrays_t. The
     *         number of which depends on the active timestepper.
     */
    GRACE_ALWAYS_INLINE std::vector<var_array_t>&
    getstagingbuffer() { return _staging_buffer ; }
    //*****************************************************************************************************
    /**
     * @brief Get the staging buffer for time-stepping. 
     * 
     * @return The staging buffer, it consists in a set of var_arrays_t. The
     *         number of which depends on the active timestepper.
     */
    GRACE_ALWAYS_INLINE std::vector<staggered_variable_arrays_t>&
    getstagstagingbuffer() { return _stag_staging_buffer ; }
    //*****************************************************************************************************
    /**
     * @brief Get the fluxes vector 
     * 
     * @return The fluxes vector, used during time 
     *         evolution to hold the fluxes. 
     */
    GRACE_ALWAYS_INLINE flux_array_t& 
    getfluxesarray() { return _fluxes ; }
    //*****************************************************************************************************
    /**
     * @brief Get the fluxes vector 
     * 
     * @return The fluxes vector, used during time 
     *         evolution to hold the fluxes. 
     */
    GRACE_ALWAYS_INLINE flux_array_t& 
    getvbararray() { return _vbar ; }
    //*****************************************************************************************************
    /**
     * @brief Get the EMF vector 
     * 
     * @return The EMF vector, used during time 
     *         evolution to hold the electromotive force at cell edges. 
     */
    GRACE_ALWAYS_INLINE emf_array_t& 
    getemfarray() { return _emf ; }
    //*****************************************************************************************************
    /**
     * @brief Get the halo state vector 
     */
    GRACE_ALWAYS_INLINE var_array_t& 
    gethalo() { return _halo ; }
    //*****************************************************************************************************
    template< typename ... ArgT >
    void realloc_state(ArgT&& ... args)
    {
        Kokkos::realloc(_state, args...) ; 
    } 
    //*****************************************************************************************************
    var_array_t
    allocate_state(std::string const& name = "temp_storage") const 
    {
        DECLARE_GRID_EXTENTS ; 
        return grace::var_array_t(name, VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz), _state.extent(GRACE_NSPACEDIM), nq) ;
    }
    //*****************************************************************************************************
    staggered_variable_arrays_t
    allocate_staggered_state(std::string const& name = "temp_storage") const 
    {
        DECLARE_GRID_EXTENTS ; 
        auto tmp_storage = grace::staggered_variable_arrays_t() ; 
        auto nvars_corner = _staggered_vars.corner_staggered_fields.extent(GRACE_NSPACEDIM) ;
        auto nvars_face  = _staggered_vars.face_staggered_fields_x.extent(GRACE_NSPACEDIM) ;
        auto nvars_edge  = _staggered_vars.edge_staggered_fields_xz.extent(GRACE_NSPACEDIM) ;
        tmp_storage.realloc(VEC(nx,ny,nz), ngz, nq, nvars_face, nvars_edge, nvars_corner) ;
        return tmp_storage ; 
    }

private: 
    //*****************************************************************************************************
    /**
     * @brief (Never) construct a new <code>variable_list_impl_t</code> object
     * 
     */
    variable_list_impl_t() ; 
    //*****************************************************************************************************
    /**
     * @brief (Never) destroy the <code>variable_list_impl_t</code> object
     * 
     */
    ~variable_list_impl_t() = default; 
    //*****************************************************************************************************
    //******** Member variables ***************************************************************************
    //*****************************************************************************************************
    scalar_array_t<GRACE_NSPACEDIM>  _coords  ;  //!< tree-logical coordinates of quadrant corners 
    scalar_array_t<GRACE_NSPACEDIM>  _coords_ispacing  ;  //!< Inverse spacing of coordinate system
    scalar_array_t<GRACE_NSPACEDIM>  _coords_spacing  ;  //!< Spacing of coordinate system
    cell_vol_array_t<GRACE_NSPACEDIM> _cell_volumes ; //!< Volume of cells 
    var_array_t _state   ;     //!< State variables 
    var_array_t _state_p ;     //!< Second timelevel, allocated at all times 
    var_array_t _halo    ;     //!< Halo exchange buffer, allocated when necessary
    var_array_t _aux     ;     //!< Auxiliary variables  
    std::vector<var_array_t> _staging_buffer; //!< Additional storage for timestepper
    staggered_coordinate_arrays_t _staggered_coords ; //!< Staggered coordinate utilities (surfaces, lengths) 
    staggered_variable_arrays_t   _staggered_vars   ; //!< Staggered variable arrays 
    staggered_variable_arrays_t   _staggered_vars_p ; //!< Staggered scratch state
    staggered_variable_arrays_t   _staggered_aux    ; //!< Auxiliary variables on staggered grids.
    std::vector<staggered_variable_arrays_t> _stag_staging_buffer ; //!< Additional storage for timestepper 
    flux_array_t   _fluxes              ; //!< Fluxes for time evolution.
    flux_array_t   _vbar              ; //!< Fluxes for time evolution.
    emf_array_t   _emf                  ; //!< EMF for time evolution of ideal MHD.
    //*****************************************************************************************************
    friend class utils::singleton_holder<variable_list_impl_t, memory::default_create> ; //!< Give access 
    friend class memory::new_delete_creator<variable_list_impl_t, memory::new_delete_allocator> ; //!< Give access 
    //*****************************************************************************************************
    static constexpr size_t longevity = GRACE_VARIABLES ; //!< Schedule destruction at appropriate time.
    //*****************************************************************************************************
} ; 
//*****************************************************************************************************
/**
 * @brief Proxy holding all variables within GRACE. Only a unique instance 
 *        of this class exists at runtime and its reference can be obtained 
 *        via the <code>get()</code> static method. See references on 
 *        <code>singleton_holder</code> for details.
 * \ingroup variables
 */
using variable_list = utils::singleton_holder<variable_list_impl_t > ; 
//*****************************************************************************************************
//*****************************************************************************************************

} /* namespace grace */

#endif /* GRACE_DATA_STRUCTURES_VARIABLES_HH */ 