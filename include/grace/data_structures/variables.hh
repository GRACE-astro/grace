/**
 * @file variables.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Global variable-list singleton: registry of evolved, auxiliary and HRSC variables with their properties (rank, staggering, boundary types).
 * @version 0.1
 * @date 2024-03-07
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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
    #if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS 
    /**
     * @brief Get the cell centered electric field 
     * 
     * @return The cell centered electric field.
     */
    GRACE_ALWAYS_INLINE var_array_t& 
    getecarray() { return _Ecenter ; }
    /**
     * @brief Get the face-staggered electric field 
     * 
     * @return The face-staggered electric field.
     */
    GRACE_ALWAYS_INLINE flux_array_t& 
    getefarray() { return _Eface ; }
    #else 
    /**
     * @brief Get the fluxes vector 
     * 
     * @return The fluxes vector, used during time 
     *         evolution to hold the fluxes. 
     */
    GRACE_ALWAYS_INLINE flux_array_t& 
    getvbararray() { return _vbar ; }
    #endif 
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
     * @brief Get the FOFC flagged-cell index lists.
     *
     * Compacted (q,i,j,k) of every interior cell whose tentative post-flux
     * c2p would have floored. Populated atomically by flag_fofc_cells;
     * consumed by apply_fofc_correction, which iterates [0, count).
     *
     * Each list is allocated to nx*ny*nz*nq (worst case = every interior
     * cell flagged in every quadrant).
     */
    GRACE_ALWAYS_INLINE Kokkos::View<int*****, grace::default_space>& getfofcfacetags() { return _fofc_face_tags ; }
    GRACE_ALWAYS_INLINE Kokkos::View<int*****, grace::default_space>& getfofcedgetags() { return _fofc_edge_tags ; }

    GRACE_ALWAYS_INLINE Kokkos::View<fofc_index_tag_t*, grace::default_space>& getfofcfx() { return _fofc_fx ; }
    GRACE_ALWAYS_INLINE Kokkos::View<fofc_index_tag_t*, grace::default_space>& getfofcfy() { return _fofc_fy ; }
    GRACE_ALWAYS_INLINE Kokkos::View<fofc_index_tag_t*, grace::default_space>& getfofcfz() { return _fofc_fz ; }
    GRACE_ALWAYS_INLINE Kokkos::View<fofc_index_tag_t*, grace::default_space>& getfofceyz() { return _fofc_eyz ; }
    GRACE_ALWAYS_INLINE Kokkos::View<fofc_index_tag_t*, grace::default_space>& getfofcexz() { return _fofc_exz ; }
    GRACE_ALWAYS_INLINE Kokkos::View<fofc_index_tag_t*, grace::default_space>& getfofcexy() { return _fofc_exy ; }

    GRACE_ALWAYS_INLINE Kokkos::View<int[3], grace::default_space>& getfofcfcnt() { return _fofc_face_cnt ; }
    GRACE_ALWAYS_INLINE Kokkos::View<int[3], grace::default_space>& getfofcecnt() { return _fofc_edge_cnt ; }
    //*****************************************************************************************************
    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    /**
     * @brief Per-cell scratch produced by the Z4c curvature-pre kernel.
     *        Holds Ricci tensor, Ricci trace, and second Christoffel
     *        (see z4c_curv_scratch_idx in variable_indices.hh).
     */
    GRACE_ALWAYS_INLINE var_array_t&
    getz4ccurvscratch() { return _z4c_curv_scratch ; }
    #endif
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
    //*****************************************************************************************************
    void resize_aux_staging_and_flux_buffers(int nq_new) ; 
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
    scalar_array_t<GRACE_NSPACEDIM>  _coords_ispacing  ;  //!< Inverse spacing of coordinate system
    scalar_array_t<GRACE_NSPACEDIM>  _coords_spacing  ;  //!< Spacing of coordinate system
    var_array_t _state   ;     //!< State variables 
    var_array_t _state_p ;     //!< Second timelevel, allocated at all times 
    var_array_t _aux     ;     //!< Auxiliary variables  
    std::vector<var_array_t> _staging_buffer; //!< Additional storage for timestepper
    staggered_variable_arrays_t   _staggered_vars   ; //!< Staggered variable arrays 
    staggered_variable_arrays_t   _staggered_vars_p ; //!< Staggered scratch state
    std::vector<staggered_variable_arrays_t> _stag_staging_buffer ; //!< Additional storage for timestepper 
    flux_array_t   _fluxes              ; //!< Fluxes for time evolution.
    #if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS 
    var_array_t  _Ecenter ; //!< Cell centered E field 
    flux_array_t _Eface   ; //!< Face staggered E field
    #else 
    flux_array_t   _vbar              ; //!< Fluxes for time evolution.
    #endif 
    emf_array_t   _emf                  ; //!< EMF for time evolution of ideal MHD.
    Kokkos::View<int*****, grace::default_space> _fofc_face_tags ;     //!< Tag faces for fofc, atomically.
    Kokkos::View<int*****, grace::default_space> _fofc_edge_tags ;     //!< Tag edges for fofc, atomically.
    Kokkos::View<fofc_index_tag_t*, grace::default_space> _fofc_fx, _fofc_fy, _fofc_fz    ; //!< Unique faces in each direction that need first order correction
    Kokkos::View<fofc_index_tag_t*, grace::default_space> _fofc_eyz, _fofc_exz, _fofc_exy ; //!< Unique edges in each direction that need first order correction
    Kokkos::View<int[3], grace::default_space> _fofc_face_cnt, _fofc_edge_cnt ; //!< Count of unique faces/edges where fofc is needed
    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    var_array_t   _z4c_curv_scratch     ; //!< Persistent intermediates between curvature-pre and curvature-update kernels.
    #endif
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