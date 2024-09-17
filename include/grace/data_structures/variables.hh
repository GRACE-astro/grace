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
#include<grace/data_structures/macros.hh>

#include<grace/utils/inline.h>
#include<grace/utils/singleton_holder.hh> 
#include<grace/utils/creation_policies.hh>
#include<grace/utils/lifetime_tracker.hh>

namespace grace { 
//*****************************************************************************************************
size_t get_variable_index(std::string const& name, bool is_aux=false) ;
//*****************************************************************************************************
size_t get_staggered_variable_index( std::string const& name, int& staggering, bool is_aux) ; 
//*****************************************************************************************************
size_t get_variable_index_ext(std::string const& name, int& staggering, bool is_aux) ; 
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
    GRACE_ALWAYS_INLINE var_array_t<GRACE_NSPACEDIM>&  
    getaux() { return _aux ; }
    //*****************************************************************************************************
    /**
     * @brief Get state vector
     * 
     * @return The state vector, containing all evolved variables
     *         on all local cells.  
     */
    GRACE_ALWAYS_INLINE var_array_t<GRACE_NSPACEDIM>&  
    getstate() { return _state ; }
    //*****************************************************************************************************
    /**
     * @brief Get the scratch state vector 
     * 
     * @return The scratch state vector, used during time 
     *         evolution to hold the previous state. 
     */
    GRACE_ALWAYS_INLINE var_array_t<GRACE_NSPACEDIM>& 
    getscratch() { return _state_p ; }
    //*****************************************************************************************************
    /**
     * @brief Get the staggered state vector
     * 
     * @return The state vector, containing all evolved variables
     *         on all local cells.  
     */
    GRACE_ALWAYS_INLINE staggered_variable_arrays_t&  
    getstaggeredstate() { return _staggered_vars ; }
    //*****************************************************************************************************
    /**
     * @brief Get the scratch staggered state vector 
     * 
     * @return The scratch state vector, used during time 
     *         evolution to hold the previous state. 
     */
    GRACE_ALWAYS_INLINE staggered_variable_arrays_t& 
    getstaggeredscratch() { return _staggered_vars_p ; }
    //*****************************************************************************************************
    /**
     * @brief Get the staggered aux vector 
     * 
     * @return The staggered aux vector. 
     */
    GRACE_ALWAYS_INLINE staggered_variable_arrays_t& 
    getstaggeredaux() { return _staggered_aux ; }
    //*****************************************************************************************************
    /**
     * @brief Get the halo state vector 
     */
    GRACE_ALWAYS_INLINE var_array_t<GRACE_NSPACEDIM>& 
    gethalo() { return _halo ; }
    //*****************************************************************************************************
    template< typename ... ArgT >
    void realloc_state(ArgT&& ... args)
    {
        Kokkos::realloc(_state, args...) ; 
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
    var_array_t<GRACE_NSPACEDIM> _state   ;     //!< State variables 
    var_array_t<GRACE_NSPACEDIM> _state_p ;     //!< Second timelevel, allocated at all times 
    var_array_t<GRACE_NSPACEDIM> _halo    ;     //!< Halo exchange buffer, allocated when necessary
    var_array_t<GRACE_NSPACEDIM> _aux     ;     //!< Auxiliary variables  
    staggered_coordinate_arrays_t _staggered_coords ; //!< Staggered coordinate utilities (surfaces, lengths) 
    staggered_variable_arrays_t   _staggered_vars   ; //!< Staggered variable arrays 
    staggered_variable_arrays_t   _staggered_vars_p ; //!< Staggered scratch state
    staggered_variable_arrays_t   _staggered_aux    ; //!< Auxiliary variables on staggered grids.
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
template< typename IndexT_x 
        , typename IndexT_y 
        , typename IndexT_z 
        , typename IndexT_q > 
static decltype(auto) 
get_variable_subview(
      std::string const& vname
    , VEC( IndexT_x xidx_subset 
         , IndexT_y yidx_subset
         , IndexT_z zidx_subset 
    , IndexT_q qidx_subset )
) 
{
    using namespace grace  ; 
    using namespace Kokkos ; 
    

    auto& state = variables::get().getstate() ; 
    auto& sstate = variables::get().getstaggeredstate() ; 
    auto& aux = variables::get().getaux() ; 
    auto& saux = variables::get().getstaggeredaux() ; 
    if ( props.is_evolved ) {
        if ( props.staggering == var_staggering_t::CELL_CENTER ) {
            return Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), props.index, ALL()) ; 
        } else if (props.staggering == var_staggering_t::FACE ) {
            auto& fstate = (props.comp_num == 0) ? sstate.face_staggered_fields_x
                : ( (props.comp_num == 1) ? sstate.face_staggered_fields_y : sstate.face_staggered_fields_z ) ; 
            return Kokkos::subview(fstate, VEC(ALL(),ALL(),ALL()), props.index, ALL())  ; 
        } else if (props.staggering == var_staggering_t::EDGE ) {
            auto& estate = (props.comp_num == 0) ? sstate.edge_staggered_fields_yz
                : ( (props.comp_num == 1) ? sstate.edge_staggered_fields_xz : sstate.edge_staggered_fields_xy ) ; 
            return Kokkos::subview(estate, VEC(ALL(),ALL(),ALL()), props.index, ALL())  ; 
        } else {
            auto& cstate = sstate.corner_staggered_fields ;
            return Kokkos::subview(cstate, VEC(ALL(),ALL(),ALL()), props.index, ALL())  ; 
        }
    } else {
        if ( props.staggering == var_staggering_t::CELL_CENTER ) {
            return Kokkos::subview(aux, VEC(ALL(),ALL(),ALL()), props.index, ALL()) ; 
        } else if (props.staggering == var_staggering_t::FACE ) {
            auto& fstate = (props.comp_num == 0) ? saux.face_staggered_fields_x
                : ( (props.comp_num == 1) ? saux.face_staggered_fields_y : saux.face_staggered_fields_z ) ; 
            return Kokkos::subview(fstate, VEC(ALL(),ALL(),ALL()), props.index, ALL())  ; 
        } else if (props.staggering == var_staggering_t::EDGE ) {
            auto& estate = (props.comp_num == 0) ? saux.edge_staggered_fields_yz
                : ( (props.comp_num == 1) ? saux.edge_staggered_fields_xz : saux.edge_staggered_fields_xy ) ; 
            return Kokkos::subview(estate, VEC(ALL(),ALL(),ALL()), props.index, ALL())  ; 
        } else {
            auto& cstate = saux.corner_staggered_fields ;
            return Kokkos::subview(cstate, VEC(ALL(),ALL(),ALL()), props.index, ALL())  ; 
        }
    }
} ; 

static decltype(auto)
get_variable_subview(std::string const& vname) 
{
    return get_variable_subview(vname, VEC(Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), Kokkos::ALL() ) ;
}
} /* namespace grace */

#endif /* GRACE_DATA_STRUCTURES_VARIABLES_HH */ 