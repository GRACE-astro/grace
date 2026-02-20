/**
 * @file initial_data.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-15
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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

#include <grace_config.h>

#include <grace/evolution/initial_data.hh>
#include <grace/config/config_parser.hh>
#include <grace/physics/grace_physical_systems.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/data_structures/index_helpers.hh>
#ifdef GRACE_ENABLE_GRMHD
//#include <grace/physics/admbase.hh>
#include <grace/physics/grmhd.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#endif
#ifdef GRACE_ENABLE_M1
#include <grace/physics/m1_helpers.hh>
#include <grace/physics/m1.hh>
#endif 
#include <grace/physics/eos/eos_types.hh>

#include <Kokkos_Core.hpp>

namespace grace {

void set_initial_data() {
    auto const eos_type = grace::get_param<std::string>("eos", "eos_type") ;
    if( eos_type == "hybrid" ) {
        auto const cold_eos_type = 
            get_param<std::string>("eos","hybrid_eos","cold_eos_type") ;  
        if( cold_eos_type == "piecewise_polytrope" ) {
            set_initial_data_impl<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>() ; 
        } else if ( cold_eos_type == "tabulated" ) {
            ERROR("Not implemented yet.") ;
        }
    } else if ( eos_type == "tabulated" ) {
        ERROR("Not implemented yet.") ; 
    }
}

template< typename eos_t >
void set_initial_data_impl() {
    Kokkos::Profiling::pushRegion("ID") ; 
    using namespace grace ;

    #ifdef GRACE_ENABLE_SCALAR_ADV 
    set_scalar_advection_initial_data() ; 
    #endif 
    #ifdef GRACE_ENABLE_BURGERS
    set_burgers_initial_data() ; 
    #endif 
    #ifdef GRACE_ENABLE_GRMHD
    set_grmhd_initial_data<eos_t>();
    #endif 
    Kokkos::fence() ; 
    #ifdef GRACE_ENABLE_M1
    set_m1_initial_data<eos_t>();
    #endif 
    //transform_to_logical_frame() ; 
    Kokkos::Profiling::popRegion() ; 
} 

#if 0
void transform_to_logical_frame() {
    DECLARE_GRID_EXTENTS ; 
    using namespace grace  ; 
    using namespace Kokkos ;
    /************************************************/
    /* Fill coordinate jacobian matrices on device  */
    /* at cell centers                              */
    /************************************************/
    jacobian_array_t jac, invjac ; 
    fill_jacobian_matrices(jac,invjac) ;  
    /************************************************/
    /* Retrieve vector and tensor variables and     */
    /* auxiliaries                                  */
    /************************************************/
    auto const h_vec_vars = variables::get_vector_state_variables_indices()  ;
    auto const h_tens_vars = variables::get_tensor_state_variables_indices() ;
    auto const h_vec_aux = variables::get_vector_aux_variables_indices()     ;
    auto const h_tens_aux = variables::get_tensor_aux_variables_indices()    ;
    
    auto const h_vec_kind     = variables::get_vector_state_variables_kinds() ; 
    auto const h_tens_kind    = variables::get_tensor_state_variables_kinds() ;
    auto const h_vec_aux_kind = variables::get_vector_aux_variables_kinds()   ;
    auto const h_tens_aux_kind = variables::get_tensor_aux_variables_kinds()   ;


    for( int i=0; i<h_vec_vars.size(); ++i){
        GRACE_TRACE("Vector var (state): {}", variables::get_var_name(h_vec_vars[i],false)) ; 
    }
    for( int i=0; i<h_tens_vars.size(); ++i){
        GRACE_TRACE("Tensor var (state): {}", variables::get_var_name(h_tens_vars[i],false)) ; 
    }
    for( int i=0; i<h_vec_aux.size(); ++i){
        GRACE_TRACE("Vector var (aux): {}", variables::get_var_name(h_vec_aux[i],true)) ; 
    }
    for( int i=0; i<h_tens_aux.size(); ++i){
        GRACE_TRACE("Tensor var (aux): {}", variables::get_var_name(h_tens_aux[i],true)) ; 
    }
    Kokkos::View<size_t* , default_execution_space> 
        vec_vars("vector_var_indices", h_vec_vars.size()), vec_vars_idx("vector_var_kind", h_vec_vars.size())
      , tens_vars("tensor_var_indices", h_tens_vars.size()), tens_vars_idx("tensor_var_kind", h_tens_vars.size())
      , vec_aux("vector_aux_indices", h_vec_aux.size()), vec_aux_idx("vector_aux_kind", h_vec_aux.size())
      , tens_aux("tensor_aux_indices", h_tens_aux.size()), tens_aux_idx("tensor_aux_kind", h_tens_aux.size()) ; 
    size_t const n_vec_vars = h_vec_vars.size() ; 
    size_t const n_tens_vars = h_tens_vars.size() ; 
    size_t const n_vec_aux = h_vec_aux.size() ;
    size_t const n_tens_aux = h_tens_aux.size() ; 

    deep_copy_vec_to_view(vec_vars, h_vec_vars) ; 
    deep_copy_vec_to_view(tens_vars, h_tens_kind) ; 
    deep_copy_vec_to_view(vec_aux, h_vec_aux) ; 
    deep_copy_vec_to_view(tens_aux, h_tens_aux) ; 

    deep_copy_vec_to_view(vec_vars_idx, h_vec_kind) ; 
    deep_copy_vec_to_view(tens_vars_idx, h_tens_vars) ; 
    deep_copy_vec_to_view(vec_aux_idx, h_vec_aux_kind) ; 
    deep_copy_vec_to_view(tens_aux_idx, h_tens_aux_kind) ; 

    /************************************************/
    /* Launch a kernel to apply jacobians to tensor */
    /* and vector variables                         */
    /************************************************/
    auto& state = variable_list::get().getstate() ; 
    auto& aux   = variable_list::get().getaux()   ; 
    GRACE_TRACE("Starting conversion to logical coordinates.") ; 
    parallel_for(GRACE_EXECUTION_TAG("ID","convert_to_logical")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) 
                {
                    for(int iv=0; iv<n_vec_vars; ++iv){
                        auto& jview = vec_vars_idx(iv) == UP ? invjac : jac ; 
                        std::array<double,3> const vin
                        {
                              state(VEC(i,j,k),vec_vars(iv)  ,q)
                            , state(VEC(i,j,k),vec_vars(iv)+1,q)
                            , state(VEC(i,j,k),vec_vars(iv)+2,q)
                        } ; 
                        auto J = Kokkos::subview(jview, VEC(i,j,k), ALL(), ALL(),  q) ; 
                        auto const vout = detail::apply_jacobian_vec(vin, J) ;
                        state(VEC(i,j,k),vec_vars(iv)  ,q) = vout[0] ; 
                        state(VEC(i,j,k),vec_vars(iv)+1,q) = vout[1] ; 
                        #ifdef GRACE_3D 
                        state(VEC(i,j,k),vec_vars(iv)+2,q) = vout[2] ; 
                        #endif
                    }

                    for(int iv=0; iv<n_tens_vars; ++iv){
                        auto& jview = tens_vars_idx(iv) == UPUP ? invjac : jac ; 
                        std::array<double,6> const vin
                        {
                              state(VEC(i,j,k),tens_vars(iv)  ,q)
                            , state(VEC(i,j,k),tens_vars(iv)+1,q)
                            , state(VEC(i,j,k),tens_vars(iv)+2,q)
                            , state(VEC(i,j,k),tens_vars(iv)+3,q)
                            , state(VEC(i,j,k),tens_vars(iv)+4,q)
                            , state(VEC(i,j,k),tens_vars(iv)+5,q)
                        } ; 
                        auto J = Kokkos::subview(jview, VEC(i,j,k), ALL(), ALL(),  q) ; 
                        auto const vout = detail::apply_jacobian_symtens(vin, J) ;
                        state(VEC(i,j,k),tens_vars(iv)  ,q) = vout[0] ; 
                        state(VEC(i,j,k),tens_vars(iv)+1,q) = vout[1] ; 
                        #ifdef GRACE_3D
                        state(VEC(i,j,k),tens_vars(iv)+2,q) = vout[2] ; 
                        #endif 
                        state(VEC(i,j,k),tens_vars(iv)+3,q) = vout[3] ; 
                        #ifdef GRACE_3D 
                        state(VEC(i,j,k),tens_vars(iv)+4,q) = vout[4] ; 
                        state(VEC(i,j,k),tens_vars(iv)+5,q) = vout[5] ;
                        #endif   
                    }

                    for(int iv=0; iv<n_vec_aux; ++iv){
                        auto& jview = vec_aux_idx(iv) == UP ? invjac : jac ; 
                        std::array<double,3> const vin
                        {
                              aux(VEC(i,j,k),vec_aux(iv)  ,q)
                            , aux(VEC(i,j,k),vec_aux(iv)+1,q)
                            , aux(VEC(i,j,k),vec_aux(iv)+2,q)
                        } ; 
                        auto J = Kokkos::subview(jview, VEC(i,j,k), ALL(), ALL(),  q) ; 
                        auto const vout = detail::apply_jacobian_vec(vin, J) ;
                        aux(VEC(i,j,k),vec_aux(iv)  ,q) = vout[0] ; 
                        aux(VEC(i,j,k),vec_aux(iv)+1,q) = vout[1] ; 
                        #ifdef GRACE_3D
                        aux(VEC(i,j,k),vec_aux(iv)+2,q) = vout[2] ;
                        #endif
                    }

                    for(int iv=0; iv<n_tens_aux; ++iv){
                        auto& jview = tens_aux_idx(iv) == UPUP ? invjac : jac ; 
                        std::array<double,6> const vin
                        {
                              aux(VEC(i,j,k),tens_aux(iv)  ,q)
                            , aux(VEC(i,j,k),tens_aux(iv)+1,q)
                            , aux(VEC(i,j,k),tens_aux(iv)+2,q)
                            , aux(VEC(i,j,k),tens_aux(iv)+3,q)
                            , aux(VEC(i,j,k),tens_aux(iv)+4,q)
                            , aux(VEC(i,j,k),tens_aux(iv)+5,q)
                        } ; 
                        auto J = Kokkos::subview(jview, VEC(i,j,k), ALL(), ALL(),  q) ; 
                        auto const vout = detail::apply_jacobian_symtens(vin, J) ;
                        aux(VEC(i,j,k),tens_aux(iv)  ,q) = vout[0] ; 
                        aux(VEC(i,j,k),tens_aux(iv)+1,q) = vout[1] ; 
                        #ifdef GRACE_3D
                        aux(VEC(i,j,k),tens_aux(iv)+2,q) = vout[2] ; 
                        #endif 
                        aux(VEC(i,j,k),tens_aux(iv)+3,q) = vout[3] ; 
                        #ifdef GRACE_3D
                        aux(VEC(i,j,k),tens_aux(iv)+4,q) = vout[4] ; 
                        aux(VEC(i,j,k),tens_aux(iv)+5,q) = vout[5] ;
                        #endif 
                    }
                } ) ; 
}
#endif 
#define INSTANTIATE_TEMPLATE(EOS)   \
template                            \
void set_initial_data_impl<EOS>()
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}