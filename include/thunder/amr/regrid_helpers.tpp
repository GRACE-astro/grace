/**
 * @file regrid_helpers.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-20
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

#ifndef THUNDER_AMR_REGRID_HELPERS_TPP 
#define THUNDER_AMR_REGRID_HELPERS_TPP

#include <thunder_config.h>

#include <thunder/amr/amr_functions.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/data_structures/variables.hh>
#include <thunder/data_structures/memory_defaults.hh>
#include <thunder/amr/regridding_policy_kernels.tpp>

#include <thunder/utils/interpolators.hh>

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

namespace thunder { namespace amr {

#ifdef THUNDER_3D 
using subview_t =  Kokkos::View<double****, default_space> ; 
#else 
using subview_t =  Kokkos::View<double***, default_space> ; 
#endif 

template< typename ViewT 
    , typename KerT 
    , typename ... KerArgT> 
void evaluate_regrid_criterion( ViewT flag_view
                              , KerT kernel  
                              , KerArgT&& ... kernel_args) 
{
    using namespace thunder  ;  
    auto& params = config_parser::get() ; 
    auto  state  = variable_list::get().getstate() ; 
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    size_t nq = amr::get_local_num_quadrants() ; 
    size_t ngz = amr::get_n_ghosts() ; 

    size_t REFINE_FLAG  = amr::quadrant_flags_t::REFINE  ;  
    size_t COARSEN_FLAG = amr::quadrant_flags_t::COARSEN ; 

    double CTORE = params["amr"]["refinement_criterion_CTORE"].as<double>() ; 
    double CTODE = params["amr"]["refinement_criterion_CTODE"].as<double>() ;
    
    /* Each thread league deals with a single quadrant */ 
    Kokkos::TeamPolicy<default_execution_space> policy(nq, Kokkos::AUTO() ) ; 
    using member_type = Kokkos::TeamPolicy<default_execution_space>::member_type ; 
    Kokkos::parallel_for( THUNDER_EXECUTION_TAG("AMR","eval_refine_coarsen_criterion")
                        , policy 
                        , KOKKOS_LAMBDA (member_type team_member)
    {
        double eps ; 
        /* 
        * parallel reduction of regridding criterion 
        * over quadrant cells 
        */ 
        auto reduce_range = 
            Kokkos::TeamThreadMDRange<Kokkos::Rank<THUNDER_NSPACEDIM>, member_type>( 
                    team_member 
                , VEC(nx,ny,nz) ) ; 
        int const q = team_member.league_rank() ; 
        Kokkos::parallel_reduce( 
                reduce_range 
            , KOKKOS_LAMBDA (VEC(int& i, int& j, int& k), double& leps )
            {
                leps = Kokkos::fmax(leps, kernel(VEC(i+ngz,j+ngz,k+ngz), q, kernel_args...)) ; 
            } 
            , eps 
        ) ; 
        if( team_member.team_rank() == 0 ) 
        {
            flag_view(q) = REFINE_FLAG  * ( eps > CTORE )
                         + COARSEN_FLAG * ( eps < CTODE ) ; 
        } 
    }) ;
}

template< typename InterpT
        , typename InViewT
        , typename OutViewT
        , typename CoordViewT >
void prolongate_variables(    InViewT  in_state 
                            , OutViewT out_state
                            , CoordViewT out_inv_spacing    
                            , Kokkos::vector<int,thunder::default_space>& in_idx 
                            , Kokkos::vector<int,thunder::default_space>& out_idx ) 
{  
    using namespace thunder ; 
    using namespace Kokkos  ; 
    long nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts()                   ; 
    long in_n_quad  = in_idx.size()                  ; 
    long out_n_quad = out_idx.size()                 ;
    int nvar = in_state.extent(THUNDER_NSPACEDIM)  ; 

    auto p_in_idx  = in_idx.d_view  ; 
    auto p_out_idx = out_idx.d_view ; 

    amr::prolongator_t<InterpT,decltype(out_state),decltype(out_inv_spacing)> kernel{ 
          VEC(nx,ny,nz)
        , ngz
        , out_state
        , out_inv_spacing 
    } ; 
    /*
    Kokkos::TeamPolicy<default_execution_space> policy( out_n_quad, Kokkos::AUTO() ) ;
    using member_type = Kokkos::TeamPolicy<default_execution_space>::member_type ; 

    Kokkos::parallel_for( THUNDER_EXECUTION_TAG("AMR","prolongate_variables")
                        , policy 
                        , KOKKOS_LAMBDA (member_type team)
    {
        
        size_t q_out = p_out_idx[team.league_rank()] ; 
        auto team_range = 
        Kokkos::TeamThreadMDRange<Kokkos::Rank<THUNDER_NSPACEDIM+2>, member_type>( 
              team 
            , VEC(nx,ny,nz)
            , nvar
            , P4EST_CHILDREN 
        ) ;
        
        parallel_for( team_range
                    , KOKKOS_LAMBDA ( VEC(int& i, int& j, int& k), int& ivar, int& ichild )
                    {
                        size_t q_in  = p_in_idx[team.league_rank()*P4EST_CHILDREN+ichild]; 
                        in_state(VEC(i+ngz,j+ngz,k+ngz),q_in,ivar) = 
                            kernel(VEC(i,j,k),q_out,ivar,ichild) ;
                    }
        ) ; 
    }) ; 
    */
   MDRangePolicy<Rank<THUNDER_NSPACEDIM+2>,default_execution_space>
        policy( {VEC(0,0,0),0,0}, {VEC(nx,ny,nz),nvar,out_n_quad}) ; 

    parallel_for(THUNDER_EXECUTION_TAG("AMR","prolongate_variables")
                        , policy 
                        , KOKKOS_LAMBDA (
                            VEC(const unsigned int& i
                               ,const unsigned int& j
                               ,const unsigned int& k)
                            ,const unsigned int& ivar
                            ,const unsigned int& iq_parent)
                        {
                            size_t q_out = p_out_idx(iq_parent) ; 
                            for( int ichild=0; ichild<P4EST_CHILDREN; ++ichild){
                                size_t q_in  = p_in_idx(P4EST_CHILDREN*iq_parent + ichild) ; 
                                in_state(VEC(i+ngz,j+ngz,k+ngz),ivar,q_in) = 
                                    kernel(VEC(i,j,k),q_out,ivar,ichild) ;
                            }
                        }
    ); 
}; 

template< typename InterpT
        , typename InViewT
        , typename OutViewT 
        , typename CoordViewT >
void restrict_variables(      InViewT  in_state 
                            , OutViewT out_state
                            , CoordViewT out_inv_spacing    
                            , Kokkos::vector<int, thunder::default_space>& in_idx 
                            , Kokkos::vector<int, thunder::default_space>& out_idx )  
{
    using namespace thunder ; 
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents()        ;
    auto ngz = amr::get_n_ghosts()                          ;  
    size_t in_n_quad  = in_idx.size()                       ; 
    size_t out_n_quad = out_idx.size()                      ;
    int nvar = in_state.extent(THUNDER_NSPACEDIM) ; 

    auto p_in_idx  = in_idx.d_view  ; 
    auto p_out_idx = out_idx.d_view ; 

    amr::restrictor_t<InterpT,decltype(out_state),decltype(out_inv_spacing)> kernel{ VEC(nx,ny,nz), ngz, out_state, out_inv_spacing } ; 

    Kokkos::TeamPolicy<default_execution_space> policy( in_n_quad, Kokkos::AUTO() ) ;
    using member_type = Kokkos::TeamPolicy<default_execution_space>::member_type ; 

    Kokkos::parallel_for( THUNDER_EXECUTION_TAG("AMR","restrict_variables")
                        , policy 
                        , KOKKOS_LAMBDA (member_type team)
    {
        
        size_t q_in = p_in_idx[team.league_rank()] ; 
        auto team_range = 
        Kokkos::TeamThreadMDRange<Kokkos::Rank<THUNDER_NSPACEDIM+1>, member_type>( 
                team 
            , VEC(nx,ny,nz),nvar ) ;
        
        parallel_for( team_range
                    , KOKKOS_LAMBDA ( VEC(int& i, int& j, int& k), int& ivar)
                    {
                        int q_out[P4EST_CHILDREN]; 
                        for(int ichild=0;ichild<P4EST_CHILDREN;++ichild) {
                            q_out[ichild] = 
                                p_out_idx(team.league_rank()*P4EST_CHILDREN+ichild); 
                        }
                        in_state(VEC(i+ngz,j+ngz,k+ngz),ivar,q_in) = 
                            kernel(VEC(i,j,k),q_out,ivar) ;
                    }
        ) ; 
    }) ; 
} ;

}} /* namespace thunder::amr */ 

#endif 