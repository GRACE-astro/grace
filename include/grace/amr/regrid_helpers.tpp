/**
 * @file regrid_helpers.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-20
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

#ifndef GRACE_AMR_REGRID_HELPERS_TPP 
#define GRACE_AMR_REGRID_HELPERS_TPP

#include <grace_config.h>

#include <grace/amr/amr_functions.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/amr/regridding_policy_kernels.tpp>

#include <grace/utils/interpolators.hh>

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

namespace grace { namespace amr {

/**
 * @brief Decide whether a quadrant needs to be refined/coarsened
 *        based on custom criterion.
 * \ingroup amr
 * 
 * @tparam ViewT Type of variable view.
 * @tparam KerT  Type of the kernel.
 * @tparam KerArgT Type of extra arguments to the kernel.
 * @param flag_view View containing regrid flags. 
 * @param kernel    Cell-wise kernel to decide whether to regrid.
 * @param kernel_args Extra arguments to kernel.
 */
template< typename ViewT 
    , typename KerT 
    , typename ... KerArgT> 
void evaluate_regrid_criterion_kernel( ViewT flag_view
                                     , KerT kernel  
                                     , KerArgT&& ... kernel_args) 
{
    using namespace grace  ;  

    int nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int nq = amr::get_local_num_quadrants() ; 
    int const ngz = amr::get_n_ghosts() ; 

    size_t REFINE_FLAG  = amr::quadrant_flags_t::REFINE  ;  
    size_t COARSEN_FLAG = amr::quadrant_flags_t::COARSEN ; 

    double CTORE = grace::get_param<double>("amr", "refinement_criterion_CTORE") ;
    double CTODE = grace::get_param<double>("amr", "refinement_criterion_CTODE") ;
    
    /* Each thread league deals with a single quadrant */ 
    Kokkos::TeamPolicy<default_execution_space> policy(nq, Kokkos::AUTO() ) ; 
    using member_type = Kokkos::TeamPolicy<default_execution_space>::member_type ; 
    Kokkos::parallel_for( GRACE_EXECUTION_TAG("AMR","eval_refine_coarsen_criterion")
                        , policy 
                        , KOKKOS_LAMBDA (member_type team_member)
    {
        double eps ; 
        /* 
        * parallel reduction of regridding criterion 
        * over quadrant cells 
        */ 
        auto reduce_range = 
            Kokkos::TeamThreadRange( 
                    team_member 
                , EXPR(nx,*ny,*nz) ) ; 
        int const q = team_member.league_rank() ; 
        Kokkos::parallel_reduce(  
                reduce_range 
            , KOKKOS_LAMBDA (int& icell, double& leps )
            {
                int const i = icell%nx ;
                int const j = icell/nx%ny; 
                #ifdef GRACE_3D 
                int const k = icell/nx/ny ; 
                #endif  
                auto eps_new = kernel(VEC(i+ngz,j+ngz,k+ngz), q, kernel_args...) ; 
                if( eps_new > leps ) {
                    leps = eps_new ;
                }
            } 
            , Kokkos::Max<double>(eps)  
        ) ; 
        team_member.team_barrier() ; 
        if( team_member.team_rank() == 0 ) 
        {
            flag_view(q) = REFINE_FLAG  * ( eps > CTORE )
                         + COARSEN_FLAG * ( eps < CTODE ) ; 
        } 
    }) ;
}
#if 0
/**
 * @brief Prolongate all state variables on refined cells.
 * \ingroup amr
 * 
 * @tparam InterpT  Type of prolongation operator.
 * @tparam InViewT  Type of view for refiend state.
 * @tparam OutViewT Type of view for old state.
 * @tparam CoordViewT Type of quadrant coordinate view.
 * @tparam VolViewT   Type of view containing cell volumes.
 * @param in_state    New state. 
 * @param out_state   Old state.
 * @param out_spacing  Old coordinate spacings. 
 * @param in_spacing   New coordinate spacings.
 * @param out_coords   Old quadrant coordinates.
 * @param in_coords    New quadrant coordinates.
 * @param out_vol      Old cell volumes.
 * @param in_vol       New cell volumes.
 * @param in_idx       New inverse spacings.
 * @param out_idx      Old inverse spacings.
 */
template< typename InterpT
        , typename InViewT
        , typename OutViewT
        , typename CoordViewT 
        , typename VolViewT >
void prolongate_variables(    InViewT  in_state 
                            , OutViewT out_state
                            , CoordViewT out_spacing  
                            , CoordViewT in_spacing 
                            , CoordViewT out_coords 
                            , CoordViewT in_coords  
                            , VolViewT   out_vol 
                            , VolViewT   in_vol  
                            , Kokkos::vector<int,grace::default_space>& in_idx 
                            , Kokkos::vector<int,grace::default_space>& out_idx ) 
{  
    using namespace grace ; 
    using namespace Kokkos  ; 
    int nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts()                   ; 
    long in_n_quad  = in_idx.size()                  ; 
    long out_n_quad = out_idx.size()                 ;
    int nvar = in_state.extent(GRACE_NSPACEDIM)  ; 

    auto p_in_idx  = in_idx.d_view  ; 
    auto p_out_idx = out_idx.d_view ; 

    amr::prolongator_t<InterpT,decltype(out_state),decltype(out_spacing),decltype(out_vol)> 
    kernel { 
          VEC(nx,ny,nz)
        , ngz
        , out_state
        , out_spacing 
        , in_spacing 
        , out_vol
        , in_vol
        , out_coords
        , in_coords
    } ; 

   MDRangePolicy<IndexType<int>, Rank<GRACE_NSPACEDIM+2>,default_execution_space>
        policy( {VEC(0,0,0),0,0}, {VEC(nx,ny,nz),nvar,out_n_quad}) ; 
    parallel_for(GRACE_EXECUTION_TAG("AMR","prolongate_cell_centered_variables")
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
                                    kernel(VEC(i,j,k),q_out,q_in,ivar,ichild) ;
                            }
                        }
    ); 
}; 
/**
 * @brief Restrict all state variables to coarsened cells
 * \ingroup amr
 * @tparam InViewT  Type of view for new state.
 * @tparam OutViewT Type of view for old state.
 * @tparam VolViewT Type of view for cell volumes.
 * @param in_state  New state.
 * @param out_state Old state.
 * @param out_vol   Old cell volumes.
 * @param in_vol    New cell volumes. 
 * @param in_idx    New inverse spacings.
 * @param out_idx   Old inverse spacings.
 */
template< typename InViewT
        , typename OutViewT 
        , typename VolViewT >
void restrict_variables(      InViewT  in_state 
                            , OutViewT out_state
                            , VolViewT out_vol 
                            , VolViewT in_vol   
                            , Kokkos::vector<int, grace::default_space>& in_idx 
                            , Kokkos::vector<int, grace::default_space>& out_idx )  
{
    using namespace grace ; 
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents()        ;
    auto ngz = amr::get_n_ghosts()                          ;  
    size_t in_n_quad  = in_idx.size()                       ; 
    size_t out_n_quad = out_idx.size()                      ;
    int nvar = in_state.extent(GRACE_NSPACEDIM)             ; 

    auto p_in_idx  = in_idx.d_view  ; 
    auto p_out_idx = out_idx.d_view ; 

    amr::restrictor_t<decltype(out_state),decltype(out_vol)> 
        kernel{ VEC(nx,ny,nz), ngz, out_state, out_vol, in_vol } ; 

    Kokkos::TeamPolicy<default_execution_space> policy( in_n_quad, Kokkos::AUTO() ) ;
    using member_type = Kokkos::TeamPolicy<default_execution_space>::member_type ; 

    Kokkos::parallel_for( GRACE_EXECUTION_TAG("AMR","restrict_variables")
                        , policy 
                        , KOKKOS_LAMBDA (member_type team)
    {
        
        size_t q_in = p_in_idx[team.league_rank()] ; 
        auto team_range = 
        Kokkos::TeamThreadMDRange<Kokkos::Rank<GRACE_NSPACEDIM+1>, member_type>( 
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
                            kernel(VEC(i,j,k),q_out,q_in,ivar) ;
                    }
        ) ; 
    }) ; 
} ;

template< typename InViewT
        , typename OutViewT >
void restrict_corner_staggered_variables( InViewT  in_state 
                                        , OutViewT out_state  
                                        , Kokkos::vector<int, grace::default_space>& in_idx 
                                        , Kokkos::vector<int, grace::default_space>& out_idx )  
{
    using namespace grace ; 
    using namespace Kokkos; 

    int nx,ny,nz                                         ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents()     ;
    auto ngz = amr::get_n_ghosts()                       ;  
    int in_n_quad  = in_idx.size()                       ; 
    int out_n_quad = out_idx.size()                      ;

    int nvar = in_state.extent(GRACE_NSPACEDIM)          ;

    auto p_in_idx  = in_idx.d_view  ; 
    auto p_out_idx = out_idx.d_view ; 

    MDRangePolicy<IndexType<int>, Rank<GRACE_NSPACEDIM+2>, default_execution_space>
        policy( {VEC(0,0,0),0,0}, {VEC(nx,ny,nz),nvar,in_n_quad}) ; 
    parallel_for(GRACE_EXECUTION_TAG("AMR","regrid_restrict_corner_staggered"), policy, 
        KOKKOS_LAMBDA( VEC(int i, int j, int k), int ivar, int icoarse )
    {
        int64_t qid_coarse = p_in_idx(icoarse) ; 
        int8_t ichild = EXPRD( math::floor_int((2*j)/nx)
                           , + math::floor_int((2*k)/ny) * 2 ) ;
        int64_t qid_fine = p_out_idx(icoarse*P4EST_CHILDREN + ichild ) ; 
        in_state(VEC(i+ngz,j+ngz,k+ngz),ivar,qid_coarse) 
            = out_state(VEC((2*i)%nx+ngz, (2*j)%ny+ngz, (2*k)%nz+ngz), ivar, qid_fine) ; 
    } ) ;


}
#endif
}} /* namespace grace::amr */ 

#endif 