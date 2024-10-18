/**
 * @file grid_transfer_operators.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Prolongation and restriction for regridding.
 * @date 2024-09-24
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#include <grace/amr/regrid_helpers.hh>
#include <grace/amr/amr_functions.hh> 
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/config/config_parser.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/device/device_vector.hh>
#include <grace/utils/numerics/limiters.hh>
#include <grace/utils/numerics/prolongation.hh>
#include <grace/utils/numerics/restriction.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/system/grace_system.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <string> 

namespace grace { namespace amr {
 
template< typename LimT > 
void prolongate_variables_cell_centered(
    grace::var_array_t<GRACE_NSPACEDIM>& in_state,
    grace::var_array_t<GRACE_NSPACEDIM>& out_state,
    grace::cell_vol_array_t<GRACE_NSPACEDIM> in_vol,
    grace::device_vector<int>& in_idx,
    grace::device_vector<int>& out_idx
)
{
    using namespace grace ; 
    using namespace Kokkos  ;
    using InterpT = utils::linear_prolongator_t<LimT> ; 

    int nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts()                   ; 


    in_idx.host_to_device()  ; 
    out_idx.host_to_device() ; 
    long in_n_quad  = in_idx.size()                  ; 
    long out_n_quad = out_idx.size()                 ;


    int nvar = in_state.extent(GRACE_NSPACEDIM)  ; 

    auto p_in_idx  = in_idx.d_view  ; 
    auto p_out_idx = out_idx.d_view ; 

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
                /* 
                * First we need to find the index 
                * in the parent quadrant closest 
                * to the requested index in the child
                * quadrant. 
                */ 
                EXPR( 
                int const iquad_x = (ichild>>0) & 1 ;, 
                int const iquad_y = (ichild>>1) & 1;,
                int const iquad_z = (ichild>>2) & 1;
                )
                EXPR(
                int const i0 = 
                    math::floor_int((iquad_x * nx + i ) / 2) ;,

                int const j0 = 
                    math::floor_int((iquad_y * ny + j ) / 2) ;,

                int const k0 = 
                    math::floor_int((iquad_z * nz + k ) / 2) ; 
                )
                EXPR(
                int const sign_x = (i%2==1) - (i%2==0) ;, 
                int const sign_y = (j%2==1) - (j%2==0) ;, 
                int const sign_z = (k%2==1) - (k%2==0) ; 
                )
                #ifndef GRACE_CARTESIAN_COORDINATES
                in_state(VEC(i+ngz,j+ngz,k+ngz),ivar,q_in) = 
                    InterpT::interpolate(
                        VEC(i+ngz,j+ngz,k+ngz),
                        VEC(i0+ngz,j0+ngz,k0+ngz),
                        q_in,q_out,ngz,ivar,
                        VEC(sign_x,sign_y,sign_z),
                        out_state,
                        in_vol
                    ) ;
                #else
                in_state(VEC(i+ngz,j+ngz,k+ngz),ivar,q_in) = 
                    InterpT::interpolate(
                        VEC(i+ngz,j+ngz,k+ngz),
                        VEC(i0+ngz,j0+ngz,k0+ngz),
                        q_in,q_out,ngz,ivar,
                        VEC(sign_x,sign_y,sign_z),
                        out_state                    
                    ) ;
                #endif 
            }
        }
    ) ; 
}


template< int order > 
void prolongate_variables_corner_staggered(
    grace::var_array_t<GRACE_NSPACEDIM>& in_state,
    grace::var_array_t<GRACE_NSPACEDIM>& out_state,
    grace::device_vector<int> & in_idx,
    grace::device_vector<int> & out_idx
)
{
    using namespace grace ; 
    using namespace Kokkos  ;
    using interp_t = utils::lagrange_prolongator_t<order> ; 

    int nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts()                   ; 


    in_idx.host_to_device()  ; 
    out_idx.host_to_device() ; 
    long in_n_quad  = in_idx.size()                  ; 
    long out_n_quad = out_idx.size()                 ;


    int nvar = in_state.extent(GRACE_NSPACEDIM)  ; 

    auto p_in_idx  = in_idx.d_view  ; 
    auto p_out_idx = out_idx.d_view ; 
    /*******************************************/
    /* The idea here is that we loop over the  */
    /* coarse cells and let the Lagrange       */
    /* interpolator fill all the corresponding */
    /* fine quadrants.                         */
    /*******************************************/
    MDRangePolicy<IndexType<int>, Rank<GRACE_NSPACEDIM+2>,default_execution_space>
        policy( {VEC(0,0,0),0,0}, {VEC(nx,ny,nz),nvar,out_n_quad}) ; 
    parallel_for(GRACE_EXECUTION_TAG("AMR","prolongate_corner_staggered_variables")
        , policy 
        , KOKKOS_LAMBDA (
            VEC(const unsigned int& i
                ,const unsigned int& j
                ,const unsigned int& k)
            ,const unsigned int& ivar
            ,const unsigned int& iq_parent)
        {
            // Fine quadrant child index 
            int const ichild = 
                EXPR( math::floor_int((2*i)/nx), 
                    + math::floor_int((2*j)/ny) * 2, 
                    + math::floor_int((2*k)/nz) * 2 * 2 ) ; 
            // Fine quadrant index 
            int const q_in = 
                p_in_idx( P4EST_CHILDREN * iq_parent + ichild ) ; 
            // Coarse quadrant index
            int const q_out  = p_out_idx(iq_parent) ; 
            // Fine cell indices 
            EXPR(
            const int i0 = (2*i) % nx + ngz;,
            const int j0 = (2*j) % ny + ngz;,
            const int k0 = (2*k) % nz + ngz;
            )
            // Fine subview (quad and var specialized)
            auto in_view = subview(in_state, VEC(ALL(),ALL(),ALL()), ivar, q_in ) ; 
            // Coarse subview (quad and var specialized)
            auto out_view = subview(out_state, VEC(ALL(),ALL(),ALL()), ivar, q_out ) ; 
            interp_t::interpolate(
                VEC( i0,j0,k0 ),
                VEC( i + ngz, j + ngz, k + ngz ),
                out_view,
                in_view
            ) ;
        }
    ) ; 
}


void grace_prolongate_refined_quadrants(
    grace::var_array_t<GRACE_NSPACEDIM>& state,
    grace::var_array_t<GRACE_NSPACEDIM>& state_swap,
    grace::staggered_variable_arrays_t & sstate,
    grace::staggered_variable_arrays_t & sstate_swap,
    grace::cell_vol_array_t<GRACE_NSPACEDIM> in_vol,
    grace::device_vector<int> & refine_incoming,
    grace::device_vector<int> & refine_outgoing
) 
{
    using namespace grace ; 
    using namespace Kokkos ; 

    auto const limiter = get_param<std::string>("amr", "prolongation_limiter_type") ; 

    GRACE_VERBOSE("Initiating prolongation on refined quadrants.") ; 
    if ( limiter == "minmod" ) {
        prolongate_variables_cell_centered<grace::minmod> ( 
            state_swap,
            state,
            in_vol,
            refine_incoming,
            refine_outgoing
        ) ;
    } else if ( limiter == "monotonized-central") {
        prolongate_variables_cell_centered<grace::MCbeta> ( 
            state_swap,
            state,
            in_vol,
            refine_incoming,
            refine_outgoing
        ) ; 
    } else {
        ERROR("Requested limiter for prolongation is not implemented.") ;
    }

    auto const order = get_param<int>("amr", "prolongation_order") ; 
    if ( order == 2 ) {
        prolongate_variables_corner_staggered<2>(
            sstate_swap.corner_staggered_fields,
            sstate.corner_staggered_fields,
            refine_incoming,
            refine_outgoing
        ) ; 
    } else if ( order == 4 ) {
        prolongate_variables_corner_staggered<4>(
            sstate_swap.corner_staggered_fields,
            sstate.corner_staggered_fields,
            refine_incoming,
            refine_outgoing
        ) ;
    }
    

}

void restrict_variables_cell_centered(
    grace::var_array_t<GRACE_NSPACEDIM>& in_state,
    grace::var_array_t<GRACE_NSPACEDIM>& out_state,
    grace::cell_vol_array_t<GRACE_NSPACEDIM> out_vol,
    grace::device_vector<int> & in_idx,
    grace::device_vector<int> & out_idx
)
{
    using namespace grace  ;
    using namespace Kokkos ;
    
    int nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts()                   ; 


    in_idx.host_to_device()  ; 
    out_idx.host_to_device() ; 
    long in_n_quad  = in_idx.size()                  ; 
    long out_n_quad = out_idx.size()                 ;


    int nvar = in_state.extent(GRACE_NSPACEDIM)  ; 

    auto p_in_idx  = in_idx.d_view  ; 
    auto p_out_idx = out_idx.d_view ; 

    MDRangePolicy<IndexType<int>, Rank<GRACE_NSPACEDIM+2>,default_execution_space>
        policy( {VEC(0,0,0),0,0}, {VEC(nx,ny,nz),nvar,in_n_quad}) ; 
    parallel_for(GRACE_EXECUTION_TAG("AMR","prolongate_cell_centered_variables")
        , policy 
        , KOKKOS_LAMBDA (
            VEC(const unsigned int& i
                ,const unsigned int& j
                ,const unsigned int& k)
            ,const unsigned int& ivar
            ,const unsigned int& iq_parent)
        {
            int const ichild = 
                EXPR( math::floor_int((2*i)/nx), 
                    + math::floor_int((2*j)/ny) * 2, 
                    + math::floor_int((2*k)/nz) * 2 * 2 ) ; 
            int const q_out = 
                p_out_idx( P4EST_CHILDREN * iq_parent + ichild  ) ; 
            int const q_in  = p_in_idx(iq_parent) ; 

            EXPR(
            const int i0 = (2*i) % nx + ngz;,
            const int j0 = (2*j) % ny + ngz;,
            const int k0 = (2*k) % nz + ngz;
            )
            #ifndef GRACE_CARTESIAN_COORDINATES
            in_state(VEC(i+ngz,j+ngz,k+ngz),ivar,q_in) 
                = utils::vol_average_restrictor_t::apply(VEC(i0,j0,k0),out_state,out_vol,q_out,ivar) ; 
            #else
            in_state(VEC(i+ngz,j+ngz,k+ngz),ivar,q_in) 
                = utils::vol_average_restrictor_t::apply(VEC(i0,j0,k0),out_state,q_out,ivar) ; 
            #endif 
        }
    ); 
}

void restrict_variables_corner_staggered(
    grace::var_array_t<GRACE_NSPACEDIM>& in_state,
    grace::var_array_t<GRACE_NSPACEDIM>& out_state,
    grace::device_vector<int> & in_idx,
    grace::device_vector<int> & out_idx
)
{
    using namespace grace ; 
    using namespace Kokkos;

    using namespace grace  ;
    using namespace Kokkos ;
    
    int nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts()                   ; 


    in_idx.host_to_device()  ; 
    out_idx.host_to_device() ; 
    long in_n_quad  = in_idx.size()                  ; 
    long out_n_quad = out_idx.size()                 ;


    int nvar = in_state.extent(GRACE_NSPACEDIM)  ; 

    auto p_in_idx  = in_idx.d_view  ; 
    auto p_out_idx = out_idx.d_view ; 

    /***************************************************/
    /*  Here we:                                       */
    /*  Loop over the incoming (coarse) quadrants      */
    /*  and cells. Find which child we are in and fill */
    /*  the 8 vertices of the coarse cell with the     */
    /*  corresponding fine data.                       */
    /***************************************************/
    MDRangePolicy<IndexType<int>, Rank<GRACE_NSPACEDIM+2>,default_execution_space>
        policy( {VEC(0,0,0),0,0}, {VEC(nx,ny,nz),nvar,in_n_quad}) ; 
    parallel_for(GRACE_EXECUTION_TAG("AMR","prolongate_corner_staggered_variables")
        , policy 
        , KOKKOS_LAMBDA (
            VEC(const unsigned int& i
                ,const unsigned int& j
                ,const unsigned int& k)
            ,const unsigned int& ivar
            ,const unsigned int& iq_parent)
        {
            // Fine quadrant child index 
            int const ichild = 
                EXPR( math::floor_int((2*i)/nx), 
                    + math::floor_int((2*j)/ny) * 2, 
                    + math::floor_int((2*k)/nz) * 2 * 2 ) ; 
            // Fine quadrant index 
            int const q_out = 
                p_out_idx( P4EST_CHILDREN * iq_parent + ichild ) ; 
            // Coarse quadrant index
            int const q_in  = p_in_idx(iq_parent) ; 
            // Fine cell indices 
            EXPR(
            const int i0 = (2*i) % nx + ngz;,
            const int j0 = (2*j) % ny + ngz;,
            const int k0 = (2*k) % nz + ngz;
            )
            // Copy data fine to coarse 
            in_state(VEC(i+ngz,j+ngz,k+ngz),ivar,q_in) 
                = out_state(VEC(i0,j0,k0),ivar,q_out) ; 
            in_state(VEC(i+1+ngz,j+ngz,k+ngz),ivar,q_in) 
                = out_state(VEC(i0+2,j0,k0),ivar,q_out) ;
            in_state(VEC(i+ngz,j+1+ngz,k+ngz),ivar,q_in) 
                = out_state(VEC(i0,j0+2,k0),ivar,q_out) ;
            in_state(VEC(i+ngz,j+ngz,k+ngz+1),ivar,q_in) 
                = out_state(VEC(i0,j0,k0+2),ivar,q_out) ;

            in_state(VEC(i+1+ngz,j+1+ngz,k+ngz),ivar,q_in) 
                = out_state(VEC(i0+2,j0+2,k0),ivar,q_out) ; 
            in_state(VEC(i+1+ngz,j+ngz,k+1+ngz),ivar,q_in) 
                = out_state(VEC(i0+2,j0,k0+2),ivar,q_out) ;
            in_state(VEC(i+ngz,j+1+ngz,k+1+ngz),ivar,q_in) 
                = out_state(VEC(i0,j0+2,k0+2),ivar,q_out) ;
            
            in_state(VEC(i+1+ngz,j+1+ngz,k+ngz+1),ivar,q_in) 
                = out_state(VEC(i0+2,j0+2,k0+2),ivar,q_out) ;
        }
    ); 
}

void grace_restrict_coarsened_quadrants(
    grace::var_array_t<GRACE_NSPACEDIM>& state,
    grace::var_array_t<GRACE_NSPACEDIM>& state_swap,
    grace::staggered_variable_arrays_t & sstate,
    grace::staggered_variable_arrays_t & sstate_swap,
    grace::cell_vol_array_t<GRACE_NSPACEDIM> out_vol,
    grace::device_vector<int> & coarsen_incoming,
    grace::device_vector<int> & coarsen_outgoing
) 
{
    using namespace grace ; 

    GRACE_VERBOSE("Initiating restriction on coarsened quadrants.") ; 
    
    restrict_variables_cell_centered(
        state_swap,
        state,
        out_vol,  
        coarsen_incoming,
        coarsen_outgoing
    ) ; 

    restrict_variables_corner_staggered(
        sstate_swap.corner_staggered_fields,
        sstate.corner_staggered_fields,
        coarsen_incoming,
        coarsen_outgoing
    ) ;

}

/***********************************************************/
/*                  Intantiate templates                   */
/***********************************************************/
#define INSTANTIATE_TEMPLATE1(limiter)              \
template                                            \
void prolongate_variables_cell_centered<limiter>(   \
    grace::var_array_t<GRACE_NSPACEDIM>& ,          \
    grace::var_array_t<GRACE_NSPACEDIM>& ,          \
    grace::cell_vol_array_t<GRACE_NSPACEDIM> ,      \
    grace::device_vector<int> & ,                   \
    grace::device_vector<int> &                     \
)                                             
#define INSTANTIATE_TEMPLATE2(order)                \
template                                            \
void prolongate_variables_corner_staggered<order>(  \
    grace::var_array_t<GRACE_NSPACEDIM>& ,          \
    grace::var_array_t<GRACE_NSPACEDIM>& ,          \
    grace::device_vector<int> & ,                   \
    grace::device_vector<int> &                     \
) 

INSTANTIATE_TEMPLATE1(grace::minmod) ; 
INSTANTIATE_TEMPLATE2(2) ; 
INSTANTIATE_TEMPLATE1(grace::MCbeta) ; 
INSTANTIATE_TEMPLATE2(4) ; 


}} /* namespace grace::amr */