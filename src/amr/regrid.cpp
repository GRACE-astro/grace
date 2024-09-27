/**
 * @file regrid.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
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

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

#include <grace/amr/regrid.hh>
#include <grace/amr/regridding_policy_kernels.tpp> 
#include <grace/amr/prolongation_kernels.tpp> 
#include <grace/amr/restriction_kernels.tpp> 
#include <grace/amr/regrid_helpers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/config/config_parser.hh>
#include <grace/utils/prolongation.hh>
#include <grace/utils/limiters.hh>
#include <grace/utils/device_vector.hh>

namespace grace { namespace amr { 

void regrid() {
    auto const criterion = grace::get_param<std::string>("amr", "refinement_criterion") ; 
    auto const criterion_var = grace::get_param<std::string>("amr", "refinement_criterion_var") ; 
    regrid(criterion,criterion_var) ; 
}


void regrid( std::string const& regrid_criterion
           , std::string const& regrid_criterion_var ) 
{

    Kokkos::Profiling::pushRegion("regrid") ; 
    using namespace grace  ; 
    using namespace Kokkos ;
    /***************************************************/
    /*                 Get var arrays                  */ 
    /***************************************************/
    auto& state  = variable_list::get().getstate()          ; 
    auto& sstate = variable_list::get().getstaggeredstate() ; 
    auto& aux = variable_list::get().getaux()               ;

    int nvars_cell_centered      = state.extent(GRACE_NSPACEDIM)               ; 
    int nvars_face_staggered     = variables::get_n_evolved_face_staggered()   ; 
    int nvars_edge_staggered     = variables::get_n_evolved_edge_staggered()   ; 
    int nvars_corner_staggered   = variables::get_n_evolved_corner_staggered() ; 

    /***************************************************/
    /*                Get grid properties              */
    /***************************************************/
    size_t grace_maxlevel = 
        grace::get_param<size_t>("amr", "max_refinement_level") ; 
    size_t nx,ny,nz                                        ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents()       ; 
    auto ngz = amr::get_n_ghosts()                         ; 
    size_t nq = amr::get_local_num_quadrants()             ;
    GRACE_VERBOSE("Starting regrid with {} local quadrants", nq) ; 
    /* create host and device views to hold refinement / coarsening flags */ 
    Kokkos::View<int *, default_space> d_regrid_flags("regrid_flags", nq) ; 
    evaluate_regrid_criterion(regrid_criterion, regrid_criterion_var, d_regrid_flags) ; 
    /* copy flags from device to host   */ 
    auto h_regrid_flags = Kokkos::create_mirror_view(d_regrid_flags)      ;
    Kokkos::deep_copy(h_regrid_flags, d_regrid_flags) ; 
    /* Set data where p4est can read it */
    for( size_t iq=0UL; iq<amr::get_local_num_quadrants(); ++iq)
    {
        auto quad = amr::get_quadrant(iq) ;
        quad.set_user_data(
            amr::amr_flags_t{ static_cast<quadrant_flags_t>(
                (h_regrid_flags(iq) == REFINE)  * REFINE 
            +   (h_regrid_flags(iq) == COARSEN) * COARSEN 
            +   ((h_regrid_flags(iq) != REFINE) and (h_regrid_flags(iq) != COARSEN)) * DEFAULT_STATE )
            }
        ) ; 
    }  
    /******************************************************************************************/
    /* Call to p4est_refine                                                                   */  
    /* The arguments are:                                                                     */
    /* p4est_t* p4est   --> The forest object pointer.                                        */
    /* refine_recursive --> Wether we allow for recursive refinement (never).                 */
    /* maxlevel         --> Maximum allowed refinement level (parameter).                     */
    /* refine_fn        --> Function called on each quadrant to determine                     */
    /*                      whether it should be refined (see amr_flags.hh).                  */
    /* init_fn          --> Function to initialize new quadrants.                             */
    /* replace_fn       --> Function to modify the new quadrants.                             */
    /******************************************************************************************/ 
    p4est_refine_ext( amr::forest::get().get() 
                    , 0, grace_maxlevel 
                    , amr::refine_cback
                    , amr::initialize_quadrant 
                    , amr::set_quadrant_flag ) ; 
    /******************************************************************************************/
    /* Call to p4est_coarsen                                                                  */
    /* The arguments are:                                                                     */
    /* p4est_t* p4est    --> The forest object pointer.                                       */
    /* coarsen_recursive --> Wether we allow for recursive coarsening (never).                */
    /* callback_orphans  --> Allow passing orphan nodes into coarsen_fn.                      */
    /* coarsen_fn        --> Function called on each quadrant family to determine             */
    /*                       whether it should be coarsened (see amr_flags.hh).               */
    /* init_fn           --> Function to initialize new quadrants.                            */
    /* replace_fn        --> Function to modify the new quadrants.                            */
    /******************************************************************************************/
    p4est_coarsen_ext( amr::forest::get().get() 
                      , 0, 0 
                      , amr::coarsen_cback
                      , amr::initialize_quadrant 
                      , amr::set_quadrant_flag ) ; 
    /******************************************************************************************/
    /* Call to p4est_balance                                                                  */
    /* This ensures the grid is 2:1 balanced.                                                 */
    /******************************************************************************************/
    p4est_balance_ext( amr::forest::get().get() 
                      , P4EST_CONNECT_FULL
                      , amr::initialize_quadrant 
                      , amr::set_quadrant_flag ) ;
    /******************************************************************************************/
    /*                       Resize variable arrays, we use state_p                           */
    /*                       as swap state, then copy data over                               */
    /******************************************************************************************/
    size_t nq_new = amr::get_local_num_quadrants() ; 
    auto& state_swap = variable_list::get().getscratch() ;
    Kokkos::realloc(state_swap   , VEC( nx + 2*ngz 
                                 ,      ny + 2*ngz 
                                 ,      nz + 2*ngz )
                                 , nvars_cell_centered
                                 , nq_new                          
    ) ; 
    auto& sstate_swap = variable_list::get().getstaggeredscratch() ; 
    sstate_swap.realloc(
          VEC( nx 
             , ny  
             , nz  )
             , ngz
             , nq_new
             , nvars_face_staggered
             , nvars_edge_staggered
             , nvars_corner_staggered
    ) ; 
    /******************************************************************************************/
    /*                      Collect indices of outgoing and incoming                          */
    /*                      quadrants in their respective z-ordering.                         */
    /******************************************************************************************/
    grace::device_vector<int> refine_incoming, coarsen_incoming ;
    grace::device_vector<int> refine_outgoing, coarsen_outgoing ; 
    unsigned long iq_new{0UL}, iq_old{0UL} ;  
    GRACE_VERBOSE("Initiating copy of data after regrid.") ; 
    while(iq_new < nq_new)
    {       
        quadrant_t quadrant = amr::get_quadrant(iq_new) ; 
        int flag = 
            static_cast<int>(quadrant.get_user_data<amr_flags_t>()->quadrant_status); 
        if ( (flag == DEFAULT_STATE) or (flag==REFINE) or (flag==COARSEN) )
        {
            /* Copy over data that does not need anything done */
            auto sview_state = Kokkos::subview( state
                                              , VEC( Kokkos::ALL()
                                                   , Kokkos::ALL()
                                                   , Kokkos::ALL())
                                              , Kokkos::ALL()
                                              , iq_old) ;
            auto sview_swap  = Kokkos::subview( state_swap
                                              , VEC( Kokkos::ALL()
                                                   , Kokkos::ALL()
                                                   , Kokkos::ALL())
                                              , Kokkos::ALL()
                                              , iq_new) ;
            
            auto ssview = sstate.subview(VEC(ALL(),ALL(),ALL()), ALL(), iq_old ) ; 
            auto ssview_swap = sstate_swap.subview(VEC(ALL(),ALL(),ALL()), ALL(), iq_new ) ; 
            ssview_swap.deep_copy_async(ssview) ; 
            Kokkos::deep_copy(default_execution_space{}, sview_swap, sview_state) ; 
            iq_new++; iq_old++ ; 
        } else if ( flag == NEED_PROLONGATION )
        {
            refine_outgoing.push_back(static_cast<int>(iq_old)) ; 
            iq_old++ ; 
            for( int ichild=0; ichild<P4EST_CHILDREN; ++ichild){
                refine_incoming.push_back(static_cast<int>(iq_new)) ; 
                iq_new++ ;
            }
        } else if ( flag == NEED_RESTRICTION )
        {
            coarsen_incoming.push_back(iq_new) ; 
            iq_new++ ; 
            for( int ichild=0; ichild<P4EST_CHILDREN; ++ichild){
                coarsen_outgoing.push_back(static_cast<int>(iq_old)) ; 
                iq_old++ ;
            }
        } else if (flag == INVALID_STATE) {
            ERROR("Invalid state " << flag << " for quadrant " << iq_new << '\n') ;
        } 
    }
    refine_incoming.host_to_device() ; refine_outgoing.host_to_device() ; 
    coarsen_incoming.host_to_device() ; coarsen_outgoing.host_to_device() ;
    GRACE_VERBOSE("Incoming (coarsen) {}, outgoing (coarsen) {}", coarsen_incoming.size(), coarsen_outgoing.size()) ; 
    ASSERT_DBG( iq_old == nq, 
              "Something went really wrong. "
              "nq= " << nq << " iq= " << iq_old <<".") ;

    auto& dx = variable_list::get().getspacings()    ;
    auto& coords = variable_list::get().getcoords()  ; 
    auto& vol = variable_list::get().getvolumes()    ;
    auto& staggered_coords = variable_list::get().getstaggeredcoords() ;
    /******************************************************************************************/
    /*                     Allocate temporary coordinate arrays                               */
    /******************************************************************************************/
    cell_vol_array_t<GRACE_NSPACEDIM> in_vol( 
        "temporary_cell_volumes", VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz), nq_new 
    ) ; 
    scalar_array_t<GRACE_NSPACEDIM> in_dx(
        "temporary_cell_spacing", GRACE_NSPACEDIM, nq_new 
    ) ; 
    scalar_array_t<GRACE_NSPACEDIM> in_idx(
        "temporary_cell_inv_spacing", GRACE_NSPACEDIM, nq_new 
    ) ;
    scalar_array_t<GRACE_NSPACEDIM> in_coords(
        "temporary_quadrant_coordinates", GRACE_NSPACEDIM, nq_new 
    ) ;
    staggered_coordinate_arrays_t in_staggered_coords(
        VEC(nx,ny,nz), ngz, nq_new 
    ) ; 
    fill_cell_coordinates(in_coords,in_idx,in_dx,in_vol,in_staggered_coords) ; 
    /******************************************************************************************/
    /*                      Prolongate data on refined quadrants                              */
    /******************************************************************************************/
    grace_prolongate_refined_quadrants(
        state, state_swap, sstate, sstate_swap, in_vol, refine_incoming, refine_outgoing
    ) ; 
    /******************************************************************************************/
    /*                      Restrict data on coarsened quadrants                              */
    /******************************************************************************************/
    grace_restrict_coarsened_quadrants(
        state, state_swap, sstate, sstate_swap, vol, coarsen_incoming, coarsen_outgoing
    ) ; 
    /******************************************************************************************/
    /*                      Partition the new forest in parallel                              */
    /******************************************************************************************/
    auto context = grace_partition_begin(
        state,
        state_swap,
        sstate,
        sstate_swap
    ) ; 
    /******************************************************************************************/
    /*                          Get new local forest size                                     */
    /******************************************************************************************/
    size_t const nq_local = amr::get_local_num_quadrants() ; 
    /******************************************************************************************/
    /*                          Recompute coordinates                                         */
    /******************************************************************************************/
    auto& idx = variable_list::get().getinvspacings()  ; 
    
    Kokkos::resize( coords      ,   GRACE_NSPACEDIM
                                ,   nq_local 
                                 ) ;
    Kokkos::realloc( idx        , GRACE_NSPACEDIM
                                ,   nq_local 
                                 ) ;
    Kokkos::realloc(  dx        , GRACE_NSPACEDIM
                                ,   nq_local 
                                 ) ;
    Kokkos::realloc( vol        , VEC(  nx + 2*ngz 
                                      , ny + 2*ngz 
                                      , nz + 2*ngz )
                                ,  nq_local 
                                 ) ;
    staggered_coords.realloc(VEC(nx,ny,nz),ngz,nq_local) ; 
    fill_cell_coordinates(coords, idx, dx, vol,staggered_coords) ;
    /******************************************************************************************/
    /*                            Auxiliary vars are reallocated                              */
    /*                            but not re-initialized.                                     */
    /******************************************************************************************/
    int nvars_aux = grace::variables::get_n_auxiliary() ; 
    GRACE_TRACE("Resizing aux array {} aux vars registered, new quad count {}", nvars_aux, nq_local) ; 
    Kokkos::realloc( aux        ,   VEC(  nx + 2*ngz 
                                        , ny + 2*ngz 
                                        , nz + 2*ngz )
                                ,   nvars_aux
                                ,   nq_local 
                                ) ;    
    // todo realloc staggered aux
    /******************************************************************************************/
    /*                                Synchronize everything                                  */
    /******************************************************************************************/
    grace_partition_finalize(context) ;  
    GRACE_VERBOSE("received.") ; 
    /******************************************************************************************/
    /*                                Copy state to scratch                                   */
    /******************************************************************************************/
    Kokkos::realloc( state_swap ,   VEC(  nx + 2*ngz 
                                        , ny + 2*ngz 
                                        , nz + 2*ngz )
                                ,   nvars_cell_centered
                                ,   nq_local 
                                ) ;
    Kokkos::deep_copy(state_swap, state) ; 
    sstate_swap.realloc(
        VEC(nx,ny,nz),
        ngz,nq_local,
        nvars_face_staggered,
        nvars_edge_staggered,
        nvars_corner_staggered 
    ) ; 
    sstate_swap.deep_copy(sstate) ; 
    /******************************************************************************************/
    /*                         Reset quadrants to default state                               */
    /******************************************************************************************/
    set_quadrants_to_default(); 
    GRACE_VERBOSE("Finished regrid with {} local quadrants", nq_local) ; 
    /******************************************************************************************/
    /*                                      All done                                          */
    /******************************************************************************************/
    Kokkos::Profiling::popRegion() ;
}; 


void set_quadrants_to_default()  
{
    for(int itree=forest::get().first_local_tree();
            itree<=forest::get().last_local_tree();
            ++itree) 
    {
        auto quadrants = forest::get().tree(itree).quadrants() ; 
        for( int iquad=0; iquad<quadrants.size(); ++iquad) {
            quadrant_t quad{ &(quadrants[iquad]) } ;
            quad.set_user_data( amr_flags_t{DEFAULT_STATE} ) ; 
        }
    }
}
}} /* namespace grace::amr */ 