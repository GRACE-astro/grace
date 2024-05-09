/**
 * @file regrid.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
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

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

#include <thunder/amr/regrid.hh>
#include <thunder/amr/regridding_policy_kernels.tpp> 
#include <thunder/amr/prolongation_kernels.tpp> 
#include <thunder/amr/restriction_kernels.tpp> 
#include <thunder/amr/regrid_helpers.tpp>
#include <thunder/amr/amr_functions.hh>
#include <thunder/coordinates/coordinates.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/utils/prolongation.hh>
#include <thunder/utils/limiters.hh>

namespace thunder { namespace amr { 

void regrid() {
    Kokkos::Profiling::pushRegion("regrid") ; 
    using namespace thunder ; 
    auto& params = config_parser::get()             ; 
    auto&  state  = variable_list::get().getstate() ; 
    int nvars = state.extent(THUNDER_NSPACEDIM)   ; 
    size_t thunder_maxlevel = 
        params["amr"]["max_refinement_level"].as<size_t>() ; 
    size_t nx,ny,nz                                        ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents()       ; 
    auto ngz = amr::get_n_ghosts()                         ; 
    size_t nq = amr::get_local_num_quadrants()             ; 
    /* create host and device views to hold refinement / coarsening flags */ 
    Kokkos::View<int *, default_space> d_regrid_flags("regrid_flags", nq) ; 
    auto h_regrid_flags = Kokkos::create_mirror_view(d_regrid_flags)      ;
     
    std::string ref_criterion = 
        params["amr"]["refinement_criterion"].as<std::string>() ;
    auto varname = params["amr"]["refinement_criterion_var"].as<std::string>() ;
    auto varidx = get_variable_index(varname) ; 
    ASSERT(varidx>=0, "Index of variable " << varname << " not found.") ; 
    auto u = Kokkos::subview(state, VEC(  Kokkos::ALL() 
                                        , Kokkos::ALL() 
                                        , Kokkos::ALL() )
                                        , varidx
                                        , Kokkos::ALL() 
                                    ) ; 
    if( ref_criterion == "FLASH_second_deriv") {
        double eps = params["amr"]["FLASH_criterion_eps"].as<double>() ; 
        amr::flash_second_deriv_criterion<decltype(u)> kernel{ u } ; 
        evaluate_regrid_criterion(
                  d_regrid_flags
                , kernel 
                , eps) ;
    } else if ( ref_criterion == "simple_threshold" ) {
        amr::simple_threshold_criterion<decltype(u)> kernel{ u } ; 
        evaluate_regrid_criterion(
                  d_regrid_flags
                , kernel) ;
    } else {
        ERROR("Unsupported refinement criterion.") ; 
    }
    /* copy flags from device to host */ 
    Kokkos::deep_copy(h_regrid_flags, d_regrid_flags) ; 
    
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
                    , 0, thunder_maxlevel 
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
                                 , nvars
                                 , nq_new                          
    ) ; 
    /******************************************************************************************/
    /*                      Collect indices of outgoing and incoming                          */
    /*                      quadrants in their respective z-ordering.                         */
    /******************************************************************************************/
    Kokkos::vector<int,default_space> refine_incoming, coarsen_incoming ;
    Kokkos::vector<int,default_space> refine_outgoing, coarsen_outgoing ; 
    unsigned long iq_new{0UL}, iq_old{0UL} ;  
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
    cell_vol_array_t<THUNDER_NSPACEDIM> in_vol( 
        "temporary_cell_volumes", VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz), nq_new 
    ) ; 
    scalar_array_t<THUNDER_NSPACEDIM> in_dx(
        "temporary_cell_spacing", THUNDER_NSPACEDIM, nq_new 
    ) ; 
    scalar_array_t<THUNDER_NSPACEDIM> in_idx(
        "temporary_cell_inv_spacing", THUNDER_NSPACEDIM, nq_new 
    ) ;
    scalar_array_t<THUNDER_NSPACEDIM> in_coords(
        "temporary_quadrant_coordinates", THUNDER_NSPACEDIM, nq_new 
    ) ;
    staggered_coordinate_arrays_t in_staggered_coords(
        VEC(nx,ny,nz), ngz, nq_new 
    ) ; 
    fill_cell_coordinates(in_coords,in_idx,in_dx,in_vol,in_staggered_coords) ; 
    /******************************************************************************************/
    /*                      Prolongate data on refined quadrants                              */
    /******************************************************************************************/
    std::string interp = params["amr"]["prolongation_interpolator_type"].as<std::string>(); 
    std::string limiter = params["amr"]["prolongation_limiter_type"].as<std::string>();
    if( interp == "linear" ) 
    {
        if( limiter == "minmod"){
            prolongate_variables<utils::linear_prolongator_t<thunder::minmod>> ( 
                state_swap
                , state 
                , dx
                , in_dx 
                , coords
                , in_coords
                , vol
                , in_vol
                , refine_incoming
                , refine_outgoing 
            ) ;
        } else if (limiter == "monotonized-central") {
            prolongate_variables<utils::linear_prolongator_t<thunder::MCbeta>> ( 
                state_swap
                , state 
                , dx
                , in_dx 
                , coords
                , in_coords
                , vol
                , in_vol
                , refine_incoming
                , refine_outgoing 
            ) ;
        } else {
            ERROR("Requested limiter for prolongation is not implemented.") ;
        }
    } else {
        ERROR("Requested interpolator for prolongation is not implemented.") ; 
    }
    /******************************************************************************************/
    /*                      Restrict data on coarsened quadrants                              */
    /******************************************************************************************/
    std::string reduce = params["amr"]["restriction_interpolator_type"].as<std::string>(); 
    if (reduce=="linear")
    {
        restrict_variables(
              state_swap
            , state
            , vol
            , in_vol
            , coarsen_incoming
            , coarsen_outgoing
        ) ;
    } else {
        ERROR("Requested interpolator for restriction is not implemented.") ; 
    }
    Kokkos::fence(); 
    /******************************************************************************************/
    /*                      Partition the new forest in parallel                              */
    /*                      we store global quadrant offsets, then                            */
    /*                      partition the forest, transfer state data                         */
    /*                      asynchronously, and realloc other fields                          */
    /*                      in the meanwhile. Coordinates are recomputed                      */
    /*                      but the auxiliary fields are left empty.                          */
    /******************************************************************************************/
    auto const glob_qoffsets = amr::get_global_quadrant_offsets() ;
    /******************************************************************************************/
    /*                                    Partition forest                                    */
    /******************************************************************************************/
    size_t transfer_count = p4est_partition_ext( forest::get().get()
                                               , 0
                                               , nullptr  ) ; 
    auto const new_glob_qoffsets = amr::get_global_quadrant_offsets() ; 
    size_t const quadrant_data_size = EXPR(   (nx+2*ngz)
                                          , * (ny+2*ngz)
                                          , * (nz+2*ngz)  ) * nvars * sizeof(double); 
    size_t const nq_local = amr::get_local_num_quadrants() ; 
    /******************************************************************************************/
    /*                              Realloc data and partition forest                         */
    /******************************************************************************************/ 
    Kokkos::realloc( state      ,   VEC(  nx + 2*ngz 
                                        , ny + 2*ngz 
                                        , nz + 2*ngz )
                                ,   nvars
                                ,   nq_local 
                                 ) ;
    /******************************************************************************************/
    /*                                Transfer data                                           */
    /******************************************************************************************/
    auto context = 
        p4est_transfer_fixed_begin (
                new_glob_qoffsets.data() 
                , glob_qoffsets.data()
                , parallel::get_comm_world() 
                , parallel::THUNDER_PARTITION_TAG 
                , reinterpret_cast<void*>(state.data())
                , reinterpret_cast<void*>(state_swap.data())
                , quadrant_data_size 
        ) ; 

    auto& idx = variable_list::get().getinvspacings()  ; 
    
    Kokkos::resize( coords      ,   THUNDER_NSPACEDIM
                                ,   nq_local 
                                 ) ;
    Kokkos::realloc( idx        , THUNDER_NSPACEDIM
                                ,   nq_local 
                                 ) ;
    Kokkos::realloc(  dx        , THUNDER_NSPACEDIM
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
    auto & aux = variable_list::get().getaux() ; 
    int nvars_aux = aux.extent(THUNDER_NSPACEDIM) ; 
    Kokkos::resize( aux         ,   VEC(  nx + 2*ngz 
                                        , ny + 2*ngz 
                                        , nz + 2*ngz )
                                ,   nvars_aux
                                ,   nq_local 
                                ) ;    
    /******************************************************************************************/
    /*                                Synchronize everything                                  */
    /******************************************************************************************/
    p4est_transfer_fixed_end(context) ;  
    /******************************************************************************************/
    /*                                Copy state to scratch                                   */
    /******************************************************************************************/
    Kokkos::realloc( state_swap ,   VEC(  nx + 2*ngz 
                                        , ny + 2*ngz 
                                        , nz + 2*ngz )
                                ,   nvars
                                ,   nq_local 
                                ) ;
    Kokkos::deep_copy(state_swap, state) ; 
    /******************************************************************************************/
    /*                         Reset quadrants to default state                               */
    /******************************************************************************************/
    set_quadrants_to_default(); 
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
}} /* namespace thunder::amr */ 