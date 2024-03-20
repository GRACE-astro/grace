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

#include <thunder/amr/regrid.hh>
#include <thunder/amr/regriding_policy_kernels.tpp> 
#include <thunder/amr/prolongation_kernels.tpp> 
#include <thunder/amr/restriction_kernels.tpp> 
#include <thunder/amr/regrid_helpers.tpp>

#include <thunder/data_structures/thunder_data_structures.hh>

namespace thunder { namespace amr { 

void regrid() {

    using namespace thunder ; 

    auto& params = config_parser::get() ; 
    auto  state  = variable_list::get().getstate() ; 
    size_t thunder_maxlevel = params["amr"]["max_refinement_level"].as<size_t>() ; 
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    size_t nq = amr::get_local_num_quadrants() ; 
    /* create host and device views to hold refinement / coarsening flags */ 
    Kokkos::View<int *, DefaultSpace> d_regrid_flags("regrid_flags", nq) ; 
    auto h_regrid_flags = Kokkos::create_mirror_view(d_regrid_flags)             ;

    std::string ref_criterion = params["amr"]["refinement_criterion"].as<std::string>() ; 

    if( ref_criterion == "FLASH_second_deriv") {
        double eps = params["amr"]["FLASH_criterion_eps"].as<double>() ; 
        evaluate_regrid_criterion(
                  d_regrid_flags
                , amr::flash_second_deriv_criterion<subview_t>
                , eps) ;
    } else {
        ERROR("Unsupported refinement criterion.") ; 
    }
    /* copy flags from device to host */ 
    Kokkos::deep_copy(h_regrid_flags, d_regrid_flags) ; 
    
    for( size_t iq=0UL; iq<amr::get_local_num_quadrants(); ++iq)
    {
        auto quad = amr::get_quadrant(iq) ;
        quad.set_user_data(
            amr::amr_flags_t{
                (h_regrid_flags(q) == REGRID)  * REGRID 
            +   (h_regrid_flags(q) == COARSEN) * COARSEN 
            +   ((h_regrid_flags(q) != REGRID) and (h_regrid_flags(q) != COARSEN)) * DEFAULT_STATE 
            }
        ) ; 
    }  
    /*
    * Call to p4est_refine 
    * The arguments are: 
    * p4est_t* p4est   --> The forest object pointer.
    * refine_recursive --> Wether we allow for recursive refinement (never).
    * maxlevel         --> Maximum allowed refinement level (parameter).
    * refine_fn        --> Function called on each quadrant to determine 
    *                      whether it should be refined (see amr_flags.hh).
    * init_fn          --> Function to initialize new quadrants.  
    * replace_fn       --> Function to modify the new quadrants. 
    */ 
    p4est_refine_ext( amr::forest::get().get() 
                    , 0, thunder_maxlevel 
                    , amr::refine_cback
                    , amr::initialize_quadrant 
                    , amr::set_quadrant_flag ) ; 

    /*
    * Call to p4est_coarsen 
    * The arguments are: 
    * p4est_t* p4est    --> The forest object pointer.
    * coarsen_recursive --> Wether we allow for recursive coarsening (never).
    * callback_orphans  --> Allow passing orphan nodes into coarsen_fn.
    * coarsen_fn        --> Function called on each quadrant family to determine 
    *                       whether it should be coarsened (see amr_flags.hh).
    * init_fn           --> Function to initialize new quadrants.  
    * replace_fn        --> Function to modify the new quadrants. 
    */ 
    p4est_coarsen_ext( amr::forest::get().get() 
                      , 0, 1 
                      , amr::coarsen_cback
                      , amr::initialize_quadrant 
                      , amr::set_quadrant_flag ) ; 

    /* Initialize new data */
    /* State: we use state_p as swap space */ 
    size_t nq_new = amr::local_num_quadrants() ;
    Kokkos::realloc(state,    VEC( nx + 2*ngz 
                                 , nx + 2*ngz 
                                 , nz + 2*ngz )
                            , nq_new 
                            , variables::nvars_state() 
    ) ; 

    auto state_swap = variable_list::get().getscratch() ;

    Kokkos::vector<int, Device> refine_incoming, coarsen_incoming ;
    Kokkos::vector<int, Device> refine_outgoing, coarsen_outgoing ; 
    size_t iq_new{0UL}, iq_old{0UL} ;  
    while(iq_new < nq_new)
    {       
        quadrant_t quadrant = amr::get_quadrant(iq_new) ; 
        auto flag = quadrant.get_user_data<amr_flags_t>()->quadrant_status; 
        if ( flag == DEFAULT_STATE )
        {
            iq_new++; iq_old++ ; 
        } else if ( flag == NEED_PROLONGATION )
        {
            refine_outgoing.push_back(iq_old) ; 
            iq_old++ ; 
            for( int ichild=0; ichild<P4EST_CHILDREN; ++ichild){
                refine_incoming.push_back(iq_new) ; 
                iq_new++ ;
            }
        } else if ( flag == NEED_RESTRICTION )
        {
            coarsen_incoming.push_back(iq_new) ; 
            iq_new++ ; 
            for( int ichild=0; ichild<P4EST_CHILDREN; ++ichild){
                coarsen_outgoing.push_back(old) ; 
                iq_old++ ;
            }
        }
    }
    auto idx = variable_list::get().getinvspacings() ; 

    std::string interp = params["amr"]["prolongation_interpolator_type"].as<std::string>(); 

    if( interp == "linear" ) 
    {
        prolongate_variables<utils::linear_interp_t<THUNDER_NSPACEDIM>>
            ( 
              state
            , state_swap 
            , idx 
            , refine_outgoing 
            , refine_incoming 
            ) ;
    } else {
        ERROR("Requested interpolator for prolongation is not implemented.") ; 
    }



}; 

void refine() 
{
    
}; 

void coarsen() ; 

void partition() ; 

}} /* namespace thunder::amr */ 