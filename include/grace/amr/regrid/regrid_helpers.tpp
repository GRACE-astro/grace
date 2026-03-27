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
#include <grace/amr/regrid/regridding_policy_kernels.tpp>

#include <grace/utils/device_vector.hh>
#include <grace/utils/reductions.hh>

#include <grace/coordinates/coordinate_systems.hh>
#include <grace/IO/diagnostics/ns_tracker.hh>
#ifdef GRACE_ENABLE_Z4C_METRIC
#include <grace/IO/diagnostics/puncture_tracker.hh>
#endif 

#include <Kokkos_Core.hpp>

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
 */
template< typename ViewT 
    , typename KerT 
    , typename ... KerArgT> 
void evaluate_regrid_criterion( ViewT flag_view
                              , KerT kernel) 
{
    using namespace grace  ;  
    // Get arrays 
    auto  state  = variable_list::get().getstate() ; 
    // Get array sizes 
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    size_t nq = amr::get_local_num_quadrants() ; 
    size_t ngz = amr::get_n_ghosts() ; 
    // Store flags 
    size_t REFINE_FLAG  = amr::quadrant_flags_t::REFINE  ;  
    size_t COARSEN_FLAG = amr::quadrant_flags_t::COARSEN ; 
    // Store parameters 
    double CTORE = get_param<double>("amr","refinement_criterion_CTORE");  
    double CTODE = get_param<double>("amr","refinement_criterion_CTODE");
    auto reduction = get_param<std::string>("amr", "refinement_criterion_reduction") ; 
    // Policy 
    Kokkos::TeamPolicy<default_execution_space> policy(nq, Kokkos::AUTO() ) ; 
    using member_type = Kokkos::TeamPolicy<default_execution_space>::member_type ;

    if ( reduction == "max" ) {
        /* Each thread league deals with a  single quadrant */ 
     
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
                , [=] (int64_t const icell, double& leps )
                {
                    int const i = icell%nx ;
                    int const j = icell/nx%ny; 
                    #ifdef GRACE_3D 
                    int const k = icell/nx/ny ; 
                    #endif  
                    auto eps_new = kernel(VEC(i+ngz,j+ngz,k+ngz), q) ; 
                    if( eps_new > leps ) {
                        leps = eps_new ;
                    }
                } 
                , Kokkos::Max<double>(eps)  
            ) ; 
            team_member.team_barrier() ; 
            Kokkos::single( 
                Kokkos::PerTeam(team_member),
                [&] () {
                    flag_view(q) = REFINE_FLAG  * ( eps >= CTORE )  
                                 + COARSEN_FLAG * ( eps <= CTODE ) ; 
                }
            ) ; 
        }) ;
    } else if ( reduction == "min" ) {
        /* Each thread league deals with a  single quadrant */ 
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
                , [=] (int64_t const icell, double& leps )
                {
                    int const i = icell%nx ;
                    int const j = icell/nx%ny; 
                    #ifdef GRACE_3D 
                    int const k = icell/nx/ny ; 
                    #endif  
                    auto eps_new = kernel(VEC(i+ngz,j+ngz,k+ngz), q) ; 
                    if( eps_new < leps ) {
                        leps = eps_new ;
                    }
                } 
                , Kokkos::Min<double>(eps)  
            ) ; 
            team_member.team_barrier() ; 
            Kokkos::single( 
                Kokkos::PerTeam(team_member),
                [&] () {
                    flag_view(q) = REFINE_FLAG  * ( eps <= CTORE )  
                                 + COARSEN_FLAG * ( eps >= CTODE ) ; 
                }
            ) ; 
        }) ;
    } else {
        ERROR("Unrecognized reduction for refinement.") ; 
    }
    
}

#ifdef GRACE_ENABLE_Z4C_METRIC
enum co_t {
    NS, BH
} ; 

template< co_t co1, co_t co2, typename view_t >
void evaluate_binary_tracker_criterion(view_t & flag_view )
{
    DECLARE_GRID_EXTENTS; 

    using namespace Kokkos ; 
    using namespace grace  ; 
    auto& ns_tracker = grace::ns_tracker::get() ; 
    auto& bh_tracker = grace::puncture_tracker::get() ; 

    if ( co1 == co_t::NS or co2 == co_t::NS ) {
        ASSERT(ns_tracker.is_active(), "NS tracker specified as AMR criterion is inactive.") ; 
    }

    if ( co1 == co_t::BH or co2 == co_t::BH ) {
        ASSERT(bh_tracker.is_active(), "BH tracker specified as AMR criterion is inactive.") ; 
    }

    double co1_re_radius = get_param<double>("amr","binary_tracker_amr_criterion","compact_object_1_refine_radius") ; 
    double co1_de_radius = get_param<double>("amr","binary_tracker_amr_criterion","compact_object_1_coarsen_radius") ; 

    double co2_re_radius = get_param<double>("amr","binary_tracker_amr_criterion","compact_object_2_refine_radius") ; 
    double co2_de_radius = get_param<double>("amr","binary_tracker_amr_criterion","compact_object_2_coarsen_radius") ; 

    double min_separation = get_param<double>("amr", "binary_tracker_amr_criterion", "minimum_coordinate_separation") ; 
    double post_merger_re_radius = get_param<double>("amr", "binary_tracker_amr_criterion", "post_merger_refine_radius") ; 
    double post_merger_de_radius = get_param<double>("amr", "binary_tracker_amr_criterion", "post_merger_coarsen_radius") ; 

    Kokkos::View<double[6], grace::default_space> co_locations ; 
    auto co_locations_h = create_mirror_view(co_locations) ; 


    auto ns_locations = ns_tracker.get_ns_locations() ; 
    auto ns_locations_h = create_mirror_view_and_copy(Kokkos::HostSpace{}, ns_locations) ; 

    auto bh_locations = bh_tracker.get_puncture_locations() ; 


    if ( co1 == co_t::NS ) {
        co_locations_h(0) = ns_locations_h(0,0); co_locations_h(1) = ns_locations_h(0,1); co_locations_h(2) = ns_locations_h(0,2);
        if ( co2 == co_t::NS ) {
            ASSERT(ns_tracker.get_n_ns() > 1, "BNS tracker amr requested but only one NS is tracked.") ;  
            co_locations_h(3) = ns_locations_h(1,0); co_locations_h(4) = ns_locations_h(1,1); co_locations_h(5) = ns_locations_h(1,2);
        } else {
            co_locations_h(3) = bh_locations[0][0]; co_locations_h(4) = bh_locations[0][1]; co_locations_h(5) = bh_locations[0][2];
        }
    } else {
        co_locations_h(0) = bh_locations[0][0]; co_locations_h(1) = bh_locations[0][1]; co_locations_h(2) = bh_locations[0][2];
        if ( co2 == co_t::NS ) {
            co_locations_h(3) = ns_locations_h(0,0); co_locations_h(4) = ns_locations_h(0,1); co_locations_h(5) = ns_locations_h(0,2);
        } else {
            ASSERT(bh_tracker.get_n_punctures() > 1, "BBH tracker amr requested but only one BH is tracked.") ; 
            co_locations_h(3) = bh_locations[1][0]; co_locations_h(4) = bh_locations[1][1]; co_locations_h(5) = bh_locations[1][2];
        }
    }   

    double distance = 0.0 ; 
    for( int ii=0; ii<3; ++ii) distance+=SQR(co_locations_h(ii)-co_locations_h(3+ii));
    distance = sqrt(distance) ; 

    Kokkos::deep_copy(co_locations,co_locations_h) ; 

    // Policy 
    Kokkos::TeamPolicy<default_execution_space> policy(nq, Kokkos::AUTO() ) ; 
    using member_type = Kokkos::TeamPolicy<default_execution_space>::member_type ;

    // coordinates 
    auto dc = grace::coordinate_system::get().get_device_coord_system() ;
    
    // Store flags 
    size_t REFINE_FLAG  = amr::quadrant_flags_t::REFINE  ;  
    size_t COARSEN_FLAG = amr::quadrant_flags_t::COARSEN ; 

    if (distance < min_separation) {

        Kokkos::parallel_for( GRACE_EXECUTION_TAG("AMR","eval_refine_coarsen_criterion")
                            , policy 
                            , KOKKOS_LAMBDA (member_type team_member)
        {
            double rmin ; 
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
                , [=] (int64_t const icell, double& lrmin )
                {
                    int const i = icell%nx ;
                    int const j = icell/nx%ny; 
                    #ifdef GRACE_3D 
                    int const k = icell/nx/ny ; 
                    #else 
                    int const k = 0 
                    #endif   
                    double xyz[3];
                    dc.get_physical_coordinates(i+ngz,j+ngz,k+ngz,q,xyz) ; 
                    double r = Kokkos::sqrt(
                        SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
                    ) ; 
                    if( r < lrmin ) {
                        lrmin = r ;
                    }
                } 
                , Kokkos::Min<double>(rmin)  
            ) ; 
            team_member.team_barrier() ; 
            Kokkos::single( 
                Kokkos::PerTeam(team_member),
                [&] () {
                    flag_view(q) = ( rmin <= post_merger_re_radius ) ? REFINE_FLAG 
                                        : (( rmin >= post_merger_de_radius ) ? COARSEN_FLAG : 0) ;
                }
            ) ; 
        }) ;

    } else {
        Kokkos::parallel_for( GRACE_EXECUTION_TAG("AMR","eval_refine_coarsen_criterion")
                            , policy 
                            , KOKKOS_LAMBDA (member_type team_member)
        {
            block_dist_t dist ; 
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
                , [=] (int64_t const icell, block_dist_t& ldist )
                {
                    int const i = icell%nx ;
                    int const j = icell/nx%ny; 
                    #ifdef GRACE_3D 
                    int const k = icell/nx/ny ; 
                    #endif  
                    double xyz[3];
                    dc.get_physical_coordinates(i+ngz,j+ngz,k+ngz,q,xyz) ; 
                    double d1 = Kokkos::sqrt(
                        SQR(xyz[0]-co_locations[0]) + SQR(xyz[1]-co_locations[1]) + SQR(xyz[2]-co_locations[2])
                    ) ;
                    double d2 = Kokkos::sqrt(
                        SQR(xyz[0]-co_locations[3]) + SQR(xyz[1]-co_locations[4]) + SQR(xyz[2]-co_locations[5])
                    ) ; 
                    
                    if ( d1 < ldist.min_d0 ) {
                        ldist.min_d0 = d1 ; 
                    }

                    if ( d2 < ldist.min_d1 ) {
                        ldist.min_d1 = d2 ; 
                    }
                } 
                , Kokkos::Sum<block_dist_t>(dist)  
            ) ; 
            team_member.team_barrier() ; 
            Kokkos::single( 
                Kokkos::PerTeam(team_member),
                [&] () {
                    bool refine   = (dist.min_d0 < co1_re_radius) || (dist.min_d1 < co2_re_radius) ;
                    bool derefine = (dist.min_d0 > co1_de_radius) && (dist.min_d1 > co2_de_radius) ;
                    flag_view(q) = refine ? REFINE_FLAG : (derefine ? COARSEN_FLAG : 0) ;
                }
            ) ; 
        }) ;
    }
}
#endif 
}} /* namespace grace::amr */ 

#endif 