/**
 * @file grmhd.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-28
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
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/id/shocktube.hh>
//#include <grace/physics/id/blastwave.hh>
#include <grace/physics/id/tov.hh>
#include <grace/physics/id/fmtorus.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/evolution/hrsc_evolution_system.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/grmhd.hh>

#ifdef GRACE_DO_MHD
#ifdef GRACE_ENABLE_B_FIELD_GLM
#include <grace/physics/id/shocktube_mhd.hh>
#include <grace/physics/id/linear_mhd_waves.hh>
#include <grace/physics/id/orszag_tang_vortex.hh>
#include <grace/physics/id/boosted_loop_advection.hh>
#include <grace/physics/id/circularly_polarized_alfven_wave.hh>
#include <grace/physics/id/magnetic_rotor.hh>
#include <grace/physics/id/blast_wave.hh>
#include <grace/physics/id/bondi_accretion.hh>
#endif
#endif

#include <grace/config/config_parser.hh>
#include <Kokkos_Core.hpp>

#include <string>

namespace grace{

template< typename eos_t >
static void set_grmhd_shocktube_initial_data() {
    using namespace grace ;
    using namespace Kokkos ; 

    GRACE_VERBOSE("Setting Shocktube initial data.") ; 

    auto const rho_L = get_param<double>("grmhd","shocktube_rho_L") ; 
    auto const rho_R = get_param<double>("grmhd","shocktube_rho_R") ; 
    auto const press_L = get_param<double>("grmhd","shocktube_press_L") ; 
    auto const press_R = get_param<double>("grmhd","shocktube_press_R") ;

    auto& aux   = variable_list::get().getaux() ; 
    auto& state = variable_list::get().getstate() ;
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;

    auto& coord_system = grace::coordinate_system::get() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(aux) ; 


    int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
    
    #pragma omp parallel for 
    for( int64_t icell=0; icell<ncells; ++icell) {
        size_t const i = icell%(nx+2*ngz); 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef GRACE_3D 
        size_t const k = 
            (icell/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ; 
        size_t const q = 
            (icell/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ;
        #else 
        size_t const q = (icell/(nx+2*ngz)/(ny+2*ngz)) ; 
        #endif 
        /* Physical coordinates of cell center */
        auto pcoords = coord_system.get_physical_coordinates(
            {VEC(i,j,k)},
            q,
            true
        ) ; 
        h_state_mirror(VEC(i,j,k),VELX,q) = 0. ;
        h_state_mirror(VEC(i,j,k),VELY,q) = 0. ;
        h_state_mirror(VEC(i,j,k),VELZ,q) = 0. ;
        h_state_mirror(VEC(i,j,k),ZVECX,q) = 0. ;
        h_state_mirror(VEC(i,j,k),ZVECY,q) = 0. ;
        h_state_mirror(VEC(i,j,k),ZVECZ,q) = 0. ;
        if ( pcoords[0] <= 0 ) {
            h_state_mirror(VEC(i,j,k),RHO,q) = rho_L ;
            h_state_mirror(VEC(i,j,k),PRESS,q) = press_L ;
        } else {
            h_state_mirror(VEC(i,j,k),RHO,q) = rho_R ;
            h_state_mirror(VEC(i,j,k),PRESS,q) = press_R ;
        }
    }
    Kokkos::deep_copy(aux,h_state_mirror) ;
    
    auto const& _eos = eos::get().get_eos<eos_t>() ; 
    parallel_for( GRACE_EXECUTION_TAG("ID","shocktube_ID")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    double h, csnd2; 
                    double ye = _eos.ye_atmosphere();
                    unsigned int err ; 
                    /* Set eps temp and entropy */
                    aux(VEC(i,j,k),EPS_,q) = 
                        _eos.eps_h_csnd2_temp_entropy__press_rho_ye( h, csnd2, aux(VEC(i,j,k),TEMP_,q)
                                                                   , aux(VEC(i,j,k),ENTROPY_,q)
                                                                   , aux(VEC(i,j,k),PRESS_,q)
                                                                   , aux(VEC(i,j,k),RHO_,q)
                                                                   , ye,err);
                    /* Set ye */
                    aux(VEC(i,j,k),YE_,q) = ye ; 
                }) ;
}


template< typename eos_t >
static void set_grmhd_spherical_blastwave_initial_data() {
    using namespace grace ;
    using namespace Kokkos ; 

    GRACE_VERBOSE("Setting Shocktube initial data.") ; 

    auto const press_fact = get_param<double>("grmhd","blastwave_over_pressure_factor") ; 
    auto const rblast     = get_param<double>("grmhd","blastwave_initial_radius") ; 

    auto& aux   = variable_list::get().getaux() ; 
    auto& state = variable_list::get().getstate() ;
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;

    auto& coord_system = grace::coordinate_system::get() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(aux) ; 


    int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
    #pragma omp parallel for 
    for( int64_t icell=0; icell<ncells; ++icell) {
        size_t const i = icell%(nx+2*ngz); 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef GRACE_3D 
        size_t const k = 
            (icell/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ; 
        size_t const q = 
            (icell/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ;
        #else 
        size_t const q = (icell/(nx+2*ngz)/(ny+2*ngz)) ; 
        #endif 
        /* Physical coordinates of cell center */
        auto pcoords = coord_system.get_physical_coordinates(
            {VEC(i,j,k)},
            q,
            true
        ) ; 

        double const r = 
	  Kokkos::sqrt( EXPR( pcoords[0]*pcoords[0], + pcoords[1]*pcoords[1], + pcoords[2] * pcoords[2])) ;

        
        h_state_mirror(VEC(i,j,k),VELX,q) = 0. ;
        h_state_mirror(VEC(i,j,k),VELY,q) = 0. ;
        h_state_mirror(VEC(i,j,k),VELZ,q) = 0. ;
        h_state_mirror(VEC(i,j,k),ZVECX,q) = 0. ;
        h_state_mirror(VEC(i,j,k),ZVECY,q) = 0. ;
        h_state_mirror(VEC(i,j,k),ZVECZ,q) = 0. ;
        h_state_mirror(VEC(i,j,k),RHO,q) = 1. ;
        h_state_mirror(VEC(i,j,k),PRESS,q) = 0.1 ;
        if ( r <= rblast ) {
            h_state_mirror(VEC(i,j,k),PRESS,q) *= press_fact ; 
        }
    }
    Kokkos::deep_copy(aux,h_state_mirror) ;
    
    auto const& _eos = eos::get().get_eos<eos_t>() ; 
    parallel_for( GRACE_EXECUTION_TAG("ID","shocktube_ID")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    double h, csnd2; 
                    double ye = _eos.ye_atmosphere();
                    unsigned int err ; 
                    /* Set eps temp and entropy */
                    aux(VEC(i,j,k),EPS_,q) = 
                        _eos.eps_h_csnd2_temp_entropy__press_rho_ye( h, csnd2, aux(VEC(i,j,k),TEMP_,q)
                                                                   , aux(VEC(i,j,k),ENTROPY_,q)
                                                                   , aux(VEC(i,j,k),PRESS_,q)
                                                                   , aux(VEC(i,j,k),RHO_,q)
                                                                   , ye,err);
                    /* Set ye */
                    aux(VEC(i,j,k),YE_,q) = ye ; 
                }) ;
}

template< typename eos_t
        , typename id_t 
        , typename ... arg_t > 
static void set_grmhd_initial_data_impl(arg_t ... kernel_args)
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace  ; 
    using namespace Kokkos ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    grace::fill_physical_coordinates(pcoords) ; 
    GRACE_TRACE("Filled physical coordinates array.") ; 
    auto& state = grace::variable_list::get().getstate() ; 
    auto& aux   = grace::variable_list::get().getaux()   ; 
    auto& idx   = grace::variable_list::get().getinvspacings()   ; 

    auto const& _eos = eos::get().get_eos<eos_t>() ; 

    // Avec |---> B field in case magnetic field is set from the vector potential
    // 1. Note that for now calling the id_kernel sets:
    //      -   the vector potential ALSO at cell centres
    //      -   followed by computing the densitized B vector field at cell centres from 2nd order derivatives of Avec
    // 2. When and if we implement Avec-constrained-transport, we would instead
    //      -   evaluate the id_kernel at all cell edges
    //      (notabene inefficient if all the metric & hydro fields are traversed just to get 1 (ONE!) Avec component at the end, but at least it makes density,
    //         metric etc consistent at that point where Avec^i is evaluated)
    //      -  in a subsequent call, after the grmhd_ID parallel_for, we would call an appropriate version of Avec--->Bfield transform [same as it happens for 1 now]
    
    const bool set_Bfield_from_Avec = get_param<bool>("grmhd","set_B_from_Avec") ;

    id_t id_kernel{ _eos, pcoords, kernel_args... } ; 
    Kokkos::fence() ; 
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {

                    auto const id = id_kernel(VEC(i,j,k), q) ; 
                    
                    aux(VEC(i,j,k),RHO_,q)   = id.rho; 
                    aux(VEC(i,j,k),PRESS_,q) = id.press ; 

		    #ifdef GRACE_DO_MHD
                    if(set_Bfield_from_Avec){
                        aux(VEC(i,j,k),AVECX_,q) = id.ax ; 
                        aux(VEC(i,j,k),AVECY_,q) = id.ay ; 
                        aux(VEC(i,j,k),AVECZ_,q) = id.az ; 
                        aux(VEC(i,j,k),PHI_EM_,q) = id.phi_em ; 
                    }
                    else{
                        // is this really necessary? 
                        aux(VEC(i,j,k),AVECX_,q) = 0.0 ; 
                        aux(VEC(i,j,k),AVECY_,q) = 0.0 ; 
                        aux(VEC(i,j,k),AVECZ_,q) = 0.0 ; 
                        aux(VEC(i,j,k),PHI_EM_,q) = 0.0 ; 
                    }
                    aux(VEC(i,j,k),BX_,q) = id.bx ;  // this is redundant if set_B_from_Avec, but we leave it for compatibility with other examples
                    aux(VEC(i,j,k),BY_,q) = id.by ; 
                    aux(VEC(i,j,k),BZ_,q) = id.bz ; 
                    #ifdef GRACE_ENABLE_B_FIELD_GLM
                    aux(VEC(i,j,k),PHI_GLM_,q) = 0.0 ; 
                    #endif
		    #endif 

                    state(VEC(i,j,k),ALP_,q) = id.alp ;

                    state(VEC(i,j,k),BETAX_,q) = id.betax ;
                    state(VEC(i,j,k),BETAY_,q) = id.betay ;
                    state(VEC(i,j,k),BETAZ_,q) = id.betaz ;

                    state(VEC(i,j,k),GXX_,q) = id.gxx ; 
                    state(VEC(i,j,k),GXY_,q) = id.gxy ; 
                    state(VEC(i,j,k),GXZ_,q) = id.gxz ; 
                    state(VEC(i,j,k),GYY_,q) = id.gyy ; 
                    state(VEC(i,j,k),GYZ_,q) = id.gyz ;
                    state(VEC(i,j,k),GZZ_,q) = id.gzz ;

                    state(VEC(i,j,k),KXX_,q) = id.kxx ; 
                    state(VEC(i,j,k),KXY_,q) = id.kxy ; 
                    state(VEC(i,j,k),KXZ_,q) = id.kxz ; 
                    state(VEC(i,j,k),KYY_,q) = id.kyy ; 
                    state(VEC(i,j,k),KYZ_,q) = id.kyz ;
                    state(VEC(i,j,k),KZZ_,q) = id.kzz ;

                    auto const v2 = id.gxx * id.vx * id.vx +
                                    id.gyy * id.vy * id.vy +
                                    id.gzz * id.vz * id.vz +
                                    2. * ( 
                                        id.gxy * id.vx * id.vy +
                                        id.gxz * id.vx * id.vz +
                                        id.gyz * id.vy * id.vz 
                                    ) ; 
                    auto const w = 1./Kokkos::sqrt( 1 - v2  ) ; 

                    aux(VEC(i,j,k),ZVECX_,q)  = w * id.vx ; 
                    aux(VEC(i,j,k),ZVECY_,q)  = w * id.vy ; 
                    aux(VEC(i,j,k),ZVECZ_,q)  = w * id.vz ; 

                    aux(VEC(i,j,k),VELX_,q)  = id.alp * id.vx - id.betax ; 
                    aux(VEC(i,j,k),VELY_,q)  = id.alp * id.vy - id.betay ; 
                    aux(VEC(i,j,k),VELZ_,q)  = id.alp * id.vz - id.betaz ; 
                    
                    double h, csnd2; 
                    double ye = _eos.ye_atmosphere();
                    unsigned int err ; 
                    /* Set eps temp and entropy */
                    aux(VEC(i,j,k),EPS_,q) = 
                        _eos.eps_h_csnd2_temp_entropy__press_rho_ye( h, csnd2, aux(VEC(i,j,k),TEMP_,q)
                                                                   , aux(VEC(i,j,k),ENTROPY_,q)
                                                                   , aux(VEC(i,j,k),PRESS_,q)
                                                                   , aux(VEC(i,j,k),RHO_,q)
                                                                   , ye,err);
                    /* Set ye */
                    aux(VEC(i,j,k),YE_,q) = ye ; 
                }) ; 

    // at this point, the vector potential (or magnetic field in some cases) is already set up at cell-centres (if applicable)
    // in the future, we might want to schedule an edge-centred initialization for Avec components here:
        
    // for (size_t idir = 0 ; idir < 3 ; idir++)
    //  set_edge_staggered_Avec<idir>(id_kernel, state, cstate, idx); 

    // finally, we launch a separate loop to set the magnetic field from vector potential
    // in GLM, A is only needed in the ID; vector potential is then never touched
    if(set_Bfield_from_Avec){
        GRACE_INFO("Computing magnetic field from vector potential"); 
        compute_B_field_from_Avec(state, aux, idx);
    }

}

template< typename eos_t >
void set_grmhd_initial_data() {
    auto const id_type = get_param<std::string>("grmhd","id_type") ;
    GRACE_VERBOSE("Setting grmhd initial data of type {}.", id_type) ;  
    /* Set requested initial data */
    if( id_type == "shocktube" ) {
        auto const rho_L = get_param<double>("grmhd","shocktube_rho_L") ; 
        auto const rho_R = get_param<double>("grmhd","shocktube_rho_R") ; 
        auto const press_L = get_param<double>("grmhd","shocktube_press_L") ; 
        auto const press_R = get_param<double>("grmhd","shocktube_press_R") ;
        set_grmhd_initial_data_impl<eos_t,shocktube_id_t<eos_t>>(rho_L, rho_R, press_L, press_R) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with hydro ID.") ;  
    }	  
    #ifdef GRACE_DO_MHD
    else if( id_type == "shocktube_mhd" ) {
        auto const rho_L = get_param<double>("grmhd","shocktube_mhd_rho_L") ; 
        auto const rho_R = get_param<double>("grmhd","shocktube_mhd_rho_R") ; 
        auto const press_L = get_param<double>("grmhd","shocktube_mhd_press_L") ; 
        auto const press_R = get_param<double>("grmhd","shocktube_mhd_press_R") ;

        auto const vel_x_L = get_param<double>("grmhd","shocktube_mhd_vel_x_L") ; 
        auto const vel_y_L = get_param<double>("grmhd","shocktube_mhd_vel_y_L") ; 
        auto const vel_z_L = get_param<double>("grmhd","shocktube_mhd_vel_z_L") ; 
        
        auto const vel_x_R = get_param<double>("grmhd","shocktube_mhd_vel_x_R") ; 
        auto const vel_y_R = get_param<double>("grmhd","shocktube_mhd_vel_y_R") ; 
        auto const vel_z_R = get_param<double>("grmhd","shocktube_mhd_vel_z_R") ;

        auto const B_x_L = get_param<double>("grmhd","shocktube_mhd_B_x_L") ; 
        auto const B_y_L = get_param<double>("grmhd","shocktube_mhd_B_y_L") ; 
        auto const B_z_L = get_param<double>("grmhd","shocktube_mhd_B_z_L") ; 
       
        auto const B_x_R = get_param<double>("grmhd","shocktube_mhd_B_x_R") ; 
        auto const B_y_R = get_param<double>("grmhd","shocktube_mhd_B_y_R") ; 
        auto const B_z_R = get_param<double>("grmhd","shocktube_mhd_B_z_R") ; 
        
        ASSERT(std::abs(B_x_L-B_x_R)<1e-10, "What are you doing?");

        set_grmhd_initial_data_impl<eos_t,shocktube_mhd_id_t<eos_t>>(rho_L, rho_R, 
                                                                     press_L, press_R,
                                                                     vel_x_L, vel_y_L, vel_z_L,
                                                                     vel_x_R, vel_y_R, vel_z_R,
                                                                     B_x_L, B_y_L, B_z_L,
                                                                     B_x_R, B_y_R, B_z_R
                                                                      ) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with shocktube MHD ID.") ;  
    }
    else if( id_type == "linear_mhd_wave" ) {
        auto const str_wave_type     = get_param<std::string>("grmhd","linear_mhd_wave_type") ; 
        auto const str_wave_movement = get_param<std::string>("grmhd","linear_mhd_wave_movement") ; 
        auto const ampl          = get_param<double>("grmhd","linear_mhd_wave_ampl") ; 
        auto const wavelength    = get_param<double>("grmhd","linear_mhd_wave_wavelength") ;
        
        using namespace linear_mhd_utils;
        WAVE_TYPE wave_type;
        if(str_wave_type=="alfven"){
            wave_type = WAVE_TYPE::ALFVEN;
            GRACE_INFO("Wave type: Alfven wave");
        }
        else if(str_wave_type=="slow_magnetosonic"){
            wave_type = WAVE_TYPE::SLOW_MAGNETOSONIC;
            GRACE_INFO("Wave type: slow magnetosonic wave");
        }
        else if(str_wave_type=="fast_magnetosonic"){
            wave_type = WAVE_TYPE::FAST_MAGNETOSONIC;
            GRACE_INFO("Wave type: fast magnetosonic wave");
        }
        else if(str_wave_type=="contact"){
            wave_type = WAVE_TYPE::CONTACT;
            GRACE_INFO("Wave type: fast magnetosonic wave");
        }
        else {
            GRACE_INFO("Setting constant MHD state");
        }

        WAVE_DIRECTION wave_movement;
        if(str_wave_movement=="right"){wave_movement = WAVE_DIRECTION::RIGHT;}
        else if(str_wave_movement=="left"){wave_movement = WAVE_DIRECTION::LEFT;}
        else if(str_wave_movement=="standing"){wave_movement = WAVE_DIRECTION::STANDING;}
        else {GRACE_INFO("Unknown wave direction"); }

        set_grmhd_initial_data_impl<eos_t,linear_mhd_wave_id_t<eos_t>>(wave_type,ampl,wavelength,wave_movement
                                                                      ) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with linear wave MHD ID.") ;  
    }
    else if( id_type == "orszag_tang_vortex" ) {
        auto const rho          = get_param<double>("grmhd","orszag_tang_vortex_rho") ; 
        auto const press    = get_param<double>("grmhd","orszag_tang_vortex_press") ;
        set_grmhd_initial_data_impl<eos_t,orszag_tang_vortex_mhd_id_t<eos_t>>(rho,press) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with Orszag-Tang Vortex MHD ID.") ;  
    }
    else if( id_type == "boosted_loop_advection" ) {
        auto const rho          = get_param<double>("grmhd","boosted_loop_advection_rho") ; 
        auto const press    = get_param<double>("grmhd","boosted_loop_advection_press") ;
        auto const beta0    = get_param<double>("grmhd","boosted_loop_advection_beta0") ;
        auto const vc    = get_param<double>("grmhd","boosted_loop_advection_vc") ;
        auto const B0    = get_param<double>("grmhd","boosted_loop_advection_B0") ;
        bool compensate    = get_param<bool>("grmhd","boosted_loop_advection_compensate_shift") ;
        set_grmhd_initial_data_impl<eos_t,boosted_loop_advection_mhd_id_t<eos_t>>(rho,press,beta0,vc,B0,compensate) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with Boosted Loop Advection MHD ID.") ;  
    }
    else if( id_type == "cp_alfven_wave" ) {
        auto const rho          = get_param<double>("grmhd","cp_alfven_wave_rho") ; 
        auto const press    = get_param<double>("grmhd","cp_alfven_wave_press") ;
        auto const B0    = get_param<double>("grmhd","cp_alfven_wave_B0") ;
        auto const vel    = get_param<double>("grmhd","cp_alfven_wave_vel") ;
        auto const kmag    = get_param<double>("grmhd","cp_alfven_wave_mag_k") ;
        auto const ampl    = get_param<double>("grmhd","cp_alfven_wave_ampl") ;
        set_grmhd_initial_data_impl<eos_t,cp_alfven_wave_id_t<eos_t>>(rho,press,B0,ampl, kmag, vel) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with Large-Amplitude Circularly Polarized Alfven Wave MHD ID.") ;  
    }
    else if( id_type == "magnetic_rotor" ) {
        auto const rho_in          = get_param<double>("grmhd","magnetized_rotor_rho_in") ; 
        auto const rho_out    = get_param<double>("grmhd","magnetized_rotor_rho_out") ;
        auto const press    = get_param<double>("grmhd","magnetized_rotor_press") ;
        auto const B0    = get_param<double>("grmhd","magnetized_rotor_B0") ;
        set_grmhd_initial_data_impl<eos_t,magnetic_rotor_id_t<eos_t>>(rho_in, rho_out, press, B0) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with Magnetized Rotor MHD ID.") ;  
    }
    else if( id_type == "bondi_accretion" ) {
        auto const M          = get_param<double>("grmhd","Bondi_accretion_M") ; 
        auto const mdot       = get_param<double>("grmhd","Bondi_accretion_M_dot") ;
        auto const r_sonic    = get_param<double>("grmhd","Bondi_accretion_r_sonic") ;
        auto const gamma      = get_param<double>("grmhd","Bondi_accretion_gamma") ;
        auto const rmin       = get_param<double>("grmhd","Bondi_accretion_rmin") ;
        auto const rmax       = get_param<double>("grmhd","Bondi_accretion_rmax") ;
        auto const bmag       = get_param<double>("grmhd","Bondi_accretion_bmag") ;
        auto const beta_sonic = get_param<double>("grmhd","Bondi_accretion_beta_sonic") ;
         set_grmhd_initial_data_impl<eos_t,bondi_accretion_id_t<eos_t>>(M,mdot,r_sonic,gamma,rmin,rmax,bmag,beta_sonic) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with Bondi accretion MHD ID.") ;  
    }
    else if( id_type == "blast_wave" ) {
        auto const rho_in     = get_param<double>("grmhd","blast_wave_rho_in") ; 
        auto const rho_out    = get_param<double>("grmhd","blast_wave_rho_out") ;
        auto const press_in   = get_param<double>("grmhd","blast_wave_press_in") ;
        auto const press_out  = get_param<double>("grmhd","blast_wave_press_out") ;
        auto const B0         = get_param<double>("grmhd","blast_wave_B0") ;
        auto const phi        = get_param<double>("grmhd","blast_wave_B_phi") ;
        auto const theta      = get_param<double>("grmhd","blast_wave_B_theta") ;
        set_grmhd_initial_data_impl<eos_t,blast_wave_id_t<eos_t>>(rho_in, rho_out, press_in, press_out, B0, phi, theta) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with Blast Wave ID.") ;  
    }
    #endif
    else if ( id_type == "blastwave" ) {
        set_grmhd_spherical_blastwave_initial_data<eos_t>() ; 
    } else if ( id_type == "TOV") { 
        auto const rho_c = get_param<double>("grmhd", "TOV_central_density") ; 

        const bool set_Bfield_from_Avec = get_param<bool>("grmhd","set_B_from_Avec") ;
        if(not set_Bfield_from_Avec){
            set_grmhd_initial_data_impl<eos_t,tov_id_t<eos_t>>(rho_c) ;
        }
        else{
            const std::string Avec_type = get_param<std::string>("grmhd","Avec_type") ; // poloidal, dipole, monopole, linear (e.g. for shocktubes)
            const std::string Avec_prescription = get_param<std::string>("grmhd","Avec_prescription") ; // density/pressure based
            const double Avec_Pcut = get_param<double>("grmhd","Avec_Pcut") ;
            const int Avec_n = get_param<int>("grmhd","Avec_n") ;
            const double Avec_Ab = get_param<double>("grmhd","Avec_Ab") ;

            const int int_Avec_prescription = (Avec_prescription=="pressure_prescription" ? 0 : 1 );
            const int int_Avec_type = (Avec_type=="poloidal" ? 0 : 1 ); // add more...
            set_grmhd_initial_data_impl<eos_t,tov_id_t<eos_t>>(rho_c,
                                                               set_Bfield_from_Avec, int_Avec_type, int_Avec_prescription,
                                                               Avec_Pcut, Avec_n, Avec_Ab ) ;
        }
    }else if ( id_type == "FMtorus") { 
        bool is_eos_thermal=(get_param<double>("eos","gamma_th") > 0.0 ?  true : false) ;
        if(is_eos_thermal) ERROR("Thermal component not available for FMtorus currently");

        auto const a_BH = get_param<double>("grmhd", "FMTorus_a_BH") ; 
        auto const M_BH = get_param<double>("grmhd", "FMTorus_M_BH") ; 
        auto const rho_min = get_param<double>("grmhd", "FMTorus_rho_min") ; 
        auto const press_min = get_param<double>("grmhd", "FMTorus_press_min") ; 
        auto const lapse_min = get_param<double>("grmhd", "FMTorus_lapse_min") ; 
        auto const r_in = get_param<double>("grmhd", "FMTorus_r_rin") ; 
        auto const r_at_max_density = get_param<double>("grmhd", "FMTorus_r_at_max_density") ; 
        auto const gamma = get_param<double>("grmhd", "FMTorus_gamma") ; 
        auto const kappa = get_param<double>("grmhd", "FMTorus_kappa") ; 

        // note: setting things like this leads to a lot of code-repetition and very verbose id_kernel constructors
        // [every id kernel needs to have these in the constructor
        // if it wants to initialize B field from Avec...]
        const bool set_Bfield_from_Avec = get_param<bool>("grmhd","set_B_from_Avec") ;
        const std::string Avec_type = get_param<std::string>("grmhd","Avec_type") ; // poloidal, dipole, monopole, linear (e.g. for shocktubes)
        const std::string Avec_prescription = get_param<std::string>("grmhd","Avec_prescription") ; // density/pressure based
        const double Avec_Pcut = get_param<double>("grmhd","Avec_Pcut") ;
        const int Avec_n = get_param<int>("grmhd","Avec_n") ;
        const double Avec_Ab = get_param<double>("grmhd","Avec_Ab") ;

        const int int_Avec_prescription = (Avec_prescription=="pressure_prescription" ? 0 : 1 );
        const int int_Avec_type = (Avec_type=="poloidal" ? 0 : 1 ); // add more...
        
        set_grmhd_initial_data_impl<eos_t,fmtorus_id_t<eos_t>>(a_BH,M_BH,rho_min, 
                        press_min,lapse_min,r_in,r_at_max_density,kappa,gamma,
                        set_Bfield_from_Avec, int_Avec_type, int_Avec_prescription,
                        Avec_Pcut, Avec_n, Avec_Ab) ;
        GRACE_TRACE("Done with magnetized FMTorus ID.") ;  
    } else {
        ERROR("Unrecognized id_type " << id_type ) ; 
    }
    set_conservs_from_prims() ;
}

void set_conservs_from_prims() {
    using namespace grace ;
    using namespace Kokkos ;

    GRACE_VERBOSE("Setting conservative variables from primitives.") ; 

    auto& state = grace::variable_list::get().getstate() ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;
    auto& aux = variable_list::get().getaux() ;

    parallel_for( GRACE_EXECUTION_TAG("ID","set_conservs_from_prims")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(metric, state, q, VEC(i,j,k)) ; 
        grmhd_prims_array_t prims ; 
        FILL_PRIMS_ARRAY(prims,aux,q,VEC(i,j,k)) ; 
        grmhd_cons_array_t cons ;
        prims_to_conservs(prims,cons,metric) ; 
        state(VEC(i,j,k),DENS_,q) = cons[DENSL] ; 
        state(VEC(i,j,k),SX_,q) = cons[STXL] ; 
        state(VEC(i,j,k),SY_,q) = cons[STYL] ; 
        state(VEC(i,j,k),SZ_,q) = cons[STZL] ; 
        state(VEC(i,j,k),TAU_,q) = cons[TAUL] ;
        state(VEC(i,j,k),YESTAR_,q) = cons[YESL] ; 
        state(VEC(i,j,k),ENTROPYSTAR_,q) = cons[ENTSL] ; 
        #ifdef GRACE_DO_MHD        
        state(VEC(i,j,k),BGX_,q) = cons[BGXL] ; 
        state(VEC(i,j,k),BGY_,q) = cons[BGYL] ; 
        state(VEC(i,j,k),BGZ_,q) = cons[BGZL] ; 
        #ifdef GRACE_ENABLE_B_FIELD_GLM
        state(VEC(i,j,k),PHI_GLM_,q) = cons[PHIG_GLML] ; 
        #endif
        #endif
    }) ;
}
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)                                       \
template                                                                \
void set_grmhd_initial_data<EOS>( )

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}
