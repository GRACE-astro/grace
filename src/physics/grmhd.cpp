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
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/physics/grmhd_helpers.hh>
#ifdef GRACE_ENABLE_BSSN_METRIC
#include <grace/physics/bssn_helpers.hh>
#endif 
#include <grace/physics/id/shocktube.hh>
//#include <grace/physics/id/blastwave.hh>
#include <grace/physics/id/tov.hh>
#include <grace/physics/id/puncture.hh>
#include <grace/physics/id/gauge_wave.hh>
#include <grace/physics/id/linear_gw.hh>
#include <grace/physics/id/robust_stability.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/evolution/hrsc_evolution_system.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/grmhd.hh>
#include <grace/physics/grmhd_metric_utils.hh>

#include <grace/config/config_parser.hh>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp> 

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
    auto& cstate = grace::variable_list::get().getstaggeredstate().corner_staggered_fields ;
    auto& aux   = grace::variable_list::get().getaux()   ; 
    auto& idx   = grace::variable_list::get().getinvspacings()   ; 

    auto const& _eos = eos::get().get_eos<eos_t>() ; 

    id_t id_kernel{ _eos, pcoords, kernel_args... } ; 
    Kokkos::fence() ; 
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {

                    auto const id = id_kernel(VEC(i,j,k), q) ; 
                    
                    aux(VEC(i,j,k),RHO_,q)   = id.rho; 
                    aux(VEC(i,j,k),PRESS_,q) = id.press ; 

                    #ifdef GRACE_ENABLE_COWLING_METRIC
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
                    #elif defined(GRACE_ENABLE_BSSN_METRIC)
                    aux(VEC(i,j,k),ALPC_,q) = id.alp ;

                    aux(VEC(i,j,k),BETAXC_,q) = id.betax ;
                    aux(VEC(i,j,k),BETAYC_,q) = id.betay ;
                    aux(VEC(i,j,k),BETAZC_,q) = id.betaz ;

                    aux(VEC(i,j,k),GXX_,q) = id.gxx ; 
                    aux(VEC(i,j,k),GXY_,q) = id.gxy ; 
                    aux(VEC(i,j,k),GXZ_,q) = id.gxz ; 
                    aux(VEC(i,j,k),GYY_,q) = id.gyy ; 
                    aux(VEC(i,j,k),GYZ_,q) = id.gyz ;
                    aux(VEC(i,j,k),GZZ_,q) = id.gzz ;

                    aux(VEC(i,j,k),KXX_,q) = id.kxx ; 
                    aux(VEC(i,j,k),KXY_,q) = id.kxy ; 
                    aux(VEC(i,j,k),KXZ_,q) = id.kxz ; 
                    aux(VEC(i,j,k),KYY_,q) = id.kyy ; 
                    aux(VEC(i,j,k),KYZ_,q) = id.kyz ;
                    aux(VEC(i,j,k),KZZ_,q) = id.kzz ;
                    #endif 

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
    #ifdef GRACE_ENABLE_BSSN_METRIC
    init_bssn_metric(id_kernel,state,cstate,idx) ; 
    #endif 
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
    } else if ( id_type == "blastwave" ) {
        set_grmhd_spherical_blastwave_initial_data<eos_t>() ; 
    } else if ( id_type == "TOV") { 
        auto const rho_c = get_param<double>("grmhd", "TOV_central_density") ; 
        set_grmhd_initial_data_impl<eos_t,tov_id_t<eos_t>>(rho_c) ;
    } else if ( id_type == "gauge_wave" ) {
        auto const A = get_param<double>("grmhd", "gauge_wave_amplitude"  ) ; 
        auto const d = get_param<double>("grmhd", "gauge_wave_wavelength" ) ; 
        set_grmhd_initial_data_impl<eos_t, gauge_wave_id_t<eos_t>>(
            A, d
        ) ; 
    } else if (id_type == "linear_gw" ) {
        auto const A = get_param<double>("grmhd", "gauge_wave_amplitude"  ) ; 
        auto const d = get_param<double>("grmhd", "gauge_wave_wavelength" ) ; 
        set_grmhd_initial_data_impl<eos_t, linear_wave_id_t<eos_t>>(
            A, d
        ) ; 
    } else if ( id_type == "robust_stability_test" ) {
        auto const rho = get_param<double>("grmhd", "robust_stability_rho") ; 
        using gen_t = Kokkos::Random_XorShift64_Pool<grace::default_execution_space> ;
        gen_t gen(12345) ;
        set_grmhd_initial_data_impl<eos_t, robust_stability_test_id_t<eos_t,gen_t>>(
            gen, rho
        ) ;
    } else if ( id_type == "puncture" ) {
      double const mass = 0 ; 
      set_grmhd_initial_data_impl<eos_t,puncture_id_t<eos_t>>(
          mass
      ) ;
    } else {    ERROR("Unrecognized id_type " << id_type ) ; 
    }
    set_conservs_from_prims() ;
}

void set_conservs_from_prims() {
    using namespace grace ;
    using namespace Kokkos ;

    GRACE_VERBOSE("Setting conservative variables from primitives.") ; 

    auto& state = grace::variable_list::get().getstate() ; 
    auto& cstate = grace::variable_list::get().getstaggeredstate().corner_staggered_fields ;
    auto& aux = grace::variable_list::get().getaux() ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;

    parallel_for( GRACE_EXECUTION_TAG("ID","set_conservs_from_prims")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        #ifdef GRACE_ENABLE_BSSN_METRIC
        auto mview = aux ; 
        #elif defined(GRACE_ENABLE_COWLING_METRIC)
        auto mview = state ; 
        #endif 
        #if 1
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(
            metric, mview, q, VEC(i,j,k)
        ) ; 
        #endif 
        
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
    }) ;
}
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)                                       \
template                                                                \
void set_grmhd_initial_data<EOS>( )

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}
