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
#ifdef GRACE_ENABLE_BSSN_METRIC
#include <grace/physics/bssn_helpers.hh>
#endif
#include <grace/physics/id/shocktube.hh>
#include <grace/physics/id/vacuum.hh>
//#include <grace/physics/id/blastwave.hh>
#include <grace/physics/id/kelvin_helmholtz.hh>
#include <grace/physics/id/tov.hh>
#include <grace/physics/id/magnetic_rotor.hh>
#include <grace/physics/id/orszag_tang_vortex.hh>
#include <grace/physics/id/fmtorus.hh>
#include <grace/physics/id/Avec_id.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/evolution/hrsc_evolution_system.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/grmhd.hh>

#include <grace/config/config_parser.hh>
#include <Kokkos_Core.hpp>

#include <string>

namespace grace{

static void rescale_B_field(double max_betam1, double max_press) {

    DECLARE_GRID_EXTENTS ; 
    using namespace grace ;
    using namespace Kokkos ;

    auto& aux   = variable_list::get().getaux() ; 
    auto& state   = variable_list::get().getstate() ;
    auto& stag_state = variable_list::get().getstaggeredstate() ; 
    MinMaxScalar<double> b2_max_loc ;
    double b2_max ; 
    auto policy =
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
    parallel_reduce( GRACE_EXECUTION_TAG("IO","find_max_press_loc") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, MinMaxScalar<double>& lres)
        {
            metric_array_t metric ; 
            FILL_METRIC_ARRAY(metric,state,q,VEC(i,j,k)) ; 
            grmhd_prims_array_t prims ; 
            FILL_PRIMS_ARRAY_ZVEC(prims,aux,q,VEC(i,j,k)) ; 
            auto W = Kokkos::sqrt(1+metric.square_vec({prims[VXL],prims[VYL],prims[VZL]}));

            std::array<double,3> const B = {
                0.5 * (stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) + stag_state.face_staggered_fields_x(VEC(i+1,j,k),BSX_,q)),
                0.5 * (stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) + stag_state.face_staggered_fields_y(VEC(i,j+1,k),BSY_,q)),
                0.5 * (stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) + stag_state.face_staggered_fields_z(VEC(i,j,k+1),BSZ_,q))
            } ; 
            auto b2 = metric.square_vec(B) / W / W / metric.sqrtg() / metric.sqrtg() ; // assume B_i v^i == 0 ! 

            lres.max_val = lres.max_val < b2 ? b2 : lres.max_val    ; 
        }, MinMax<double>(b2_max_loc)) ; 
    parallel::mpi_allreduce( &b2_max_loc.max_val
                            , &b2_max
                            , 1
                            , sc_MPI_MAX) ;
    auto max_beta_now = 2 * max_press / b2_max ; 
    double fact ; 
    if ( b2_max > 1e-15 ) {
        fact = Kokkos::sqrt( 2 * max_press / b2_max / max_betam1 ) ; 
    } else {
        fact = 1 ;
    }
    GRACE_INFO("B2_max {} fact {}", b2_max, fact) ; 
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BX")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) *= fact ; 
                    });
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BY")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                { 
                    stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) *= fact ; 
                });
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BZ")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz+1),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) *= fact ; 
                });
}

static double get_max_press()
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ;
    using namespace Kokkos ;

    auto& aux   = variable_list::get().getaux() ; 

    MinMaxScalar<double> pmax_loc ;
    double pmax ; 
    auto policy =
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
    parallel_reduce( GRACE_EXECUTION_TAG("IO","find_max_press_loc") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, MinMaxScalar<double>& lres)
        {
            lres.max_val = lres.max_val < aux(VEC(i,j,k),PRESS_,q) ? aux(VEC(i,j,k),PRESS_,q) : lres.max_val    ; 
        }, MinMax<double>(pmax_loc)) ; 
    parallel::mpi_allreduce( &pmax_loc.max_val
                            , &pmax
                            , 1
                            , sc_MPI_MAX) ; 
    return pmax ; 
}

static double get_max_rho()
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ;
    using namespace Kokkos ;

    auto& aux   = variable_list::get().getaux() ; 

    MinMaxScalar<double> rhomax_loc ;
    double rhomax ; 
    auto policy =
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
    parallel_reduce( GRACE_EXECUTION_TAG("IO","find_max_rho_loc") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, MinMaxScalar<double>& lres)
        {
            lres.max_val = lres.max_val < aux(VEC(i,j,k),RHO_,q) ? aux(VEC(i,j,k),RHO_,q) : lres.max_val    ; 
        }, MinMax<double>(rhomax_loc)) ; 
    parallel::mpi_allreduce( &rhomax_loc.max_val
                            , &rhomax
                            , 1
                            , sc_MPI_MAX) ; 
    return rhomax ; 
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
                    double ye = 0;
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
    auto& stag_state = grace::variable_list::get().getstaggeredstate() ; 
    auto& aux   = grace::variable_list::get().getaux()   ; 

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
                    adm_to_bssn(id,state,VEC(i,j,k),q);
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

                    aux(VEC(i,j,k),YE_,q) = id.ye ; 
                    
                    double h, csnd2; 
                    unsigned int err ;
                    /* Set eps temp and entropy */
                    aux(VEC(i,j,k),EPS_,q) = 
                        _eos.eps_h_csnd2_temp_entropy__press_rho_ye( h, csnd2, aux(VEC(i,j,k),TEMP_,q)
                                                                   , aux(VEC(i,j,k),ENTROPY_,q)
                                                                   , aux(VEC(i,j,k),PRESS_,q)
                                                                   , aux(VEC(i,j,k),RHO_,q)
                                                                   , aux(VEC(i,j,k),YE_,q)
                                                                   ,err);                    
                    /* Set B field */
                    aux(VEC(i,j,k),BX_,q) = id.bx ;
                    aux(VEC(i,j,k),BY_,q) = id.by ;
                    aux(VEC(i,j,k),BZ_,q) = id.bz ; 
                }) ; 
    #ifdef GRACE_ENABLE_BSSN_METRIC 
    {
        Kokkos::fence(); 
        auto& idx = variable_list::get().getinvspacings() ; 
        parallel_for( GRACE_EXECUTION_TAG("ID","BSSN_Gamma")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        std::array<double,3> _idx{idx(0,q),idx(1,q),idx(2,q)} ; 
                        compute_gamma_tilde<BSSN_DER_ORDER>(state,VEC(i,j,k),q,_idx,VEC(nx,ny,nz),ngz) ; 
                    });
    }
    #endif 
    auto B_from_A = grace::get_param<bool>("grmhd","B_field_from_Avec") ; 
    if ( ! B_from_A ) {
        // get staggered coordinates 
        fill_physical_coordinates(pcoords,STAG_FACEX) ; 
        id_kernel = id_t( _eos, pcoords, kernel_args...) ; 
        // now we set the staggered fields 
        parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BX")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        auto const id = id_kernel(VEC(i,j,k), q) ;  
                        metric_array_t metric({id.gxx,id.gxy,id.gxz,id.gyy,id.gyz,id.gzz},{id.betax,id.betay,id.betaz},id.alp) ;
                        stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) = id.bx * metric.sqrtg() ; 
                    });
        // get staggered coordinates 
        fill_physical_coordinates(pcoords,STAG_FACEY) ; 
        id_kernel = id_t( _eos, pcoords, kernel_args...) ; 
        parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BY")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        auto const id = id_kernel(VEC(i,j,k), q) ;  
                        metric_array_t metric({id.gxx,id.gxy,id.gxz,id.gyy,id.gyz,id.gzz},{id.betax,id.betay,id.betaz},id.alp) ;
                        stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) = id.by * metric.sqrtg() ; 
                    });
        fill_physical_coordinates(pcoords,STAG_FACEZ) ; 
        id_kernel = id_t( _eos, pcoords, kernel_args...) ; 
        parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BZ")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz+1),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        auto const id = id_kernel(VEC(i,j,k), q) ;  
                        metric_array_t metric({id.gxx,id.gxy,id.gxz,id.gyy,id.gyz,id.gzz},{id.betax,id.betay,id.betaz},id.alp) ; 
                        stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) = id.bz * metric.sqrtg() ; 
                    });
    } else {

        auto par = get_param<YAML::Node>("grmhd","Avec_ID") ; 
        auto kind = par["kind"].as<std::string>() ; 
        ASSERT(kind == "current_loop", "Only current_loop Avec initialization supported.") ; 
        auto cutoff_var = par["cutoff_var"].as<std::string>() ; 
        bool use_rho = cutoff_var == "rho" ; 
        ASSERT(cutoff_var=="press" or cutoff_var=="rho", "Only pressure and density-based cutoff supported.") ; 
        auto A_pcut = par["cutoff_fact"].as<double>() ; 
        auto A_phi = par["A_phi"].as<double>() ; 
        auto A_n = par["A_n"].as<double>() ; 
        double vmax ; 
        if ( use_rho  ) {
            vmax = get_max_rho() ; 
        } else {
            vmax = get_max_press() ; 
        }
        auto A_id = Avec_toroidal_id_t(
            vmax * A_pcut, A_phi, A_n
        ) ; 
        // Initialize Avec 
        grace::var_array_t Ax("Ax", VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz+1),1,nq) 
                         , Ay("Ay", VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz+1),1,nq) 
                         , Az("Az", VEC(nx+2*ngz+1,ny+2*ngz+1,nz+2*ngz),1,nq) ; 
        // Ax 
        parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_AX")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz+1),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        auto const id = id_kernel(VEC(i,j,k), q) ;
                        auto var = use_rho ? id.rho : id.press ; 
                        Ax(VEC(i,j,k),0,q) = A_id.template get<0>({pcoords(VEC(i,j,k),0,q), pcoords(VEC(i,j,k),1,q), pcoords(VEC(i,j,k),2,q)}, var); 
                    });
        // Ay
        fill_physical_coordinates(pcoords,STAG_EDGEXZ) ;
        id_kernel = id_t( _eos, pcoords, kernel_args...) ; 
        parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_AY")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz+1),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        auto const id = id_kernel(VEC(i,j,k), q) ; 
                        auto var = use_rho ? id.rho : id.press ;  
                        Ay(VEC(i,j,k),0,q) = A_id.template get<1>({pcoords(VEC(i,j,k),0,q), pcoords(VEC(i,j,k),1,q), pcoords(VEC(i,j,k),2,q)}, var); 
                    });
        // Az
        fill_physical_coordinates(pcoords,STAG_EDGEXY) ;
        id_kernel = id_t( _eos, pcoords, kernel_args...) ; 
        parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_AZ")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz+1,ny+2*ngz+1,nz+2*ngz),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        auto const id = id_kernel(VEC(i,j,k), q) ;  
                        auto var = use_rho ? id.rho : id.press ; 
                        Az(VEC(i,j,k),0,q) = A_id.template get<2>({pcoords(VEC(i,j,k),0,q), pcoords(VEC(i,j,k),1,q), pcoords(VEC(i,j,k),2,q)}, var); 
                    });
        // Now set B from A:
        // B^k = \epsilon^{ijk} d/dx^j A_k = gamma^{-1/2} [ijk] d/dx^j A_k
        // we want sqrt(gamma) B^k so the metric factors cancel out 
        auto& idx = variable_list::get().getinvspacings() ; 
        // Bx
        fill_physical_coordinates(pcoords,STAG_FACEX) ; 
        id_kernel._pcoords = pcoords ; 
        parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BX")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        auto const id = id_kernel(VEC(i,j,k), q) ;  
                        metric_array_t metric_face({id.gxx,id.gxy,id.gxz,id.gyy,id.gyz,id.gzz},{id.betax,id.betay,id.betaz},id.alp) ;  
                        // B^x = d/dy A^z - d/dz A^y
                        stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) = (
                            (Az(VEC(i  ,j+1,k  ),0,q) - Az(VEC(i  ,j  ,k  ),0,q)) * idx(1,q)
                          + (Ay(VEC(i  ,j  ,k  ),0,q) - Ay(VEC(i  ,j  ,k+1),0,q)) * idx(2,q)
                        ) ; 
                    });
        // By
        fill_physical_coordinates(pcoords,STAG_FACEY) ; 
        id_kernel._pcoords = pcoords ; 
        parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BY")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        auto const id = id_kernel(VEC(i,j,k), q) ;  
                        metric_array_t metric_face({id.gxx,id.gxy,id.gxz,id.gyy,id.gyz,id.gzz},{id.betax,id.betay,id.betaz},id.alp) ;   
                        // B^y = d/dz A^x - d/dx A^z
                        stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) = (
                            (Ax(VEC(i  ,j  ,k+1),0,q) - Ax(VEC(i  ,j  ,k  ),0,q)) * idx(2,q)
                          + (Az(VEC(i  ,j  ,k  ),0,q) - Az(VEC(i+1,j  ,k  ),0,q)) * idx(0,q)
                        ) ; 
                    });
        // Bz
        fill_physical_coordinates(pcoords,STAG_FACEZ) ; 
        id_kernel._pcoords = pcoords ; 
        parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BZ")
                    , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz+1),nq})
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        auto const id = id_kernel(VEC(i,j,k), q) ;  
                        metric_array_t metric_face({id.gxx,id.gxy,id.gxz,id.gyy,id.gyz,id.gzz},{id.betax,id.betay,id.betaz},id.alp) ;  
                        // B^z = d/dx A^y - d/dy A^x
                        stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) = (
                            (Ay(VEC(i+1,j  ,k  ),0,q) - Ay(VEC(i  ,j  ,k  ),0,q)) * idx(0,q)
                          + (Ax(VEC(i  ,j  ,k  ),0,q) - Ax(VEC(i  ,j+1,k  ),0,q)) * idx(1,q)
                        ) ; 
                    });
    }
    
}

template< typename eos_t >
void set_grmhd_initial_data() {
    auto const id_type = get_param<std::string>("grmhd","id_type") ;
    GRACE_VERBOSE("Setting grmhd initial data of type {}.", id_type) ;  
    /* Set requested initial data */
    if ( id_type == "minkowski_vacuum" ) { 
        auto const rho_bg = get_param<double>("grmhd","vacuum","rho_floor") ; 
        auto const press_bg = get_param<double>("grmhd","vacuum","press_floor") ; 
        auto const vx_bg = get_param<double>("grmhd","vacuum","velocity_x") ; 
        auto const vy_bg = get_param<double>("grmhd","vacuum","velocity_y") ; 
        auto const vz_bg = get_param<double>("grmhd","vacuum","velocity_z") ; 
        set_grmhd_initial_data_impl<eos_t, vacuum_id_t<eos_t>(
            rho_bg,press_bg,vx_bg,vy_bg,vz_bg
        ) ; 
    } else if( id_type == "shocktube" ) {
        auto pars = get_param<YAML::Node>("grmhd","shocktube") ; 
        auto const rho_L = pars["rho_L"].as<double>() ; 
        auto const rho_R = pars["rho_R"].as<double>() ; 
        auto const press_L = pars["press_L"].as<double>() ; 
        auto const press_R = pars["press_R"].as<double>()  ;
        auto const Bx_L = pars["Bx_L"].as<double>() ; 
        auto const Bx_R = pars["Bx_R"].as<double>()  ;
        auto const By_L = pars["By_L"].as<double>() ; 
        auto const By_R = pars["By_R"].as<double>()  ;
        auto const Bz_L = pars["Bz_L"].as<double>() ; 
        auto const Bz_R = pars["Bz_R"].as<double>()  ;
        set_grmhd_initial_data_impl<eos_t,shocktube_id_t<eos_t>>(
            rho_L, rho_R, 
            press_L, press_R, 
            Bx_L, Bx_R,
            By_L, By_R,
            Bz_L, Bz_R
        ) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with hydro ID.") ;  
    } else if ( id_type == "blastwave" ) {
        set_grmhd_spherical_blastwave_initial_data<eos_t>() ; 
    } else if ( id_type == "TOV") { 
        auto const rho_c = get_param<double>("grmhd", "TOV_central_density") ; 
        auto atmo_pars = get_param<YAML::Node>("grmhd","atmosphere"); 

        atmo_params_t atmo_params ;
        atmo_params.rho_fl = atmo_pars["rho_fl"].as<double>() ; 
        atmo_params.ye_fl = atmo_pars["ye_fl"].as<double>() ; 
        atmo_params.temp_fl = atmo_pars["temp_fl"].as<double>() ; 
        atmo_params.rho_fl_scaling = atmo_pars["rho_scaling"].as<double>() ;
        atmo_params.temp_fl_scaling = atmo_pars["temp_scaling"].as<double>() ; 

        set_grmhd_initial_data_impl<eos_t,tov_id_t<eos_t>>(atmo_params,rho_c) ;
    } else if ( id_type == "KHI") {
        set_grmhd_initial_data_impl<eos_t, kelvin_helmholtz_id_t<eos_t>>() ; 
    } else if( id_type == "magnetic_rotor" ) {
        auto pars = get_param<YAML::Node>("grmhd","magnetic_rotor") ; 
        auto const rho_in  = pars["rho_in"].as<double>() ; 
        auto const rho_out = pars["rho_out"].as<double>() ; 
        auto const press   = pars["press"].as<double>() ; 
        auto const B0      = pars["B0"].as<double>() ; 
        set_grmhd_initial_data_impl<eos_t,magnetic_rotor_id_t<eos_t>>(rho_in, rho_out, press, B0) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with Magnetized Rotor MHD ID.") ;  
    } else if ( id_type == "orszag_tang_vortex") {
        auto pars = get_param<YAML::Node>("grmhd","orszag_tang_vortex") ;
        auto const rho  = pars["rho"].as<double>() ; 
        auto const press = pars["press"].as<double>() ;  
        set_grmhd_initial_data_impl<eos_t,orszag_tang_vortex_mhd_id_t<eos_t>>(rho, press) ;
        Kokkos::fence() ; 
        GRACE_TRACE("Done with Orszag-Tang MHD ID.") ;  
    } else if ( id_type == "fmtorus") {
        auto pars = get_param<YAML::Node>("grmhd","fmtorus") ;
        auto a_BH = pars["a_BH"].as<double>() ; 
        auto rho_min = pars["rho_min"].as<double>() ; 
        auto lapse_min = pars["lapse_min"].as<double>() ; 
        auto press_min = pars["press_min"].as<double>() ;
        auto r_in = pars["r_in"].as<double>() ; 
        auto r_at_max_rho = pars["r_at_max_density"].as<double>() ; 
        auto gamma = pars["gamma"].as<double>() ; 
        auto rho_pow = pars["rho_power"].as<double>() ; 
        auto press_pow = pars["press_power"].as<double>() ; 
        torus_params_t torus ;
        torus.spin = a_BH ; 
        torus.gamma_adi = gamma;
        torus.prograde = true ;
        torus.r_edge = r_in ; 
        torus.r_peak = r_at_max_rho ; 
        torus.rho_max = grace::get_param<double>("grmhd","fmtorus","rho_max") ; 
        torus.psi = 0.0 ; 
        torus.is_vertical_field = false ;
        torus.fm_torus = true ; 
        torus.chakrabarti_torus = false ;

        torus.rho_min = rho_min ; 
        torus.rho_pow = rho_pow ; 

        torus.pgas_min = press_min ; 
        torus.pgas_pow = press_pow ;

        torus.lapse_excision = lapse_min ; 
        torus.rho_excise = grace::get_param<double>("grmhd","excision","rho_excision") ;
        double const temp_excise = grace::get_param<double>("grmhd","excision","temp_excision") ; 
        torus.pgas_excise  =  temp_excise * torus.rho_excise ; 

        double pert = pars["perturbation_amplitude"].as<double>() ; 
        set_grmhd_initial_data_impl<eos_t,fmtorus_id_t<eos_t>>(torus,pert) ;
        double const P_max   = get_max_press() ;
        GRACE_INFO("Pmax {}", P_max) ; 
        auto max_betam1 = pars["max_inverse_beta"].as<double>() ; 
        rescale_B_field(max_betam1, P_max) ; 
        GRACE_TRACE("Done with magnetized FMTorus ID.") ;
    } else if (id_type == "khi") { 
        set_grmhd_initial_data_impl<eos_t,kelvin_helmholtz_id_t<eos_t>>() ;
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
    auto& sstate = grace::variable_list::get().getstaggeredstate() ; 
    auto& idx     = grace::variable_list::get().getinvspacings() ;

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
        // note here we reset B-center since it is outdated 
        auto Bx = Kokkos::subview(sstate.face_staggered_fields_x,
                                 VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), BSX_, q) ; 
        auto By = Kokkos::subview(sstate.face_staggered_fields_y,
                                 VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), BSY_, q) ; 
        auto Bz = Kokkos::subview(sstate.face_staggered_fields_z,
                                 VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), BSZ_, q) ;
        aux(VEC(i,j,k),BX_,q) = 0.5 * (sstate.face_staggered_fields_x(VEC(i,j,k),BSX_,q) + sstate.face_staggered_fields_x(VEC(i+1,j,k),BSX_,q)) / metric.sqrtg();
        aux(VEC(i,j,k),BY_,q) = 0.5 * (sstate.face_staggered_fields_y(VEC(i,j,k),BSY_,q) + sstate.face_staggered_fields_y(VEC(i,j+1,k),BSY_,q)) / metric.sqrtg();
        aux(VEC(i,j,k),BZ_,q) = 0.5 * (sstate.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) + sstate.face_staggered_fields_z(VEC(i,j,k+1),BSZ_,q)) / metric.sqrtg();
        aux(VEC(i,j,k),BDIV_,q) = ( (Bx(VEC(i+1,j,k)) - Bx(VEC(i,j,k))) * idx(0,q) 
                                  + (By(VEC(i,j+1,k)) - By(VEC(i,j,k))) * idx(1,q)
                                  + (Bz(VEC(i,j,k+1)) - Bz(VEC(i,j,k))) * idx(2,q))/metric.sqrtg() ; 
        

        grmhd_prims_array_t prims ; 
        FILL_PRIMS_ARRAY(prims,aux,q,VEC(i,j,k)) ;
        
        std::array<double,4> dummy; 
        compute_smallb(dummy, aux(VEC(i,j,k),SMALLB2_,q), prims,metric) ; 
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
