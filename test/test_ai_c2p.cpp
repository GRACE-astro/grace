/**
 * @file test_c2p.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
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

#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/scalar_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/system/grace_system.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <grace/ML/ml_networks.hh>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>


#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>

#define N 100
#define DUMP_RESIDUAL_TO_FILE

static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
conservs_from_prims(grace::grmhd_cons_array_t& cons, grace::grmhd_prims_array_t& prims, grace::metric_array_t const& metric)
{
    double const v2 = metric.square_vec({prims[VXL],prims[VYL],prims[VZL]}) ; 
    double const alp_sqrtgamma = metric.alp() * metric.sqrtg() ;
    double const W  = 1./Kokkos::sqrt(1-v2) ; 
    double const u0 = W / metric.alp();
    cons[DENSL] = alp_sqrtgamma * u0 * prims[RHOL] ; 
    double const b2{0.}, smallbt{0.} ; 
    double const one_over_alp2 = 1./math::int_pow<2>(metric.alp());
    double const rho0_h_plus_b2 = (prims[RHOL]*(1+prims[EPSL])) + prims[PRESSL] + b2 ;
    double const alp2_sqrtgamma = math::int_pow<2>(metric.alp()) * metric.sqrtg() ;
    double const g4uptt = -one_over_alp2 ; 
    
    double const P_plus_half_b2 = (prims[PRESSL] + 0.5*b2);
    double const Tuptt = rho0_h_plus_b2 * math::int_pow<2>(u0) + P_plus_half_b2 * g4uptt - math::int_pow<2>(smallbt) ; 
    cons[TAUL] = alp2_sqrtgamma * Tuptt - cons[DENSL] ;

    std::array<double,4> smallb{0.,0.,0.,0.}, smallbD{0.,0.,0.,0.} ;
    /* After initialization this is the Eulerian 3-vel (not that it matters in Minkowski)*/ 
    auto vD = metric.lower({prims[VXL],prims[VYL],prims[VZL]}) ; 

    cons[STXL] = metric.sqrtg() * (rho0_h_plus_b2*math::int_pow<2>(W)*vD[0]-smallb[0]*smallbD[1]) ; 
    cons[STYL] = metric.sqrtg() * (rho0_h_plus_b2*math::int_pow<2>(W)*vD[1]-smallb[0]*smallbD[2]) ; 
    cons[STZL] = metric.sqrtg() * (rho0_h_plus_b2*math::int_pow<2>(W)*vD[2]-smallb[0]*smallbD[3]) ;
    cons[YESL] = prims[YEL] * cons[DENSL] ; 
    cons[ENTSL] = cons[DENSL] * prims[ENTL] ;
    return ; 
}


static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
compute_residual(grace::grmhd_prims_array_t const& new_prims, grace::grmhd_prims_array_t& old_prims)
{
    double err{0.} ; 
    std::array<unsigned int, 5> indices {RHOL, EPSL,VXL,VYL,VZL} ; 
    for( auto const i: indices ) {
        err += math::abs((new_prims[i]-old_prims[i])/(old_prims[i]+1e-50)) ; 
    }
    return 1./indices.size() * (err) ; 
}

static void GRACE_ALWAYS_INLINE
fill_primitive_views(Kokkos::View<double *> lrho, Kokkos::View<double *> ltemp, const double nx, const double ny) {
    double const start_logrho{-8};
    double const end_logrho{-2.7}   ; 
    double const start_logT{-1}   ; 
    double const end_logT{2.3}    ;
    double const dlrho{(end_logrho-start_logrho)/nx}, dlT{(end_logT-start_logT)/ny} ; 
    Kokkos::parallel_for("fill_views", N, 
    KOKKOS_LAMBDA(int const& i) {
        lrho(i) = start_logrho + static_cast<double>(i)*dlrho ; 
        ltemp(i) = start_logT + static_cast<double>(i)*dlT ; 
    }) ; 
}

// Function to generate a random unit vector of 3 components
static void generateRandomUnitVector(double &x, double &y, double &z) {
    // Seed the random number generator
    std::srand(std::time(0));

    // Generate random angles for spherical coordinates
    double theta = ((double) std::rand() / RAND_MAX) * 2.0 * M_PI; // Random angle theta in [0, 2*pi)
    double phi = ((double) std::rand() / RAND_MAX) * M_PI;         // Random angle phi in [0, pi)

    // Convert spherical coordinates to Cartesian coordinates
    x = std::sin(phi) * std::cos(theta);
    y = std::sin(phi) * std::sin(theta);
    z = std::cos(phi);
}

static void get_velocity_from_W(double const& W, Kokkos::View<double **> vel, const double ny) {
    double v_max = Kokkos::sqrt( math::int_pow<2>(W) -  1) / W; 
    auto h_vel = Kokkos::create_mirror_view(vel) ; 
    for( int i=0; i<ny;++i) { 
        generateRandomUnitVector(h_vel(i,0),h_vel(i,1),h_vel(i,2)) ; 
        //h_vel(i,j,0) = 0.; h_vel(i,j,1)=1.; h_vel(i,j,2) = 0.;
        const double v = v_max * i/(ny-1);
        for(int iv=0; iv<3; ++iv)
            h_vel(i,iv) *= v ; 
    }
    Kokkos::deep_copy(vel,h_vel) ; 
}

template<typename eos_t>
static void check_c2p(eos_t eos){
    using namespace grace ;
    metric_array_t minkowski_metric ({1.,0.,0.,1.,0.,1.},{0.,0.,0.},1.) ; 
        
    double const W = 2. ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nq = amr::get_local_num_quadrants() ;
    int nvars_hrsc = variables::get_n_hrsc() ;

    Kokkos::View<double *> d_logrho("logrho", nx) ; 
    Kokkos::View<double *> d_logT("logT", nz) ;
    Kokkos::View<double **> d_vel("vel", ny,3) ; 
    Kokkos::View<double ***> d_res("residual", nx,ny,nz) ;
    Kokkos::View<double ***> d_eps("eps", nx,ny,nz) ;
    Kokkos::View<double ***> d_press("press", nx,ny,nz) ;
    fill_primitive_views(d_logrho,d_logT, nx, ny) ; 
    get_velocity_from_W(W, d_vel, ny) ; 
    double const ye = 0.1 ; 
    
    // States on GPU, but Host functions
    var_array_t<GRACE_NSPACEDIM>& state = grace::variable_list::get().getstate() ;
    auto& ml_list = ml::ml_network_list::get(); 

    
    auto advance_policy = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+2>> (
              {VEC(0,0,0),0,0}
            , {VEC(nx,ny,nz),nvars_hrsc,nq}
        ) ; 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
                , advance_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& ivar, int const& q)
    {
        grmhd_prims_array_t prims ; 
        grmhd_prims_array_t new_prims ; 
        grmhd_cons_array_t cons ; 

        prims[RHOL] = Kokkos::pow(10.,d_logrho(i)) ; 
        prims[TEMPL] = Kokkos::pow(10.,d_logT(k))  ; 
        prims[YEL] = ye; 
        prims[VXL] = d_vel(j,0) ; 
        prims[VYL] = d_vel(j,1) ; 
        prims[VZL] = d_vel(j,2) ; 

        double csnd2 ;
        unsigned int err ;  
        double temp{0} ; 
        prims[PRESSL] = eos.press_eps_csnd2__temp_rho_ye(prims[EPSL],csnd2,prims[TEMPL],prims[RHOL],prims[YEL],err) ; 
        conservs_from_prims(cons,prims,minkowski_metric) ; 

        int const VEC(I{i+ngz},J{j+ngz},K{k+ngz}) ; 
        state(VEC(I,J,K),ivar,q) = cons[ivar];
    }) ; 

    //// AI - Addition
    auto& c2p_view = ml_list.get_c2p_device_view();
    auto& ml_model = c2p_view(0);  // Access the single network inside the view

    if (!ml_list.is_c2p_initialized()) {
        GRACE_WARN("C2P model not initialized!");
        return;
    }
    ml_model.fill_input(state);
    unsigned long ncells = (nx+2*ngz)*(ny+2*ngz)*(nz+2*ngz)*nq;
    ml_model.batched_forward(ncells);
    Kokkos::fence();
    //// ----------------

    //conservs_to_prims<eos_t>(cons,new_prims,minkowski_metric,eos,0.) ; 
}

TEST_CASE("c2p", "[c2p-hydro]") {
    Kokkos::printf("Starting Test \n");
    auto eos = grace::eos::get().get_hybrid_pwpoly() ; 
    check_c2p(eos) ; 
}    

