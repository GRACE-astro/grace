/**
 * @file test_scalar_reductions.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-22
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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
#include <grace/IO/vtk_output.hh>
#include <grace/IO/scalar_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Reductions", "[reductions]")
{
    using namespace grace::variables ; 
    using namespace grace ; 
    using namespace Kokkos ; 
    #if defined(GRACE_ENABLE_SCALAR_ADV) or defined(GRACE_ENABLED_BURGERS)
    int const DENS = U ; 
    int const DENS_ = U ; 
    #endif

    auto& state  = grace::variable_list::get().getstate()  ;
    auto& coord_system = grace::coordinate_system::get() ; 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    int ngz = grace::amr::get_n_ghosts() ; 
    auto ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ; 

    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    deep_copy(h_state_mirror,state) ; 

    auto const h_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        return EXPR(8.5 * x, - 5.1 * y, -2*z) - 3.14 ; 
    } ; 
    /*************************************************/
    /*                   fill data                   */
    /*     here we fill the ghost zones as well.     */
    /*************************************************/
    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
        size_t const i = icell%(nx+2*ngz) ; 
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
        double const sigma = 0.1 ;
        double const r = sqrt( EXPR(
            math::int_pow<2>(pcoords[0]), + math::int_pow<2>(pcoords[1]), + math::int_pow<2>(pcoords[2])
        )) ; 
        h_state_mirror(VEC(i,j,k),DENS,q) = exp(- 0.5 * math::int_pow<2>(r)/math::int_pow<2>(sigma)) / sigma / sqrt(2*M_PI) ; 
    }
    /* copy data to device */
    Kokkos::deep_copy(state,h_state_mirror); 
    /* Compute reductions  */
    IO::compute_reductions() ; 
    /* Check results */
    #if defined(GRACE_ENABLE_SCALAR_ADV) or defined(GRACE_ENABLED_BURGERS)
    std::string const vname = "U" ; 
    #else 
    std::string const vname = "dens" ; 
    #endif  
    double const umax = IO::detail::_minmax_reduction_vars_results[vname].max_val; 
    double const umin = IO::detail::_minmax_reduction_vars_results[vname].min_val;
    double const unorm =  IO::detail::_norm2_reduction_vars_results[vname] ; 
    double const uintegral = IO::detail::_integral_reduction_vars_results[vname] ; 

    #ifdef GRACE_3D 
    double constexpr umax_exact  = 3.845968535513439      ; 
    double constexpr unorm_exact = 53.888666613           ; 
    double constexpr uint_exact  = 0.0628318531           ; 
    #else 
    double constexpr umax_exact  = 3.8932041101     ; 
    double constexpr unorm_exact = 22.627416998     ; 
    double constexpr uint_exact  = 0.2506628275     ; 
    #endif 

    CHECK_THAT(umax, Catch::Matchers::WithinRel(umax_exact,1e-03)) ; 
    CHECK_THAT(umin, Catch::Matchers::WithinAbs(0.,1e-12)) ; 
    CHECK_THAT(unorm, Catch::Matchers::WithinRel(unorm_exact,1e-03)) ; 
    CHECK_THAT(uintegral, Catch::Matchers::WithinRel(uint_exact,1e-03)) ; 

}