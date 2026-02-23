/**
 * @file test_regridding.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-04-12
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
#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/gridloop.hh>
#include <grace/IO/cell_output.hh>

#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

inline double fill_func(std::array<double,GRACE_NSPACEDIM> const& c)
{
    double const x = c[0] ; 
    double const y = c[1] ; 
    #ifdef GRACE_3D 
    double const z = c[2] ; 
    #else 
    double const z = 0 ;
    #endif  
    return x - 3.14 * y + 11 * z - 2.22 ; 
}

inline double fill_func_stagger(std::array<double,GRACE_NSPACEDIM> const& c, int idir)
{
    double const x = c[0] ; 
    double const y = c[1] ; 
    #ifdef GRACE_3D 
    double const z = c[2] ; 
    #else 
    double const z = 0 ;
    #endif  
    double L = 2 ;
    double kx = 2 * M_PI * 1 / L ; 
    double ky = 2 * M_PI * 2 / L ; 
    double kz = 2 * M_PI * 3 / L ; 
    double A{0.7},B{1.1},C{0.9} ; 
    if ( idir == 0 ) {
        return A * sin(kz*z) + C * cos(ky*y) ; 
    } else if ( idir == 1 ) {
        return B * sin(kx*x) + A * cos(kz*z) ; 
    } else {
        return C * sin(ky*y) + B * cos(kx*x) ; 
    }
}

template< grace::var_staggering_t stag, typename view_t>
static void setup_initial_data(
    view_t host_data 
) 
{
    DECLARE_GRID_EXTENTS ;
    using namespace grace ; 
    auto& coord_system = grace::coordinate_system::get() ; 

    std::array<bool,3> stagger {false,false,false}; 
    std::array<double,3> lcoord {0.5,0.5,0.5} ; 
    int nvars = host_data.extent(GRACE_NSPACEDIM); 
    if ( stag == STAG_FACEX ) { 
        stagger[0] = true ; 
        lcoord[0] = 0 ; 

    }
    if ( stag == STAG_FACEY ) {
        stagger[1] = true ; 
        lcoord[1] = 0 ; 
    }
    if ( stag == STAG_FACEZ ) {
        stagger[2] = true ;
        lcoord[2] = 0 ; 
    }; 
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto const itree = grace::amr::get_quadrant_owner(q) ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            if ( stag == STAG_CENTER ) {
                host_data(VEC(i,j,k), DENS_, q) = fill_func(pcoords) ; 
            } else if ( stag == STAG_FACEX ) {
                host_data(VEC(i,j,k), 0, q) = fill_func_stagger(pcoords,0) ; 
            } else if ( stag == STAG_FACEY ) {
                host_data(VEC(i,j,k), 0, q) = fill_func_stagger(pcoords,1) ; 
            } else if ( stag == STAG_FACEZ ) {
                host_data(VEC(i,j,k), 0, q) = fill_func_stagger(pcoords,2) ; 
            }
            
        }, stagger, true 
    ) ; 

    
}

static inline bool is_ghostzone(VEC(int i, int j, int k), VEC(int nx, int ny, int nz), int ngz)
{
    return (EXPR((i<ngz) + (i>nx+ngz-1), + (j<ngz) + (j>ny+ngz-1), + (k<ngz) + (k>nz+ngz-1))) > 0 ; 
}

void fill_b_field() {
    DECLARE_GRID_EXTENTS;
    using namespace grace ; 
    using namespace Kokkos; 

    auto& aux = variable_list::get().getaux() ;
    auto aux_h = Kokkos::create_mirror_view(aux) ; 

    auto& sstate = variable_list::get().getstaggeredstate() ;
    auto bx = create_mirror_view(sstate.face_staggered_fields_x) ; 
    auto by = create_mirror_view(sstate.face_staggered_fields_y) ; 
    auto bz = create_mirror_view(sstate.face_staggered_fields_z) ;
    deep_copy(bx,sstate.face_staggered_fields_x) ; 
    deep_copy(by,sstate.face_staggered_fields_y) ; 
    deep_copy(bz,sstate.face_staggered_fields_z) ;
    auto idx_d = grace::variable_list::get().getinvspacings() ; 
    auto idx = Kokkos::create_mirror_view(idx_d) ; 
    Kokkos::deep_copy(idx,idx_d) ;
    grace::host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            aux_h(VEC(i,j,k),BX_,q) = (bx(VEC(i+1,j,k),0,q) + bx(VEC(i,j,k),0,q))/2 ; 
            aux_h(VEC(i,j,k),BY_,q) = (by(VEC(i,j+1,k),0,q) + by(VEC(i,j,k),0,q))/2 ; 
            aux_h(VEC(i,j,k),BZ_,q) = (bz(VEC(i,j,k+1),0,q) + bz(VEC(i,j,k),0,q))/2 ;
            aux_h(VEC(i,j,k),BDIV_,q) =   (bx(VEC(i+1,j,k),0,q) - bx(VEC(i,j,k),0,q)) * idx(0,q)
                                     + (by(VEC(i,j+1,k),0,q) - by(VEC(i,j,k),0,q)) * idx(1,q)
                                     + (bz(VEC(i,j,k+1),0,q) - bz(VEC(i,j,k),0,q)) * idx(2,q) ; 
        }, {false,false,false}, false ) ;

    deep_copy(aux,aux_h) ; 
}

template<typename view_t>
static void setup_initial_B_field(
    view_t stag_state  
) 
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ; 
    using namespace Kokkos ; 
    auto& coord_system = grace::coordinate_system::get() ; 
    Kokkos::fence() ; 

    View<double ****> Axd("Ax", nx + 2*ngz, ny+2*ngz +1, nz+2*ngz+1,nq) ; 
    View<double ****> Ayd("Ay", nx + 2*ngz+1, ny+2*ngz, nz+2*ngz+1,nq) ; 
    View<double ****> Azd("Az", nx + 2*ngz+1, ny+2*ngz +1, nz+2*ngz,nq) ; 
    auto Ax = create_mirror_view(Axd) ; 
    auto Ay = create_mirror_view(Ayd) ; 
    auto Az = create_mirror_view(Azd) ; 
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.5,0.,0.} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            auto rm3 = std::pow(pcoords[0] * pcoords[0] + pcoords[1] * pcoords[1] + pcoords[2] * pcoords[2],-3/2) ; 
            Ax(i,j,k,q) = -pcoords[1] * rm3 ; 
        }, {false,true,true}, true 
    ) ; 
    deep_copy(Axd,Ax) ;
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.,0.5,0.} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            auto rm3 = std::pow(pcoords[0] * pcoords[0] + pcoords[1] * pcoords[1] + pcoords[2] * pcoords[2],-3/2) ; 
            Ay(i,j,k,q) = pcoords[0] * rm3 ; 
        }, {true,false,true}, true 
    ) ;
    deep_copy(Ayd,Ay) ;
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.,0.,0.5} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            Az(i,j,k,q) = 0 ; 
        }, {true,true,false}, true 
    ) ;
    deep_copy(Azd,Az) ;

    auto& idx = variable_list::get().getinvspacings() ; 
    parallel_for(
        "fill_bx", 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz),nq}),
        KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        // B^x = d/dy A^z - d/dz A^y
                        stag_state.face_staggered_fields_x(VEC(i,j,k),0,q) = (
                            (Azd(VEC(i  ,j+1,k  ),q) - Azd(VEC(i  ,j  ,k  ),q)) * idx(1,q)
                          + (Ayd(VEC(i  ,j  ,k  ),q) - Ayd(VEC(i  ,j  ,k+1),q)) * idx(2,q)
                        ) ; 
                    }
    );

    parallel_for(
        "fill_by", 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz),nq}),
        KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    {
                        // B^x = d/dy A^z - d/dz A^y
                        stag_state.face_staggered_fields_y(VEC(i,j,k),0,q) = (
                            (Axd(VEC(i  ,j  ,k+1),q) - Axd(VEC(i  ,j  ,k  ),q)) * idx(2,q)
                          + (Azd(VEC(i  ,j  ,k  ),q) - Azd(VEC(i+1,j  ,k  ),q)) * idx(0,q)
                        ) ; 
                    }
    );

    parallel_for(
        "fill_bz", 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz+1),nq}),
        KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                    { 
                        // B^x = d/dy A^z - d/dz A^y
                        stag_state.face_staggered_fields_z(VEC(i,j,k),0,q) = (
                            (Ayd(VEC(i+1,j  ,k  ),q) - Ayd(VEC(i  ,j  ,k  ),q)) * idx(0,q)
                          + (Axd(VEC(i  ,j  ,k  ),q) - Axd(VEC(i  ,j+1,k  ),q)) * idx(1,q)
                        ) ;
                    }
    );
            

    
}

template<typename view_t> 
static void check(
      view_t host_data
    , view_t host_data_x
    , view_t host_data_y
    , view_t host_data_z
) 
{
    using namespace grace ; 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    size_t ngz = static_cast<size_t>(grace::amr::get_n_ghosts()) ; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 

    auto& coord_system = grace::coordinate_system::get() ; 
    std::array<bool,3> stagger {false,false,false}; 
    std::array<double,3> lcoord {0.5,0.5,0.5} ; 
    int nvars = host_data.extent(GRACE_NSPACEDIM);     

    auto idx_d = grace::variable_list::get().getinvspacings() ; 
    auto idx = Kokkos::create_mirror_view(idx_d) ; 
    Kokkos::deep_copy(idx,idx_d) ; 

    grace::host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            if (! is_ghostzone(i,j,k,nx,ny,nz,ngz))
            {
                auto pcoords = coord_system.get_physical_coordinates(
                    {VEC(i,j,k)}, q, lcoord, true 
                ) ;
                double ground_truth = fill_func(pcoords);
                if ( std::isnan(host_data(VEC(i,j,k),DENS_,q)) or (fabs(host_data(VEC(i,j,k),DENS_,q)-ground_truth)>1e-13)) {
                    auto quad = grace::amr::get_quadrant(q).get() ; 
                    GRACE_TRACE("NaN, level {} ijk {},{},{}, q {}, ground_truth {} val {}", static_cast<int>(quad->level),i,j,k,q, ground_truth, host_data(VEC(i,j,k),DENS_,q)) ;
                }
                
                REQUIRE_THAT(
                host_data(VEC(i,j,k),DENS_,q),
                Catch::Matchers::WithinAbs(ground_truth,
                    1e-13 ) ) ; 
                
                // compute divergence of B 
                double divB = (host_data_x(VEC(i+1,j,k),0,q) - host_data_x(VEC(i,j,k),0,q)) * idx(0,q)
                            + (host_data_y(VEC(i,j+1,k),0,q) - host_data_y(VEC(i,j,k),0,q)) * idx(1,q)
                            + (host_data_z(VEC(i,j,k+1),0,q) - host_data_z(VEC(i,j,k),0,q)) * idx(2,q) ; 
                REQUIRE( fabs(divB) < 1e-14 ) ; 
                
            }
            
        }, stagger, false 
    ) ; 
}

TEST_CASE("Simple regrid", "[regrid]")
{
    using namespace grace ;
    using namespace grace::variables ; 

    Kokkos::fence() ; 

    auto params = grace::config_parser::get()["amr"] ; 
    params["refinement_criterion_var"] = "dens" ; 
    
    auto& coords = grace::variable_list::get().getcoords() ; 
    auto& dx     = grace::variable_list::get().getspacings(); 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    int ngz = grace::amr::get_n_ghosts() ; 
    
    auto& coord_system = grace::coordinate_system::get() ; 

    auto& state = grace::variable_list::get().getstate() ; 
    auto& stag_state = grace::variable_list::get().getstaggeredstate() ; 
    auto state_mirror = Kokkos::create_mirror_view(state) ; 
    auto fx_mirror = Kokkos::create_mirror_view(stag_state.face_staggered_fields_x) ; 
    auto fy_mirror = Kokkos::create_mirror_view(stag_state.face_staggered_fields_y) ; 
    auto fz_mirror = Kokkos::create_mirror_view(stag_state.face_staggered_fields_z) ; 

    /*************************************************/
    /*                     ID                        */
    /*************************************************/
    setup_initial_data<STAG_CENTER>(state_mirror) ; 
    Kokkos::deep_copy(state, state_mirror) ; 
    setup_initial_B_field(stag_state) ; 
    Kokkos::deep_copy(fx_mirror, stag_state.face_staggered_fields_x) ; 
    Kokkos::deep_copy(fy_mirror, stag_state.face_staggered_fields_y) ; 
    Kokkos::deep_copy(fz_mirror, stag_state.face_staggered_fields_z) ; 
    fill_b_field() ; 
    /*write output and regrid*/
    grace::IO::write_cell_output(true,false,false) ; 
    grace::amr::regrid() ;  
    Kokkos::fence() ; 
    grace::runtime::get().increment_iteration() ;
    grace::runtime::get().set_simulation_time(1) ; 
    fill_b_field() ; 
    grace::IO::write_cell_output(true,true,true) ; 
    
    auto state_mirror_2 = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(state_mirror_2, state) ; 
    
    auto fx_mirror_2 = Kokkos::create_mirror_view(stag_state.face_staggered_fields_x) ; 
    Kokkos::deep_copy(fx_mirror_2, stag_state.face_staggered_fields_x) ; 
    auto fy_mirror_2 = Kokkos::create_mirror_view(stag_state.face_staggered_fields_y) ; 
    Kokkos::deep_copy(fy_mirror_2, stag_state.face_staggered_fields_y) ; 
    auto fz_mirror_2 = Kokkos::create_mirror_view(stag_state.face_staggered_fields_z) ; 
    Kokkos::deep_copy(fz_mirror_2, stag_state.face_staggered_fields_z) ; 
    check( state_mirror_2 
         , fx_mirror_2 
         , fy_mirror_2
         , fz_mirror_2 ) ;
    
}
