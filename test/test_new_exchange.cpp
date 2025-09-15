/**
 * @file test_new_exchange.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2025-09-09
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
#include <grace/amr/amr_ghosts.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/gridloop.hh>
#include <grace/IO/vtk_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <grace/data_structures/variable_utils.hh>

#include <grace/utils/task_queue.hh>

#include <string>

inline double fill_func(std::array<double,GRACE_NSPACEDIM> const& c)
{
    double const x = c[0] ; 
    double const y = c[1] ; 
    #ifdef GRACE_3D 
    double const z = c[2] ; 
    #else 
    double const z = 0 ;
    #endif  
    return x - 3.14 * y + 1.1 * z - 2.22 ; 
}

static inline bool is_ghostzone(VEC(int i, int j, int k), VEC(size_t nx, size_t ny, size_t nz), size_t ngz)
{
    return (EXPR((i<ngz) or (i>nx+ngz-1), or (j<ngz) or (j>ny+ngz-1), or (k<ngz) or (k>nz+ngz-1))) ; 
}
static inline bool is_corner_ghostzone(VEC(int i, int j, int k), VEC(size_t nx, size_t ny, size_t nz), size_t ngz)
{
    return (EXPR((i<ngz) + (i>nx+ngz-1), + (j<ngz) + (j>ny+ngz-1), + (k<ngz) + (k>nz+ngz-1))) > 1 ; 
}
static inline bool is_outside_grid(VEC(size_t i,size_t j, size_t k), int64_t q)
{
    auto params = grace::config_parser::get()["amr"] ; 
    auto pcoords = grace::get_physical_coordinates({VEC(i,j,k)},q,{VEC(0.5,0.5,0.5)}, true) ;
    #ifdef GRACE_CARTESIAN_COORDINATES 
        double xmin = params["xmin"].as<double>() ;
        double ymin = params["ymin"].as<double>() ;
        double zmin = params["zmin"].as<double>() ;

        double xmax = params["xmax"].as<double>() ;
        double ymax = params["ymax"].as<double>() ;
        double zmax = params["zmax"].as<double>() ; 

        return (pcoords[0]<xmin) || (pcoords[0]>xmax) || pcoords[1]<ymin || pcoords[1]>ymax 
        #ifdef GRACE_3D 
        || (pcoords[2]<zmin) || (pcoords[2]>zmax)
        #endif 
        ;
    #else    
        auto const Ro = params["outer_region_radius"].as<double>() ;
        auto r2 = EXPR(
              math::int_pow<2>(pcoords[0]),
            + math::int_pow<2>(pcoords[1]),
            + math::int_pow<2>(pcoords[2])
        );

        return r2 > Ro*Ro ;
    #endif 
}

template< typename view_t >
static void setup_initial_data(
    view_t host_data 
) 
{
    auto& coord_system = grace::coordinate_system::get() ; 

    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto const itree = grace::amr::get_quadrant_owner(q) ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, {VEC(0.5,0.5,0.5)}, true 
            ) ; 
            host_data(VEC(i,j,k), 0, q) = fill_func(pcoords) ; 
        }, {VEC(false,false,false)}, true 
    ) ; 
}

template< typename view_t >
static void invalidate_ghostzones(
    view_t device_data
)
{
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    size_t ngz = static_cast<size_t>(grace::amr::get_n_ghosts()) ; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 

    auto host_data = Kokkos::create_mirror_view(device_data) ; 
    Kokkos::deep_copy(host_data, device_data) ; 
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            if (is_ghostzone(VEC(i,j,k),VEC(nx,ny,nz),ngz )) {
                host_data(VEC(i,j,k), 0, q) = std::numeric_limits<double>::quiet_NaN() ; 
            }
        }, {VEC(false,false,false)}, true 
    ) ; 
    Kokkos::deep_copy(device_data, host_data) ; 
}

template< typename view_t > 
static void check_ghostzones(
    view_t host_data, view_t ground_truth 
) 
{
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    size_t ngz = static_cast<size_t>(grace::amr::get_n_ghosts()) ; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 

    grace::host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            if ( ! is_corner_ghostzone(
                VEC(i,j,k), VEC(nx,ny,nz), ngz
            ) and ! is_outside_grid(VEC(i,j,k),q)){
                CHECK_THAT(
                host_data(VEC(i,j,k),0,q),
                Catch::Matchers::WithinAbs(ground_truth(VEC(i,j,k),0,q),
                    1e-12 ) ) ; 
            }
            
        }, {VEC(false,false,false)}, true 
    ) ; 
}

TEST_CASE("Unigrid exchange", "[unigrid]")
{
    using namespace grace ; 
    auto& ghost = grace::amr_ghosts::get() ; 
    ghost.update() ; 
    
    auto const & layer = ghost.get_ghost_layer() ; 
    auto rank = parallel::mpi_comm_rank() ; 
    auto nproc = parallel::mpi_comm_size() ;
    auto nq = grace::amr::get_local_num_quadrants() ; 
    std::cout << "Neighbor list updated" << std::endl ;
    size_t qid = 0 ;
    if ( rank == 0 ) {
    for( auto const& q: layer ) {
        std::cout << "quad-id " << qid << std::endl ;
        std::cout << "face neighbors: " << std::endl ;
        for( int i=0; i<P4EST_FACES; ++i) {
            std::cout << std::endl ; 
            auto face = q.faces[i] ; 
            std::string face_kind =  face.kind == grace::interface_kind_t::PHYS ? "phys bound" : "internal" ; 
            std::cout << "     " << i << " kind " << face_kind << '\n' ; 
            if( ! (face.kind == grace::interface_kind_t::PHYS) ){ 
                    std::cout << "     level diff " << (int) face.level_diff << '\n'
                        << "     neighbor quadid " << face.data.full.quad_id << '\n' 
                        << "     neighbor is remote " << face.data.full.is_remote << std::endl ;
                if (face.data.full.is_remote  ){
                    std::cout << "     owner rank id " << face.data.full.owner_rank << std::endl ;  
                }  
            }
        }
        qid ++ ; 
    }
    std::vector<std::size_t> send_rank_offsets, recv_rank_offsets ; //!< In # of elements
    std::vector<std::size_t> send_rank_sizes, recv_rank_sizes ; //!< In # of elements
    ghost.get_rank_offsets(send_rank_offsets, recv_rank_offsets) ; 
    ghost.get_rank_sizes(send_rank_sizes,recv_rank_sizes) ; 

    auto nq = amr::get_local_num_quadrants() ; 
    std::size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts() ; 
    std::size_t nvars = variables::get_n_evolved() ; 

    std::size_t face_size = nx*nx * ngz * nvars ; 

    std::cout << "Send/Recv buffer sizes and offsets per rank:\n";
    for (int r = 0; r < nproc; ++r) {
        std::cout << "  Rank " << r << ":\n"
                << "    send size   = " << send_rank_sizes[r] / face_size 
                << ", offset = " << send_rank_offsets[r] << "\n"
                << "    recv size   = " << recv_rank_sizes[r] / face_size 
                << ", offset = " << recv_rank_offsets[r] << "\n";
    }

    auto& task_list = ghost.get_task_list() ; 
    std::cout << "We have " << task_list.size() << " tasks." << std::endl ;
    }

    GRACE_TRACE("Rank {} send buffer {} recv buffer {}", rank, ghost.get_send_buf_size(), ghost.get_recv_buf_size()) ; 

    auto& runtime = ghost.get_task_executor() ; 
    // runtime.run() ; 


    // now the real test 
    auto& state = grace::variable_list::get().getstate() ; 
    auto state_mirror = Kokkos::create_mirror_view(state) ; 

    setup_initial_data(state_mirror) ; 

    Kokkos::deep_copy(state, state_mirror) ; 

    invalidate_ghostzones(state) ; 
    runtime.run() ; 
    auto state_mirror_2 = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(state_mirror_2, state) ; 
    check_ghostzones(state_mirror_2, state_mirror) ; 

}