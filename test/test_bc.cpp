#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/amr/amr_ghosts.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/gridloop.hh>

#include <grace/IO/cell_output.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numeric>
#include <fstream>
#include <string>
#include <string>
#include <utility>
#include <stdexcept>

#define DBG_GHOSTZONE_TEST 

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

static inline bool is_outside_grid(VEC(size_t i,size_t j, size_t k), int64_t q, VEC(double xoff,double yoff, double zoff))
{
    auto params = grace::config_parser::get()["amr"] ; 
    auto pcoords = grace::get_physical_coordinates({VEC(i,j,k)},q,{VEC(xoff,yoff,zoff)}, true) ;
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

static inline bool is_affected_by_boundary(
    VEC(size_t i,size_t j, size_t k), int64_t q, int offset, VEC(double xoff, double yoff, double zoff)
)
{
    return is_outside_grid(VEC(i,j,k),q,VEC(xoff,yoff,zoff)) 
           or is_outside_grid(VEC(i+offset,j,k),q,VEC(xoff,yoff,zoff))  
           or is_outside_grid(VEC(i-offset,j,k),q,VEC(xoff,yoff,zoff)) 
           or is_outside_grid(VEC(i,j+offset,k),q,VEC(xoff,yoff,zoff)) 
           or is_outside_grid(VEC(i,j-offset,k),q,VEC(xoff,yoff,zoff)) 
           #ifdef GRACE_3D 
           or is_outside_grid(VEC(i,j,k+offset),q,VEC(xoff,yoff,zoff)) 
           or is_outside_grid(VEC(i,j,k-offset),q,VEC(xoff,yoff,zoff)) 
           #endif   
    ;
}

static inline bool is_corner_ghostzone(VEC(long i, long j, long k), VEC(long nx, long ny, long nz), int ngz)
{
    return (EXPR((i<ngz) + (i>nx+ngz-1), + (j<ngz) + (j>ny+ngz-1), + (k<ngz) + (k>nz+ngz-1))) == GRACE_NSPACEDIM ; 
}

static inline bool is_edge_ghostzone(VEC(long i, long j, long k), VEC(long nx, long ny, long nz), int ngz)
{
    return (EXPR((i<ngz) + (i>nx+ngz-1), + (j<ngz) + (j>ny+ngz-1), + (k<ngz) + (k>nz+ngz-1))) == 2 ; 
}

static inline bool is_ghostzone(VEC(int i, int j, int k), VEC(int nx, int ny, int nz), int ngz)
{
    return (EXPR((i<ngz) + (i>nx+ngz-1), + (j<ngz) + (j>ny+ngz-1), + (k<ngz) + (k>nz+ngz-1))) > 0 ; 
}

static inline std::string 
elem_kind(size_t i, size_t j, size_t k, size_t n, size_t g)
{
    if ( ! is_ghostzone(i,j,k,n,n,n,g) ) {
        return "interior" ; 
    }

    int nn = (EXPR((i<g) + (i>n+g-1), + (j<g) + (j>n+g-1), + (k<g) + (k>n+g-1))) ; 
    if ( nn == 3 ) {
        return "corner" ;
    } else if ( nn == 2 ) {
        return "edge" ; 
    } else if (nn==1) {
        return "face" ;
    } else {
        return "interior" ; 
    }
    
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




static void collect_info(
    std::vector<grace::quad_neighbors_descriptor_t> const& ghost_array
)
{
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    size_t ngz = static_cast<size_t>(grace::amr::get_n_ghosts()) ; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    // collect some info 
    for( int q=0; q<nq; ++q){
        for( int f=0; f<P4EST_FACES; ++f) {
            auto& face = ghost_array[q].faces[f] ; 
            if (face.level_diff == grace::FINER) GRACE_TRACE("Coarse face {} {}", q, f) ; 
            if (face.level_diff == grace::COARSER ) {
                if (face.data.full.is_remote ) GRACE_TRACE("Remote fine face {} {}", q, f) ; 
            } 
        }
        for( int e=0; e<12; ++e) {
            auto& edge = ghost_array[q].edges[e];
            if (!edge.filled) GRACE_TRACE("Virtual edge {} {}", q, e) ; 
            if (edge.level_diff == grace::FINER) GRACE_TRACE("Coarse edge {} {}", q, e) ; 
            if (edge.level_diff == grace::COARSER ) {
                if (edge.data.full.is_remote ) GRACE_TRACE("Remote fine edge {} {}", q, e) ; 
            } 
        }
        for( int c=0; c<P4EST_CHILDREN; ++c) {
            auto& corner = ghost_array[q].corners[c];
            if (!corner.filled) GRACE_TRACE("Virtual corner {} {}", q, c) ; 
            if (corner.level_diff == grace::FINER) GRACE_TRACE("Coarse corner {} {}", q, c) ;
            if (corner.level_diff == grace::COARSER ) {
                if (corner.data.is_remote ) GRACE_TRACE("Remote fine corner {} {}", q, c) ; 
            }
        }     
    }

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
            if (
            #if 1
                ! is_corner_ghostzone(
                VEC(i,j,k), VEC(nx,ny,nz), ngz
            ) 
            and 
            ! is_edge_ghostzone(
                VEC(i,j,k), VEC(nx,ny,nz), ngz
            ) 
            and
            #endif  
            ! is_affected_by_boundary(VEC(i,j,k),q,2,VEC(0.5,0.5,0.5))){
                if ( std::isnan(host_data(VEC(i,j,k),0,q)) or (fabs(host_data(VEC(i,j,k),0,q)-ground_truth(VEC(i,j,k),0,q))>1e-14)) {
                    auto quad = grace::amr::get_quadrant(0, q).get() ; 
                    GRACE_TRACE("NaN at {}, level {} ijk {},{},{}, q {}", elem_kind(i,j,k,nx,ngz), static_cast<int>(quad->level),i,j,k,q) ;
                }
                
                CHECK_THAT(
                host_data(VEC(i,j,k),0,q),
                Catch::Matchers::WithinAbs(ground_truth(VEC(i,j,k),0,q),
                    1e-14 ) ) ; 
            }
            
        }, {VEC(false,false,false)}, true 
    ) ; 
}

TEST_CASE("Apply BC", "[boundaries]")
{
    using namespace grace ; 
    auto& ghost = grace::amr_ghosts::get() ; 
    //ghost.update() ; 
    
    auto& runtime = ghost.get_task_executor() ; 
    // now the real test 
    auto& state = grace::variable_list::get().getstate() ; 
    auto state_mirror = Kokkos::create_mirror_view(state) ; 

    /*************************************************/
    /*                     ID                        */
    /*************************************************/
    setup_initial_data(state_mirror) ; 
    Kokkos::deep_copy(state, state_mirror) ; 

    /*************************************************/
    /*                   Regrid                      */
    /*************************************************/
    bool do_regrid = grace::get_param<bool>("amr","do_regrid_test") ; 
    if( do_regrid ) {
        grace::amr::regrid() ;
        // size has changed
        state_mirror = Kokkos::create_mirror_view(state) ; 
        // reset ground truth 
        setup_initial_data(state_mirror) ; 
        Kokkos::deep_copy(state, state_mirror) ; 
    }
    collect_info(ghost.get_ghost_layer()) ; 
    invalidate_ghostzones(state) ; 
    view_alias_t alias{&state} ;
    runtime.run(alias) ; 
    auto state_mirror_2 = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(state_mirror_2, state) ; 
    check_ghostzones(state_mirror_2, state_mirror) ; 
}