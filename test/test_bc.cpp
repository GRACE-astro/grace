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
    double L = 32 ;
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
            aux_h(VEC(i,j,k),BX,q) = (bx(VEC(i+1,j,k),0,q) + bx(VEC(i,j,k),0,q))/2 ; 
            aux_h(VEC(i,j,k),BY,q) = (by(VEC(i,j+1,k),0,q) + by(VEC(i,j,k),0,q))/2 ; 
            aux_h(VEC(i,j,k),BZ,q) = (bz(VEC(i,j,k+1),0,q) + bz(VEC(i,j,k),0,q))/2 ;
            aux_h(VEC(i,j,k),BDIV,q) =   (bx(VEC(i+1,j,k),0,q) - bx(VEC(i,j,k),0,q)) * idx(0,q)
                                     + (by(VEC(i,j+1,k),0,q) - by(VEC(i,j,k),0,q)) * idx(1,q)
                                     + (bz(VEC(i,j,k+1),0,q) - bz(VEC(i,j,k),0,q)) * idx(2,q) ; 
        }, {false,false,false}, false ) ;

    deep_copy(aux,aux_h) ; 
}

template< grace::var_staggering_t stag, typename view_t>
static void setup_initial_data(
    view_t host_data 
) 
{
    using namespace grace ; 
    auto& coord_system = grace::coordinate_system::get() ; 
    Kokkos::fence() ; 
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
            for( int ivar=0; ivar<nvars; ++ivar) {
                if ( stag == STAG_CENTER ) {
                    host_data(VEC(i,j,k), ivar, q) = fill_func(pcoords) ; 
                } else if ( stag == STAG_FACEX ) {
                    host_data(VEC(i,j,k), ivar, q) = fill_func_stagger(pcoords,0) ; 
                } else if ( stag == STAG_FACEY ) {
                    host_data(VEC(i,j,k), ivar, q) = fill_func_stagger(pcoords,1) ; 
                } else if ( stag == STAG_FACEZ ) {
                    host_data(VEC(i,j,k), ivar, q) = fill_func_stagger(pcoords,2) ; 
                }
            }
        }, stagger, true 
    ) ; 
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
            Ax(i,j,k,q) = pcoords[0] * SQR(pcoords[2]-pcoords[1])  ; 
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
            Ay(i,j,k,q) = pcoords[1] * SQR(pcoords[0] + 2*pcoords[2]) ; 
        }, {true,false,true}, true 
    ) ;
    deep_copy(Ayd,Ay) ;
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.,0.,0.5} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            Az(i,j,k,q) = pcoords[2] * SQR(2*pcoords[0] - pcoords[1]); 
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

template< grace::var_staggering_t stag, typename view_t >
static void invalidate_ghostzones(
    view_t device_data
)
{
    using namespace grace ; 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    size_t ngz = static_cast<size_t>(grace::amr::get_n_ghosts()) ; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 

    std::array<bool,3> stagger {false,false,false}; 
    int nvars = device_data.extent(GRACE_NSPACEDIM); 
    if ( stag == STAG_FACEX ) { 
        stagger[0] = true ; 
        nx ++;  
    }
    if ( stag == STAG_FACEY ) {
        stagger[1] = true ; 
        ny ++ ; 
    }
    if ( stag == STAG_FACEZ ) {
        stagger[2] = true ;
        nz ++ ; 
    }; 

    auto host_data = Kokkos::create_mirror_view(device_data) ; 
    Kokkos::deep_copy(host_data, device_data) ; 
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            if (is_ghostzone(VEC(i,j,k),VEC(nx,ny,nz),ngz )) {
                host_data(VEC(i,j,k), 0, q) = std::numeric_limits<double>::quiet_NaN() ; 
            }
        }, stagger, true 
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
                if (face.data.full.is_remote ) GRACE_TRACE("Remote fine face {} {} {}", q, f, ghost_array[q].cbuf_id) ; 
            } 
        }
        for( int e=0; e<12; ++e) {
            auto& edge = ghost_array[q].edges[e];
            if (!edge.filled) GRACE_TRACE("Virtual edge {} {}", q, e) ; 
            if (edge.level_diff == grace::FINER) GRACE_TRACE("Coarse edge {} {}", q, e) ; 
            if (edge.level_diff == grace::COARSER ) {
                if (edge.data.full.is_remote ) GRACE_TRACE("Remote fine edge {} {} {}", q, e, ghost_array[q].cbuf_id) ; 
            } 
        }
        for( int c=0; c<P4EST_CHILDREN; ++c) {
            auto& corner = ghost_array[q].corners[c];
            if (!corner.filled) GRACE_TRACE("Virtual corner {} {}", q, c) ; 
            if (corner.level_diff == grace::FINER) GRACE_TRACE("Coarse corner {} {} {}", q, c, ghost_array[q].cbuf_id) ;
            if (corner.level_diff == grace::COARSER ) {
                if (corner.data.is_remote ) GRACE_TRACE("Remote fine corner {} {}", q, c) ; 
            }
        }     
    }

}

template<typename view_t> 
static void check_ghostzones(
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
            if ( ! is_ghostzone(VEC(i,j,k),VEC(nx,ny,nz),ngz)) return ; 
            if (
            #if 0
                ! is_corner_ghostzone(
                VEC(i,j,k), VEC(nx,ny,nz), ngz
            ) 
            and 
            ! is_edge_ghostzone(
                VEC(i,j,k), VEC(nx,ny,nz), ngz
            ) 
            and
            #endif  
            ! is_affected_by_boundary(VEC(i,j,k),q,2,lcoord[0],lcoord[1],lcoord[2])){
                auto pcoords = coord_system.get_physical_coordinates(
                    {VEC(i,j,k)}, q, lcoord, true 
                ) ;
                double ground_truth = fill_func(pcoords);
                if ( std::isnan(host_data(VEC(i,j,k),0,q)) or (fabs(host_data(VEC(i,j,k),0,q)-ground_truth)>1e-13)) {
                    auto quad = grace::amr::get_quadrant(q).get() ; 
                    GRACE_TRACE("NaN at {}, level {} ijk {},{},{}, q {}", elem_kind(i,j,k,nx,ngz), static_cast<int>(quad->level),i,j,k,q) ;
                }
                static constexpr double macheps = std::numeric_limits<double>::epsilon() ; 
                #if 0
                REQUIRE_THAT(
                    fabs(host_data(VEC(i,j,k),0,q)-ground_truth),
                    Catch::Matchers::WithinAbs(0.0,
                        2*macheps*fabs(ground_truth) + 1e-16 ) 
                ) ; 
                #endif 
                REQUIRE_THAT(
                    host_data(VEC(i,j,k),0,q),
                    Catch::Matchers::WithinULP(ground_truth, 4)
                );

                // compute divergence of B 
                double divB = (host_data_x(VEC(i+1,j,k),0,q) - host_data_x(VEC(i,j,k),0,q)) * idx(0,q)
                            + (host_data_y(VEC(i,j+1,k),0,q) - host_data_y(VEC(i,j,k),0,q)) * idx(1,q)
                            + (host_data_z(VEC(i,j,k+1),0,q) - host_data_z(VEC(i,j,k),0,q)) * idx(2,q) ; 
                //GRACE_TRACE("DivB problem ijk {},{},{}, q {}, divB {}, Bx {}",i,j,k,q,divB,host_data_x(VEC(i,j,k),0,q)) ;
                if ( fabs(divB) > 1e-14 or std::isnan(divB)) {
                    auto quad = grace::amr::get_quadrant(q).get() ; 
                    GRACE_TRACE("DivB problem at {}, level {} ijk {},{},{}, q {}, divB {}", elem_kind(i,j,k,nx,ngz), static_cast<int>(quad->level),i,j,k,q,divB) ;
                }
                REQUIRE( fabs(divB) < 1e-13 ) ; 
                double Bx = (host_data_x(VEC(i+1,j,k),0,q) + host_data_x(VEC(i,j,k),0,q)) * 0.5 ;
                
            }
            
        }, stagger, true 
    ) ; 
}

TEST_CASE("Apply BC", "[boundaries]")
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ; 
    Kokkos::fence() ; 
    auto& ghost = grace::amr_ghosts::get() ; 
    auto& runtime = ghost.get_task_executor() ; 
    // now the real test 
    auto& state = grace::variable_list::get().getstate() ; 
    auto& stag_state = grace::variable_list::get().getstaggeredstate() ; 
    auto state_mirror = Kokkos::create_mirror_view(state) ; 
    auto fx_mirror = Kokkos::create_mirror_view(stag_state.face_staggered_fields_x) ; 
    auto fy_mirror = Kokkos::create_mirror_view(stag_state.face_staggered_fields_y) ; 
    auto fz_mirror = Kokkos::create_mirror_view(stag_state.face_staggered_fields_z) ; 

    GRACE_TRACE("In BC: size of fx {} {} {} {} {}, fy {} {} {} {} {}, fz {} {} {} {} {}"
                , stag_state.face_staggered_fields_x.extent(0), stag_state.face_staggered_fields_x.extent(1), stag_state.face_staggered_fields_x.extent(2), stag_state.face_staggered_fields_x.extent(3), stag_state.face_staggered_fields_x.extent(4),
                stag_state.face_staggered_fields_y.extent(0), stag_state.face_staggered_fields_y.extent(1), stag_state.face_staggered_fields_y.extent(2), stag_state.face_staggered_fields_y.extent(3), stag_state.face_staggered_fields_y.extent(4),
                stag_state.face_staggered_fields_z.extent(0), stag_state.face_staggered_fields_z.extent(1), stag_state.face_staggered_fields_z.extent(2), stag_state.face_staggered_fields_z.extent(3), stag_state.face_staggered_fields_z.extent(4)) ;  

    /*************************************************/
    /*                     ID                        */
    /*************************************************/
    setup_initial_data<STAG_CENTER>(state_mirror) ; 
    Kokkos::deep_copy(state, state_mirror) ; 
    setup_initial_B_field(stag_state) ; 
    Kokkos::deep_copy(fx_mirror, stag_state.face_staggered_fields_x) ; 
    Kokkos::deep_copy(fy_mirror, stag_state.face_staggered_fields_y) ; 
    Kokkos::deep_copy(fz_mirror, stag_state.face_staggered_fields_z) ; 
    /*************************************************/
    /*                   Regrid                      */
    /*************************************************/
    bool do_regrid = grace::get_param<bool>("amr","do_regrid_test") ; 
    if( do_regrid ) {
        grace::amr::regrid() ;
        // size has changed
        state_mirror = Kokkos::create_mirror_view(state) ; 
        fx_mirror = Kokkos::create_mirror_view(stag_state.face_staggered_fields_x) ; 
        fy_mirror = Kokkos::create_mirror_view(stag_state.face_staggered_fields_y) ; 
        fz_mirror = Kokkos::create_mirror_view(stag_state.face_staggered_fields_z) ; 
    }
    auto& layer = ghost.get_ghost_layer() ; 
    
    fill_b_field() ; 
    grace::IO::write_cell_output(true,false,false) ; 
    invalidate_ghostzones<STAG_CENTER>(state) ; 
    invalidate_ghostzones<STAG_FACEX>(stag_state.face_staggered_fields_x) ; 
    invalidate_ghostzones<STAG_FACEY>(stag_state.face_staggered_fields_y) ; 
    invalidate_ghostzones<STAG_FACEZ>(stag_state.face_staggered_fields_z) ; 

    GRACE_VERBOSE("Filling ghostzones") ;
    view_alias_t alias{&state,&state,&stag_state,&stag_state,1.0,1.0} ;
    runtime.run(alias) ; 
    GRACE_VERBOSE("Done filling ghostzones") ;

    auto state_mirror_2 = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(state_mirror_2, state) ; 
    auto fx_mirror_2 = Kokkos::create_mirror_view(stag_state.face_staggered_fields_x) ; 
    Kokkos::deep_copy(fx_mirror_2, stag_state.face_staggered_fields_x) ; 
    auto fy_mirror_2 = Kokkos::create_mirror_view(stag_state.face_staggered_fields_y) ; 
    Kokkos::deep_copy(fy_mirror_2, stag_state.face_staggered_fields_y) ; 
    auto fz_mirror_2 = Kokkos::create_mirror_view(stag_state.face_staggered_fields_z) ; 
    Kokkos::deep_copy(fz_mirror_2, stag_state.face_staggered_fields_z) ; 

    GRACE_VERBOSE("Checking ghostzones") ; 
    check_ghostzones( state_mirror_2
                    , fx_mirror_2
                    , fy_mirror_2
                    , fz_mirror_2 ) ;
}
