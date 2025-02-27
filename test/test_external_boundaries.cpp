#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/numerics/gridloop.hh>

#include <grace/IO/cell_output.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numeric>

static inline bool is_ghostzone(VEC(int i, int j, int k), VEC(int nx, int ny, int nz), int ngz)
{
    return (EXPR((i<ngz) + (i>nx+ngz-1), + (j<ngz) + (j>ny+ngz-1), + (k<ngz) + (k>nz+ngz-1))) > 0 ; 
}


/***********************************************/
// Note to reader: 
// In this test we EXPLICITLY 
// assume that the grid consists of a 
// single quadrant. This test is purely 
// designed to check the logic of extrapolating 
// external boundaries, and it should only 
// be ran with the accompanying parameter file.
/***********************************************/

TEST_CASE("External boundaries", "[external_boundaries]") 
{
    using namespace grace::variables ; 
    using namespace grace ; 

    DECLARE_GRID_EXTENTS ; 

    /*************************************************/
    /*                Fetch arrays                   */
    /*************************************************/
    auto& state  = grace::variable_list::get().getstate()  ;
    auto& sstate  = grace::variable_list::get().getstaggeredstate()  ;
    auto& coord_system = grace::coordinate_system::get() ;
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    auto h_corner_mirror = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 

    double const dx = 0.1 ;  
    double const x_max{1-dx/2}, x_min{-1+dx/2}, y_max{1-dx/2}, y_min{-1+dx/2}, z_max{1-dx/2}, z_min{-1+dx/2} ; 
    /*************************************************/
    /*            Define filling func                */
    /*************************************************/
    auto const h_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        #if 1
        return EXPR(8.5 * x, - 5.1 * y, + 2*z) - 3.14 ; 
        #else 
        auto const r2 = x*x+y*y+z*z ; 
        return 1.-Kokkos::fabs(x) ; 
        #endif 
    } ;
    // This function returns the "correct" value in the boundaries
    auto const h_func_bnd = [&] (VEC(const double& x, const double&y, const double& z)) {
        double xx = std::max(std::min(x_max,x), x_min) ; 
        double yy = std::max(std::min(y_max,y), y_min) ; 
        #ifdef GRACE_3D
        double zz = std::max(std::min(z_max,z), z_min) ; 
        #endif 
        return EXPR(8.5 * xx, - 5.1 * yy, + 2*zz) - 3.14 ; 
    } ; 
    /*************************************************/
    /*                   fill data                   */
    /*     here we fill the ghost zones as well.     */
    /*************************************************/
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                true
            ) ;
            h_state_mirror(VEC(i,j,k),DENS,q) = 
                h_func(VEC(pcoords[0],pcoords[1],pcoords[2])) ;
        },
        {VEC(false,false,false)},
        true
    ) ;
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                {VEC(0,0,0)}, 
                true
            ) ;
            h_corner_mirror(VEC(i,j,k),DENS,q) = 
                h_func(VEC(pcoords[0],pcoords[1],pcoords[2])) ;
        },
        {VEC(true,true,true)},
        true
    ) ;
    /*************************************************/
    /*                 Copy H2D                      */
    /*************************************************/
    Kokkos::deep_copy(state,h_state_mirror); 
    Kokkos::deep_copy(sstate.corner_staggered_fields,h_corner_mirror); 
    auto& swap = grace::variable_list::get().getscratch() ; 
    auto& sswap = grace::variable_list::get().getstaggeredscratch() ; 
    Kokkos::deep_copy(swap, state) ; 
    Kokkos::deep_copy(sswap.corner_staggered_fields, sstate.corner_staggered_fields) ;
    /*************************************************/
    /* Set ghostzone values to NaN before filling    */
    /*************************************************/
    h_state_mirror = Kokkos::create_mirror_view(state) ; 
    h_corner_mirror = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ;
    Kokkos::deep_copy(h_state_mirror, state) ; 
    Kokkos::deep_copy(h_corner_mirror, sstate.corner_staggered_fields) ;
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            
            if(   is_ghostzone(VEC(i,j,k),VEC(nx,ny,nz),ngz) ) 
            {
            h_state_mirror(VEC(i,j,k),DENS,q) = 
                std::numeric_limits<double>::quiet_NaN();
            }
        },
        {VEC(false,false,false)},
        true
    ) ;
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {

            if(   is_ghostzone(VEC(i,j,k),VEC(nx+1,ny+1,nz+1),ngz) ) 
            {
                h_corner_mirror(VEC(i,j,k),DENS,q) = 
                    std::numeric_limits<double>::quiet_NaN();
            }
        },
        {VEC(true,true,true)},
        true
    ) ;
    /*************************************************/
    /*                 Copy H2D                      */
    /*************************************************/
    Kokkos::deep_copy(state,h_state_mirror); 
    Kokkos::deep_copy(sstate.corner_staggered_fields,h_corner_mirror); 

    /*************************************************/
    /*                 Apply BCs                     */
    /*************************************************/
    grace::IO::write_cell_output(true,true,true) ; 
    grace::amr::apply_boundary_conditions() ; 

    /*************************************************/
    /*                 Copy D2H                      */
    /*************************************************/
    auto h_state_mirror_new = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state_mirror_new,state); 
    auto h_corner_mirror_new = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 
    Kokkos::deep_copy(h_corner_mirror_new,sstate.corner_staggered_fields);  

    /*************************************************/
    /*                   CHECK                       */
    /*************************************************/
    // here we do a loop over all ghostzones, cause 
    // all of them are external. 
    // First we define some shorthand 
    // for the directions in which ghostzones
    // run onto each face/edge/corner:
    // 
    // face_dir [iface][idir] == 1 
    // means that ghostzones of face face_id run into positive index direction
    // in coordinate direction idir. 
    // 
    // face_dir [iface][idir] == -1 
    // means that ghostzones of face face_id run into negative index direction 
    // in coordinate direction idir. 
    // 
    // face_dir [iface][idir] == 0 
    // means that idir is not a ghostzone direction for face iface.

    int face_dir[6][3] = {
        {-1,0,0}, // FACE 0 
        {1,0,0} , // FACE 1 
        {0,-1,0}, // FACE 2 
        {0,1,0} , // FACE 3 
        {0,0,-1}, // FACE 4 
        {0,0,1}  // FACE 5 
    } ; 

    int corner_dir[8][3] = {
        {-1,-1,-1}, // 0
        {1,-1,-1},  // 1 
        {-1,1,-1},  // 2 
        {1,1,-1},   // 3 
        {-1,-1,1},  // 4 
        {1,-1,1},   // 5
        {-1,1,1},   // 6 
        {1,1,1}     // 7
    } ; 

    int edge_dir[12][3] {
        {0,-1,-1}, // 0
        {0,1,-1},  // 1
        {0,-1,1}, // 2 
        {0,1,1}, // 3 
        {-1,0,-1}, // 4 
        {1,0,-1}, // 5
        {-1,0,1}, // 6
        {1,0,1}, // 7
        {-1,-1,0}, // 8
        {1,-1,0}, // 9
        {-1,1,0}, // 10
        {1,1,0} // 11 
    } ; 

    
    for( int face=0; face<6; ++face) {
        unsigned long lbnd[3] ; 
        unsigned long ubnd[3] ; 
        unsigned long step[3] ; 
        for( int idir=0; idir<3; ++idir ) {
            if ( face_dir[face][idir] == -1 ) {
                lbnd[idir] = ngz - 1; 
                ubnd[idir] = -1 ; 
                step[idir] = -1 ;
            } else if ( face_dir[face][idir] == 1 ) {
                lbnd[idir] = nx + ngz ; 
                ubnd[idir] = nx + 2 * ngz ; 
                step[idir] = + 1  ;
            } else {
                lbnd[idir] = ngz ;
                ubnd[idir] = nx + ngz ; 
                step[idir] = +1 ; 
            }
        }
        int ii{0}, jj{0}, kk{0} ; 
        for( unsigned long ig=lbnd[0]; ig!=ubnd[0]; ig+=step[0], ii++ ) 
        for( unsigned long jg=lbnd[1]; jg!=ubnd[1]; jg+=step[1], jj++ )
        for( unsigned long kg=lbnd[2]; kg!=ubnd[2]; kg+=step[2], kk++ ) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(ig,jg,kg)},
                0,
                {VEC(0.5,0.5,0.5)}, 
                true
            ) ;            
            if ( std::fabs(h_state_mirror_new(VEC(ig,jg,kg),DENS,0)-h_func_bnd(pcoords[0],pcoords[1],pcoords[2])) > 1e-10 ) {
                std::cout << "Error found at (i,j,k): " << ig << " " << jg << " " << kg << std::endl 
                          << "Coordinates: " << pcoords[0] << " " << pcoords[1] << " " << pcoords[2] << std::endl ; 
                std::cout << "Mirrored coords: " <<  std::min(std::max(pcoords[0],x_min), x_max) << " " 
                                                 <<  std::min(std::max(pcoords[1],y_min), y_max) <<  " "
                                                 <<  std::min(std::max(pcoords[2],z_min), z_max) << std::endl ; 
            }
            REQUIRE_THAT(
                h_state_mirror_new(VEC(ig,jg,kg),DENS,0),
                Catch::Matchers::WithinAbs(h_func_bnd(pcoords[0],pcoords[1],pcoords[2]), 1e-10)
            ) ; 
        }
    }

    for( int corner=0; corner<8; ++corner) {
        unsigned long lbnd[3] ; 
        unsigned long ubnd[3] ; 
        unsigned long step[3] ; 
        for( int idir=0; idir<3; ++idir ) {
            if ( corner_dir[corner][idir] == -1 ) {
                lbnd[idir] = ngz - 1; 
                ubnd[idir] = -1 ; 
                step[idir] = -1 ;
            } else if ( corner_dir[corner][idir] == 1 ) {
                lbnd[idir] = nx + ngz ; 
                ubnd[idir] = nx + 2 * ngz ; 
                step[idir] = + 1  ;
            } else {
                lbnd[idir] = ngz ;
                ubnd[idir] = nx + ngz ; 
                step[idir] = +1 ; 
            }
        }
        int ii{0}, jj{0}, kk{0} ; 
        for( unsigned long ig=lbnd[0]; ig!=ubnd[0]; ig+=step[0], ii++ ) 
        for( unsigned long jg=lbnd[1]; jg!=ubnd[1]; jg+=step[1], jj++ )
        for( unsigned long kg=lbnd[2]; kg!=ubnd[2]; kg+=step[2], kk++ ) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(ig,jg,kg)},
                0,
                {VEC(0.5,0.5,0.5)}, 
                true
            ) ;            
            if ( std::fabs(h_state_mirror_new(VEC(ig,jg,kg),DENS,0)-h_func_bnd(pcoords[0],pcoords[1],pcoords[2])) > 1e-10 ) {
                std::cout << "Error found at (i,j,k): " << ig << " " << jg << " " << kg << std::endl 
                          << "Coordinates: " << pcoords[0] << " " << pcoords[1] << " " << pcoords[2] << std::endl ; 
                std::cout << "Mirrored coords: " <<  std::min(std::max(pcoords[0],x_min), x_max) << " " 
                                                 <<  std::min(std::max(pcoords[1],y_min), y_max) <<  " "
                                                 <<  std::min(std::max(pcoords[2],z_min), z_max) << std::endl ; 
            }
            REQUIRE_THAT(
                h_state_mirror_new(VEC(ig,jg,kg),DENS,0),
                Catch::Matchers::WithinAbs(h_func_bnd(pcoords[0],pcoords[1],pcoords[2]), 1e-10)
            ) ; 
        }
    }


    for( int edge=0; edge<12; ++edge) {
        unsigned long lbnd[3] ; 
        unsigned long ubnd[3] ; 
        unsigned long step[3] ; 
        for( int idir=0; idir<3; ++idir ) {
            if ( edge_dir[edge][idir] == -1 ) {
                lbnd[idir] = ngz - 1; 
                ubnd[idir] = -1 ; 
                step[idir] = -1 ;
            } else if ( edge_dir[edge][idir] == 1 ) {
                lbnd[idir] = nx + ngz ; 
                ubnd[idir] = nx + 2 * ngz ; 
                step[idir] = + 1  ;
            } else {
                lbnd[idir] = ngz ;
                ubnd[idir] = nx + ngz ; 
                step[idir] = +1 ; 
            }
        }
        int ii{0}, jj{0}, kk{0} ; 
        for( unsigned long ig=lbnd[0]; ig!=ubnd[0]; ig+=step[0], ii++ ) 
        for( unsigned long jg=lbnd[1]; jg!=ubnd[1]; jg+=step[1], jj++ )
        for( unsigned long kg=lbnd[2]; kg!=ubnd[2]; kg+=step[2], kk++ ) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(ig,jg,kg)},
                0,
                {VEC(0.5,0.5,0.5)}, 
                true
            ) ;            
            if ( std::fabs(h_state_mirror_new(VEC(ig,jg,kg),DENS,0)-h_func_bnd(pcoords[0],pcoords[1],pcoords[2])) > 1e-10 ) {
                std::cout << "Error found at (i,j,k): " << ig << " " << jg << " " << kg << std::endl 
                          << "Coordinates: " << pcoords[0] << " " << pcoords[1] << " " << pcoords[2] << std::endl ; 
                std::cout << "Mirrored coords: " <<  std::min(std::max(pcoords[0],x_min), x_max) << " " 
                                                 <<  std::min(std::max(pcoords[1],y_min), y_max) <<  " "
                                                 <<  std::min(std::max(pcoords[2],z_min), z_max) << std::endl ; 
            }
            REQUIRE_THAT(
                h_state_mirror_new(VEC(ig,jg,kg),DENS,0),
                Catch::Matchers::WithinAbs(h_func_bnd(pcoords[0],pcoords[1],pcoords[2]), 1e-10)
            ) ; 
        }
    }

    
}