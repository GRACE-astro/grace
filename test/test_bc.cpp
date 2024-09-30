#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/gridloop.hh>

#include <grace/IO/vtk_output.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numeric>

#define DBG_GHOSTZONE_TEST 

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

TEST_CASE("Apply BC", "[boundaries]")
{
    using namespace grace::variables ; 
    using namespace grace ;

    #ifdef GRACE_ENABLE_BURGERS 
    int const DENS = U ; 
    int const DENS_ = U ; 
    auto params = grace::config_parser::get()["amr"] ; 
    params["refinement_criterion_var"] = "U" ; 
    #endif 
    #ifdef GRACE_ENABLE_SCALAR_ADV
    int const DENS = U ; 
    int const DENS_ = U ; 
    auto params = grace::config_parser::get()["amr"] ; 
    params["refinement_criterion_var"] = "U" ; 
    #endif 

    DECLARE_GRID_EXTENTS ; 
    auto const interp_order = grace::get_param<uint32_t>("amr","prolongation_order") ;
    /*************************************************/
    /*                Fetch arrays                   */
    /*************************************************/
    auto& state  = grace::variable_list::get().getstate()  ;
    auto& sstate  = grace::variable_list::get().getstaggeredstate()  ;
    auto& coord_system = grace::coordinate_system::get() ;
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    auto h_corner_mirror = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 
    /*************************************************/
    /*            Define filling func                */
    /*************************************************/
    auto const h_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        return EXPR(8.5 * x, - 5.1 * y, + 2*z) - 3.14 ; 
    } ; 
    
    auto const h_corner_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        #if 0
        if( interp_order == 2 ) {
            return EXPR(8.5 * x, - 5.1 * y, -2*z) - 3.14 ; 
        } else if ( interp_order == 4) {
            #ifdef GRACE_3D
            return 0.09645987612683005 + 0.9689256995609989*x + 0.9280564240107632*y - 0.27263220791463016*x*y + 1.6557688148274297*z - 1.8293477262261941*x*z + 
   1.8321409249345644*y*z - 0.6168312325224381*x*y*z + 1.7146635117999285*pow(x,2) - 0.8323622181656987*y*pow(x,2) - 1.1983364369285372*z*pow(x,2) + 
   0.11344784791220963*pow(x,3) + 1.46241660443817*pow(y,2) + 1.9071800878186975*x*pow(y,2) + 1.7912912453890968*z*pow(y,2) - 0.37430580597888685*pow(y,3) - 
   0.07020440743423961*pow(z,2) + 1.0902200536627111*x*pow(z,2) + 1.2434145608397085*y*pow(z,2) + 0.6321621456866486*pow(z,3) ; 
            #else 
            return 1.0354333039152808 + 1.6630034246569636*x + 1.3491577540970425*y - 1.6695252008930153*x*y - 1.205160193337056*pow(x,2) - 1.6913180599507545*y*pow(x,2) - 
   0.4452976970948681*pow(x,3) + 0.3512878541919209*pow(y,2) + 0.17773874176068194*x*pow(y,2) - 0.7151254966832106*pow(y,3) ; 
            #endif 
        } else {
            return - 1.; 
        }
        #endif 
        return EXPR( cos(2*M_PI*x), + cos(2*M_PI*y), + cos(2*M_PI*z) ) ; 
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
                h_corner_func(VEC(pcoords[0],pcoords[1],pcoords[2])) ;
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
    /*                   Regrid                      */
    /*************************************************/
    bool do_regrid = grace::get_param<bool>("amr","do_regrid_test") ; 
    if( do_regrid ) {
        grace::amr::regrid() ;
    }
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
    grace::amr::apply_boundary_conditions() ; 

    /*************************************************/
    /*                 Copy D2H                      */
    /*************************************************/
    auto h_state_mirror_new = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state_mirror_new,state); 
    auto h_corner_mirror_new = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 
    Kokkos::deep_copy(h_corner_mirror_new,sstate.corner_staggered_fields);  

    /*************************************************/
    /*                   Check                       */
    /*************************************************/
    host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                true
            ) ;
            REQUIRE_THAT( h_state_mirror_new(VEC(i,j,k),DENS,q)
                      , Catch::Matchers::WithinAbs(
                                  h_func(VEC(pcoords[0],pcoords[1],pcoords[2]))
                                , 1e-12 )) ;
        },
        {VEC(false,false,false)},
        true
    ) ; 
    #if 1 
    host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                {VEC(0,0,0)},
                true
            ) ;
            if(
                std::fabs(h_corner_mirror_new(VEC(i,j,k),DENS,q) - h_corner_func(VEC(pcoords[0],pcoords[1],pcoords[2])))>1e-10
            ) {
                std::cout << pcoords[0] << ", " << pcoords[1] << ", " << pcoords[2] << std::endl ;
            }
            REQUIRE_THAT( h_corner_mirror_new(VEC(i,j,k),DENS,q)
                      , Catch::Matchers::WithinAbs(
                                  h_corner_func(VEC(pcoords[0],pcoords[1],pcoords[2]))
                                , 1e-12 )) ;
        },
        {VEC(true,true,true)},
        true
    ) ;
    #endif 
}