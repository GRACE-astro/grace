#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
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
    using namespace Kokkos ; 
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
    auto& state  = grace::variable_list::get().getstate()  ;
    auto& coords = grace::variable_list::get().getcoords() ; 
    long nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    int ngz = grace::amr::get_n_ghosts() ; 
    auto ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ; 
    auto ncells_noghost = EXPR(nx,*ny,*nz)*nq ; 
    auto const h_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        return EXPR(8.5 * x, - 5.1 * y, + 2*z) - 3.14 ; 
    } ; 
    auto const h_func_derivative = [&] (VEC(const double& x,const double& y,const double &z))
    {
        return std::array<double,GRACE_NSPACEDIM>{
            VEC(
                8.5,
                -5.1,
                2. 
            )
        } ; 
    } ; 
    auto h_state = Kokkos::create_mirror_view( state ) ; 
    auto& coord_system = grace::coordinate_system::get() ;
    /*************************************************/
    /*                   fill data                   */
    /*************************************************/
    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
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
        h_state(VEC(i,j,k),DENS,q) = h_func(VEC(pcoords[0],pcoords[1],pcoords[2])) ; 
    }
    Kokkos::deep_copy(state, h_state) ; 
    auto& swap = grace::variable_list::get().getscratch() ; 
    Kokkos::deep_copy(swap, state) ; 
    /* Regrid */
    grace::amr::regrid() ; 
    /* Set ghostzone values to NaN before filling */
    nq = grace::amr::get_local_num_quadrants() ;
    ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ; 
    ncells_noghost = EXPR((nx),*(ny),*(nz))*nq ;
    h_state = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state, state) ; 
    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
        long const i = icell%(nx+2*ngz); 
        long const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef GRACE_3D 
        long const k = 
            (icell/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ; 
        long const q = 
            (icell/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ;
        #else 
        long const q = (icell/(nx+2*ngz)/(ny+2*ngz)) ; 
        #endif 
        ASSERT(!std::isnan(h_state(VEC(i,j,k),DENS,q))
        , "We have a NaN at " << q << ", " EXPR(<< i ,<< ", " << j ,<< ", " << k) << '\n' ) ; 
        if(   is_ghostzone(VEC(i,j,k),VEC(nx,ny,nz),ngz) ) 
        {
            h_state(VEC(i,j,k),DENS,q) = std::numeric_limits<double>::quiet_NaN() ; 
        }
    }
    Kokkos::deep_copy(state, h_state) ; 
    //grace::IO::write_volume_cell_data() ; 
    /* Fill boundaries and ghost-zones */
    grace::amr::apply_boundary_conditions() ; 
    //grace::runtime::get().increment_iteration() ; 
    //grace::IO::write_cell_output(true,true,true) ; 

    /* Check values in ghost-zones */
    auto& idx = grace::variable_list::get().getinvspacings() ; 
    auto h_idx = Kokkos::create_mirror_view(idx) ; 
    Kokkos::deep_copy(h_idx,idx)  ; 
    Kokkos::deep_copy(h_state, state) ; 
    grace::var_array_t<GRACE_NSPACEDIM> dxdens(
        "Density derivatives", VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz), GRACE_NSPACEDIM, nq  
    ) ; 
    auto h_dxdens = Kokkos::create_mirror_view(dxdens) ; 

    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
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
        
        if(   is_outside_grid(VEC(i,j,k),q) 
           or is_outside_grid(VEC(i+2,j,k),q)  
           or is_outside_grid(VEC(i-2,j,k),q) 
           or is_outside_grid(VEC(i,j+2,k),q) 
           or is_outside_grid(VEC(i,j-2,k),q) 
           #ifdef GRACE_3D 
           or is_outside_grid(VEC(i,j,k+2),q) 
           or is_outside_grid(VEC(i,j,k-2),q) 
           #endif 
          //or  is_corner_ghostzone(VEC(i,j,k),VEC(nx,ny,nz),ngz)) 
        )
        {
            continue ; 
        }
        #ifdef DBG_GHOSTZONE_TEST
        if(  std::isnan(h_state(VEC(i,j,k),DENS,q)) || std::fabs(h_state(VEC(i,j,k),DENS,q) - h_func(VEC(pcoords[0],pcoords[1],pcoords[2]) ) ) > 1e-12 ) {
            std::cout << "Rank: " << parallel::mpi_comm_rank() << '\n'
                      << "Quadrant, indices " << q EXPR(<< ", " << i ,<< ", " << j ,<< ", " << k) << '\n'  
                      << "Coordinates " << EXPR(pcoords[0] ,<< ", " << pcoords[1], << ", " << pcoords[2]) << '\n'
                      << "Quadrant level " << grace::amr::get_quadrant(q).level() << std::endl 
                      << ( is_corner_ghostzone(VEC(i,j,k),VEC(nx,ny,nz), ngz) ? "in corners"
                            :  ( is_edge_ghostzone(VEC(i,j,k),VEC(nx,ny,nz), ngz) ? "in edges" : "in face") ) << std::endl 
                     << h_state(VEC(i,j,k),DENS,q) << std::endl  ; 
        }
        if ( (q == 42 || q == 44) and not is_ghostzone(VEC(i,j,k),VEC(nx,ny,nz), ngz) and i < 2*ngz and k < 2*ngz ) {
            std::cout << "Qid, indices  " << q EXPR(<< ", " << i ,<< ", " << j ,<< ", " << k) << '\n'
                      << "Coordinates " << EXPR(pcoords[0] ,<< ", " << pcoords[1], << ", " << pcoords[2]) << '\n'
                      << "Quadrant level " << grace::amr::get_quadrant(q).level() << std::endl 
                      << h_state(VEC(i,j,k),DENS,q) << std::endl ; 
        }
        #endif 
        CHECK_THAT(
            h_state(VEC(i,j,k),DENS,q),
            Catch::Matchers::WithinAbs(h_func(VEC(pcoords[0],pcoords[1],pcoords[2])),
                1e-12 ) ) ; 

    }
    #if 0 
    for( size_t icell=0UL; icell<ncells_noghost; icell+=1UL)
    {
        size_t const i = icell%(nx) + ngz ; 
        size_t const j = (icell/(nx)) % (ny) + ngz ;
        #ifdef GRACE_3D 
        size_t const k = 
            (icell/(nx)/(ny)) % (nz) + ngz ; 
        size_t const q = 
            (icell/(nx)/(ny)/(nz)) ;
        #else 
        size_t const q = (icell/(nx)/(ny)) ; 
        #endif 
        auto pcoords = coord_system.get_physical_coordinates(
            {VEC(i,j,k)},
            q,
            false
        ) ; 
        #ifdef DBG_GHOSTZONE_TEST
        std::cout << "Cell " << icell << std::endl ;
        std::cout << "Quadrant, indices " << q EXPR(<< ", " << i ,<< ", " << j ,<< ", " << k) << '\n'  
                  << "Coordinates " << EXPR(pcoords[0] ,<< ", " << pcoords[1], << ", " << pcoords[2]) << '\n' ; 
        #endif 
        if(   is_outside_grid(VEC(i,j,k),q) 
           or is_outside_grid(VEC(i+2,j,k),q)  
           or is_outside_grid(VEC(i-2,j,k),q) 
           or is_outside_grid(VEC(i,j+2,k),q) 
           or is_outside_grid(VEC(i,j-2,k),q) 
           #ifdef GRACE_3D 
           or is_outside_grid(VEC(i,j,k+2),q) 
           or is_outside_grid(VEC(i,j,k-2),q) 
           #endif 
           ) 
        {
            continue ; 
        }

        
        auto itree = grace::amr::get_quadrant_owner(q) ; 
        auto dx_tree = grace::amr::get_tree_spacing(itree) ; 
        std::array<double,GRACE_NSPACEDIM> idxphys{
            VEC(
                h_idx(0,q) / dx_tree[0],
                h_idx(1,q) / dx_tree[1],
                h_idx(2,q) / dx_tree[2]
            ) 
        } ; 
        auto der_exact = h_func_derivative(VEC(pcoords[0],pcoords[1],pcoords[2])) ; 
        
        for(int idim=0; idim<GRACE_NSPACEDIM; ++idim){ 
            h_dxdens(VEC(i,j,k),idim,q) = 
              0.5*(h_state(VEC(i+utils::delta(idim,0),j+utils::delta(idim,1),k+utils::delta(idim,2)), DENS, q)
            - h_state(VEC(i-utils::delta(idim,0),j-utils::delta(idim,1),k-utils::delta(idim,2)), DENS, q))*idxphys[idim]; 
            CHECK_THAT( h_dxdens(VEC(i,j,k),idim,q)
                , Catch::Matchers::WithinAbs(
                  der_exact[idim]
                , 1e-3)) ;
        }
    } 
    #endif 
}