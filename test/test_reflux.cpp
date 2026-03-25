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
#include <grace/evolution/refluxing.hh>

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


static void setup_initial_emf() 
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ; 
    using namespace Kokkos ; 
    auto& coord_system = grace::coordinate_system::get() ; 
    auto& emf = grace::variable_list::get().getemfarray() ; 
    Kokkos::fence() ; 

    auto emf_h = create_mirror_view(emf) ; 

    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.5,0.,0.} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            emf_h(i,j,k,0,q) = pcoords[0]  * (SQR(pcoords[1])-1.333*SQR(pcoords[2])); 
        }, {false,true,true}, true 
    ) ; 
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.,0.5,0.} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            emf_h(i,j,k,1,q) = pcoords[1] * (SQR(pcoords[0])-4.333*pcoords[2]); 
        }, {true,false,true}, true 
    ) ;
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.,0.,0.5} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            emf_h(i,j,k,2,q) = pcoords[2] * (SQR(pcoords[0])+pcoords[1]); 
        }, {true,true,false}, true 
    ) ;
    deep_copy(emf,emf_h) ;
}

void dump(const std::vector<grace::hanging_edge_reflux_desc_t>& vec, const std::string& fname)
{
    std::ofstream out(fname);

    for (size_t i = 0; i < vec.size(); i++) {
        const auto& d = vec[i];

        out << "DESC " << i << "\n";
        out << "  NFINE "   << d.n_fine   << "\n";
        out << "  NCOARSE " << d.n_coarse << "\n";
        out << "  NSIDES "  << d.n_sides  << "\n";

        for (int s = 0; s < d.n_sides; s++) {
            const auto& side = d.sides[s];

            out << "  SIDE " << s 
                << " is_fine=" << side.is_fine
                << " edge_id=" << int(side.edge_id)
                << " off_i="   << side.off_i
                << " off_j="   << side.off_j
                << "\n";

            if (side.is_fine) {
                out << "    FINE quad_id=[" 
                    << side.octants.fine.quad_id[0] << ","
                    << side.octants.fine.quad_id[1] << "]"
                    << " owner=[" 
                    << side.octants.fine.owner_rank[0] << ","
                    << side.octants.fine.owner_rank[1] << "]"
                    << " remote=["
                    << side.octants.fine.is_remote[0] << ","
                    << side.octants.fine.is_remote[1] << "]"
                    << "\n";
            } else {
                out << "    COARSE quad_id="  << side.octants.coarse.quad_id
                    << " owner="              << side.octants.coarse.owner_rank
                    << " remote="             << side.octants.coarse.is_remote
                    << "\n";
            }
        }
    }
}



static void check() 
{
    DECLARE_GRID_EXTENTS ; 
    auto& coord_system = grace::coordinate_system::get() ; 
    auto& emf = grace::variable_list::get().getemfarray() ; 
    auto& ghost_layer = grace::amr_ghosts::get();
    auto const& edge_desc = ghost_layer.get_reflux_edge_descriptors() ; 
    auto const& coarse_edge_desc = ghost_layer.get_reflux_coarse_edge_descriptors() ; 

    auto rank = parallel::mpi_comm_rank() ; 
    dump(edge_desc, "edge_desc" + std::to_string(rank) + ".out");
    dump(coarse_edge_desc, "coarse_edge_desc" + std::to_string(rank) + ".out");

    using namespace grace ; 

    auto emf_h = create_mirror_view(emf) ; 
    deep_copy(emf_h,emf) ; 


    grace::host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.5,0.,0.} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            double ground_truth = (pcoords[0]  * (SQR(pcoords[1])-1.333*SQR(pcoords[2])));
            double check = fabs(emf_h(i,j,k,0,q) - (pcoords[0]  * (SQR(pcoords[1])-1.333*SQR(pcoords[2]))));
            if ( check > 1e-10 * fabs(ground_truth)) {
                GRACE_VERBOSE("Issue (E^x) at i {} j {} k {} q {} target {} actual {}", 
                    i,j,k,q, (pcoords[0]  * (SQR(pcoords[1])-1.333*SQR(pcoords[2]))),emf_h(i,j,k,0,q));
            }
            REQUIRE_THAT( emf_h(i,j,k,0,q),
             Catch::Matchers::WithinULP(ground_truth, 4));
        }, {false,true,true}, true 
    ) ; 

    grace::host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.,0.5,0.} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            double ground_truth = (pcoords[1] * (SQR(pcoords[0])-4.333*pcoords[2]));
            double check = fabs(emf_h(i,j,k,1,q) - (pcoords[1] * (SQR(pcoords[0])-4.333*pcoords[2])));
            if ( check > 1e-10* fabs(ground_truth) ) {
                GRACE_VERBOSE("Issue (E^y) at i {} j {} k {} q {} target {} actual {}", 
                    i,j,k,q, (pcoords[1] * (SQR(pcoords[0])-4.333*pcoords[2])),emf_h(i,j,k,1,q));
            }
            REQUIRE_THAT( emf_h(i,j,k,1,q),
             Catch::Matchers::WithinULP(ground_truth, 4));
        }, {true,false,true}, true 
    ) ;

    grace::host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.,0.,0.5} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            double ground_truth = (pcoords[2] * (SQR(pcoords[0])+pcoords[1]));
            double check = fabs(emf_h(i,j,k,2,q) - (pcoords[2] * (SQR(pcoords[0])+pcoords[1])));
            if ( check > 1e-10 * fabs(ground_truth) ) {
                GRACE_VERBOSE("Issue (E^z) at i {} j {} k {} q {} target {} actual {}", 
                    i,j,k,q, (pcoords[2] * (SQR(pcoords[0])+pcoords[1])),emf_h(i,j,k,2,q));
            }
            REQUIRE_THAT( emf_h(i,j,k,2,q),
             Catch::Matchers::WithinULP(ground_truth, 4));
        }, {true,true,false}, true 
    ) ;
}

TEST_CASE("Test_EMF_REFLUX", "[refluxing]")
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ; 
    Kokkos::fence() ; 
    
    setup_initial_emf() ; 
    auto context = reflux_fill_emf_buffers() ; 
    Kokkos::fence() ; 
    reflux_correct_emfs(context) ;
    check() ; 
}
