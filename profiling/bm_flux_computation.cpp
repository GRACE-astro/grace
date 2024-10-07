

#include <yaml-cpp/yaml.h>

#include <grace_config.h>
#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#define ROCPROFILER_PLUGIN_SETUP
#include <grace/profiling/gpu_profiling.hh>
#include "../src/profiling/gpu_profiling.cpp"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#define GRACE_REAL double

#define INTERP_ORDER 1
//#define PARTIALLY_UNROLL_LOOPS
//#define PARTIALLY_UNROLL_X_LOOP 
#define UNROLL_SIZE 4
#define X_DIR 0
#define Y_DIR 1
#define Z_DIR 2

template< int idir, int jdir >
static constexpr inline int delta() {
    return static_cast<int>(idir==jdir) ; 
}

template< int order > 
struct lagrange_recon_t {} ; 


template<> 
struct lagrange_recon_t<1>
{
    template< int idir > 
    static void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
    interp(Kokkos::View<GRACE_REAL ****> u, int i, int j, int k, int q, double& uR, double& uL) {
        double const um1 = u(i-delta<idir,0>(),j-delta<idir,1>(),k-delta<idir,2>(),q) ; 
        double const um2 = u(i-2*delta<idir,0>(),j-2*delta<idir,1>(),k-2*delta<idir,2>(),q) ; 
        uL = 0.25 * (um1-um2) + um1 ; 
        double const up0 = u(i,j,k,q) ; 
        double const up1 = u(i+delta<idir,0>(),j+delta<idir,1>(),k+delta<idir,2>(),q) ; 
        uR = 0.25 * (up0-up1) + up0 ;
    }       
} ; 


int main(int argc, char** argv) {

    using namespace Kokkos ;
    // Check if a config file was provided as an argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file.yaml>" << std::endl;
        return 1;
    }

    // Load the YAML file from the first command line argument
    std::string config_file = argv[1];
    YAML::Node params;
    try {
        params = YAML::LoadFile(config_file);
    } catch (const YAML::BadFile& e) {
        std::cerr << "Error: Could not open the file '" << config_file << "'" << std::endl;
        return 1;
    }

    // Access the "amr" sub-node
    if (!params["amr"]) {
        std::cerr << "Error: 'amr' key not found in the YAML file." << std::endl;
        return 1;
    }
    YAML::Node amr_params = params["amr"];
    int nx,ny,nz, ngz,nq,n_repetitions ; 
    bool do_profiling{false} ; 
    std::vector<std::string> _counters ; 
    // Extract and check the individual values
    try {
        nx = amr_params["nx"].as<int>();
        ny = amr_params["ny"].as<int>();
        nz = amr_params["nz"].as<int>();
        ngz = amr_params["ngz"].as<int>();
        nq = amr_params["nq"].as<int>();
        n_repetitions = amr_params["n_repetitions"].as<int>();

        _counters =
            amr_params["counters"].as<std::vector<std::string>>() ; 
        // Print the extracted values
        std::cout << "nx: " << nx << std::endl;
        std::cout << "ny: " << ny << std::endl;
        std::cout << "nz: " << nz << std::endl;
        std::cout << "ngz: " << ngz << std::endl;
        std::cout << "nq: " << nq << std::endl;
        std::cout << "n_repetitions: " << n_repetitions << std::endl;
    } catch (const YAML::BadConversion& e) {
        std::cerr << "Error: Bad conversion in YAML file." << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }
    auto hasher = std::hash<std::string>{} ;
    for( auto const& x: _counters )
    {
        std::cout << "Counter " << x << " active." << std::endl ;
        counter_names.emplace(hasher(x),x) ; 
    }
    int const ncells_total = 256 * 256 * 256 ; 

    double const dx = 1./nx ; 
    double const dy = 1./ny ;
    double const dz = 1./nz ; 
    if( do_profiling )
        rocprofiler_initialize();
    Kokkos::initialize(argc,argv) ; 
    {   

        View<GRACE_REAL ****, Kokkos::LayoutLeft> u("u",nx+2*ngz,ny+2*ngz,nz+2*ngz, nq) ; 
        View<GRACE_REAL *****, Kokkos::LayoutLeft> f("flux",nx+1,ny+1,nz+1, 3, nq) ;

        auto const compute_x_flux = KOKKOS_LAMBDA (int i, int j, int k, int q) 
        {
            double uR,uL ; 
            lagrange_recon_t<INTERP_ORDER>::template interp<X_DIR>(u,i+ngz,j+ngz,k+ngz,q,uR,uL) ; 
            // just do llf flux we ain't here for precision
            f(i,j,k,X_DIR,q) = (uL*uL + uR*uR - (uR-uL) ) ;
        } ;

        auto const compute_y_flux = KOKKOS_LAMBDA (int i, int j, int k, int q) 
        {
            double uR,uL ; 
            lagrange_recon_t<INTERP_ORDER>::template interp<Y_DIR>(u,i+ngz,j+ngz,k+ngz,q,uR,uL) ; 
            // just do llf flux we ain't here for precision
            f(i,j,k,Y_DIR,q) = (uL*uL + uR*uR - (uR-uL) ) ;
        } ;

        auto const compute_z_flux = KOKKOS_LAMBDA (int i, int j, int k, int q) 
        {
            double uR,uL ; 
            lagrange_recon_t<INTERP_ORDER>::template interp<Z_DIR>(u,i+ngz,j+ngz,k+ngz,q,uR,uL) ; 
            // just do llf flux we ain't here for precision
            f(i,j,k,Z_DIR,q) = (uL*uL + uR*uR - (uR-uL) ) ;
        } ; 

        MDRangePolicy<Rank<4>,IndexType<int>>
        policy( {0,0,0,0}, {nx+2*ngz,ny+2*ngz,nz+2*ngz,nq} ) ; 
        parallel_for(
            "initial_data",
            policy,
            KOKKOS_LAMBDA(int i, int j, int k, int q) {
                double const x = (i-ngz+0.5) * dx ; 
                double const y = (i-ngz+0.5) * dy ; 
                double const z = (i-ngz+0.5) * dz ;

                double const r2 = x*x+y*y+z*z ; 

                u(i,j,k,q) = exp(-0.5 * r2 / 0.1 ) ;  
            }
        ) ; 
        for( int rep=0; rep<n_repetitions; ++rep )
        {
            rocm_profiling_context_t context{} ;  
            if( do_profiling ) {
                rocm_initiate_profiling_session(context, counter_names) ; 
            }
            // X flux 
            MDRangePolicy<Rank<4>,IndexType<int>>
                policy_x( {0,0,0,0}, {static_cast<int>(floor((nx+1)/UNROLL_SIZE)),ny,nz,nq} ) ; 
            parallel_for(
                "compute_x_flux",
                policy_x,
                KOKKOS_LAMBDA( int i, int j, int k, int q) {
                    int ii = UNROLL_SIZE*i;
                    #pragma unroll UNROLL_SIZE 
                    for( int offset=0; offset<UNROLL_SIZE; ++offset) {
                        if( ii+offset < nx+1 ) {
                            compute_x_flux(ii+offset,j,k,q) ;
                        }
                    }
                }
            ) ; 
            // Y flux 
            #ifndef PARTIALLY_UNROLL_LOOPS
            MDRangePolicy<Rank<4>,IndexType<int>>
                policy_y( {0,0,0,0}, {nx,ny+1,nz,nq} ) ; 
            parallel_for(
                "compute_y_flux",
                policy_y,
                KOKKOS_LAMBDA( int i, int j, int k, int q) {
                    compute_y_flux(i,j,k,q) ; 
                }
            ) ;
            #else
            MDRangePolicy<Rank<4>,IndexType<int>>
                policy_y( {0,0,0,0}, {nx,static_cast<int>(floor((ny+1)/UNROLL_SIZE)),nz,nq} ) ; 
            parallel_for(
                "compute_y_flux",
                policy_y,
                KOKKOS_LAMBDA( int i, int j, int k, int q) {
                    int jj = UNROLL_SIZE * j ; 
                    #pragma unroll UNROLL_SIZE 
                    for( int offset=0; offset<UNROLL_SIZE; ++offset) {
                        if( jj+offset < ny+1 ) {
                            compute_y_flux(i,jj+offset,k,q) ;
                        }
                    }
                }
            ) ;
            #endif 

            // Z flux 
            #ifndef PARTIALLY_UNROLL_LOOPS
            MDRangePolicy<Rank<4>,IndexType<int>>
                policy_z( {0,0,0,0}, {nx,ny,nz+1,nq} ) ; 
            parallel_for(
                "compute_z_flux",
                policy_z,
                KOKKOS_LAMBDA( int i, int j, int k, int q) {
                    double uR,uL ; 
                    lagrange_recon_t<INTERP_ORDER>::template interp<Z_DIR>(u,i+ngz,j+ngz,k+ngz,q,uR,uL) ; 
                    // just do llf flux we ain't here for precision
                    f(i,j,k,2,q) = (uL*uL + uR*uR - (uR-uL) ) ; 
                }
            ) ;
            #else
            MDRangePolicy<Rank<4>,IndexType<int>>
                policy_z( {0,0,0,0}, {nx,static_cast<int>(floor((nz+1)/UNROLL_SIZE)),ny,nq} ) ; 
            parallel_for(
                "compute_z_flux",
                policy_z,
                KOKKOS_LAMBDA( int i, int k, int j, int q) {
                    int kk = k * UNROLL_SIZE ; 
                    #pragma unroll UNROLL_SIZE 
                    for( int offset=0; offset<UNROLL_SIZE; ++offset) {
                        if( kk+offset < nz+1 ) {
                            compute_z_flux(i,j,kk+offset,q) ;
                        }
                    }
                }
            ) ;
            #endif 
            #if 0
            parallel_for(
                "compute_flux_derivatives",
                policy,
                KOKKOS_LAMBDA( int i, int k, int j, int q) {
                    u(i,j,k,q) +=
                        ( f(i,j,k,X_DIR,q) - f(i+1,j,k,X_DIR,q) ) / dx 
                      + ( f(i,j,k,X_DIR,q) - f(i,j+1,k,Y_DIR,q) ) / dx 
                      + ( f(i,j,k,X_DIR,q) - f(i,j,k+1,Z_DIR,q) ) / dx ; 
                }
            ) ;
            #endif 
            if( do_profiling )
                rocm_terminate_profiling_session(context) ; 
        }

    }
    Kokkos::finalize() ; 
    if( do_profiling )
        rocprofiler_finalize();
}