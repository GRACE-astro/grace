/**
 * @file puncture_tracker.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2026-01-27
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

#ifndef GRACE_IO_NS_TRACKER_HH
#define GRACE_IO_NS_TRACKER_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>

#include <grace/utils/metric_utils.hh>

#include <Kokkos_Core.hpp>


#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <grace/system/grace_runtime.hh>

#include <grace/coordinates/coordinate_systems.hh>

#include <grace/utils/reductions.hh>

#include <vector>
#include <memory>

namespace grace {

/**
 * @brief Given an initial location of n neutron stars, integrates sqrtg D x^i to find the CoM
 *        of each star. The integral is restricted to rho > thresh and points are assigned to 
 *        the star whose last known position is closest. 
 */
struct ns_tracker_impl_t 
{
    //**************************************************************************************************
    public:
    //**************************************************************************************************
    void update_and_write() {
        auto& grace_runtime = grace::runtime::get(); 
        size_t const iter = grace_runtime.iteration() ;
        if ( (output_every > 0) and (iter%output_every == 0) ) {
            update() ;
            output() ; 
        }
    }
    //**************************************************************************************************
    Kokkos::View<double*[3], grace::default_space>
    get_ns_locations() const {
        return ns_centers; 
    }
    //**************************************************************************************************
    void set_ns_locations(
        Kokkos::View<double*[3], grace::default_space> new_centers
    )
    {
        ASSERT(new_centers.extent(0) == n_ns, "Trying to set ns locations with a wrong sized array!") ; 
    }
    //**************************************************************************************************
    bool is_active() const {
        return (output_every>0) && (n_ns>0) ; 
    }
    //**************************************************************************************************
    int get_n_ns() const {
        return n_ns ; 
    }
    //**************************************************************************************************
    private:
    //**************************************************************************************************
    void update() {
        using namespace grace  ; 
        using namespace Kokkos ;
        
        DECLARE_GRID_EXTENTS ; 

        auto state = variable_list::get().getstate() ; 
        auto dx = variable_list::get().getspacings() ; 

        auto dc = coordinate_system::get().get_device_coord_system() ; 

        MDRangePolicy<Rank<4>> policy(
            {ngz,ngz,ngz,0},
            {nx+ngz,ny+ngz,nz+ngz,nq}
        ) ; 


        int local_n = n_ns ;
        double local_dthresh = dens_thresh ; 
        auto local_ns_centers = ns_centers ;

        array_sum_t<double,8> com_integrals ; 

        parallel_reduce(
            GRACE_EXECUTION_TAG("DIAG", "ns_tracker_update_positions"),
            policy,
            KOKKOS_LAMBDA(int const i, int const j, int const k, int q, array_sum_t<double,8>& integrals) {

                double densL = state(i,j,k,DENS_,q) > local_dthresh ? state(i,j,k,DENS_,q) : 0.0 ; 

                double xyz[3] ; 
                dc.get_physical_coordinates(i,j,k,q,xyz) ; 

                double d[2] = {std::numeric_limits< double >::max(),std::numeric_limits< double >::max()} ; 
                d[0] = Kokkos::sqrt(
                    SQR(xyz[0]-local_ns_centers(0,0)) + SQR(xyz[1]-local_ns_centers(0,1)) + SQR(xyz[2]-local_ns_centers(0,2))
                ); 
                if (local_n>1) {
                    d[1] = Kokkos::sqrt(
                        SQR(xyz[0]-local_ns_centers(1,0)) + SQR(xyz[1]-local_ns_centers(1,1)) + SQR(xyz[2]-local_ns_centers(1,2))
                    ); 
                }

                double cell_vol = dx(0,q) * dx(1,q) * dx(2,q) ; 

                double x_int = densL * xyz[0] * cell_vol; 
                double y_int = densL * xyz[1] * cell_vol; 
                double z_int = densL * xyz[2] * cell_vol; 
                double norm_int = densL * cell_vol;                 

                if ( local_n > 1 ) {
                    if ( d[0] < d[1] ) {
                        integrals.data[0] += norm_int; 
                        integrals.data[1] += x_int ; 
                        integrals.data[2] += y_int ; 
                        integrals.data[3] += z_int ; 
                    } else {
                        integrals.data[4] += norm_int ; 
                        integrals.data[5] += x_int ; 
                        integrals.data[6] += y_int ; 
                        integrals.data[7] += z_int ; 
                    }
                } else {
                    integrals.data[0] += norm_int ; 
                    integrals.data[1] += x_int ; 
                    integrals.data[2] += y_int ; 
                    integrals.data[3] += z_int ;
                }

            }, Kokkos::Sum<array_sum_t<double,8>>(com_integrals)
        ) ; 

        array_sum_t<double,8> com_integrals_global;

        MPI_Allreduce(
            com_integrals.data,         // sendbuf
            com_integrals_global.data,  // recvbuf
            8,                          // count
            MPI_DOUBLE,                 // datatype
            MPI_SUM,                    // op
            MPI_COMM_WORLD              // comm
        );

        auto hc = Kokkos::create_mirror_view(ns_centers) ;
        hc(0,0) = com_integrals_global.data[1]/com_integrals_global.data[0] ; 
        hc(0,1) = com_integrals_global.data[2]/com_integrals_global.data[0] ; 
        hc(0,2) = equatorial_symm ? 0 : com_integrals_global.data[3]/com_integrals_global.data[0] ; 

        if (local_n>1) {
            hc(1,0) = com_integrals_global.data[5]/com_integrals_global.data[4] ; 
            hc(1,1) = com_integrals_global.data[6]/com_integrals_global.data[4] ; 
            hc(1,2) = equatorial_symm ? 0 : com_integrals_global.data[7]/com_integrals_global.data[4] ; 
        }
        
        Kokkos::deep_copy(ns_centers, hc) ; 
    }
    //**************************************************************************************************
    void output() 
    {
        int proc = parallel::mpi_comm_rank() ; 
        if ( proc == 0 ) {
            auto hc = Kokkos::create_mirror_view(ns_centers) ;
            Kokkos::deep_copy(hc,ns_centers) ; 
            auto& grace_runtime = grace::runtime::get() ; 
            size_t const iter = grace_runtime.iteration() ; 
            double const time = grace_runtime.time()      ;
            for ( int ip=0; ip<n_ns; ++ip ) {
                auto fname = outpaths[ip] ; 
                std::ofstream outfile(fname.string(), std::ios::app) ;
                outfile << std::fixed << std::setprecision(15) ; 
                outfile << std::left << iter << '\t'
                    << std::left << time << '\t' 
                    << std::left << hc(ip,0) << '\t'
                    << std::left << hc(ip,1) << '\t'
                    << std::left << hc(ip,2) << '\n' ; 
            }
        }
        parallel::mpi_barrier() ; 
    }
    //**************************************************************************************************
    void initialize_files() {
        static constexpr const size_t width = 20 ; 
        int proc = parallel::mpi_comm_rank() ; 
        auto& grace_runtime = grace::runtime::get() ; 
        std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ;  
        for( int i=0; i<n_ns; ++i) {
            std::string pfname = "ns_loc_" + std::to_string(i) + ".dat";
            std::filesystem::path fpath = bdir / pfname ; 
            if ( !std::filesystem::exists(fpath) and (proc == 0) ) {
                std::ofstream outfile(fpath.string());
                outfile << std::fixed << std::setprecision(15) ; 
                outfile << std::left << std::setw(width) << "Iteration" 
                        << std::left << std::setw(width) << "Time" 
                        << std::left << std::setw(width) << "X [M]" 
                        << std::left << std::setw(width) << "Y [M]"
                        << std::left << std::setw(width) << "Z [M]" << '\n' ;  
            }
            outpaths.push_back(fpath) ; 
        }
        parallel::mpi_barrier() ; 
    }
    //**************************************************************************************************
    Kokkos::View<double*[3], grace::default_space> ns_centers ;
    //**************************************************************************************************
    double dens_thresh ; 
    //**************************************************************************************************
    int n_ns ;
    //**************************************************************************************************
    int output_every ;  
    //**************************************************************************************************
    bool equatorial_symm ; 
    //**************************************************************************************************
    // output files 
    std::vector<std::filesystem::path> outpaths ;  
    //**************************************************************************************************
    protected:
    //**************************************************************************************************
    ~ns_tracker_impl_t() = default ; 
    //**************************************************************************************************
    ns_tracker_impl_t() : ns_centers("ns_tracker_positions",0) 
    {
        n_ns = get_param<int>("ns_tracker", "ns_number") ; 

        ASSERT(n_ns<3, "No more than 2 NSs can be tracked") ; 

        Kokkos::realloc(ns_centers, n_ns); 

        auto hc = Kokkos::create_mirror_view(ns_centers) ; 

        auto x_initial = get_param<std::vector<double>>("ns_tracker", "ns_x_initial") ; 
        auto y_initial = get_param<std::vector<double>>("ns_tracker", "ns_y_initial") ; 
        auto z_initial = get_param<std::vector<double>>("ns_tracker", "ns_z_initial") ; 

        ASSERT(x_initial.size() == n_ns, "Initial position needed for all neutron stars") ; 
        ASSERT(y_initial.size() == n_ns, "Initial position needed for all neutron stars") ; 
        ASSERT(z_initial.size() == n_ns, "Initial position needed for all neutron stars") ; 

        for( int ip=0; ip<n_ns; ++ip) {
            hc(ip,0) = x_initial[ip] ; 
            hc(ip,1) = y_initial[ip] ; 
            hc(ip,2) = z_initial[ip] ; 
        }

        Kokkos::deep_copy(ns_centers,hc) ; 
        
        // output frequency 
        output_every = get_param<int>("ns_tracker", "output_frequency") ;

        // density theshold 
        dens_thresh = get_param<double>("ns_tracker", "density_threshold") ; 

        // find out if the run is bitant 
        equatorial_symm = get_param<bool>("amr", "reflection_symmetries", "z") ; 
 
        // initialize files for output 
        initialize_files() ;
    }
    //**************************************************************************************************
    static constexpr unsigned long longevity = unique_objects_lifetimes::GRACE_SPHERICAL_SURFACES ; 
    //**************************************************************************************************
    //**************************************************************************************************
    friend class utils::singleton_holder<ns_tracker_impl_t> ;
    friend class memory::new_delete_creator<ns_tracker_impl_t, memory::new_delete_allocator> ; 
    //**************************************************************************************************
} ; 

//**************************************************************************************************
using ns_tracker = utils::singleton_holder<ns_tracker_impl_t> ; 
//**************************************************************************************************


}

#endif /* GRACE_IO_NS_TRACKER_HH */