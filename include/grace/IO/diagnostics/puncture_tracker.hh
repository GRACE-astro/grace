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
#ifndef GRACE_IO_PUNCTURE_TRACKER_HH
#define GRACE_IO_PUNCTURE_TRACKER_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>

#include <grace/utils/metric_utils.hh>

#include <grace/utils/device_vector.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/config/config_parser.hh>

#include <grace/IO/diagnostics/diagnostic_base.hh>
#include <grace/utils/lagrange_interpolation.hh>

#include <Kokkos_Core.hpp>


#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <vector>
#include <memory>


namespace grace {

struct puncture_tracker_impl_t 
{

    public:
        //**************************************************************************************************
        void update_and_write() {
            auto& grace_runtime = grace::runtime::get() ; 
            auto const dt = grace_runtime.timestep() ; 
            size_t const iter = grace_runtime.iteration() ;
            if ( (out_every > 0) and (iter%out_every == 0) and (n_punctures > 0) ) {
                // note that we pass out_every x dt to the pusher 
                update(out_every * dt) ;  
                output() ; 
            } 
        }
        //**************************************************************************************************
    private:
        //**************************************************************************************************
        void update(double dt) {
            auto beta = interpolate_shift() ; 
            std::vector<point_host_t> local_updates ; 
            for( int ip=0; ip<beta.size(); ++ip ) {
                auto pidx = local_puncture_ids[ip] ; 
                point_host_t pl; 
                pl.first = pidx ;
                for( int idir=0; idir<3; ++idir) {
                    pl.second[idir] = _coords[pidx][idir] - dt * beta[ip][idir] ; 
                }
                local_updates.push_back(pl) ; 
            }

            // now exchange across ranks the updated positions
            exchange_new_positions(local_updates) ;             
        }; 
        //**************************************************************************************************
        void output() 
        {
            int proc = parallel::mpi_comm_rank() ; 
            if ( proc == 0 ) {
                auto& grace_runtime = grace::runtime::get() ; 
                size_t const iter = grace_runtime.iteration() ; 
                double const time = grace_runtime.time()      ;
                for ( int ip=0; ip<n_punctures; ++ip ) {
                    auto fname = outpaths[ip] ; 
                    std::ofstream outfile(fname.string(), std::ios::app) ;
                    outfile << std::fixed << std::setprecision(15) ; 
                    outfile << std::left << iter << '\t'
                        << std::left << time << '\t' 
                        << std::left << _coords[ip][0] << '\t'
                        << std::left << _coords[ip][1] << '\t'
                        << std::left << _coords[ip][2] << '\n' ; 
                }
            }
            parallel::mpi_barrier() ; 
        }; 
        //**************************************************************************************************
        void initialize_files() {
            static constexpr const size_t width = 20 ; 
            int proc = parallel::mpi_comm_rank() ; 
            auto& grace_runtime = grace::runtime::get() ; 
            std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ;  
            for( int i=0; i<n_punctures; ++i) {
                std::string pfname = "puncture_loc_" + std::to_string(i) + ".dat";
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

        void exchange_new_positions(
            std::vector<point_host_t> const& local
        ) {
            int nprocs = parallel::mpi_comm_size();

            int n_local = local.size();

            std::vector<int> recv_counts(nprocs);
            std::vector<int> recv_offsets(nprocs);

            /* gather element counts */
            MPI_Allgather(&n_local, 1, MPI_INT,
                        recv_counts.data(), 1, MPI_INT,
                        MPI_COMM_WORLD);

            /* convert counts to bytes + compute offsets */
            recv_offsets[0] = 0;
            for (int r = 1; r < nprocs; ++r) {
                recv_offsets[r] = recv_offsets[r-1]
                                + recv_counts[r-1] * sizeof(point_host_t);
            }

            int total_bytes =
                recv_offsets[nprocs-1]
            + recv_counts[nprocs-1] * sizeof(point_host_t);

            ASSERT(total_bytes == n_punctures * sizeof(point_host_t),
                "Mismatch number of punctures in MPI comm");

            std::vector<point_host_t> global(n_punctures);

            /* convert counts to bytes */
            for (int r = 0; r < nprocs; ++r)
                recv_counts[r] *= sizeof(point_host_t);

            MPI_Allgatherv(
                local.data(),
                n_local * sizeof(point_host_t),
                MPI_BYTE,
                global.data(),
                recv_counts.data(),
                recv_offsets.data(),
                MPI_BYTE,
                MPI_COMM_WORLD
            );

            // store everything back 
            for( int ip=0; ip<n_punctures; ++ip ) {
                point_host_t const& pl = global[ip]; 
                _coords[pl.first] = pl.second ; 
            }
        }


        std::vector<point_host_t> make_point_vector()
        {
            std::vector<point_host_t> out{n_punctures} ;
            for ( int ip=0; ip<n_punctures; ++ip) {
                point_host_t c ;
                c.second[0] = _coords[ip][0] ; c.second[1] = _coords[ip][1] ; c.second[2] = _coords[ip][2] ; 
                c.first = ip ;
                out[ip] = c ; 
            }
            return out ; 
        }   
        //**************************************************************************************************
        std::vector<std::array<double,3>>
        interpolate_shift() {
            DECLARE_GRID_EXTENTS;
            // get the centers 
            auto centers = make_point_vector() ;
            auto centers_array = sc_array_new_data(
                centers.data(), sizeof(point_host_t), centers.size() 
            ) ; 
            // Create a descriptor for cells containing 
            // the centers 
            auto p4est = grace::amr::forest::get().get() ; 
            std::vector<intersected_cell_descriptor_t> intersected_cells_h;
            std::vector<size_t> intersecting_points_h;
            intersected_cell_set_t set{
                &intersected_cells_h,
                &intersecting_points_h
            }; 
            p4est->user_pointer = static_cast<void*>(&set) ; 
            p4est_search_local(
                p4est, 
                false, 
                nullptr, 
                &grace_search_points,
                centers_array
            ) ; 
            n_punctures_loc = intersecting_points_h.size() ;
            local_puncture_ids = intersecting_points_h ; 

            std::vector<std::array<double,3>> beta_p{n_punctures_loc} ; 

            if ( n_punctures_loc > 0 ) {
                lagrange_interpolator_t<LAGRANGE_INTERP_ORDER> interpolator{ngz} ; 
                interpolator.compute_weights(
                    centers, intersecting_points_h, intersected_cells_h
                ) ; 
                Kokkos::View<double**,grace::default_space> vals("ptracker_shift",0,0); 
                auto& state = grace::variable_list::get().getstate() ; 
                interpolator.interpolate(
                    state, {BETAX_,BETAY_,BETAZ_}, vals
                ) ; 
                auto vals_h = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), vals 
                ) ; 
                for( int i=0; i<n_punctures_loc; ++i) {
                    beta_p[i][0] = vals_h(i,0) ; 
                    beta_p[i][1] = vals_h(i,1) ;
                    beta_p[i][2] = vals_h(i,2) ;  
                }
            }
            return beta_p ; 
        }
        //**************************************************************************************************
    protected: 
        //**************************************************************************************************
        ~puncture_tracker_impl_t() = default ; 
        //**************************************************************************************************
        puncture_tracker_impl_t() {
            n_punctures = get_param<int>("puncture_tracker", "n_punctures") ;
            _coords.resize(n_punctures) ; 

            auto x_initial = get_param<std::vector<double>>("puncture_tracker", "puncture_x_initial") ; 
            auto y_initial = get_param<std::vector<double>>("puncture_tracker", "puncture_y_initial") ; 
            auto z_initial = get_param<std::vector<double>>("puncture_tracker", "puncture_z_initial") ; 

            ASSERT(x_initial.size() == n_punctures, "Initial position needed for all punctures") ; 
            ASSERT(y_initial.size() == n_punctures, "Initial position needed for all punctures") ; 
            ASSERT(z_initial.size() == n_punctures, "Initial position needed for all punctures") ; 

            for( int ip=0; ip<n_punctures; ++ip) {
                _coords[ip][0] = x_initial[ip] ; 
                _coords[ip][1] = y_initial[ip] ; 
                _coords[ip][2] = z_initial[ip] ; 
            }
            // output frequency 
            out_every = get_param<int>("puncture_tracker", "output_frequency") ;
            // initialize files for output 
            initialize_files() ; 
        }
        //**************************************************************************************************
        size_t n_punctures, n_punctures_loc ; //!< Total and local number of punctures 
        int out_every ; 
        std::vector<size_t> local_puncture_ids;
        std::vector<std::array<double,3>> _coords ; 
        // output files 
        std::vector<std::filesystem::path> outpaths ; 
        //**************************************************************************************************
        static constexpr unsigned long longevity = unique_objects_lifetimes::GRACE_SPHERICAL_SURFACES ; 
        //**************************************************************************************************
        //**************************************************************************************************
        friend class utils::singleton_holder<puncture_tracker_impl_t> ;
        friend class memory::new_delete_creator<puncture_tracker_impl_t, memory::new_delete_allocator> ; 
        //**************************************************************************************************
} ; 

//**************************************************************************************************
using puncture_tracker = utils::singleton_holder<puncture_tracker_impl_t> ; 
//**************************************************************************************************

}


#endif 