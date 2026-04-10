/**
 * @file co_tracker.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Compact object tracker
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

#ifndef GRACE_IO_CO_TRACKER_HH
#define GRACE_IO_CO_TRACKER_HH

#include <grace_config.h> 

#include <grace/system/grace_runtime.hh>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>

#include <grace/utils/metric_utils.hh>

#include <grace/utils/device_vector.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/config/config_parser.hh>

#include <grace/IO/diagnostics/diagnostic_base.hh>
#include <grace/utils/lagrange_interpolation.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>
#include <grace/utils/reductions.hh>

#include <grace/system/checkpoint_handler.hh>

#include <string> 
#include <array> 
#include <vector>
#include <memory>
#include <filesystem> 

#include <Kokkos_Core.hpp>

namespace grace {

class co_tracker_t {

    public:

    co_tracker_t()                           = default ; 
    co_tracker_t(co_tracker_t const& other ) = default ; 
    co_tracker_t(co_tracker_t &&     other ) = default ; 

    co_tracker_t(
        std::array<double,3> l,
        double r,
        std::string const& n
    ) : name(n), location(l), radius(r) 
    {
        initialize_file() ; 
        initialized = true ; 
    }

    virtual ~co_tracker_t() ; 

    std::string get_name() const {return name;}
    virtual std::string get_kind() const = 0;
    std::array<double,3> get_loc() const {return location;}
    double get_radius() const { return radius; }

    virtual void update(double dt) = 0; 

    void output()
    {
        ASSERT(initialized, "Attempting to output co location before inititialization") ; 
        int proc = parallel::mpi_comm_rank() ; 
        if ( proc == 0 ) {
            auto& grace_runtime = grace::runtime::get() ; 
            size_t const iter = grace_runtime.iteration() ; 
            double const time = grace_runtime.time()      ;
            std::ofstream outfile(outfilepath.string(), std::ios::app) ;
            outfile << std::fixed << std::setprecision(15) ; 
            outfile << std::left << iter << '\t'
                    << std::left << time << '\t' 
                    << std::left << location[0] << '\t'
                    << std::left << location[1] << '\t'
                    << std::left << location[2] << '\t'
                    << std::left << radius << '\n' ; 
        }
    }

    private: 

    void initialize_file()
    {
        static constexpr const size_t width = 20 ; 
        int proc = parallel::mpi_comm_rank() ; 
        auto& grace_runtime = grace::runtime::get() ; 
        std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ;
        std::string pfname = "co_" + name + "_loc.dat" ;
        outfilepath = bdir / pfname ;  
        if ( (!std::filesystem::exists(outfilepath)) && (proc==0) ) {
            std::ofstream outfile(outfilepath.string());
            outfile << std::fixed << std::setprecision(15) ; 
            outfile << std::left << std::setw(width) << "Iteration" 
                    << std::left << std::setw(width) << "Time" 
                    << std::left << std::setw(width) << "X [M]" 
                    << std::left << std::setw(width) << "Y [M]"
                    << std::left << std::setw(width) << "Z [M]" 
                    << std::left << std::setw(width) << "R [M]" << '\n' ; 
        }
        parallel::mpi_barrier() ; 
    }

    protected: 

    std::string name                  ; 
    std::array<double,3> location     ; 
    double radius                     ; 
    bool initialized{false}           ; 
    std::filesystem::path outfilepath ; 
} ; 

/**
 * @brief Tracks a NS CoM 
 */
class ns_tracker_t : public co_tracker_t 
{
    public: 
    using co_tracker_t::co_tracker_t;  // inherits all base ctors

    std::string get_kind() const override final {return "ns";}

    void update(double /*unused*/ dt) override final {
        auto l = this->location ; 
        auto r = this->radius   ; 

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
        
        array_sum_t<double,4> com_integrals ; 
        parallel_reduce(
            GRACE_EXECUTION_TAG("DIAG", "ns_tracker_update_positions"),
            policy,
            KOKKOS_LAMBDA(int const i, int const j, int const k, int q, array_sum_t<double,4>& integrals) {
                double xyz[3] ; 
                dc.get_physical_coordinates(i,j,k,q,xyz) ; 

                double dist = Kokkos::sqrt(
                    SQR(xyz[0]-l[0]) + SQR(xyz[1]-l[1]) + SQR(xyz[2]-l[2])
                ) ; 

                double densL = state(i,j,k,DENS_,q); 

                double cell_vol = dx(0,q) * dx(1,q) * dx(2,q) ; 

                double x_int = densL * xyz[0] * cell_vol; 
                double y_int = densL * xyz[1] * cell_vol; 
                double z_int = densL * xyz[2] * cell_vol; 
                double norm_int = densL * cell_vol;  

                if ( dist <= r ) {
                    integrals.data[0] += x_int ; 
                    integrals.data[1] += y_int ; 
                    integrals.data[2] += z_int ; 
                    integrals.data[3] += norm_int ; 
                }
            }, Kokkos::Sum<array_sum_t<double,4>>(com_integrals)
        ); 

        array_sum_t<double,4> com_integrals_global;

        MPI_Allreduce(
            com_integrals.data,         // sendbuf
            com_integrals_global.data,  // recvbuf
            4,                          // count
            MPI_DOUBLE,                 // datatype
            MPI_SUM,                    // op
            MPI_COMM_WORLD              // comm
        );

        bool eq_symm = get_param<bool>("amr","reflection_symmetries","z") ; 
        this->location[0] = com_integrals_global.data[0]/com_integrals_global.data[3];
        this->location[1] = com_integrals_global.data[1]/com_integrals_global.data[3];
        this->location[2] = eq_symm ? 0 : com_integrals_global.data[2]/com_integrals_global.data[3];
    } 
} ; 


/**
 * @brief Tracks a puncture
 */
class puncture_tracker_t : public co_tracker_t 
{
    public: 
    using co_tracker_t::co_tracker_t;  // inherits all base ctors

    std::string get_kind() const override final {return "bh";}

    void update(double dt) override final {
        auto l = this->location ; 

        using namespace grace  ; 
        using namespace Kokkos ;
        
        DECLARE_GRID_EXTENTS ; 
        
        // interpolate shift at old location 
        bool puncture_is_here ; 
        auto beta = interpolate_shift(l, puncture_is_here) ; 
        
        
        if (puncture_is_here) {
            l[0] -= dt * beta[0] ; 
            l[1] -= dt * beta[1] ; 
            l[2] -= dt * beta[2] ; 
        }

        // now we need to exchange via MPI 
        // Find who owns it - everyone needs to agree on the root rank
        auto my_rank = parallel::mpi_comm_rank() ; 
        int owner = -1;
        if (puncture_is_here) owner = my_rank;
        MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (owner<0) {
            GRACE_WARN("Puncture location ({} {} {}) outside of the computational domain!", l[0], l[1], l[2]) ; 
            return ; 
        }
        MPI_Bcast(l.data(), 3, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        this->location = l;
    } 

    private:

    std::array<double,3> interpolate_shift(
        std::array<double,3> const& center,
        bool& have_it
    ) const 
    {
        DECLARE_GRID_EXTENTS ; 
        // very inefficient on a single point 
        // but we need to get by somehow 
        point_host_t c ; 
        c.first = 0 ; 
        c.second = center ; 
        auto carr = sc_array_new_data(
            &c, sizeof(point_host_t), 1
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
            carr
        ) ; 

        if (intersecting_points_h.size()==0) {
            have_it = false ; 
            return {0,0,0} ; 
        }
        have_it = true ; 

        lagrange_interpolator_t<LAGRANGE_INTERP_ORDER> interpolator{ngz} ; 
        interpolator.compute_weights(
            std::vector<point_host_t>{{c}},
            intersecting_points_h,
            intersected_cells_h
        ) ; 

        Kokkos::View<double**,grace::default_space> vals("ptracker_shift",0,0); 
        auto& state = grace::variable_list::get().getstate() ; 
        interpolator.interpolate(
            state, {BETAX_,BETAY_,BETAZ_}, vals
        ) ; 
        auto vals_h = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), vals 
        ) ; 

        bool eq_symm = get_param<bool>("amr","reflection_symmetries","z") ; 
        std::array<double,3> beta ; 
        beta[0] = vals_h(0,0) ; 
        beta[1] = vals_h(0,1) ; 
        beta[2] = eq_symm ? 0 : vals_h(0,2) ; 

        return beta ; 
    }
} ; 


class co_tracker_list_impl_t {
    public:

    void update_and_write() {
        auto& grace_runtime = grace::runtime::get() ; 
        size_t const iter = grace_runtime.iteration() ;
        auto const dt = grace_runtime.timestep() ; 
        if ( (update_every>0) && (iter%update_every == 0)) {
            for( auto const& co: COs) {
                co->update(dt * update_every) ;
                co->output() ; 
            }
            if (n_cos>1) {
                auto l1 = COs[0]->get_loc() ; 
                auto l2 = COs[1]->get_loc() ; 
                cur_distance = Kokkos::sqrt(
                    SQR(l1[0]-l2[0]) + 
                    SQR(l1[1]-l2[1]) + 
                    SQR(l1[2]-l2[2]) 
                ) ; 
                // only set once, after merger locations get inaccurate
                if (! merged ) {
                    merged = cur_distance < merge_distance ; 
                }
            }
        }
    }

    bool is_active() const { return (n_cos>0) && (update_every>0) ; }

    // checkpointing utilities 
    int get_n_cos()  const { return n_cos  ; }
    int get_merged() const { return merged ; }

    // accessor 
    std::unique_ptr<co_tracker_t> const& 
    get(int i) const { return COs[i] ; }

    protected: 
    //**************************************************************************************************
    int n_cos ; 
    //**************************************************************************************************
    bool merged{false} ; 
    //**************************************************************************************************
    double merge_distance, cur_distance ; 
    //**************************************************************************************************
    int update_every ; 
    //**************************************************************************************************
    std::vector<std::unique_ptr<co_tracker_t>> COs ; 
    //**************************************************************************************************
    ~co_tracker_list_impl_t() = default ; 
    //**************************************************************************************************
    co_tracker_list_impl_t() {
        using namespace grace ; 

        update_every   = get_param<int>("co_tracker", "update_every") ; 

        n_cos          = get_param<int>("co_tracker", "n_cos") ; 
        merge_distance = get_param<double>("co_tracker", "merge_distance") ; 

        merged = false ;

        cur_distance = std::numeric_limits<double>::max() ; 

        auto co_list = grace::config_parser::get().get()["co_tracker"]["cos"] ; 

        ASSERT( co_list.size() == n_cos, "Mismatched number of cos provided") ; 

        COs.clear() ; 
        COs.reserve(n_cos) ; 

        std::set<std::string> co_names; 
        for( int i=0; i<n_cos; ++i) {
            auto co_pars = co_list[i] ; 
            // this must succeed, the par checked would have caught it 
            auto kind = co_pars["kind"].as<std::string>() ; 
            std::array<double,3> loc_0 ; 
            loc_0[0] = co_pars["x_i"].as<double>() ; 
            loc_0[1] = co_pars["y_i"].as<double>() ; 
            loc_0[2] = co_pars["z_i"].as<double>() ; 
            double radius = co_pars["radius"].as<double>() ; 
            std::string name = co_pars["name"].as<std::string>() ; 
            auto [it,inserted] = co_names.insert(name) ; 
            if (!inserted) ERROR("Duplicate co name " << name) ;  
            if ( kind == "ns" ) {
                COs.push_back(
                    std::make_unique<ns_tracker_t>(
                        loc_0, radius, name
                    )
                ) ; 
            } else {
                COs.push_back(
                    std::make_unique<puncture_tracker_t>(
                        loc_0, radius, name
                    )
                ) ;
            }
            
        }


    }
    //**************************************************************************************************
    void clear_co_list() { COs.clear() ; }
    //**************************************************************************************************
    void set_co_list(std::vector<std::unique_ptr<co_tracker_t>>&& cos) {COs = std::move(cos);}
    //**************************************************************************************************
    void set_merged(bool _merged) {
        merged=_merged;
    }
    //**************************************************************************************************
    static constexpr unsigned long longevity = unique_objects_lifetimes::GRACE_SPHERICAL_SURFACES ; 
    //**************************************************************************************************
    //**************************************************************************************************
    friend class utils::singleton_holder<co_tracker_list_impl_t> ;
    friend class memory::new_delete_creator<co_tracker_list_impl_t, memory::new_delete_allocator> ;
    friend class grace::checkpoint_handler_impl_t ;  
    //**************************************************************************************************
} ; 

//**************************************************************************************************
//**************************************************************************************************
using co_tracker = utils::singleton_holder<co_tracker_list_impl_t> ; 
//**************************************************************************************************

}

#endif
