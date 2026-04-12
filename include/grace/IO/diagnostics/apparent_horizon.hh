/**
 * @file apparent_horizon.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Apparent horizon finder using spectral methods.
 * @date 2026-04-11
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

#ifndef GRACE_IO_APPARENT_HORIZON_HH
#define GRACE_IO_APPARENT_HORIZON_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>

#include <grace/utils/metric_utils.hh>
#include <grace/utils/lagrange_interpolation.hh>
#include <grace/utils/spherical_harmonics.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/config/config_parser.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/grace_runtime.hh>
#include <grace/coordinates/coordinate_systems.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <fstream>
#include <cmath>

namespace grace {

class checkpoint_handler_impl_t ;

struct ah_config_t {
    int l_max               = 8     ;
    double tolerance         = 1e-10 ;
    int max_iterations       = 50    ;
    double initial_radius    = 0.5   ;
    std::array<double,3> center = {0,0,0} ;
    int find_every           = 1     ;
    std::string name         = "AH0" ;
    double jacobian_eps      = 1e-6  ;
    bool verbose             = false ;
};

struct ah_result_t {
    bool found               = false ;
    int iterations           = 0     ;
    double residual          = 1e30  ;
    double area              = 0     ;
    double irreducible_mass  = 0     ;
    std::array<double,3> centroid = {0,0,0} ;
    std::vector<double> coefficients ;
};

struct ah_flux_result_t {
    double mdot = 0 ;   // mass accretion rate
    double edot = 0 ;   // energy flux
    double ldot = 0 ;   // angular momentum flux
    double phi  = 0 ;   // magnetic flux
};

/**
 * @brief Apparent horizon finder via spectral expansion.
 *
 * Represents the horizon surface as r = h(theta,phi) in real
 * spherical harmonics and solves Theta[h] = 0 via Newton-Raphson,
 * where Theta is the expansion of the outgoing null normal.
 *
 * Metric data is obtained by Lagrange interpolation from the
 * evolution grid. Metric gradients are computed via centered
 * finite differences on staggered interpolation points (6 extra
 * points per collocation point, offset by +/- delta in x,y,z).
 */
class apparent_horizon_finder_t {

public:

    apparent_horizon_finder_t(ah_config_t const& cfg) ;

    ah_result_t find() ;

    void initialize_output_file() ;
    void write_result(ah_result_t const& res) ;

    ah_flux_result_t compute_surface_fluxes() ;
    void initialize_flux_output_file() ;
    void write_flux_result(ah_flux_result_t const& res) ;

    ah_config_t const& config() const { return cfg_ ; }
    ah_result_t const& last_result() const { return last_result_ ; }

    std::array<double,3> get_center() const { return cfg_.center ; }
    void set_center(std::array<double,3> const& c) { cfg_.center = c ; }

    // checkpoint accessors
    std::vector<double> const& get_coefficients() const { return coeffs_ ; }
    void set_coefficients(std::vector<double> const& c) { coeffs_ = c ; has_initial_guess_ = true ; }
    bool was_found() const { return last_result_.found ; }

private:

    // --- configuration ---
    ah_config_t cfg_ ;
    ah_result_t last_result_ ;

    // --- spectral basis ---
    std::unique_ptr<spherical_harmonic_basis_t> basis_ ;

    // --- interpolation infrastructure ---
    lagrange_interpolator_t<LAGRANGE_INTERP_ORDER> interpolator_ ;

    // --- state ---
    std::vector<double> coeffs_ ;    // current spectral coefficients
    bool has_initial_guess_ = false ;

    // --- output ---
    std::filesystem::path outfilepath_ ;
    std::filesystem::path flux_outfilepath_ ;

    // --- local enums for interpolated variables ---
    // Center points: 13 Z4c state vars
    enum loc_var_idx_t : int {
        GTXXL=0, GTXYL, GTXZL, GTYYL, GTYZL, GTZZL,
        CHIL,
        ATXXL, ATXYL, ATXZL, ATYYL, ATYZL, ATZZL,
        KHATL,
        THETAL,
        NUM_INTERP_VARS
    };

    static constexpr int N_STAG_DIRS = 6 ; // +x,-x,+y,-y,+z,-z

    // --- internal methods ---

    void generate_surface_points(
        std::vector<point_host_t>& points,
        std::vector<std::array<double,2>>& angles,
        int& n_colloc,
        double delta
    ) const ;

    void setup_interpolation(
        std::vector<point_host_t> const& points,
        std::vector<size_t>& intersecting_points_out
    ) ;

    struct metric_data_t {
        // per collocation point
        std::array<double,6> gamma ;     // physical metric (XX,XY,XZ,YY,YZ,ZZ)
        std::array<double,6> gamma_inv ; // inverse metric
        std::array<double,6> K_ij ;      // extrinsic curvature
        double K_trace ;                 // trace of K_ij
        double sqrtg ;                   // sqrt(det(gamma))
        // metric gradients: d_k gamma_ij (k=0,1,2 for x,y,z; ij packed as 6)
        double dgamma[3][6] ;
    };

    void extract_metric_data(
        Kokkos::View<double**, Kokkos::HostSpace> const& ivals_h,
        int n_colloc,
        double delta,
        std::vector<metric_data_t>& mdata
    ) const ;

    double compute_expansion(
        int pt,
        metric_data_t const& md,
        double const* h_vals,
        double const* dh_dth,
        double const* dh_dph,
        double const* d2h_dth2,
        double const* d2h_dthdph,
        double const* d2h_dph2,
        double theta,
        double phi
    ) const ;

    void compute_residual(
        std::vector<metric_data_t> const& mdata,
        double const* a,
        std::vector<double>& residual
    ) const ;

    void compute_jacobian(
        std::vector<metric_data_t> const& mdata,
        double const* a,
        std::vector<double> const& residual,
        std::vector<double>& J
    ) const ;

    bool solve_normal_equations(
        std::vector<double> const& J,
        std::vector<double> const& residual,
        int n_rows,
        int n_cols,
        std::vector<double>& delta_a
    ) const ;

    ah_result_t compute_diagnostics(double const* a) const ;

    void compute_area_with_metric(
        std::vector<metric_data_t> const& mdata,
        ah_result_t& result
    ) const ;

    double get_stagger_delta() const ;
};

/**
 * @brief Singleton manager for all apparent horizon finders.
 *
 * Constructed from YAML config. Call find_all() each iteration
 * from the main loop when iter % find_every == 0.
 */
class ah_finder_manager_impl_t {
public:

    void find_all() {
        if (!active_) return ;
        auto& grace_runtime = grace::runtime::get() ;
        size_t const iter = grace_runtime.iteration() ;
        if (find_every_ <= 0) return ;
        if (iter % find_every_ != 0) return ;

        for (auto& ah : finders_) {
            auto res = ah->find() ;
            ah->write_result(res) ;
            // if found, optionally compute surface fluxes
            if (res.found && compute_fluxes_) {
                auto fluxes = ah->compute_surface_fluxes() ;
                ah->write_flux_result(fluxes) ;
            }
            // if found and tracking enabled, update center for next call
            if (res.found && track_center_flags_[&ah - &finders_[0]]) {
                ah->set_center(res.centroid) ;
            }
        }
    }

    bool is_active() const { return active_ ; }
    int n_horizons() const { return static_cast<int>(finders_.size()) ; }

    apparent_horizon_finder_t const& get(int i) const { return *finders_[i] ; }
    apparent_horizon_finder_t& get(int i) { return *finders_[i] ; }

protected:

    ah_finder_manager_impl_t() {
        int n = get_param<int>("apparent_horizon", "n_horizons") ;
        find_every_ = get_param<int>("apparent_horizon", "find_every") ;
        compute_fluxes_ = get_param<bool>("apparent_horizon", "compute_fluxes") ;
        bool verbose = get_param<bool>("apparent_horizon", "verbose") ;

        active_ = (n > 0) && (find_every_ > 0) ;
        if (!active_) return ;

        auto horizon_list = grace::config_parser::get().get()["apparent_horizon"]["horizons"] ;

        for (int i = 0; i < n; ++i) {
            auto h = horizon_list[i] ;
            ah_config_t cfg ;
            cfg.name            = h["name"].as<std::string>() ;
            cfg.l_max           = h["l_max"].as<int>() ;
            cfg.initial_radius  = h["initial_radius"].as<double>() ;
            cfg.center          = {
                h["center_x"].as<double>(),
                h["center_y"].as<double>(),
                h["center_z"].as<double>()
            } ;
            cfg.tolerance       = h["tolerance"].as<double>() ;
            cfg.max_iterations  = h["max_iterations"].as<int>() ;
            cfg.jacobian_eps    = h["jacobian_eps"].as<double>() ;
            cfg.find_every      = find_every_ ;
            cfg.verbose         = verbose ;

            finders_.push_back(std::make_unique<apparent_horizon_finder_t>(cfg)) ;
            track_center_flags_.push_back(h["track_center"].as<bool>()) ;
        }

        // initialize output files
        for (auto& ah : finders_) {
            ah->initialize_output_file() ;
            if (compute_fluxes_) ah->initialize_flux_output_file() ;
        }
    }

    ~ah_finder_manager_impl_t() = default ;

private:

    std::vector<std::unique_ptr<apparent_horizon_finder_t>> finders_ ;
    std::vector<bool> track_center_flags_ ;
    int find_every_ = -1 ;
    bool active_ = false ;
    bool compute_fluxes_ = false ;

    static constexpr unsigned long longevity = unique_objects_lifetimes::GRACE_SPHERICAL_SURFACES ;

    friend class utils::singleton_holder<ah_finder_manager_impl_t> ;
    friend class memory::new_delete_creator<ah_finder_manager_impl_t, memory::new_delete_allocator> ;
    friend class grace::checkpoint_handler_impl_t ;
};

using ah_finder_manager = utils::singleton_holder<ah_finder_manager_impl_t> ;

} // namespace grace

#endif /* GRACE_IO_APPARENT_HORIZON_HH */
