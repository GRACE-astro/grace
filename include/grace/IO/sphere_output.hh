

/**
 * @file sphere_output.hh
 * @authors Konrad Topolski, Kenneth Miller 
 * @brief 
 * @date 2025-03-21
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#ifndef GRACE_SPHERE_OUTPUT
#define GRACE_SPHERE_OUTPUT

#include <grace_config.h> 
#include <hdf5.h>

#include <array>
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/utils/make_string.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <cmath>

#include <grace/healpix/detectors.hh>


/**
 * @brief This file governs the scheduling and computation of spherical surface output
 * 
 *  TODO: There is a number of different kinds of output we want:
 *        1. 2D healpix output of a registered variable (interpolation onto the sphere)
 *        2. 2D healpix output of a named variable (e.g. integrand consisting of several fields)
 *        3. 2D healpix output for vector-type variables in the form of a contraction with the normal to the surface
 *        4. 0D multipole output for (1), (2), (3)
 *        5. Surface integral of (1), (2) and (3) above
 * 
 *        For each of the above types, we may additionally be dealing with scalar, vector, tensor
 *        or symmetric tensor variables.
 *        The structure of this code should reflect the aforementioned variety of 
 *        request and be designed flexible enough, without code duplication.
 */

namespace grace { namespace IO {


    using namespace healpix;
    using namespace Kokkos;
    extern std::map<std::string, spherical_detector> detectors;
    extern std::vector<std::vector<std::vector<double>>> spherical_harmonics_re; 
    extern std::vector<std::vector<std::vector<double>>> spherical_harmonics_im; 

    using Complex = Kokkos::complex<double>;
    using HostM   = Kokkos::HostSpace;
    // the size of all the used spin-weighted spherical harmonics (SWSH)
    // is dictated by max_ell
    // and nside 
    // dim0 - (max_ell+1)*(max_ell+1)  // flattened indexing of SWSH space
    // dim1 - 12 * nside * nside       // indexing of individual SWSH values at healpix points  
    // extern Kokkos::View<Complex**, HostM> sw_sph_harmonics;


    void initialize_spherical_detectors(const int n_detectors, 
                                        const int nside,
                                        const int ntheta,
                                        const int nphi, 
                                        const std::vector<std::array<double,3>> output_spheres_centres,
                                        const std::vector<double> output_spheres_radii,
                                        const std::vector<std::string> output_spheres_names,
                                        const std::vector<std::string> output_sphere_types);

    void update_spherical_detectors();

    void compute_spherical_surface_variable_data(std::set<std::string> corner_scalar_vars,
                                                 std::set<std::string> corner_vector_vars, 
                                                 std::set<std::string> corner_tensor_vars, 
                                                 std::set<std::string> cell_scalar_vars, 
                                                 std::set<std::string> cell_vector_vars,
                                                 std::set<std::string> cell_tensor_vars );

    void write_sphere_cell_data_hdf5() ; 


    void save_multipole_timeseries_ascii(const std::string& parent_path,
                                        const double radius,
                                        const int spin_weight,
                                        const int max_ell,
                                        const std::set<std::string>& vars_names,
                                        const std::map<std::string, Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>> all_multipoles, 
                                        const double iter,
                                        const double current_time );

   void save_multipole_timeseries_hdf5_init(const std::string& abs_path,
                                             const double radius,
                                             const int spin_weight, 
                                             const int max_ell,
                                             std::set<std::string> vars_names);
                                             
  
    void save_multipole_timeseries_hdf5(const std::string& abs_path,
                                        const double spin_weight,
                                        const double max_ell,
                                        const std::set<std::string>& vars_names,
                                        const std::map<std::string, View<Complex*, HostM>> all_multipoles, 
                                        const double current_time );

    void update_spin_weighted_spherical_harmonics(Kokkos::View<Complex**, HostM>& sph_harmonics, 
                                                  const int spin_weight, 
                                                  const int max_ell,
                                                  const int ntheta,
                                                  const int nphi);
    

    std::map<std::string, View<Complex*, HostM>> complexify_detector_data(std::map<std::string,std::vector<double>> const& det_surface_data);

    std::map<std::string, View<Complex*,HostM>> get_all_multipoles(const int spin_weight, 
                                                                   const int max_ell,
                                                                   const int ntheta,
                                                                   const int nphi,
                                                                   const std::vector<int>& det_healpix_indices,
                                                                   const View<Complex**, HostM>& sw_sph_harmonics,
                                                                   const std::map< std::string, View<Complex*, HostM>>& complex_det_surface_data);

    void write_multipole_timeseries() ;
     

    void save_surface_integral_timeseries_ascii(const std::string& abs_path,
                                        const double radius,
                                        const std::map<std::string, double>& surface_integrals, 
                                        const int current_iteration,
                                        const double current_time  );

    void write_spherical_integrals_timeseries();


    void write_multipole_timeseries_2() ; 

    }


}



#endif /** GRACE_SPHERE_OUTPUT  */