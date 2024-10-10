/**
 * @file import_kadath.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * Subject to GPL and adapted from the work of author(s)/maintainer(s):
 * Samuel David Tootle <tootle@itp.uni-frankfurt.de>
 * Ludwig Jens Papenfort <papenfort@th.physik.uni-frankfurt.de>
 * @brief 
 * @date 2024-08-29
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


#include <cmath>
#include <functional>
#include <string>
#include <chrono>
#include <vector>
#include <array>
#include <type_traits>


#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/profiling/profiling.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/runtime_functions.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/error.hh>
#include <grace/system/print.hh>


// #include <grace/physics/grace_physical_systems.hh>
// #include <grace/amr/amr_functions.hh>
// #include <grace/data_structures/variable_indices.hh>
// #include <grace/data_structures/variables.hh>
// #include <grace/system/grace_system.hh>
// #include <grace/coordinates/coordinate_systems.hh>

#include <grace/physics/id/kadath_helpers.hh>


#include <Kokkos_Core.hpp>

// TO DO: 
// With the rewrite of import_data, it actually no longer matters 
// whether we call import_data or import_data_wmatter.
// It used to matter in ETK, where rho,press,... grid functions 
// had to be explicitly invoked on the LHS of the assignment operator.

//template function that handles moving the space-time values to GRACE state
template<typename T, typename S>
void import_data(std::vector<std::reference_wrapper<T>>& state_ref, S& exported_vals) {
    //using namespace KadathImport;

    GRACE_INFO("Moving Vacuum Data to output vars");
    //const int nfields = state_ref.size();
    const int nfields = NUM_VOUT;
    const int npoints = int(state_ref.size()/nfields);
    GRACE_INFO("Initializing {} fields on {} points", nfields, npoints) ;

    #pragma omp parallel for collapse(2) // is this optimal? which is faster?
    for (int i = 0; i < npoints; ++i) {
      for (int idx_field=0; idx_field<nfields; idx_field++){
        state_ref[i*nfields+idx_field].get() = exported_vals[idx_field][i]; 
        }
      }
}

//template function that handles moving the space-time and matter field values to their GRACE couterparts
template<typename T, typename S>
void import_data_wmatter(std::vector<std::reference_wrapper<T>>& state_ref, S& exported_vals) {
    GRACE_INFO("Moving Matter Data to output vars");

    const int nfields = NUM_OUT;
    const int npoints = int(state_ref.size()/nfields);
    
    //#pragma omp parallel for collapse 2 // is this optimal? 
    for (int i = 0; i < npoints; ++i) {
      for (int idx_field=0; idx_field<nfields; idx_field++){

        state_ref[i*nfields+idx_field].get() = exported_vals[idx_field][i]; 
      }
    }

}



void KadathImporter(const std::string kadath_id, const std::string  filename,
                    std::vector<std::reference_wrapper<double>>& state_ref,
                     const std::vector<std::array<double,GRACE_NSPACEDIM>>& pcoords,
                     const int nfields, const int npoints) {
 
  GRACE_INFO("Setting up KADATH initial data");
  GRACE_INFO("Setting up coordinates");
  //using namespace KadathImport;

  std::vector<double> xx(npoints), yy(npoints), zz(npoints);

  //#pragma omp parallel for
  for (int i = 0; i < npoints; ++i) {
    xx[i] = pcoords[i][0];
    yy[i] = pcoords[i][1];
    zz[i] = pcoords[i][2];
  }

  GRACE_INFO("Starting the import of {} ID", kadath_id);
  GRACE_INFO("Absolute path: {}", filename);

  const double interpolation_offset = grace::get_param<double>("kadath","id_interpolation_offset");
  const int interp_order = grace::get_param<int>("kadath","junk_interp_order");
  const double delta_r_rel = grace::get_param<double>("kadath","delta_r_rel");

  std::string id_type{kadath_id};
  auto clock_start = std::chrono::high_resolution_clock::now() ; 

    if(id_type == "BH") {
    auto exported_vals = std::move(KadathExportBH(npoints, xx.data(), yy.data(), zz.data(),
                         filename.c_str(), interpolation_offset, interp_order, delta_r_rel));
    import_data(state_ref,exported_vals);
  } else if(id_type == "BBH") {
    auto exported_vals = std::move(KadathExportBBH(npoints, xx.data(), yy.data(), zz.data(), 
                         filename.c_str(), interpolation_offset, interp_order, delta_r_rel));
    import_data(state_ref,exported_vals);
  } else if(id_type == "NS") {
    auto exported_vals = std::move(KadathExportNS(npoints, xx.data(), yy.data(), zz.data(), filename.c_str()));
    import_data_wmatter(state_ref,exported_vals);
  } else if(id_type == "BNS") {
    auto exported_vals = std::move(KadathExportBNS(npoints, xx.data(), yy.data(), zz.data(), filename.c_str()));
    import_data_wmatter(state_ref,exported_vals);
  } else if(id_type == "BHNS") {
    auto exported_vals = std::move(KadathExportBHNS(npoints, xx.data(), yy.data(), zz.data(), 
                         filename.c_str(), interpolation_offset, interp_order, delta_r_rel));
    import_data_wmatter(state_ref,exported_vals);
  } 
  
  GRACE_INFO("Finished Kadath import") ;
  auto clock_end = std::chrono::high_resolution_clock::now() ; 
  auto currentTime = double(std::chrono::duration_cast <std::chrono::seconds> (clock_end - clock_start).count());
  GRACE_VERBOSE("Filling Kadath ID took {:.3e} s.",currentTime) ;

}
