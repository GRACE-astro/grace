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
#include <utility>
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

#include <grace/physics/id/kadath_helpers.hh>


#include <Kokkos_Core.hpp>

//template function that handles moving the space-time values to GRACE state
template<typename T, typename S>
void import_data(std::vector<std::reference_wrapper<T>>& state_ref, S& exported_vals) {

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

    bool set_shift_to_zero_on_import = grace::get_param<bool>("kadath","set_shift_to_zero_on_import");
    if(set_shift_to_zero_on_import){
        for (int i = 0; i < npoints; ++i) {
            state_ref[i*nfields+K_BETAX].get() = 0.0; 
            state_ref[i*nfields+K_BETAY].get() = 0.0; 
            state_ref[i*nfields+K_BETAZ].get() = 0.0; 
        }   
      }
    
}

//template function that handles moving the space-time and matter field values to their GRACE couterparts
// note that the two are separate in case we ever need to add matter-pecific statements here:

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

    //const bool set_shift_to_zero_on_import = grace::get_param<const bool>("kadath","set_shift_to_zero_on_import");
    bool set_shift_to_zero_on_import = grace::get_param<bool>("kadath","set_shift_to_zero_on_import");
    if(set_shift_to_zero_on_import){
        for (int i = 0; i < npoints; ++i) {
          // + 1, then +2 ? 
            state_ref[i*nfields+K_BETAX].get() = 0.0; 
            state_ref[i*nfields+K_BETAY].get() = 0.0; 
            state_ref[i*nfields+K_BETAZ].get() = 0.0; 
        }   
      }
}



void KadathImporter(const std::string kadath_id, const std::string  filename,
                    std::vector<std::reference_wrapper<double>>& state_ref,
                     const std::vector<std::array<double,GRACE_NSPACEDIM>>& pcoords,
                     const int nfields, const int npoints) {
 
  GRACE_INFO("Setting up coordinates");

  std::vector<double> xx(npoints), yy(npoints), zz(npoints);

  //#pragma omp parallel for
  for (int i = 0; i < npoints; ++i) {
    xx[i] = pcoords[i][0];
    yy[i] = pcoords[i][1];
    zz[i] = pcoords[i][2];
  }

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
  GRACE_VERBOSE("Filling FUKA ID took {:.3e} s.",currentTime) ;

}
