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


#include <iostream>


#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/parallel/mpi_wrappers.hh>
// #include <grace/system/runtime_functions.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/error.hh>
#include <grace/system/print.hh>
#include <Kokkos_Core.hpp>

#include <grace/physics/id/import_kadath.hh>

#ifdef KADATH_EXPORTERS_PARALLEL
#include <omp.h>
#include "Solvers/bh_3d_xcts/bh_exporter.hpp"
#include "Solvers/ns_3d_xcts/ns_exporter.hpp"
#include "Solvers/bbh_xcts/bbh_exporter.hpp"
#include "Solvers/bns_xcts/bns_exporter.hpp"
#include "Solvers/bhns_xcts/bhns_exporter.hpp"
#endif 

//template function that handles moving the space-time values to GRACE state
template<typename T, typename S>
void import_data(std::vector<std::reference_wrapper<T>>& state_ref, S& exported_vals) {

    GRACE_INFO("Moving Vacuum Data to output vars");
    const int nfields = NUM_VOUT;
    const int npoints = int(state_ref.size()/nfields);
    GRACE_INFO("Initializing {} fields on {} points", nfields, npoints) ;

    #pragma omp parallel for collapse(2) 
    for (int i = 0; i < npoints; ++i) {
      for (int idx_field=0; idx_field<nfields; idx_field++){
        state_ref[i*nfields+idx_field].get() = exported_vals[i][idx_field]; 
        }
      }

    bool set_shift_to_zero_on_import = grace::get_param<bool>("grmhd","fuka", "set_shift_to_zero_on_import");
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
    
    #pragma omp parallel for collapse(2) 
    for (int i = 0; i < npoints; ++i) {
      for (int idx_field=0; idx_field<nfields; idx_field++){
        state_ref[i*nfields+idx_field].get() = exported_vals[i][idx_field]; 
      }
    }

    //const bool set_shift_to_zero_on_import = grace::get_param<const bool>("kadath","set_shift_to_zero_on_import");
    bool set_shift_to_zero_on_import = grace::get_param<bool>("grmhd","fuka", "set_shift_to_zero_on_import");
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
  std::string id_type{kadath_id};
  std::vector<double> xx(npoints), yy(npoints), zz(npoints);
                
  #pragma omp parallel for
  for (int i = 0; i < npoints; ++i) {
    xx[i] = pcoords[i][0];
    yy[i] = pcoords[i][1];
    zz[i] = pcoords[i][2];
  }


  auto clock_start = std::chrono::high_resolution_clock::now() ; 

  // legacy serial version 
  #ifdef KADATH_EXPORTERS_SERIAL
    const double interpolation_offset = grace::get_param<double>("grmhd", "fuka", "id_interpolation_offset");
    const int interp_order = grace::get_param<int>("grmhd", "fuka","junk_interp_order");
    const double delta_r_rel = grace::get_param<double>("grmhd", "fuka","delta_r_rel");
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
  #endif

  // new exporters with OpenMP support
  #ifdef KADATH_EXPORTERS_PARALLEL

    // templated lambdas feature template parameters right after the closure bracket
    auto interp_data_helper = []<typename reader_t, bool has_matter = false>(reader_t& input_reader,
                                    std::vector<double>& xgrid,
                                    std::vector<double>& ygrid,
                                    std::vector<double>& zgrid)
      {
        using kadath_output_t =
          std::vector<std::array<double, NUM_VOUT + NUM_MATTER * static_cast<int>(has_matter)>>;

        kadath_output_t all_data(xgrid.size());
      
	#pragma omp parallel for firstprivate(input_reader), schedule(static)
	for (size_t idx = 0; idx < xgrid.size(); ++idx) {
	  all_data[idx] = input_reader.export_pointwise(xgrid[idx], ygrid[idx], zgrid[idx]);
	}

        return std::move(all_data);
      };
                                
  if(id_type == "BH") {
    using config_t = Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BCO_BH_INFO>;
    using reader_t = Kadath::FUKA_Solvers::CFMS_BH_Exporter;
    reader_t input_reader(filename.c_str()); 
    auto exported_vals = std::move(interp_data_helper.template operator()<reader_t, false>(input_reader, xx,yy,zz)); // for a templated lambda 
    // auto exported_vals = std::move(interp_data_helper<reader_t, false>(input_reader, xx,yy,zz));   // for a separate function
    import_data(state_ref,exported_vals);
  } else if(id_type == "BBH") {
    using config_t = Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BIN_INFO>;
    using reader_t = Kadath::FUKA_Solvers::CFMS_BBH_Exporter;
    reader_t input_reader(filename.c_str()); 
    auto exported_vals = std::move(interp_data_helper.template operator()<reader_t, false>(input_reader, xx,yy,zz));
    import_data(state_ref,exported_vals);
  } else if(id_type == "NS") {
    using config_t = Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BCO_NS_INFO>;
    using reader_t = Kadath::FUKA_Solvers::CFMS_NS_Exporter;
    reader_t input_reader(filename.c_str()); 
    auto exported_vals = std::move(interp_data_helper.template operator()<reader_t, true>(input_reader, xx,yy,zz)); 
    import_data_wmatter(state_ref,exported_vals);
  } else if(id_type == "BNS") {
    using config_t = Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BIN_INFO>;
    using reader_t = Kadath::FUKA_Solvers::CFMS_BNS_Exporter;
    reader_t input_reader(filename.c_str()); 
    auto exported_vals = std::move(interp_data_helper.template operator()<reader_t, true>(input_reader, xx,yy,zz));
    import_data_wmatter(state_ref,exported_vals);
  } else if(id_type == "BHNS") {
    using config_t = Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BIN_INFO>;
    using reader_t = Kadath::FUKA_Solvers::CFMS_BHNS_Exporter;
    reader_t input_reader(filename.c_str()); 
    auto exported_vals = std::move(interp_data_helper.template operator()<reader_t, true>(input_reader, xx,yy,zz));
    import_data_wmatter(state_ref,exported_vals);
  } 
  #endif
  
  GRACE_INFO("Finished Kadath import") ;
  auto clock_end = std::chrono::high_resolution_clock::now() ; 
  auto currentTime = double(std::chrono::duration_cast <std::chrono::seconds> (clock_end - clock_start).count());
  GRACE_VERBOSE("Filling FUKA ID took {:.3e} s.",currentTime) ;

}

 
