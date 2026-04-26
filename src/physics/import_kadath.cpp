/**
 * @file import_kadath.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * Modified by Carlo Musolino <carlo.musolino@aei.mpg.de>: remove copies of data and coordinates,
 * read e instead of rho and eps.
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
#include <grace/data_structures/memory_defaults.hh>
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


 
void KadathImporter(const std::string kadath_id, const std::string  filename,
                    const std::vector<double> & xx, 
                    const std::vector<double> & yy, 
                    const std::vector<double> & zz,
                    Kokkos::View<double *****, grace::default_space>& ddata, 
                    const int nfields, const int npoints, size_t nx, size_t ny, size_t nz, size_t ngz) {
 
  GRACE_INFO("Importing FUKA data onto the grid...");
  std::string id_type{kadath_id};
  auto clock_start = std::chrono::high_resolution_clock::now() ; 
  auto data = Kokkos::create_mirror_view(ddata) ; 

  // new exporters with OpenMP support
  #ifndef KADATH_EXPORTERS_PARALLEL
  ERROR("Only OpenMP parallel exporters are supported.") ; 
  #endif 
  GRACE_TRACE("Utilizing parallel exporters");

  auto interp_data_helper = [&]<typename reader_t, bool has_matter = false>(reader_t& input_reader,
                                  std::vector<double> const & xgrid,
                                  std::vector<double> const & ygrid,
                                  std::vector<double> const & zgrid
                                )
    {
      using kadath_output_t =
        std::vector<std::array<double, NUM_VOUT + NUM_MATTER * static_cast<int>(has_matter)>>;
    
      #pragma omp parallel for firstprivate(input_reader)
      for (size_t idx = 0; idx < xgrid.size(); ++idx) {
        size_t const i = idx%(nx+2*ngz); 
        size_t const j = (idx/(nx+2*ngz)) % (ny+2*ngz) ;
        size_t const k = 
                (idx/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ;
        size_t const q = 
                (idx/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ; 
        auto all_data_pt = input_reader.export_pointwise(xgrid[idx], ygrid[idx], zgrid[idx]) ;
        data(K_ALPHA,i,j,k,q) = all_data_pt[K_ALPHA] ; 

        data(K_BETAX,i,j,k,q) = all_data_pt[K_BETAX] ;
        data(K_BETAY,i,j,k,q) = all_data_pt[K_BETAY] ;
        data(K_BETAZ,i,j,k,q) = all_data_pt[K_BETAZ] ;

        data(K_GXX,i,j,k,q) = all_data_pt[K_GXX] ;
        data(K_GXY,i,j,k,q) = all_data_pt[K_GXY] ;
        data(K_GXZ,i,j,k,q) = all_data_pt[K_GXZ] ;
        data(K_GYY,i,j,k,q) = all_data_pt[K_GYY] ;
        data(K_GYZ,i,j,k,q) = all_data_pt[K_GYZ] ;
        data(K_GZZ,i,j,k,q) = all_data_pt[K_GZZ] ;

        data(K_KXX,i,j,k,q) = all_data_pt[K_KXX] ;
        data(K_KXY,i,j,k,q) = all_data_pt[K_KXY] ;
        data(K_KXZ,i,j,k,q) = all_data_pt[K_KXZ] ;
        data(K_KYY,i,j,k,q) = all_data_pt[K_KYY] ;
        data(K_KYZ,i,j,k,q) = all_data_pt[K_KYZ] ;
        data(K_KZZ,i,j,k,q) = all_data_pt[K_KZZ] ;
        if (has_matter) {
          // Store FUKA's rho and eps directly so the full matter state is
          // preserved bit-for-bit; previously we packed e = rho*(1+eps) and
          // re-inverted via GRACE's cold table, which biased rho whenever the
          // two cold EOSs disagreed.
          data(K_RHO,i,j,k,q) =  all_data_pt[K_RHO] * ( 1. + all_data_pt[K_EPS] ) ;

          data(K_RHO+1,i,j,k,q) =  all_data_pt[K_VELX] ;
          data(K_RHO+2,i,j,k,q) =  all_data_pt[K_VELY] ;
          data(K_RHO+3,i,j,k,q) =  all_data_pt[K_VELZ] ;
        }
        

      }
    };
                                
  if(id_type == "BH") {
    using config_t = Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BCO_BH_INFO>;
    using reader_t = Kadath::FUKA_Solvers::CFMS_BH_Exporter;
    reader_t input_reader(filename.c_str()); 
    interp_data_helper.template operator()<reader_t, true>(input_reader, xx,yy,zz);
  } else if(id_type == "BBH") {
    using config_t = Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BIN_INFO>;
    using reader_t = Kadath::FUKA_Solvers::CFMS_BBH_Exporter;
    reader_t input_reader(filename.c_str()); 
    interp_data_helper.template operator()<reader_t, true>(input_reader, xx,yy,zz);
  } else if(id_type == "NS") {
    using config_t = Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BCO_NS_INFO>;
    using reader_t = Kadath::FUKA_Solvers::CFMS_NS_Exporter;
    reader_t input_reader(filename.c_str()); 
    interp_data_helper.template operator()<reader_t, true>(input_reader, xx,yy,zz);
  } else if(id_type == "BNS") {
    using config_t = Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BIN_INFO>;
    using reader_t = Kadath::FUKA_Solvers::CFMS_BNS_Exporter;
    reader_t input_reader(filename.c_str()); 
    interp_data_helper.template operator()<reader_t, true>(input_reader, xx,yy,zz);
  } else if(id_type == "BHNS") {
    using config_t = Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BIN_INFO>;
    using reader_t = Kadath::FUKA_Solvers::CFMS_BHNS_Exporter;
    reader_t input_reader(filename.c_str()); 
    interp_data_helper.template operator()<reader_t, true>(input_reader, xx,yy,zz);
  } 
  
  GRACE_INFO("Finished Kadath import") ;
  auto clock_end = std::chrono::high_resolution_clock::now() ; 
  auto currentTime = double(std::chrono::duration_cast <std::chrono::seconds> (clock_end - clock_start).count());
  GRACE_VERBOSE("Filling FUKA ID took {:.3e} s.",currentTime) ;
  Kokkos::deep_copy(ddata,data) ; 
}
