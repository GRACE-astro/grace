/**
 * @file healpix.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-03-12
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

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/utils/format_utils.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/system/grace_system.hh>
#include <grace/config/config_parser.hh>

#include <grace/healpix/healpix_state.hh> 



namespace grace {

        namespace healpix {

                // definition of the constructor 
                healpix_impl_t::healpix_impl_t(){
                        auto& params        = grace::config_parser::get() ;
                        detectors_radii     = params["healpix"]["healpix_radii"].as<std::vector<double>>() ;
                        num_of_detectors    = detectors_radii.size();
                        const size_t nside  = params["healpix"]["healpix_nside"].as<int>() ;
                        _healpix_detectors.reserve(num_of_detectors);

                        for (size_t idx_det=0; idx_det < num_of_detectors ; idx_det++) {
                                _healpix_detectors.emplace_back(nside , detectors_radii[idx_det]);
                        }
                }

                // definition of the destructor 
                healpix_impl_t::~healpix_impl_t(){}

                void GRACE_HOST
                healpix_impl_t::update_detectors_info(){
                        for(auto det: _healpix_detectors) det.update_detector_info();
                }



        }
}