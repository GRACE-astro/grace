/**
 * @file eos_storage.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
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

#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
//#include <grace/physics/eos/physical_constants.hh> //! Todo 
#include <grace/physics/eos/eos_setup.hh>
#include <grace/physics/eos/eos_storage.hh>

#include <Kokkos_Core.hpp>

namespace grace {

eos_storage_t::eos_storage_t() {
    auto& params = grace::config_parser::get() ;

    std::string eos_type = 
        params["eos"]["eos_type"].as<std::string>() ;

    double c2p_ye_atm   = params["eos"]["ye_atmosphere"].as<double>(); 
    double c2p_rho_atm  = params["eos"]["rho_atmosphere"].as<double>(); 
    double c2p_temp_atm = params["eos"]["temp_atmosphere"].as<double>(); 
    double c2p_eps_max  = params["eos"]["eps_maximum"].as<double>();

    bool atm_is_beta_eq = params["eos"]["atm_is_beta_eq"].as<bool>();

    bool extend_table_high = params["eos"]["extend_table_high"].as<bool>(); ; 

    if ( eos_type == "hybrid" ) {
        std::string cold_eos_type =
            params["eos"]["cold_eos_type"].as<std::string>() ; 
        double gamma_th = params["eos"]["gamma_th"].as<double>() ;

        if( cold_eos_type == "piecewise_polytrope" ) {

            auto _pwpoly = grace::setup_cold_politrope() ; 

            _hybrid_pwpoly = hybrid_eos_t<piecewise_polytropic_eos_t>{
                  _pwpoly 
                , gamma_th - 1. 
                , grace::physical_constants::mnuc_CGS
                , c2p_rho_atm 
                , c2p_eps_max 
            } ; 

        } else {
            ERROR("Unsupported cold_eos_type.") ; 
        }
    } else if (eos_type == "tabulated") { 

        std::string table_path = "/home/it4i-kpierre/data/DD2+VQCD_soft_quark_fraction.h5";  

        _tabulated = grace::setup_tabulated_eos_compose(table_path.c_str());


    } else {
        ERROR("Unsupported eos_type") ; 
    }

}

}