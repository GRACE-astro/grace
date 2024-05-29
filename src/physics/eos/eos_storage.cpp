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
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/system/grace_system.hh>
#include <grace/config/config_parser.hh>

#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/eos/constants.hh> //! Todo 

#include <grace/physics/eos/eos_storage.hh>

#include <Kokkos_Core.hpp>

namespace grace {

eos_storage_t::eos_storage_t {
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

            unsigned int _pwpoly_n_pieces = 
                params["eos"]["piecewise_polytrope"]["n_pieces"].as<unsigned int>() ; 
            
            std::vector<double> _pwpoly_kappas_vec =
                params["eos"]["piecewise_polytrope"]["kappas"].as<std::vector<double>>() ; 
            std::vector<double> _pwpoly_gammas_vec  =
                params["eos"]["piecewise_polytrope"]["gammas"].as<std::vector<double>>() ; 
            std::vector<double> _pwpoly_rhos_vec  =
                params["eos"]["piecewise_polytrope"]["rhos"].as<std::vector<double>>() ; 
            std::vector<double> _pwpoly_press_vec  =
                params["eos"]["piecewise_polytrope"]["pressures"].as<std::vector<double>>() ; 
            std::vector<double> _pwpoly_eps_vec  =
                params["eos"]["piecewise_polytrope"]["eps"].as<std::vector<double>>() ; 

            #define DEEP_COPY_VEC_TO_VIEW(vec,view) \
            do { \
                auto host_view = Kokkos::create_mirror_view(view) ; \
                for( int i=0; i < vec.size(); ++i){                 \
                    host_view(i) = vec[i] ;                         \
                }                                                   \
                Kokkos::deep_copy(view,host_view) ;                 \
            } while(0)

            Kokkos::View<double [piecewise_polytropic_eos_t::max_n_pieces], grace::default_space> 
                _pwpoly_kappas("Piecewise polytropic indices") ; 
            Kokkos::View<double [piecewise_polytropic_eos_t::max_n_pieces], grace::default_space> 
                _pwpoly_gammas("Piecewise polytropic adiabatic compressibilities") ; 
            Kokkos::View<double [piecewise_polytropic_eos_t::max_n_pieces], grace::default_space> 
                _pwpoly_rhos("Piecewise polytropic densities") ;
            Kokkos::View<double [piecewise_polytropic_eos_t::max_n_pieces], grace::default_space> 
                _pwpoly_press("Piecewise polytropic pressures") ;
            Kokkos::View<double [piecewise_polytropic_eos_t::max_n_pieces], grace::default_space> 
                _pwpoly_eps("Piecewise polytropic specific internal energies") ;
            
            DEEP_COPY_VEC_TO_VIEW(_pwpoly_kappas_vec,_pwpoly_kappas) ;
            DEEP_COPY_VEC_TO_VIEW(_pwpoly_gammas_vec,_pwpoly_gammas) ;
            DEEP_COPY_VEC_TO_VIEW(_pwpoly_rhos_vec,_pwpoly_rhos) ;
            DEEP_COPY_VEC_TO_VIEW(_pwpoly_press_vec,_pwpoly_press) ;
            DEEP_COPY_VEC_TO_VIEW(_pwpoly_eps_vec,_pwpoly_eps) ;

            piecewise_polytropic_eos_t _pwpoly{
                  _pwpoly_kappas
                , _pwpoly_gammas
                , _pwpoly_rhos
                , _pwpoly_eps
                , _pwpoly_press
                , _pwpoly_n_pieces
                , 1
                , 1e-20
            } ; 

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
    } else {
        ERROR("Unsupported eos_type") ; 
    }

}

}