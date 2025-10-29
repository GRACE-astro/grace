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
#include <grace/physics/eos/physical_constants.hh> //! Todo 

#include <grace/physics/eos/eos_storage.hh>

#include <Kokkos_Core.hpp>

namespace grace {

static piecewise_polytropic_eos_t setup_cold_politrope() 
{
    auto& params = grace::config_parser::get() ;

    unsigned int _pwpoly_n_pieces = 
                params["eos"]["piecewise_polytrope"]["n_pieces"].as<unsigned int>() ; 
            
    double _pwpoly_kappas_0 =
        params["eos"]["piecewise_polytrope"]["kappa_0"].as<double>() ; 
    std::vector<double> _pwpoly_gammas_vec  =
        params["eos"]["piecewise_polytrope"]["gammas"].as<std::vector<double>>() ; 
    std::vector<double> _pwpoly_rhos_vec  =
        params["eos"]["piecewise_polytrope"]["rhos"].as<std::vector<double>>() ; 
    
    
    ASSERT( _pwpoly_gammas_vec.size() == _pwpoly_n_pieces
          , "Number of gammas does not coincide with n_pieces.") ;
    ASSERT( _pwpoly_rhos_vec.size() == _pwpoly_n_pieces - 1 
          , "The piecewise polytrope densities must be n_pieces-1.") ;
    /* Add 0 as first density */
    _pwpoly_rhos_vec.insert(_pwpoly_rhos_vec.begin(), 0.);
    for( int i=0; i<_pwpoly_rhos_vec.size()-1; ++i)
        ASSERT( _pwpoly_rhos_vec[i+1] > _pwpoly_rhos_vec[i]
              , "Piecewise polytrope densities must be in ascending order.") ; 
    
    /* Fill kappas eps and press */
    std::vector<double> _pwpoly_kappas_vec(_pwpoly_n_pieces)
                      , _pwpoly_press_vec(_pwpoly_n_pieces)
                      , _pwpoly_eps_vec(_pwpoly_n_pieces) ; 
    
    _pwpoly_kappas_vec[0] = _pwpoly_kappas_0 ; 
    _pwpoly_press_vec[0]  = 0 ; 
    _pwpoly_eps_vec[0]    = 0 ; 

    for( int i=1; i < _pwpoly_n_pieces; ++i ) {
        _pwpoly_kappas_vec[i] = 
            _pwpoly_kappas_vec[i-1] * 
            pow( _pwpoly_rhos_vec[i], _pwpoly_gammas_vec[i-1]-_pwpoly_gammas_vec[i]) ; 
        _pwpoly_eps_vec[i]  = 
            _pwpoly_eps_vec[i-1] +
            _pwpoly_kappas_vec[i-1] *
                pow(_pwpoly_rhos_vec[i], _pwpoly_gammas_vec[i-1]-1.) 
                / ( _pwpoly_gammas_vec[i-1]-1. ) -
            _pwpoly_kappas_vec[i] *
                pow(_pwpoly_rhos_vec[i], _pwpoly_gammas_vec[i]-1.) 
                / ( _pwpoly_gammas_vec[i]-1. ) ; 
        _pwpoly_press_vec[i] =
            _pwpoly_kappas_vec[i] *
            pow(_pwpoly_rhos_vec[i], _pwpoly_gammas_vec[i]) ; 
    }


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
    
    #define DEEP_COPY_VEC_TO_VIEW(vec,view) \
            do { \
                auto host_view = Kokkos::create_mirror_view(view) ; \
                for( int i=0; i < vec.size(); ++i){                 \
                    host_view(i) = vec[i] ;                         \
                }                                                   \
                Kokkos::deep_copy(view,host_view) ;                 \
            } while(0)
    DEEP_COPY_VEC_TO_VIEW(_pwpoly_gammas_vec,_pwpoly_gammas) ; 
    DEEP_COPY_VEC_TO_VIEW(_pwpoly_kappas_vec,_pwpoly_kappas) ;
    DEEP_COPY_VEC_TO_VIEW(_pwpoly_rhos_vec,_pwpoly_rhos) ;
    DEEP_COPY_VEC_TO_VIEW(_pwpoly_press_vec,_pwpoly_press) ;
    DEEP_COPY_VEC_TO_VIEW(_pwpoly_eps_vec,_pwpoly_eps) ; 

    GRACE_INFO("Polytropic EOS initialized.") ; 

    std::ostringstream _pwpoly_gammas_str, _pwpoly_rhos_str
                     , _pwpoly_kappas_str, _pwpoly_press_str
                     , _pwpoly_eps_str;
    _pwpoly_gammas_str << _pwpoly_gammas_vec ; 
    _pwpoly_rhos_str << _pwpoly_rhos_vec ; 
    _pwpoly_kappas_str << _pwpoly_kappas_vec ; 
    _pwpoly_press_str << _pwpoly_press_vec ; 
    _pwpoly_eps_str << _pwpoly_eps_vec ; 

    GRACE_INFO("Polytropic has {} segments.", _pwpoly_n_pieces) ;
    GRACE_INFO("Polytropic Gammas: {}.", _pwpoly_gammas_str.str()) ;
    GRACE_INFO("Polytropic rhos: {}.", _pwpoly_rhos_str.str()) ;
    GRACE_INFO("Polytropic K: {}.", _pwpoly_kappas_str.str()) ;
    GRACE_INFO("Polytropic press: {}.", _pwpoly_press_str.str()) ;
    GRACE_INFO("Polytropic eps: {}.", _pwpoly_eps_str.str()) ;

    return std::move(piecewise_polytropic_eos_t{
          _pwpoly_kappas
        , _pwpoly_gammas
        , _pwpoly_rhos
        , _pwpoly_eps
        , _pwpoly_press
        , _pwpoly_n_pieces
        , 1e+10
        , 1e-20
    }) ; 

}

eos_storage_t::eos_storage_t() {
    auto& params = grace::config_parser::get() ;

    std::string eos_type = 
        params["eos"]["eos_type"].as<std::string>() ;

    double c2p_eps_max  = params["eos"]["eps_maximum"].as<double>();
    double c2p_entropy_min = params["eos"]["entropy_minimum"].as<double>() ; 

    bool atm_is_beta_eq = params["eos"]["atm_is_beta_eq"].as<bool>();

    bool extend_table_high = params["eos"]["extend_table_high"].as<bool>(); ; 

    if ( eos_type == "hybrid" ) {
        std::string cold_eos_type =
            params["eos"]["cold_eos_type"].as<std::string>() ; 
        double gamma_th = params["eos"]["gamma_th"].as<double>() ;

        if( cold_eos_type == "piecewise_polytrope" ) {

            auto _pwpoly = setup_cold_politrope() ; 

            _hybrid_pwpoly = hybrid_eos_t<piecewise_polytropic_eos_t>{
                  _pwpoly 
                , gamma_th - 1. 
                , c2p_entropy_min
                , grace::physical_constants::mnuc_CGS
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