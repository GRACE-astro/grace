/**
 * @file variable_indices.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-12
 * 
 * @copyright This file is part of MagMA.
 * MagMA is an evolution framework that uses Discontinuous Galerkin
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

#include <code_modules.h>
#include <thunder_config.h>

#include <thunder/errors/assert.hh>
#include <thunder/data_structures/variable_indices.hh>
#include <thunder/data_structures/macros.hh>

#include <string> 

namespace thunder { namespace variables { 


namespace detail {

int num_vars      = 0 ; 
int num_evolved   = 0 ;
int last_evolved  = -1 ; 
int num_auxiliary = 0 ;
int num_fluxes    = 0 ;
int last_flux     = -1 ; 
int first_flux    = -1 ; 


std::vector<std::string> _varprops ; 
std::vector<std::string> _auxprops ; 

} /* namespace thunder::variables::detail */

#ifdef THUNDER_ENABLE_HYDROBASE 
/* Valencia GRMHD conservatives */
int DENS= -1 ; 
int SX  = -1 ; 
int SY  = -1 ; 
int SZ  = -1 ; 
int TAU = -1 ; 
#endif 
#ifdef THUNDER_ENABLE_ADMBASE 
/* AMD metric functions */
int GXX = -1 ;
int GXY = -1 ;
int GXZ = -1 ;
int GYY = -1 ;
int GYZ = -1 ;
int GZZ = -1 ;
int ALP = -1 ;
int BETAX = -1 ;
int BETAY = -1 ;
int BETAZ = -1 ;
#endif 

void register_variables() {
    #ifdef THUNDER_ENABLE_HYDROBASE 
    /* Valencia hydrodynamics */
    int DENS = register_variable( "dens"
                                , {VEC(false,false,false)}
                                , true 
                                , true 
                                , true ) ; 
    int SX = register_variable( "Sx"
                                , {VEC(false,false,false)}
                                , true 
                                , true 
                                , true ) ;
    int SY = register_variable( "Sy"
                                , {VEC(false,false,false)}
                                , true 
                                , true 
                                , true ) ;
    int SZ = register_variable( "Sz"
                                , {VEC(false,false,false)}
                                , true 
                                , true 
                                , true ) ;
    int TAU = register_variable( "tau"
                                , {VEC(false,false,false)}
                                , true 
                                , true 
                                , true ) ;
    #endif
    #ifdef THUNDER_ENABLE_ADMBASE 
    /* registration of metric variables */
    int GXX = register_variable( "gxx"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false ) ; 

    int GXY = register_variable( "gxy"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false ) ;

    int GXZ = register_variable( "gxz"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false ) ;

    int GYY = register_variable( "gyy"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false ) ;

    int GYZ = register_variable( "gyz"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false ) ;

    int GZZ = register_variable( "gzz"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false ) ;

    int ALP = register_variable( "alp"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false ) ;

    int BETAX = register_variable( "betax"
                                , {VEC(false,false,false)} 
                                , true 
                                , true
                                , false ) ;


    int BETAY = register_variable( "betay"
                                , {VEC(false,false,false)} 
                                , true 
                                , true
                                , false ) ;


    int BETAZ = register_variable( "betaz"
                                , {VEC(false,false,false)} 
                                , true 
                                , true
                                , false ) ;
    #endif 

}


/**
 * @brief Register a variable within Thunder.
 * 
 * @param name Name of the variable.
 * @param staggered Staggering of variable in each direction.
 * @param need_ghostzones Whether the variable needs to be reconstructed.
 * @param is_evolved Whether the variable is evolved.
 * @param need_fluxes Whether the variables needs fluxes. 
 * @return size_t Index of the variable in respective state array.
 */
static int register_variable(  std::string const& name
                                , std::array<bool, THUNDER_NSPACEDIM> staggered  
                                , bool need_ghostzones 
                                , bool is_evolved 
                                , bool need_fluxes ) 
{
    using namespace detail ; 

    if( need_fluxes ) {
        if( first_flux == -1 ){
            ASSERT(num_vars == 0,
                   "The first registered evolved variable"
                   " must be a flux variable." ) ; 
            first_flux = num_vars ;
        } else {
            ASSERT( last_flux == num_vars - 1,
                    "Flux variables need to be a contiguous"
                    " block at the start of the evolved array" ) ; 
        } 
        last_flux = num_vars ; 
        num_fluxes ++ ; 

    }
    num_vars++ ; 
    if( is_evolved ){
        last_evolved = num_vars; 
        num_evolved ++ ; 
        _varprops.push_back( name ) ; 
    } else {
        num_auxiliary ++ ; 
        _auxprops.push_back( name )   ; 
    }
    return is_evolved ? num_evolved-1 : num_auxiliary-1 ; 
}

} } /* namespace thunder::variables */