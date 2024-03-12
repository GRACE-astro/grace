/**
 * @file variable_indices.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-06-13
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
 * methods to simulate relativistic astrophysical systems and plasma
 * dynamics.
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

#ifndef INCLUDE_THUNDER_DATA_STRUCTURES_VARIABLE_INDICES
#define INCLUDE_THUNDER_DATA_STRUCTURES_VARIABLE_INDICES

#include <thunder_config.h>

#include <thunder/data_structures/variable_properties.hh>

#include <vector>
#include <array>

namespace thunder { namespace variables { 

/**
* @brief Register a variable within Thunder.
* 
* @param name            Name of the variable.
* @param staggered       Staggering of variable in each direction.
* @param need_ghostzones Whether the variable needs extra ghostzone storage.
* @param is_evolved      Whether the variable is evolved.
* @param need_fluxes     Whether the variables needs fluxes. 
* @return int            Index of the variable in respective state array.
*/
static int register_variable( std::string const& name
                            , std::array<bool,THUNDER_NSPACEDIM> staggered  
                            , bool need_ghostzones 
                            , bool is_evolved 
                            , bool need_fluxes ) ;

/**
 * @brief Register all variables.
 */
void register_variables() ; 

namespace detail {

extern int num_vars      ; 
extern int num_evolved   ;
extern int last_evolved  ; 
extern int num_auxiliary ; 
extern int num_fluxes    ;
extern int last_flux     ;  

extern std::vector<variable_properties_t<THUNDER_NSPACEDIM>> _varprops ; 
extern std::vector<variable_properties_t<THUNDER_NSPACEDIM>> _auxprops ; 

} /* namespace thunder::variables::detail */

#ifdef THUNDER_ENABLE_HYDROBASE 
/* Valencia GRMHD conservatives */
extern int DENS ; 
extern int SX   ; 
extern int SY   ; 
extern int SZ   ; 
extern int TAU  ; 
#endif 
#ifdef THUNDER_ENABLE_ADMBASE 
/* AMD metric functions */
extern int GXX ;
extern int GXY ;
extern int GXZ ;
extern int GYY ;
extern int GYZ ;
extern int GZZ ;
extern int ALP ;
extern int BETAX ;
extern int BETAY ;
extern int BETAZ ;
#endif 


} } /* namespace thunder::variables */


#endif /* INCLUDE_THUNDER_DATA_STRUCTURES_VARIABLE_INDICES */
