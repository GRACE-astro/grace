/**
 * @file variable_indices.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Global utilities for variable registration / indexing in GRACE.
 * @version 0.1
 * @date 2023-06-13
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#ifndef INCLUDE_GRACE_DATA_STRUCTURES_VARIABLE_INDICES
#define INCLUDE_GRACE_DATA_STRUCTURES_VARIABLE_INDICES

#include <grace_config.h>
#include <code_modules.h> 

#include <grace/utils/device/device.h>

#include <grace/data_structures/variable_properties.hh>

#include <vector>
#include <array>
#include <unordered_map> 

namespace grace { namespace variables { 
//*****************************************************************************************************
/**
 * \defgroup variables Routines and classes to hadle variables and their storage/access. 
 */
//*****************************************************************************************************
/**
 * @brief Enum for variable types in GRACE.
 * \ingroup variables
 */
enum grace_variable_types {
    EVOLVED=0,
    AUXILIARY,
    FACE_STAGGERED,
    FACE_STAGGERED_AUXILIARY,
    EDGE_STAGGERED,
    EDGE_STAGGERED_AUXILIARY,
    CORNER_STAGGERED,
    CORNER_STAGGERED_AUXILIARY,
    N_GRACE_VARIABLE_TYPES 
} ; 
//*****************************************************************************************************
/**
* @brief Register a variable within GRACE.
* \ingroup variables
* @param name            Name of the variable.
* @param staggered       Staggering of variable in each direction.
* @param is_evolved      Whether the variable is evolved.
* @param need_fluxes     Whether the variables needs fluxes. 
* @param is_vector       True if the variable is a component of a vector.
* @return int            Index of the variable in respective state array.
*/
static int register_variable( std::string const& name
                            , std::array<bool,GRACE_NSPACEDIM> staggered  
                            , bool is_evolved 
                            , bool need_fluxes
                            , std::string const& bc_type="none"
                            , bool is_vector=false
                            , bool is_tensor=false 
                            , int  comp_num=-1
                            , std::string const& vecname=""
                             ) ;
//*****************************************************************************************************xw

/**
 * @brief Register all variables.
 * \ingroup variables
 * Whenever a new physics module needs to be defined, the indices for 
 * its variables need to be defined as <code>extern int</code>s with 
 * unique uppercase identifiers in this file. These variables are then 
 * filled with values in the correct order within this routine, which 
 * needs to be updated with appropriate calls to <code>register_variable</code>
 * for the new grid functions. 
 */
void register_variables() ; 

namespace detail {

extern int num_vars      ; 
extern int last_evolved  ; 
extern int num_fluxes    ;
extern int last_flux     ;  
extern int first_flux     ;  

/****************************************************/
/*                Variable arrays sizes             */
/****************************************************/
extern int num_evolved   ;
extern int num_auxiliary ;

extern int num_face_staggered_vars ;
extern int num_face_staggered_aux  ;

extern int num_edge_staggered_vars ;
extern int num_edge_staggered_aux  ;

extern int num_corner_staggered_vars ;
extern int num_corner_staggered_aux  ;

extern int num_vector_vars ;
extern int num_tensor_vars ; 
/****************************************************/
/****************************************************/

/****************************************************/
/*                Variable name arrays              */
/****************************************************/
extern std::vector<std::string> _varnames ; 
extern std::vector<std::string> _auxnames ; 

extern std::vector<std::string> _face_staggered_varnames ;
extern std::vector<std::string> _face_staggered_auxnames ;

extern std::vector<std::string> _edge_staggered_varnames ; 
extern std::vector<std::string> _edge_staggered_auxnames ; 

extern std::vector<std::string> _corner_staggered_varnames ; 
extern std::vector<std::string> _corner_staggered_auxnames ;
/****************************************************/
/****************************************************/
 
/****************************************************/
/*             Boundary condition arrays            */
/****************************************************/
extern std::vector<std::string> _var_bc_types ;
extern std::vector<std::string> _aux_bc_types ;

extern std::vector<std::string> _face_vars_bc_types ;
extern std::vector<std::string> _face_aux_bc_types ;

extern std::vector<std::string> _edge_vars_bc_types ;
extern std::vector<std::string> _edge_aux_bc_types ;

extern std::vector<std::string> _corner_vars_bc_types ;
extern std::vector<std::string> _corner_aux_bc_types ;
/****************************************************/
/****************************************************/
 
/****************************************************/
/*              Handling of vector/tensor           */
/*                    components                    */
/****************************************************/
extern std::vector<int> _vector_var_indices ; 
extern std::vector<int> _tensor_var_indices ; 

extern std::unordered_map<std::string, variable_properties_t<GRACE_NSPACEDIM>> 
    _varprops; 
extern std::unordered_map<std::string, variable_properties_t<GRACE_NSPACEDIM>> 
    _auxprops; 

} /* namespace grace::variables::detail */

} } /* namespace grace::variables */

#ifdef GRACE_ENABLE_BURGERS 
#define VARIABLE_LIST_BURGERS \
DECLARE_VAR_INDEX_IMPL(U)     
#else 
#define VARIABLE_LIST_BURGERS
#endif 
#ifdef GRACE_ENABLE_SCALAR_ADV
#define VARIABLE_LIST_SCALAR_ADV \
DECLARE_VAR_INDEX_IMPL(U)        \
DECLARE_VAR_INDEX_IMPL(ERR)      
#else
#define VARIABLE_LIST_SCALAR_ADV
#endif 
/* Valencia GRMHD conservatives */
#define VARIABLE_LIST_HYDROBASE                     \
DECLARE_VAR_INDEX_IMPL(DENS)                        \
DECLARE_VAR_INDEX_IMPL(SX)                          \
DECLARE_VAR_INDEX_IMPL(SY)                          \
DECLARE_VAR_INDEX_IMPL(SZ)                          \
DECLARE_VAR_INDEX_IMPL(TAU)                         \
DECLARE_VAR_INDEX_IMPL(YESTAR)                      \
DECLARE_VAR_INDEX_IMPL(ENTROPYSTAR)                 \
DECLARE_VAR_INDEX_IMPL(RHO)                         \
DECLARE_VAR_INDEX_IMPL(PRESS)                       \
DECLARE_VAR_INDEX_IMPL(VELX)                        \
DECLARE_VAR_INDEX_IMPL(VELY)                        \
DECLARE_VAR_INDEX_IMPL(VELZ)                        \
DECLARE_VAR_INDEX_IMPL(ZVECX)                        \
DECLARE_VAR_INDEX_IMPL(ZVECY)                        \
DECLARE_VAR_INDEX_IMPL(ZVECZ)                        \
DECLARE_VAR_INDEX_IMPL(TEMP)                        \
DECLARE_VAR_INDEX_IMPL(YE)                          \
DECLARE_VAR_INDEX_IMPL(ENTROPY)                     \
DECLARE_VAR_INDEX_IMPL(EPS)                         
#ifdef GRACE_ENABLE_COWLING_METRIC
/* ADM metric functions */
#define VARIABLE_LIST_ADMBASE                     \
DECLARE_VAR_INDEX_IMPL(GXX)                       \
DECLARE_VAR_INDEX_IMPL(GXY)                       \
DECLARE_VAR_INDEX_IMPL(GXZ)                       \
DECLARE_VAR_INDEX_IMPL(GYY)                       \
DECLARE_VAR_INDEX_IMPL(GYZ)                       \
DECLARE_VAR_INDEX_IMPL(GZZ)                       \
DECLARE_VAR_INDEX_IMPL(ALP)                       \
DECLARE_VAR_INDEX_IMPL(BETAX)                     \
DECLARE_VAR_INDEX_IMPL(BETAY)                     \
DECLARE_VAR_INDEX_IMPL(BETAZ)                     \
DECLARE_VAR_INDEX_IMPL(KXX)                       \
DECLARE_VAR_INDEX_IMPL(KXY)                       \
DECLARE_VAR_INDEX_IMPL(KXZ)                       \
DECLARE_VAR_INDEX_IMPL(KYY)                       \
DECLARE_VAR_INDEX_IMPL(KYZ)                       \
DECLARE_VAR_INDEX_IMPL(KZZ)                       
#elif defined(GRACE_ENABLE_BSSN_METRIC)
#define VARIABLE_LIST_ADMBASE                    \
DECLARE_VAR_INDEX_IMPL(GTXX)                     \
DECLARE_VAR_INDEX_IMPL(GTXY)                     \
DECLARE_VAR_INDEX_IMPL(GTXZ)                     \
DECLARE_VAR_INDEX_IMPL(GTYY)                     \
DECLARE_VAR_INDEX_IMPL(GTYZ)                     \
DECLARE_VAR_INDEX_IMPL(GTZZ)                     \
DECLARE_VAR_INDEX_IMPL(PHI)                      \
DECLARE_VAR_INDEX_IMPL(GAMMAX)                   \
DECLARE_VAR_INDEX_IMPL(GAMMAY)                   \
DECLARE_VAR_INDEX_IMPL(GAMMAZ)                   \
DECLARE_VAR_INDEX_IMPL(ATXX)                     \
DECLARE_VAR_INDEX_IMPL(ATXY)                     \
DECLARE_VAR_INDEX_IMPL(ATXZ)                     \
DECLARE_VAR_INDEX_IMPL(ATYY)                     \
DECLARE_VAR_INDEX_IMPL(ATYZ)                     \
DECLARE_VAR_INDEX_IMPL(ATZZ)                     \
DECLARE_VAR_INDEX_IMPL(K)                        \
DECLARE_VAR_INDEX_IMPL(ALP)                      \
DECLARE_VAR_INDEX_IMPL(BETAX)                    \
DECLARE_VAR_INDEX_IMPL(BETAY)                    \
DECLARE_VAR_INDEX_IMPL(BETAZ)                    \
DECLARE_VAR_INDEX_IMPL(BX)                       \
DECLARE_VAR_INDEX_IMPL(BY)                       \
DECLARE_VAR_INDEX_IMPL(BZ)                       \
DECLARE_VAR_INDEX_IMPL(HAM)                      \
DECLARE_VAR_INDEX_IMPL(MOMX)                     \
DECLARE_VAR_INDEX_IMPL(MOMY)                     \
DECLARE_VAR_INDEX_IMPL(MOMZ)                     \
DECLARE_VAR_INDEX_IMPL(GXX)                      \
DECLARE_VAR_INDEX_IMPL(GXY)                      \
DECLARE_VAR_INDEX_IMPL(GXZ)                      \
DECLARE_VAR_INDEX_IMPL(GYY)                      \
DECLARE_VAR_INDEX_IMPL(GYZ)                      \
DECLARE_VAR_INDEX_IMPL(GZZ)                      \
DECLARE_VAR_INDEX_IMPL(ALPC)                     \
DECLARE_VAR_INDEX_IMPL(BETAXC)                   \
DECLARE_VAR_INDEX_IMPL(BETAYC)                   \
DECLARE_VAR_INDEX_IMPL(BETAZC)                   \
DECLARE_VAR_INDEX_IMPL(KXX)                      \
DECLARE_VAR_INDEX_IMPL(KXY)                      \
DECLARE_VAR_INDEX_IMPL(KXZ)                      \
DECLARE_VAR_INDEX_IMPL(KYY)                      \
DECLARE_VAR_INDEX_IMPL(KYZ)                      \
DECLARE_VAR_INDEX_IMPL(KZZ)                         
#endif 

#define DECLARE_VARIABLE_INDICES    \
VARIABLE_LIST_HYDROBASE             \
VARIABLE_LIST_ADMBASE               \
VARIABLE_LIST_BURGERS               \
VARIABLE_LIST_SCALAR_ADV

#define DECLARE_VAR_INDEX_IMPL(var) extern int var;
DECLARE_VARIABLE_INDICES
#undef DECLARE_VAR_INDEX_IMPL

#define  DECLARE_VAR_INDEX_IMPL(var) extern GRACE_DEVICE int var##_;
DECLARE_VARIABLE_INDICES
#undef DECLARE_VAR_INDEX_IMPL

#define DECLARE_VAR_INDEX_IMPL(name) 

#endif /* INCLUDE_GRACE_DATA_STRUCTURES_VARIABLE_INDICES */
