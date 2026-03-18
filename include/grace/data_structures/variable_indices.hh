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

#include <grace/utils/device.h>

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
namespace detail {
/****************************************************/
/*                Variable arrays sizes             */
/****************************************************/
extern int num_vector_vars ;
extern int num_tensor_vars ; 
/****************************************************/
/****************************************************/

/****************************************************/
/*                Variable name arrays              */
/****************************************************/
extern std::vector<std::string> _varnames ; 
extern std::vector<std::string> _auxnames ; 

extern std::vector<std::string> _facex_staggered_varnames ;

extern std::vector<std::string> _facey_staggered_varnames ;

extern std::vector<std::string> _facez_staggered_varnames ;

extern std::vector<std::string> _edgexy_staggered_varnames ; 

extern std::vector<std::string> _edgexz_staggered_varnames ; 

extern std::vector<std::string> _edgeyz_staggered_varnames ; 

extern std::vector<std::string> _corner_staggered_varnames ; 
/****************************************************/
/****************************************************/
 
/****************************************************/
/*             Boundary condition arrays            */
/****************************************************/
extern std::vector<grace::bc_t> _var_bc_types ;

extern std::vector<grace::bc_t> _facex_vars_bc_types ;

extern std::vector<grace::bc_t> _facey_vars_bc_types ;

extern std::vector<grace::bc_t> _facez_vars_bc_types ;

extern std::vector<grace::bc_t> _edgexy_vars_bc_types ;

extern std::vector<grace::bc_t> _edgexz_vars_bc_types ;

extern std::vector<grace::bc_t> _edgeyz_vars_bc_types ;

extern std::vector<grace::bc_t> _corner_vars_bc_types ;
/****************************************************/
/*      Prolong/Restrict operator arrays            */
/****************************************************/
extern std::vector<grace::var_amr_interp_t> _var_interp_types ;

extern std::vector<grace::var_amr_interp_t> _facex_vars_interp_types ;

extern std::vector<grace::var_amr_interp_t> _facey_vars_interp_types ;

extern std::vector<grace::var_amr_interp_t> _facez_vars_interp_types ;

extern std::vector<grace::var_amr_interp_t> _edgexy_vars_interp_types ;

extern std::vector<grace::var_amr_interp_t> _edgexz_vars_interp_types ;

extern std::vector<grace::var_amr_interp_t> _edgeyz_vars_interp_types ;

extern std::vector<grace::var_amr_interp_t> _corner_vars_interp_types ;
/****************************************************/
/****************************************************/
 
/****************************************************/
/*              Handling of vector/tensor           */
/*                    components                    */
/****************************************************/

extern std::unordered_map<std::string, variable_properties_t<GRACE_NSPACEDIM>> 
    _varprops; 
extern std::unordered_map<std::string, variable_properties_t<GRACE_NSPACEDIM>> 
    _auxprops; 

} /* namespace grace::variables::detail */


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

} } /* namespace grace::variables */

enum evol_hrsc_var_cc_idx : int {
    DENS_ = 0,
    SX_,
    SY_,
    SZ_,
    TAU_,
    YESTAR_,
    ENTROPYSTAR_,
    #ifdef GRACE_ENABLE_M1
    ERAD_,
    NRAD_,
    FRADX_,
    FRADY_,
    FRADZ_,
    #ifdef M1_NU_THREESPECIES
    ERAD1_,
    NRAD1_,
    FRADX1_,
    FRADY1_,
    FRADZ1_,
    ERAD2_,
    NRAD2_,
    FRADX2_,
    FRADY2_,
    FRADZ2_,
    #endif 
    #endif 
    N_HRSC_CC
} ; 

enum evol_var_fc_x_idx : int {
    BSX_=0,
    N_FC_X
} ; 

enum evol_var_fc_y_idx : int {
    BSY_=0,
    N_FC_Y
} ; 

enum evol_var_fc_z_idx : int {
    BSZ_=0,
    N_FC_Z
} ; 

enum evol_var_ec_yz_idx : int {
    N_EC_YZ=0
} ; 

enum evol_var_ec_xz_idx : int {
    N_EC_XZ=0
} ; 

enum evol_var_ec_xy_idx : int {
    N_EC_XY=0
} ; 

enum evol_var_vc_idx : int {
    N_VC=0
} ; 

enum evol_fd_var_cc_idx : int {
    #ifdef GRACE_ENABLE_COWLING_METRIC
    GXX_=N_HRSC_CC,
    GXY_,
    GXZ_,
    GYY_,
    GYZ_,
    GZZ_,
    ALP_,
    BETAX_,
    BETAY_,
    BETAZ_,
    KXX_,
    KXY_,
    KXZ_,
    KYY_,
    KYZ_,
    KZZ_,
    #endif 
    #ifdef GRACE_ENABLE_Z4C_METRIC
    GTXX_=N_HRSC_CC,
    GTXY_,
    GTXZ_,
    GTYY_,
    GTYZ_,
    GTZZ_,
    CHI_,
    THETA_,
    GAMMATX_,
    GAMMATY_,
    GAMMATZ_,
    ATXX_,
    ATXY_,
    ATXZ_,
    ATYY_,
    ATYZ_,
    ATZZ_,
    KHAT_,
    ALP_,
    BETAX_,
    BETAY_,
    BETAZ_,
    BDRIVERX_,
    BDRIVERY_,
    BDRIVERZ_,
    #endif
    N_EVOL_VARS
} ; 

enum aux_var_idx : int {
    RHO_=0,
    ZVECX_,
    ZVECY_,
    ZVECZ_,
    BX_,
    BY_,
    BZ_,
    YE_,
    TEMP_,
    ENTROPY_,
    EPS_,
    PRESS_,
    BDIV_,
    SMALLB2_,
    C2P_ERR_,
    #ifdef GRACE_ENABLE_M1
    KAPPAA_,
    KAPPAS_,
    ETA_,
    ETAN_,
    KAPPAAN_,
    #ifdef M1_NU_THREESPECIES
    KAPPAA1_,
    KAPPAS1_,
    ETA1_,
    ETAN1_,
    KAPPAAN1_,
    KAPPAA2_,
    KAPPAS2_,
    ETA2_,
    ETAN2_,
    KAPPAAN2_,
    #endif
    #endif
    #ifdef GRACE_ENABLE_Z4C_METRIC
    PSI4RE_,
    PSI4IM_,
    HAM_,
    MOMX_,
    MOMY_,
    MOMZ_,
    #endif
    N_AUX_VARS
} ; 

#endif /* INCLUDE_GRACE_DATA_STRUCTURES_VARIABLE_INDICES */
