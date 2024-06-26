/**
 * @file grmhd_helpers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-17
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

#ifndef GRACE_PHYSICS_GRMHD_HELPERS_HH
#define GRACE_PHYSICS_GRMHD_HELPERS_HH

#include <grace_config.h> 
#include <array>
//**************************************************************************************************/
/* Auxiliaries */
//**************************************************************************************************/
/**
 * @brief Helper indices for prim arrays
 * \ingroup physics
 */
enum GRMHD_PRIMS_LOC_INDICES {
    RHOL = 0,
    PRESSL,
    VXL,
    VYL,
    VZL,
    YEL,
    TEMPL,
    EPSL,
    ENTL,
    #ifdef GRACE_DO_MHD
    BXL,
    BYL,
    BZL,
    #endif 
    NUM_PRIMS_LOC
} ; 
/**
 * @brief Helper indices for cons array.
 * \ingroup physics
 */
enum GRMHD_CONS_LOC_INDICES {
    DENSL=0,
    STXL,
    STYL,
    STZL,
    TAUL,
    YESL,
    ENTSL,
    NUM_CONS_LOC
} ; 
namespace grace {
/**
 * @brief Array of GRMHD primitives.
 * \ingroup physics
 */
using grmhd_prims_array_t = std::array<double,NUM_PRIMS_LOC> ; 
/**
 * @brief Array of GRMHD conservatives.
 * \ingroup physics
 */
using grmhd_cons_array_t  = std::array<double,NUM_CONS_LOC>  ;
} /* namespace grace */

#define FILL_METRIC_ARRAY(g, view, q, ...)                    \
g = grace::metric_array_t{  { view(__VA_ARGS__,GXX_,q)   \
                          , view(__VA_ARGS__,GXY_,q)     \
                          , view(__VA_ARGS__,GXZ_,q)     \
                          , view(__VA_ARGS__,GYY_,q)     \
                          , view(__VA_ARGS__,GYZ_,q)     \
                          , view(__VA_ARGS__,GZZ_,q) }   \
                          , { view(__VA_ARGS__,BETAX_,q) \
                          , view(__VA_ARGS__,BETAY_,q)   \
                          , view(__VA_ARGS__,BETAZ_,q) } \
                          , view(__VA_ARGS__,ALP_,q) } 
                          
#define FILL_PRIMS_ARRAY(primsarr,vview,q,...)        \
primsarr[RHOL] = vview(__VA_ARGS__,RHO_,q);      \
primsarr[PRESSL] = vview(__VA_ARGS__,PRESS_,q) ; \
primsarr[VXL] = vview(__VA_ARGS__,VELX_,q) ;     \
primsarr[VYL] = vview(__VA_ARGS__,VELY_,q) ;     \
primsarr[VZL] = vview(__VA_ARGS__,VELZ_,q) ;     \
primsarr[YEL] = vview(__VA_ARGS__,YE_,q) ;       \
primsarr[TEMPL] = vview(__VA_ARGS__,TEMP_,q) ;   \
primsarr[EPSL] = vview(__VA_ARGS__,EPS_,q) ;     \
primsarr[ENTL] = vview(__VA_ARGS__,ENTROPY_,q)

#endif /* GRACE_PHYSICS_GRMHD_HELPERS_HH */