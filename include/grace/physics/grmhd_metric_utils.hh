/**
 * @file grmhd_metric_utils.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-20
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

#ifndef GRACE_PHYSICS_GRMHD_METRIC_UTILS_HH
#define GRACE_PHYSICS_GRMHD_METRIC_UTILS_HH

#include <grace_config.h> 
#include <array>

#include <grace/utils/numerics/metric_utils.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>
#include <grace/utils/numerics/prolongation.hh>
#include <grace/utils/numerics/lagrange_interpolators.hh>

#define AM2 -0.0625
#define AM1  0.5625
#define A0   0.5625
#define A1  -0.0625
#define COMPUTE_FCVAL_HELPER(mview,i,j,k,ivar,q,idir)                                               \
  AM2*mview(VEC(i-2*utils::delta(0,idir),j-2*utils::delta(1,idir),k-2*utils::delta(2,idir)),ivar,q) \
+ AM1*mview(VEC(i-utils::delta(0,idir),j-utils::delta(1,idir),k-utils::delta(2,idir)),ivar,q)       \
+ A0*mview(VEC(i,j,k),ivar,q)                                                                       \
+ A1*mview(VEC(i+utils::delta(0,idir),j+utils::delta(1,idir),k+utils::delta(2,idir)),ivar,q)             

#ifdef GRACE_ENABLE_COWLING_METRIC
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
#elif defined(GRACE_ENABLE_BSSN_METRIC)
#define FILL_METRIC_ARRAY(g, view, q, ...)                    \
g = grace::metric_array_t{  { view(__VA_ARGS__,GXX_,q)   \
                          , view(__VA_ARGS__,GXY_,q)     \
                          , view(__VA_ARGS__,GXZ_,q)     \
                          , view(__VA_ARGS__,GYY_,q)     \
                          , view(__VA_ARGS__,GYZ_,q)     \
                          , view(__VA_ARGS__,GZZ_,q) }   \
                          , { view(__VA_ARGS__,BETAXC_,q)   \
                            , view(__VA_ARGS__,BETAYC_,q)   \
                            , view(__VA_ARGS__,BETAZC_,q) } \
                            , view(__VA_ARGS__,ALPC_,q) } 
#endif 
#endif /* GRACE_PHYSICS_GRMHD_METRIC_UTILS_HH */