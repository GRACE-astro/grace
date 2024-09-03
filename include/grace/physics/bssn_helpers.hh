/**
 * @file bssn_helpers.hh
 * @author  ()
 * @brief 
 * @date 2024-09-03
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

#ifndef GRACE_PHYSICS_BSSN_HELPERS_HH 
#define GRACE_PHYSICS_BSSN_HELPERS_HH

#include <grace_config.h> 
#include <array>

namespace grace {

using bssn_state_t = std::array<double, NUM_BSSN_VARS> ;

enum BSSN_VARENUM_t {
    PHIL=0,
    GTXXL,
    GTXYL,
    GTXZL, 
    GTYYL,
    GTYZL,
    GTZZL,
    ATXXL,
    ATXYL,
    ATXZL,
    ATYYL,
    ATYZL,
    ATZZL,
    KL,
    GAMMAXL,
    GAMMAYL,
    GAMMAZL,
    NUM_BSSN_VARS
} ; 

#define FILL_BSSN_STATE(sstate, vview, q, ...)\
do{                                      \
sstate[PHIL] = vview(__VA_ARGS__, PHI_     , q); \
sstate[GTXXL] = vview(__VA_ARGS__, GTXX_   , q); \
sstate[GTXYL] = vview(__VA_ARGS__, GTXY_   , q); \
sstate[GTXZL] = vview(__VA_ARGS__, GTXZ_   , q); \
sstate[GTYYL] = vview(__VA_ARGS__, GTYY_   , q); \
sstate[GTYZL] = vview(__VA_ARGS__, GTYZ_   , q); \
sstate[GTZZL] = vview(__VA_ARGS__, GTZZ_   , q); \
sstate[ATXXL] = vview(__VA_ARGS__, ATXX_   , q); \
sstate[ATXYL] = vview(__VA_ARGS__, ATXY_   , q); \
sstate[ATXZL] = vview(__VA_ARGS__, ATXZ_   , q); \
sstate[ATYYL] = vview(__VA_ARGS__, ATYY_   , q); \
sstate[ATYZL] = vview(__VA_ARGS__, ATYZ_   , q); \
sstate[ATZZL] = vview(__VA_ARGS__, ATZZ_   , q); \
sstate[KL]    = vview(__VA_ARGS__, K_      , q); \
sstate[GAMMAXL] = vview(__VA_ARGS__,GAMMAX_, q); \
sstate[GAMMAYL] = vview(__VA_ARGS__,GAMMAY_, q); \
sstate[GAMMAZL] = vview(__VA_ARGS__,GAMMAZ_, q); \
} while(false)

}

#endif /* GRACE_PHYSICS_BSSN_HELPERS_HH */