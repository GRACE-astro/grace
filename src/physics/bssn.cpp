/**
 * @file bssn.cpp
 * @author  Christian Ecker
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

#include <grace_config.h> 

#include <grace/utils/grace_utils.hh> 

#include <grace/data_structures/grace_data_structures.hh>

#include <grace/physics/bssn.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <grace/utils/numerics/fd_utils.hh>

#include <Kokkos_Core.hpp>

namespace grace {

template< size_t der_order >
bssn_state_t GRACE_HOST_DEVICE 
compute_bssn_rhs( VEC(int i, int j, int k), int q
                , grace::var_array_t<GRACE_NSPACEDIM> const state
                , std::array<std::array<double,4>,4> const& Tmunu
                , std::array<double,GRACE_NSPACEDIM> const& idx)
{

    // local definition of variables necessary to compute the r.h.s. of the BSSN equations
    
    // conformal (tilde) metric components
    double const gtxx=state(VEC(i,j,k),GTXX_,q);
    double const gtxy=state(VEC(i,j,k),GTXY_,q);
    double const gtyy=state(VEC(i,j,k),GTYY_,q);
    double const gtxz=state(VEC(i,j,k),GTXZ_,q);
    double const gtyz=state(VEC(i,j,k),GTYZ_,q);
    double const gtzz=state(VEC(i,j,k),GTZZ_,q);
    
    // inverse conformal (tilde) metric components
    double const gtXX=(gtyz*gtyz - gtyy*gtzz)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxx*(gtyz*gtyz) + gtxy*gtxy*gtzz - gtxx*gtyy*gtzz);
    double const gtXY=(-(gtxz*gtyz) + gtxy*gtzz)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz));
    double const gtXZ=(gtxz*gtyy - gtxy*gtyz)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxx*(gtyz*gtyz) + gtxy*gtxy*gtzz - gtxx*gtyy*gtzz);
    double const gtYY=(gtxz*gtxz - gtxx*gtzz)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxx*(gtyz*gtyz) + gtxy*gtxy*gtzz - gtxx*gtyy*gtzz);
    double const gtYZ=(-(gtxy*gtxz) + gtxx*gtyz)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz));
    double const gtZZ=(gtxy*gtxy - gtxx*gtyy)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxx*(gtyz*gtyz) + gtxy*gtxy*gtzz - gtxx*gtyy*gtzz);

// first x-derivatives of the conformal (tilde) metric components
    double const gtxxdx=grace::fd_der<der_order,0>(state,GTXX_,VEC(i,j,k),q)* idx[0];
    double const gtxydx=grace::fd_der<der_order,0>(state,GTXY_,VEC(i,j,k),q)* idx[0];
    double const gtyydx=grace::fd_der<der_order,0>(state,GTYY_,VEC(i,j,k),q)* idx[0];
    double const gtxzdx=grace::fd_der<der_order,0>(state,GTXZ_,VEC(i,j,k),q)* idx[0];
    double const gtyzdx=grace::fd_der<der_order,0>(state,GTYZ_,VEC(i,j,k),q)* idx[0];
    double const gtzzdx=grace::fd_der<der_order,0>(state,GTZZ_,VEC(i,j,k),q)* idx[0];

    // first y-derivatives of the conformal (tilde) metric components
    double const gtxxdy=grace::fd_der<der_order,1>(state,GTXX_,VEC(i,j,k),q)* idx[1];
    double const gtxydy=grace::fd_der<der_order,1>(state,GTXY_,VEC(i,j,k),q)* idx[1];
    double const gtyydy=grace::fd_der<der_order,1>(state,GTYY_,VEC(i,j,k),q)* idx[1];
    double const gtxzdy=grace::fd_der<der_order,1>(state,GTXZ_,VEC(i,j,k),q)* idx[1];
    double const gtyzdy=grace::fd_der<der_order,1>(state,GTYZ_,VEC(i,j,k),q)* idx[1];
    double const gtzzdy=grace::fd_der<der_order,1>(state,GTZZ_,VEC(i,j,k),q)* idx[1];

    // first z-derivatives of the conformal (tilde) metric components
    double const gtxxdz=grace::fd_der<der_order,2>(state,GTXX_,VEC(i,j,k),q)* idx[2];
    double const gtxydz=grace::fd_der<der_order,2>(state,GTXY_,VEC(i,j,k),q)* idx[2];
    double const gtyydz=grace::fd_der<der_order,2>(state,GTYY_,VEC(i,j,k),q)* idx[2];
    double const gtxzdz=grace::fd_der<der_order,2>(state,GTXZ_,VEC(i,j,k),q)* idx[2];
    double const gtyzdz=grace::fd_der<der_order,2>(state,GTYZ_,VEC(i,j,k),q)* idx[2];
    double const gtzzdz=grace::fd_der<der_order,2>(state,GTZZ_,VEC(i,j,k),q)* idx[2];

    // second x-derivatives of the conformal (tilde) metric components
    double const gtxxdxdx=grace::fd_der<der_order,0,0>(state,GTXX_,VEC(i,j,k),q)* idx[0]*idx[0];
    double const gtxydxdx=grace::fd_der<der_order,0,0>(state,GTXY_,VEC(i,j,k),q)* idx[0]*idx[0];
    double const gtyydxdx=grace::fd_der<der_order,0,0>(state,GTYY_,VEC(i,j,k),q)* idx[0]*idx[0];
    double const gtxzdxdx=grace::fd_der<der_order,0,0>(state,GTXZ_,VEC(i,j,k),q)* idx[0]*idx[0];
    double const gtyzdxdx=grace::fd_der<der_order,0,0>(state,GTYZ_,VEC(i,j,k),q)* idx[0]*idx[0];
    double const gtzzdxdx=grace::fd_der<der_order,0,0>(state,GTZZ_,VEC(i,j,k),q)* idx[0]*idx[0];

    // second y-derivatives of the conformal (tilde) metric components
    double const gtxxdydy=grace::fd_der<der_order,1,1>(state,GTXX_,VEC(i,j,k),q)* idx[1]*idx[1];
    double const gtxydydy=grace::fd_der<der_order,1,1>(state,GTXY_,VEC(i,j,k),q)* idx[1]*idx[1];
    double const gtyydydy=grace::fd_der<der_order,1,1>(state,GTYY_,VEC(i,j,k),q)* idx[1]*idx[1];
    double const gtxzdydy=grace::fd_der<der_order,1,1>(state,GTXZ_,VEC(i,j,k),q)* idx[1]*idx[1];
    double const gtyzdydy=grace::fd_der<der_order,1,1>(state,GTYZ_,VEC(i,j,k),q)* idx[1]*idx[1];
    double const gtzzdydy=grace::fd_der<der_order,1,1>(state,GTZZ_,VEC(i,j,k),q)* idx[1]*idx[1];

    // second z-derivatives of the conformal (tilde) metric components
    double const gtxxdzdz=grace::fd_der<der_order,2,2>(state,GTXX_,VEC(i,j,k),q)* idx[2]*idx[2];
    double const gtxydzdz=grace::fd_der<der_order,2,2>(state,GTXY_,VEC(i,j,k),q)* idx[2]*idx[2];
    double const gtyydzdz=grace::fd_der<der_order,2,2>(state,GTYY_,VEC(i,j,k),q)* idx[2]*idx[2];
    double const gtxzdzdz=grace::fd_der<der_order,2,2>(state,GTXZ_,VEC(i,j,k),q)* idx[2]*idx[2];
    double const gtyzdzdz=grace::fd_der<der_order,2,2>(state,GTYZ_,VEC(i,j,k),q)* idx[2]*idx[2];
    double const gtzzdzdz=grace::fd_der<der_order,2,2>(state,GTZZ_,VEC(i,j,k),q)* idx[2]*idx[2];

    // x-y-derivatives of the conformal (tilde) metric components
    double const gtxxdxdy=grace::fd_der<der_order,0,1>(state,GTXX_,VEC(i,j,k),q)* idx[0]*idx[1];
    double const gtxydxdy=grace::fd_der<der_order,0,1>(state,GTXY_,VEC(i,j,k),q)* idx[0]*idx[1];
    double const gtyydxdy=grace::fd_der<der_order,0,1>(state,GTYY_,VEC(i,j,k),q)* idx[0]*idx[1];
    double const gtxzdxdy=grace::fd_der<der_order,0,1>(state,GTXZ_,VEC(i,j,k),q)* idx[0]*idx[1];
    double const gtyzdxdy=grace::fd_der<der_order,0,1>(state,GTYZ_,VEC(i,j,k),q)* idx[0]*idx[1];
    double const gtzzdxdy=grace::fd_der<der_order,0,1>(state,GTZZ_,VEC(i,j,k),q)* idx[0]*idx[1];

    // x-z-derivatives of the conformal (tilde) metric components
    double const gtxxdxdz=grace::fd_der<der_order,0,2>(state,GTXX_,VEC(i,j,k),q)* idx[0]*idx[2];
    double const gtxydxdz=grace::fd_der<der_order,0,2>(state,GTXY_,VEC(i,j,k),q)* idx[0]*idx[2];
    double const gtyydxdz=grace::fd_der<der_order,0,2>(state,GTYY_,VEC(i,j,k),q)* idx[0]*idx[2];
    double const gtxzdxdz=grace::fd_der<der_order,0,2>(state,GTXZ_,VEC(i,j,k),q)* idx[0]*idx[2];
    double const gtyzdxdz=grace::fd_der<der_order,0,2>(state,GTYZ_,VEC(i,j,k),q)* idx[0]*idx[2];
    double const gtzzdxdz=grace::fd_der<der_order,0,2>(state,GTZZ_,VEC(i,j,k),q)* idx[0]*idx[2];

    // y-z-derivatives of the conformal (tilde) metric components
    double const gtxxdydz=grace::fd_der<der_order,1,2>(state,GTXX_,VEC(i,j,k),q)* idx[1]*idx[2];
    double const gtxydydz=grace::fd_der<der_order,1,2>(state,GTXY_,VEC(i,j,k),q)* idx[1]*idx[2];
    double const gtyydydz=grace::fd_der<der_order,1,2>(state,GTYY_,VEC(i,j,k),q)* idx[1]*idx[2];
    double const gtxzdydz=grace::fd_der<der_order,1,2>(state,GTXZ_,VEC(i,j,k),q)* idx[1]*idx[2];
    double const gtyzdydz=grace::fd_der<der_order,1,2>(state,GTYZ_,VEC(i,j,k),q)* idx[1]*idx[2];
    double const gtzzdydz=grace::fd_der<der_order,1,2>(state,GTZZ_,VEC(i,j,k),q)* idx[1]*idx[2];

    // x-derivatives of inverse conformal (tilde) metric components
    double const gtXXdx=-(((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(-2*gtyz*gtyzdx + gtyydx*gtzz + gtyy*gtzzdx) + (gtyz*gtyz - gtyy*gtzz)*(gtxz*gtxz*gtyydx + gtxxdx*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdx - 2*gtyz*(gtxydx*gtxz + gtxy*gtxzdx - gtxx*gtyzdx) + 2*gtxy*gtxydx*gtzz - gtxx*gtyydx*gtzz + gtxy*gtxy*gtzzdx - gtyy*(-2*gtxz*gtxzdx + gtxxdx*gtzz + gtxx*gtzzdx)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))));
    double const gtXYdx=(-((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxzdx*gtyz + gtxz*gtyzdx - gtxydx*gtzz - gtxy*gtzzdx)) + (gtxz*gtyz - gtxy*gtzz)*(gtxz*gtxz*gtyydx + gtxxdx*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdx - 2*gtyz*(gtxydx*gtxz + gtxy*gtxzdx - gtxx*gtyzdx) + 2*gtxy*gtxydx*gtzz - gtxx*gtyydx*gtzz + gtxy*gtxy*gtzzdx - gtyy*(-2*gtxz*gtxzdx + gtxxdx*gtzz + gtxx*gtzzdx)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtXZdx=((gtxzdx*gtyy + gtxz*gtyydx - gtxydx*gtyz - gtxy*gtyzdx)*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)) - (gtxz*gtyy - gtxy*gtyz)*(gtxz*gtxz*gtyydx + gtxxdx*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdx - 2*gtyz*(gtxydx*gtxz + gtxy*gtxzdx - gtxx*gtyzdx) + 2*gtxy*gtxydx*gtzz - gtxx*gtyydx*gtzz + gtxy*gtxy*gtzzdx - gtyy*(-2*gtxz*gtxzdx + gtxxdx*gtzz + gtxx*gtzzdx)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtYYdx=((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(2*gtxz*gtxzdx - gtxxdx*gtzz - gtxx*gtzzdx) - (gtxz*gtxz - gtxx*gtzz)*(gtxz*gtxz*gtyydx + gtxxdx*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdx - 2*gtyz*(gtxydx*gtxz + gtxy*gtxzdx - gtxx*gtyzdx) + 2*gtxy*gtxydx*gtzz - gtxx*gtyydx*gtzz + gtxy*gtxy*gtzzdx - gtyy*(-2*gtxz*gtxzdx + gtxxdx*gtzz + gtxx*gtzzdx)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtYZdx=(-((gtxydx*gtxz + gtxy*gtxzdx - gtxxdx*gtyz - gtxx*gtyzdx)*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))) + (gtxy*gtxz - gtxx*gtyz)*(gtxz*gtxz*gtyydx + gtxxdx*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdx - 2*gtyz*(gtxydx*gtxz + gtxy*gtxzdx - gtxx*gtyzdx) + 2*gtxy*gtxydx*gtzz - gtxx*gtyydx*gtzz + gtxy*gtxy*gtzzdx - gtyy*(-2*gtxz*gtxzdx + gtxxdx*gtzz + gtxx*gtzzdx)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtZZdx=-(((-2*gtxy*gtxydx + gtxxdx*gtyy + gtxx*gtyydx)*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)) + (gtxy*gtxy - gtxx*gtyy)*(gtxz*gtxz*gtyydx + gtxxdx*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdx - 2*gtyz*(gtxydx*gtxz + gtxy*gtxzdx - gtxx*gtyzdx) + 2*gtxy*gtxydx*gtzz - gtxx*gtyydx*gtzz + gtxy*gtxy*gtzzdx - gtyy*(-2*gtxz*gtxzdx + gtxxdx*gtzz + gtxx*gtzzdx)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))));
    
    // y-derivatives of inverse conformal (tilde) metric components
    double const gtXXdy=-(((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(-2*gtyz*gtyzdy + gtyydy*gtzz + gtyy*gtzzdy) + (gtyz*gtyz - gtyy*gtzz)*(gtxz*gtxz*gtyydy + gtxxdy*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdy - 2*gtyz*(gtxydy*gtxz + gtxy*gtxzdy - gtxx*gtyzdy) + 2*gtxy*gtxydy*gtzz - gtxx*gtyydy*gtzz + gtxy*gtxy*gtzzdy - gtyy*(-2*gtxz*gtxzdy + gtxxdy*gtzz + gtxx*gtzzdy)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))));
    double const gtXYdy=(-((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxzdy*gtyz + gtxz*gtyzdy - gtxydy*gtzz - gtxy*gtzzdy)) + (gtxz*gtyz - gtxy*gtzz)*(gtxz*gtxz*gtyydy + gtxxdy*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdy - 2*gtyz*(gtxydy*gtxz + gtxy*gtxzdy - gtxx*gtyzdy) + 2*gtxy*gtxydy*gtzz - gtxx*gtyydy*gtzz + gtxy*gtxy*gtzzdy - gtyy*(-2*gtxz*gtxzdy + gtxxdy*gtzz + gtxx*gtzzdy)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtXZdy=((gtxzdy*gtyy + gtxz*gtyydy - gtxydy*gtyz - gtxy*gtyzdy)*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)) - (gtxz*gtyy - gtxy*gtyz)*(gtxz*gtxz*gtyydy + gtxxdy*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdy - 2*gtyz*(gtxydy*gtxz + gtxy*gtxzdy - gtxx*gtyzdy) + 2*gtxy*gtxydy*gtzz - gtxx*gtyydy*gtzz + gtxy*gtxy*gtzzdy - gtyy*(-2*gtxz*gtxzdy + gtxxdy*gtzz + gtxx*gtzzdy)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtYYdy=((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(2*gtxz*gtxzdy - gtxxdy*gtzz - gtxx*gtzzdy) - (gtxz*gtxz - gtxx*gtzz)*(gtxz*gtxz*gtyydy + gtxxdy*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdy - 2*gtyz*(gtxydy*gtxz + gtxy*gtxzdy - gtxx*gtyzdy) + 2*gtxy*gtxydy*gtzz - gtxx*gtyydy*gtzz + gtxy*gtxy*gtzzdy - gtyy*(-2*gtxz*gtxzdy + gtxxdy*gtzz + gtxx*gtzzdy)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtYZdy=(-((gtxydy*gtxz + gtxy*gtxzdy - gtxxdy*gtyz - gtxx*gtyzdy)*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))) + (gtxy*gtxz - gtxx*gtyz)*(gtxz*gtxz*gtyydy + gtxxdy*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdy - 2*gtyz*(gtxydy*gtxz + gtxy*gtxzdy - gtxx*gtyzdy) + 2*gtxy*gtxydy*gtzz - gtxx*gtyydy*gtzz + gtxy*gtxy*gtzzdy - gtyy*(-2*gtxz*gtxzdy + gtxxdy*gtzz + gtxx*gtzzdy)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtZZdy=-(((-2*gtxy*gtxydy + gtxxdy*gtyy + gtxx*gtyydy)*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)) + (gtxy*gtxy - gtxx*gtyy)*(gtxz*gtxz*gtyydy + gtxxdy*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdy - 2*gtyz*(gtxydy*gtxz + gtxy*gtxzdy - gtxx*gtyzdy) + 2*gtxy*gtxydy*gtzz - gtxx*gtyydy*gtzz + gtxy*gtxy*gtzzdy - gtyy*(-2*gtxz*gtxzdy + gtxxdy*gtzz + gtxx*gtzzdy)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))));

    // z-derivatives of inverse conformal (tilde) metric components
    double const gtXXdz=-(((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(-2*gtyz*gtyzdz + gtyydz*gtzz + gtyy*gtzzdz) + (gtyz*gtyz - gtyy*gtzz)*(gtxz*gtxz*gtyydz + gtxxdz*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdz - 2*gtyz*(gtxydz*gtxz + gtxy*gtxzdz - gtxx*gtyzdz) + 2*gtxy*gtxydz*gtzz - gtxx*gtyydz*gtzz + gtxy*gtxy*gtzzdz - gtyy*(-2*gtxz*gtxzdz + gtxxdz*gtzz + gtxx*gtzzdz)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))));
    double const gtXYdz=(-((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxzdz*gtyz + gtxz*gtyzdz - gtxydz*gtzz - gtxy*gtzzdz)) + (gtxz*gtyz - gtxy*gtzz)*(gtxz*gtxz*gtyydz + gtxxdz*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdz - 2*gtyz*(gtxydz*gtxz + gtxy*gtxzdz - gtxx*gtyzdz) + 2*gtxy*gtxydz*gtzz - gtxx*gtyydz*gtzz + gtxy*gtxy*gtzzdz - gtyy*(-2*gtxz*gtxzdz + gtxxdz*gtzz + gtxx*gtzzdz)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtXZdz=((gtxzdz*gtyy + gtxz*gtyydz - gtxydz*gtyz - gtxy*gtyzdz)*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)) - (gtxz*gtyy - gtxy*gtyz)*(gtxz*gtxz*gtyydz + gtxxdz*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdz - 2*gtyz*(gtxydz*gtxz + gtxy*gtxzdz - gtxx*gtyzdz) + 2*gtxy*gtxydz*gtzz - gtxx*gtyydz*gtzz + gtxy*gtxy*gtzzdz - gtyy*(-2*gtxz*gtxzdz + gtxxdz*gtzz + gtxx*gtzzdz)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtYYdz=((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(2*gtxz*gtxzdz - gtxxdz*gtzz - gtxx*gtzzdz) - (gtxz*gtxz - gtxx*gtzz)*(gtxz*gtxz*gtyydz + gtxxdz*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdz - 2*gtyz*(gtxydz*gtxz + gtxy*gtxzdz - gtxx*gtyzdz) + 2*gtxy*gtxydz*gtzz - gtxx*gtyydz*gtzz + gtxy*gtxy*gtzzdz - gtyy*(-2*gtxz*gtxzdz + gtxxdz*gtzz + gtxx*gtzzdz)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtYZdz=(-((gtxydz*gtxz + gtxy*gtxzdz - gtxxdz*gtyz - gtxx*gtyzdz)*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))) + (gtxy*gtxz - gtxx*gtyz)*(gtxz*gtxz*gtyydz + gtxxdz*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdz - 2*gtyz*(gtxydz*gtxz + gtxy*gtxzdz - gtxx*gtyzdz) + 2*gtxy*gtxydz*gtzz - gtxx*gtyydz*gtzz + gtxy*gtxy*gtzzdz - gtyy*(-2*gtxz*gtxzdz + gtxxdz*gtzz + gtxx*gtzzdz)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)));
    double const gtZZdz=-(((-2*gtxy*gtxydz + gtxxdz*gtyy + gtxx*gtyydz)*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)) + (gtxy*gtxy - gtxx*gtyy)*(gtxz*gtxz*gtyydz + gtxxdz*(gtyz*gtyz) - 2*gtxy*gtxz*gtyzdz - 2*gtyz*(gtxydz*gtxz + gtxy*gtxzdz - gtxx*gtyzdz) + 2*gtxy*gtxydz*gtzz - gtxx*gtyydz*gtzz + gtxy*gtxy*gtzzdz - gtyy*(-2*gtxz*gtxzdz + gtxxdz*gtzz + gtxx*gtzzdz)))/((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))));
    
    // conformal factor
    double const phi=state(VEC(i,j,k), PHI_,q);

    // first derivatives of the conformal factor
    double const phidx=grace::fd_der<der_order,0>(state,PHI_,VEC(i,j,k),q)* idx[0];
    double const phidy=grace::fd_der<der_order,1>(state,PHI_,VEC(i,j,k),q)* idx[1];
    double const phidz=grace::fd_der<der_order,2>(state,PHI_,VEC(i,j,k),q)* idx[2];

    // second derivatives of the conformal factor 
    double const phidxdx=grace::fd_der<der_order,0,0>(state,PHI_,VEC(i,j,k),q)* idx[0]*idx[0];
    double const phidxdy=grace::fd_der<der_order,0,1>(state,PHI_,VEC(i,j,k),q)* idx[0]*idx[1];
    double const phidydy=grace::fd_der<der_order,1,1>(state,PHI_,VEC(i,j,k),q)* idx[1]*idx[1];
    double const phidxdz=grace::fd_der<der_order,0,2>(state,PHI_,VEC(i,j,k),q)* idx[0]*idx[2];
    double const phidydz=grace::fd_der<der_order,1,2>(state,PHI_,VEC(i,j,k),q)* idx[1]*idx[2];
    double const phidzdz=grace::fd_der<der_order,2,2>(state,PHI_,VEC(i,j,k),q)* idx[2]*idx[2];

    // lapse function
    double const alp=state(VEC(i,j,k), ALP_, q);

    // first derivatives of the lapse function 
    double const alpdx=grace::fd_der<der_order,0>(state,ALP_,VEC(i,j,k),q)* idx[0];
    double const alpdy=grace::fd_der<der_order,1>(state,ALP_,VEC(i,j,k),q)* idx[1];
    double const alpdz=grace::fd_der<der_order,2>(state,ALP_,VEC(i,j,k),q)* idx[2];

    // second derivatives of the lapse function 
    double const alpdxdx=grace::fd_der<der_order,0,0>(state,ALP_,VEC(i,j,k),q)* idx[0]*idx[0];
    double const alpdxdy=grace::fd_der<der_order,0,1>(state,ALP_,VEC(i,j,k),q)* idx[0]*idx[1];
    double const alpdydy=grace::fd_der<der_order,1,1>(state,ALP_,VEC(i,j,k),q)* idx[1]*idx[1];
    double const alpdxdz=grace::fd_der<der_order,0,2>(state,ALP_,VEC(i,j,k),q)* idx[0]*idx[2];
    double const alpdydz=grace::fd_der<der_order,1,2>(state,ALP_,VEC(i,j,k),q)* idx[1]*idx[2];
    double const alpdzdz=grace::fd_der<der_order,2,2>(state,ALP_,VEC(i,j,k),q)* idx[2]*idx[2];

    // shift vector components (with upper indices)
    double const betaX=state(VEC(i,j,k),BETAX_,q);
    double const betaY=state(VEC(i,j,k),BETAY_,q);
    double const betaZ=state(VEC(i,j,k),BETAZ_,q);

    // first derivatives of the shift vector components (with upper indices)
    double const betaXdx=grace::fd_der<der_order,0>(state,BETAX_,VEC(i,j,k),q)* idx[0];
    double const betaXdy=grace::fd_der<der_order,1>(state,BETAX_,VEC(i,j,k),q)* idx[1];
    double const betaXdz=grace::fd_der<der_order,2>(state,BETAX_,VEC(i,j,k),q)* idx[2];
    double const betaYdx=grace::fd_der<der_order,0>(state,BETAY_,VEC(i,j,k),q)* idx[0];
    double const betaYdy=grace::fd_der<der_order,1>(state,BETAY_,VEC(i,j,k),q)* idx[1];
    double const betaYdz=grace::fd_der<der_order,2>(state,BETAY_,VEC(i,j,k),q)* idx[2];
    double const betaZdx=grace::fd_der<der_order,0>(state,BETAZ_,VEC(i,j,k),q)* idx[0];
    double const betaZdy=grace::fd_der<der_order,1>(state,BETAZ_,VEC(i,j,k),q)* idx[1];
    double const betaZdz=grace::fd_der<der_order,2>(state,BETAZ_,VEC(i,j,k),q)* idx[2];

    // second derivatives of the shift vector components (with upper indices)
    // x-components
    double const betaXdxdx=grace::fd_der<der_order,0,0>(state,BETAX_,VEC(i,j,k),q)* idx[0]*idx[0];
    double const betaXdxdy=grace::fd_der<der_order,0,1>(state,BETAX_,VEC(i,j,k),q)* idx[0]*idx[1];
    double const betaXdxdz=grace::fd_der<der_order,0,2>(state,BETAX_,VEC(i,j,k),q)* idx[0]*idx[2];
    double const betaXdydy=grace::fd_der<der_order,1,1>(state,BETAX_,VEC(i,j,k),q)* idx[1]*idx[1];
    double const betaXdydz=grace::fd_der<der_order,1,2>(state,BETAX_,VEC(i,j,k),q)* idx[1]*idx[2];
    double const betaXdzdz=grace::fd_der<der_order,2,2>(state,BETAX_,VEC(i,j,k),q)* idx[2]*idx[2];
    // y-components
    double const betaYdxdx=grace::fd_der<der_order,0,0>(state,BETAY_,VEC(i,j,k),q)* idx[0]*idx[0];
    double const betaYdxdy=grace::fd_der<der_order,0,1>(state,BETAY_,VEC(i,j,k),q)* idx[0]*idx[1];
    double const betaYdxdz=grace::fd_der<der_order,0,2>(state,BETAY_,VEC(i,j,k),q)* idx[0]*idx[2];
    double const betaYdydy=grace::fd_der<der_order,1,1>(state,BETAY_,VEC(i,j,k),q)* idx[1]*idx[1];
    double const betaYdydz=grace::fd_der<der_order,1,2>(state,BETAY_,VEC(i,j,k),q)* idx[1]*idx[2];
    double const betaYdzdz=grace::fd_der<der_order,2,2>(state,BETAY_,VEC(i,j,k),q)* idx[2]*idx[2];
    // z-components
    double const betaZdxdx=grace::fd_der<der_order,0,0>(state,BETAZ_,VEC(i,j,k),q)* idx[0]*idx[0];
    double const betaZdxdy=grace::fd_der<der_order,0,1>(state,BETAZ_,VEC(i,j,k),q)* idx[0]*idx[1];
    double const betaZdxdz=grace::fd_der<der_order,0,2>(state,BETAZ_,VEC(i,j,k),q)* idx[0]*idx[2];
    double const betaZdydy=grace::fd_der<der_order,1,1>(state,BETAZ_,VEC(i,j,k),q)* idx[1]*idx[1];
    double const betaZdydz=grace::fd_der<der_order,1,2>(state,BETAZ_,VEC(i,j,k),q)* idx[1]*idx[2];
    double const betaZdzdz=grace::fd_der<der_order,2,2>(state,BETAZ_,VEC(i,j,k),q)* idx[2]*idx[2];

    // components of the energy momentum tensor
    double const Ttt=Tmunu[0][0];
    double const Ttx=Tmunu[0][1];
    double const Tty=Tmunu[0][2];
    double const Ttz=Tmunu[0][3];
    double const Txx=Tmunu[1][1];
    double const Txy=Tmunu[1][2];
    double const Txz=Tmunu[1][3];
    double const Tyy=Tmunu[2][2];
    double const Tyz=Tmunu[2][3];
    double const Tzz=Tmunu[3][3];

    // spatial compoents of the energy momentum tensor
    double const Sxx=Txx;
    double const Sxy=Txy;
    double const Sxz=Txz;
    double const Syy=Tyy;
    double const Syz=Tyz;
    double const Szz=Tzz;

    // momntum density components
    double const Sx=(-Ttx + betaX*Txx + betaY*Txy + betaZ*Txz)/alp;
    double const Sy=(-Tty + betaX*Txy + betaY*Tyy + betaZ*Tyz)/alp;
    double const Sz=(-Ttz + betaX*Txz + betaY*Tyz + betaZ*Tzz)/alp;

    // trace of the spatial energy momentum tensor
    double const S=phi*phi*(gtXX*Txx + 2*gtXY*Txy + 2*gtXZ*Txz + gtYY*Tyy + 2*gtYZ*Tyz + gtZZ*Tzz);

    // energy density
    double const EE=(Ttt - 2*betaY*Tty - 2*betaZ*Ttz + betaX*betaX*Txx + 2*betaX*(-Ttx + betaY*Txy + betaZ*Txz) + betaY*betaY*Tyy + 2*betaY*betaZ*Tyz + betaZ*betaZ*Tzz)/(alp*alp);

    // trace of the extrinsic curvature
    double const K=state(VEC(i,j,k), K_, q);

    // first derivatives of the extrinsic curvature trace 
    double const Kdx=grace::fd_der<der_order,0>(state,K_,VEC(i,j,k),q) * idx[0];
    double const Kdy=grace::fd_der<der_order,1>(state,K_,VEC(i,j,k),q) * idx[1];
    double const Kdz=grace::fd_der<der_order,2>(state,K_,VEC(i,j,k),q) * idx[2];

    // conformal (tilde) trace-free extrinsic curvature
    double const Atxx=state(VEC(i,j,k),ATXX_,q);
    double const Atxy=state(VEC(i,j,k),ATXY_,q);
    double const Atyy=state(VEC(i,j,k),ATYY_,q);
    double const Atxz=state(VEC(i,j,k),ATXZ_,q);
    double const Atyz=state(VEC(i,j,k),ATYZ_,q);
    double const Atzz=state(VEC(i,j,k),ATZZ_,q);

    // in order to reduce the number of terms in the BSSN equations it is usefull to 
    // use implicit definitions of upper-lower and upper-upper index components
    // of the conformal (tilde) trace-free extrinsic curvature. (AtudCode in Mathematica notebook BSSN.nb)
    // first index raised 
    double const AtXx=(Atxx*gtXX + Atxy*gtXY + Atxz*gtXZ)*(phi*phi);
    double const AtXy=(Atxx*gtXY + Atxy*gtYY + Atxz*gtYZ)*(phi*phi);
    double const AtXz=(Atxx*gtXZ + Atxy*gtYZ + Atxz*gtZZ)*(phi*phi);
    double const AtYy=(Atxy*gtXY + Atyy*gtYY + Atyz*gtYZ)*(phi*phi);
    double const AtYz=(Atxy*gtXZ + Atyy*gtYZ + Atyz*gtZZ)*(phi*phi);
    double const AtZz=(Atxz*gtXZ + Atyz*gtYZ + Atzz*gtZZ)*(phi*phi);
    // both indices raised 
    double const AtXX=(AtXx*gtXX + AtXy*gtXY + AtXz*gtXZ)*(phi*phi);
    double const AtXY=(AtXx*gtXY + AtXy*gtYY + AtXz*gtYZ)*(phi*phi);
    double const AtXZ=(AtXx*gtXZ + AtXy*gtYZ + AtXz*gtZZ)*(phi*phi);
    double const AtYY=(AtXy*gtXY + AtYy*gtYY + AtYz*gtYZ)*(phi*phi);
    double const AtYZ=(AtXy*gtXZ + AtYy*gtYZ + AtYz*gtZZ)*(phi*phi);
    double const AtZZ=(AtXz*gtXZ + AtYz*gtYZ + AtZz*gtZZ)*(phi*phi);
   
    // first x-derivatives of the conformal (tilde) trace-free extrinsic curvature
    double const Atxxdx=grace::fd_der<der_order,0>(state,ATXX_,VEC(i,j,k),q) * idx[0];
    double const Atxydx=grace::fd_der<der_order,0>(state,ATXY_,VEC(i,j,k),q) * idx[0];
    double const Atyydx=grace::fd_der<der_order,0>(state,ATYY_,VEC(i,j,k),q) * idx[0];
    double const Atxzdx=grace::fd_der<der_order,0>(state,ATXZ_,VEC(i,j,k),q) * idx[0];
    double const Atyzdx=grace::fd_der<der_order,0>(state,ATYZ_,VEC(i,j,k),q) * idx[0];
    double const Atzzdx=grace::fd_der<der_order,0>(state,ATZZ_,VEC(i,j,k),q) * idx[0];

    // first y-derivatives of the conformal (tilde) trace-free extrinsic curvature
    double const Atxxdy=grace::fd_der<der_order,1>(state,ATXX_,VEC(i,j,k),q) * idx[1];
    double const Atxydy=grace::fd_der<der_order,1>(state,ATXY_,VEC(i,j,k),q) * idx[1];
    double const Atyydy=grace::fd_der<der_order,1>(state,ATYY_,VEC(i,j,k),q) * idx[1];
    double const Atxzdy=grace::fd_der<der_order,1>(state,ATXZ_,VEC(i,j,k),q) * idx[1];
    double const Atyzdy=grace::fd_der<der_order,1>(state,ATYZ_,VEC(i,j,k),q) * idx[1];
    double const Atzzdy=grace::fd_der<der_order,1>(state,ATZZ_,VEC(i,j,k),q) * idx[1];

    // first z-derivatives of the conformal (tilde) trace-free extrinsic curvature
    double const Atxxdz=grace::fd_der<der_order,2>(state,ATXX_,VEC(i,j,k),q) * idx[2];
    double const Atxydz=grace::fd_der<der_order,2>(state,ATXY_,VEC(i,j,k),q) * idx[2];
    double const Atyydz=grace::fd_der<der_order,2>(state,ATYY_,VEC(i,j,k),q) * idx[2];
    double const Atxzdz=grace::fd_der<der_order,2>(state,ATXZ_,VEC(i,j,k),q) * idx[2];
    double const Atyzdz=grace::fd_der<der_order,2>(state,ATYZ_,VEC(i,j,k),q) * idx[2];
    double const Atzzdz=grace::fd_der<der_order,2>(state,ATZZ_,VEC(i,j,k),q) * idx[2];

    // first derivatives of the contracted conformal Christoffel symbol (check if GAMMA is actually GAMMAT)
    double const GammatXdx=grace::fd_der<der_order,0>(state,GAMMAX_,VEC(i,j,k),q) * idx[0];
    double const GammatXdy=grace::fd_der<der_order,1>(state,GAMMAX_,VEC(i,j,k),q) * idx[1];
    double const GammatXdz=grace::fd_der<der_order,2>(state,GAMMAX_,VEC(i,j,k),q) * idx[2];
    double const GammatYdx=grace::fd_der<der_order,0>(state,GAMMAY_,VEC(i,j,k),q) * idx[0];
    double const GammatYdy=grace::fd_der<der_order,1>(state,GAMMAY_,VEC(i,j,k),q) * idx[1];
    double const GammatYdz=grace::fd_der<der_order,2>(state,GAMMAY_,VEC(i,j,k),q) * idx[2];
    double const GammatZdx=grace::fd_der<der_order,0>(state,GAMMAZ_,VEC(i,j,k),q) * idx[0];
    double const GammatZdy=grace::fd_der<der_order,1>(state,GAMMAZ_,VEC(i,j,k),q) * idx[1];
    double const GammatZdz=grace::fd_der<der_order,2>(state,GAMMAZ_,VEC(i,j,k),q) * idx[2];

    // components of conformal Christoffel symbols (GammaTCode in Mathematica notebook BSSN.nb)
    double const Gammat111=(gtXX*gtxxdx - gtxxdy*gtXY + 2*gtXY*gtxydx - gtxxdz*gtXZ + 2*gtXZ*gtxzdx)/2.;
    double const Gammat112=(gtXX*gtxxdy - gtxydz*gtXZ + gtXZ*gtxzdy + gtXY*gtyydx + gtXZ*gtyzdx)/2.;
    double const Gammat113=(gtXX*gtxxdz + gtXY*(gtxydz - gtxzdy + gtyzdx) + gtXZ*gtzzdx)/2.;
    double const Gammat122=(2*gtXX*gtxydy - gtXX*gtyydx + gtXY*gtyydy - gtXZ*gtyydz + 2*gtXZ*gtyzdy)/2.;
    double const Gammat123=(gtXY*gtyydz + gtXX*(gtxydz + gtxzdy - gtyzdx) + gtXZ*gtzzdy)/2.;
    double const Gammat133=(2*gtXX*gtxzdz + 2*gtXY*gtyzdz - gtXX*gtzzdx - gtXY*gtzzdy + gtXZ*gtzzdz)/2.;
    double const Gammat211=(gtxxdx*gtXY - gtxxdy*gtYY + 2*gtxydx*gtYY - gtxxdz*gtYZ + 2*gtxzdx*gtYZ)/2.;
    double const Gammat212=(gtxxdy*gtXY + gtYY*gtyydx + gtYZ*(-gtxydz + gtxzdy + gtyzdx))/2.;
    double const Gammat213=(gtxxdz*gtXY + gtxydz*gtYY - gtxzdy*gtYY + gtYY*gtyzdx + gtYZ*gtzzdx)/2.;
    double const Gammat222=(2*gtXY*gtxydy - gtXY*gtyydx + gtYY*gtyydy - gtyydz*gtYZ + 2*gtYZ*gtyzdy)/2.;
    double const Gammat223=(gtYY*gtyydz + gtXY*(gtxydz + gtxzdy - gtyzdx) + gtYZ*gtzzdy)/2.;
    double const Gammat233=(2*gtXY*gtxzdz + 2*gtYY*gtyzdz - gtXY*gtzzdx - gtYY*gtzzdy + gtYZ*gtzzdz)/2.;
    double const Gammat311=(gtxxdx*gtXZ - gtxxdy*gtYZ + 2*gtxydx*gtYZ - gtxxdz*gtZZ + 2*gtxzdx*gtZZ)/2.;
    double const Gammat312=(gtxxdy*gtXZ + gtyydx*gtYZ + (-gtxydz + gtxzdy + gtyzdx)*gtZZ)/2.;
    double const Gammat313=(gtxxdz*gtXZ + gtxydz*gtYZ - gtxzdy*gtYZ + gtYZ*gtyzdx + gtZZ*gtzzdx)/2.;
    double const Gammat322=(2*gtxydy*gtXZ - gtXZ*gtyydx + gtyydy*gtYZ - gtyydz*gtZZ + 2*gtyzdy*gtZZ)/2.;
    double const Gammat323=(gtxydz*gtXZ + gtyydz*gtYZ + gtXZ*(gtxzdy - gtyzdx) + gtZZ*gtzzdy)/2.;
    double const Gammat333=(2*gtXZ*gtxzdz + 2*gtYZ*gtyzdz - gtXZ*gtzzdx - gtYZ*gtzzdy + gtZZ*gtzzdz)/2.;
    // x-derivatives
    double const Gammat111dx=(gtxxdx*gtXXdx + gtXX*gtxxdxdx - gtxxdxdy*gtXY - gtxxdy*gtXYdx + 2*gtxydx*gtXYdx + 2*gtXY*gtxydxdx - gtxxdxdz*gtXZ - gtxxdz*gtXZdx + 2*gtxzdx*gtXZdx + 2*gtXZ*gtxzdxdx)/2.;
    double const Gammat112dx=(gtXX*gtxxdxdy + gtXXdx*gtxxdy + gtXYdx*gtyydx + gtXY*gtyydxdx + gtXZdx*(-gtxydz + gtxzdy + gtyzdx) + gtXZ*(-gtxydxdz + gtxzdxdy + gtyzdxdx))/2.;
    double const Gammat113dx=(gtXX*gtxxdxdz + gtXXdx*gtxxdz + gtXYdx*(gtxydz - gtxzdy + gtyzdx) + gtXY*(gtxydxdz - gtxzdxdy + gtyzdxdx) + gtXZdx*gtzzdx + gtXZ*gtzzdxdx)/2.;
    double const Gammat122dx=(2*gtXX*gtxydxdy + 2*gtXXdx*gtxydy - gtXXdx*gtyydx - gtXX*gtyydxdx + gtXY*gtyydxdy - gtXZ*gtyydxdz + gtXYdx*gtyydy - gtXZdx*gtyydz + 2*gtXZ*gtyzdxdy + 2*gtXZdx*gtyzdy)/2.;
    double const Gammat123dx=(gtXY*gtyydxdz + gtXYdx*gtyydz + gtXXdx*(gtxydz + gtxzdy - gtyzdx) + gtXX*(gtxydxdz + gtxzdxdy - gtyzdxdx) + gtXZ*gtzzdxdy + gtXZdx*gtzzdy)/2.;
    double const Gammat133dx=(2*gtXX*gtxzdxdz + 2*gtXXdx*gtxzdz + 2*gtXY*gtyzdxdz + 2*gtXYdx*gtyzdz - gtXXdx*gtzzdx - gtXX*gtzzdxdx - gtXY*gtzzdxdy + gtXZ*gtzzdxdz - gtXYdx*gtzzdy + gtXZdx*gtzzdz)/2.;
    double const Gammat211dx=(gtxxdxdx*gtXY + gtxxdx*gtXYdx - gtxxdxdy*gtYY + 2*gtxydxdx*gtYY - gtxxdy*gtYYdx + 2*gtxydx*gtYYdx - gtxxdxdz*gtYZ + 2*gtxzdxdx*gtYZ - gtxxdz*gtYZdx + 2*gtxzdx*gtYZdx)/2.;
    double const Gammat212dx=(gtxxdxdy*gtXY + gtxxdy*gtXYdx + gtyydx*gtYYdx + gtYY*gtyydxdx + (-gtxydz + gtxzdy + gtyzdx)*gtYZdx + gtYZ*(-gtxydxdz + gtxzdxdy + gtyzdxdx))/2.;
    double const Gammat213dx=(gtxxdxdz*gtXY + gtxxdz*gtXYdx + gtYYdx*(gtxydz - gtxzdy + gtyzdx) + gtYY*(gtxydxdz - gtxzdxdy + gtyzdxdx) + gtYZdx*gtzzdx + gtYZ*gtzzdxdx)/2.;
    double const Gammat222dx=(2*gtXY*gtxydxdy + 2*gtXYdx*gtxydy - gtXYdx*gtyydx - gtXY*gtyydxdx + gtYY*gtyydxdy + gtYYdx*gtyydy - gtyydxdz*gtYZ - gtyydz*gtYZdx + 2*gtYZ*gtyzdxdy + 2*gtYZdx*gtyzdy)/2.;
    double const Gammat223dx=(gtYY*gtyydxdz + gtYYdx*gtyydz + gtXYdx*(gtxydz + gtxzdy - gtyzdx) + gtXY*(gtxydxdz + gtxzdxdy - gtyzdxdx) + gtYZ*gtzzdxdy + gtYZdx*gtzzdy)/2.;
    double const Gammat233dx=(2*gtXY*gtxzdxdz + 2*gtXYdx*gtxzdz + 2*gtYY*gtyzdxdz + 2*gtYYdx*gtyzdz - gtXYdx*gtzzdx - gtXY*gtzzdxdx - gtYY*gtzzdxdy + gtYZ*gtzzdxdz - gtYYdx*gtzzdy + gtYZdx*gtzzdz)/2.;
    double const Gammat311dx=(gtxxdxdx*gtXZ + gtxxdx*gtXZdx - gtxxdxdy*gtYZ + 2*gtxydxdx*gtYZ - gtxxdy*gtYZdx + 2*gtxydx*gtYZdx - gtxxdxdz*gtZZ + 2*gtxzdxdx*gtZZ - gtxxdz*gtZZdx + 2*gtxzdx*gtZZdx)/2.;
    double const Gammat312dx=(gtxxdxdy*gtXZ + gtxxdy*gtXZdx + gtyydxdx*gtYZ + gtyydx*gtYZdx + (-gtxydxdz + gtxzdxdy + gtyzdxdx)*gtZZ + (-gtxydz + gtxzdy + gtyzdx)*gtZZdx)/2.;
    double const Gammat313dx=(gtxxdxdz*gtXZ + gtxxdz*gtXZdx + (gtxydz - gtxzdy + gtyzdx)*gtYZdx + gtYZ*(gtxydxdz - gtxzdxdy + gtyzdxdx) + gtzzdx*gtZZdx + gtZZ*gtzzdxdx)/2.;
    double const Gammat322dx=(2*gtxydxdy*gtXZ + 2*gtxydy*gtXZdx - gtXZdx*gtyydx - gtXZ*gtyydxdx + gtyydxdy*gtYZ + gtyydy*gtYZdx - gtyydxdz*gtZZ + 2*gtyzdxdy*gtZZ - gtyydz*gtZZdx + 2*gtyzdy*gtZZdx)/2.;
    double const Gammat323dx=(gtyydxdz*gtYZ + gtXZdx*(gtxydz + gtxzdy - gtyzdx) + gtyydz*gtYZdx + gtXZ*(gtxydxdz + gtxzdxdy - gtyzdxdx) + gtZZ*gtzzdxdy + gtZZdx*gtzzdy)/2.;
    double const Gammat333dx=(2*gtXZ*gtxzdxdz + 2*gtXZdx*gtxzdz + 2*gtYZ*gtyzdxdz + 2*gtYZdx*gtyzdz - gtXZdx*gtzzdx - gtXZ*gtzzdxdx - gtYZ*gtzzdxdy + gtZZ*gtzzdxdz - gtYZdx*gtzzdy + gtZZdx*gtzzdz)/2.;
    // y-derivatives
    double const Gammat111dy=(gtXX*gtxxdxdy + gtxxdx*gtXXdy - gtxxdydy*gtXY + 2*gtXY*gtxydxdy - gtxxdy*gtXYdy + 2*gtxydx*gtXYdy - gtxxdydz*gtXZ + 2*gtXZ*gtxzdxdy - gtxxdz*gtXZdy + 2*gtxzdx*gtXZdy)/2.;
    double const Gammat112dy=(gtxxdy*gtXXdy + gtXX*gtxxdydy + gtXYdy*gtyydx + gtXY*gtyydxdy + gtXZdy*(-gtxydz + gtxzdy + gtyzdx) + gtXZ*(-gtxydydz + gtxzdydy + gtyzdxdy))/2.;
    double const Gammat113dy=(gtXX*gtxxdydz + gtXXdy*gtxxdz + gtXYdy*(gtxydz - gtxzdy + gtyzdx) + gtXY*(gtxydydz - gtxzdydy + gtyzdxdy) + gtXZdy*gtzzdx + gtXZ*gtzzdxdy)/2.;
    double const Gammat122dy=(2*gtXXdy*gtxydy + 2*gtXX*gtxydydy - gtXXdy*gtyydx - gtXX*gtyydxdy + gtXYdy*gtyydy + gtXY*gtyydydy - gtXZ*gtyydydz - gtXZdy*gtyydz + 2*gtXZdy*gtyzdy + 2*gtXZ*gtyzdydy)/2.;
    double const Gammat123dy=(gtXY*gtyydydz + gtXYdy*gtyydz + gtXXdy*(gtxydz + gtxzdy - gtyzdx) + gtXX*(gtxydydz + gtxzdydy - gtyzdxdy) + gtXZdy*gtzzdy + gtXZ*gtzzdydy)/2.;
    double const Gammat133dy=(2*gtXX*gtxzdydz + 2*gtXXdy*gtxzdz + 2*gtXY*gtyzdydz + 2*gtXYdy*gtyzdz - gtXXdy*gtzzdx - gtXX*gtzzdxdy - gtXYdy*gtzzdy - gtXY*gtzzdydy + gtXZ*gtzzdydz + gtXZdy*gtzzdz)/2.;
    double const Gammat211dy=(gtxxdxdy*gtXY + gtxxdx*gtXYdy - gtxxdydy*gtYY + 2*gtxydxdy*gtYY - gtxxdy*gtYYdy + 2*gtxydx*gtYYdy - gtxxdydz*gtYZ + 2*gtxzdxdy*gtYZ - gtxxdz*gtYZdy + 2*gtxzdx*gtYZdy)/2.;
    double const Gammat212dy=(gtxxdydy*gtXY + gtxxdy*gtXYdy + gtYY*gtyydxdy + gtyydx*gtYYdy + gtYZ*(-gtxydydz + gtxzdydy + gtyzdxdy) + (-gtxydz + gtxzdy + gtyzdx)*gtYZdy)/2.;
    double const Gammat213dy=(gtxxdydz*gtXY + gtxxdz*gtXYdy + gtYYdy*(gtxydz - gtxzdy + gtyzdx) + gtYY*(gtxydydz - gtxzdydy + gtyzdxdy) + gtYZdy*gtzzdx + gtYZ*gtzzdxdy)/2.;
    double const Gammat222dy=(2*gtxydy*gtXYdy + 2*gtXY*gtxydydy - gtXYdy*gtyydx - gtXY*gtyydxdy + gtyydy*gtYYdy + gtYY*gtyydydy - gtyydydz*gtYZ - gtyydz*gtYZdy + 2*gtyzdy*gtYZdy + 2*gtYZ*gtyzdydy)/2.;
    double const Gammat223dy=(gtYY*gtyydydz + gtYYdy*gtyydz + gtXYdy*(gtxydz + gtxzdy - gtyzdx) + gtXY*(gtxydydz + gtxzdydy - gtyzdxdy) + gtYZdy*gtzzdy + gtYZ*gtzzdydy)/2.;
    double const Gammat233dy=(2*gtXY*gtxzdydz + 2*gtXYdy*gtxzdz + 2*gtYY*gtyzdydz + 2*gtYYdy*gtyzdz - gtXYdy*gtzzdx - gtXY*gtzzdxdy - gtYYdy*gtzzdy - gtYY*gtzzdydy + gtYZ*gtzzdydz + gtYZdy*gtzzdz)/2.;
    double const Gammat311dy=(gtxxdxdy*gtXZ + gtxxdx*gtXZdy - gtxxdydy*gtYZ + 2*gtxydxdy*gtYZ - gtxxdy*gtYZdy + 2*gtxydx*gtYZdy - gtxxdydz*gtZZ + 2*gtxzdxdy*gtZZ - gtxxdz*gtZZdy + 2*gtxzdx*gtZZdy)/2.;
    double const Gammat312dy=(gtxxdydy*gtXZ + gtxxdy*gtXZdy + gtyydxdy*gtYZ + gtyydx*gtYZdy + (-gtxydydz + gtxzdydy + gtyzdxdy)*gtZZ + (-gtxydz + gtxzdy + gtyzdx)*gtZZdy)/2.;
    double const Gammat313dy=(gtxxdydz*gtXZ + gtxxdz*gtXZdy + gtYZ*(gtxydydz - gtxzdydy + gtyzdxdy) + (gtxydz - gtxzdy + gtyzdx)*gtYZdy + gtZZ*gtzzdxdy + gtzzdx*gtZZdy)/2.;
    double const Gammat322dy=(2*gtxydydy*gtXZ + 2*gtxydy*gtXZdy - gtXZdy*gtyydx - gtXZ*gtyydxdy + gtyydydy*gtYZ + gtyydy*gtYZdy - gtyydydz*gtZZ + 2*gtyzdydy*gtZZ - gtyydz*gtZZdy + 2*gtyzdy*gtZZdy)/2.;
    double const Gammat323dy=(gtyydydz*gtYZ + gtXZdy*(gtxydz + gtxzdy - gtyzdx) + gtXZ*(gtxydydz + gtxzdydy - gtyzdxdy) + gtyydz*gtYZdy + gtzzdy*gtZZdy + gtZZ*gtzzdydy)/2.;
    double const Gammat333dy=(2*gtXZ*gtxzdydz + 2*gtXZdy*gtxzdz + 2*gtYZ*gtyzdydz + 2*gtYZdy*gtyzdz - gtXZdy*gtzzdx - gtXZ*gtzzdxdy - gtYZdy*gtzzdy - gtYZ*gtzzdydy + gtZZ*gtzzdydz + gtZZdy*gtzzdz)/2.;
    // z-derivatives
    double const Gammat111dz=(gtXX*gtxxdxdz + gtxxdx*gtXXdz - gtxxdydz*gtXY + 2*gtXY*gtxydxdz - gtxxdy*gtXYdz + 2*gtxydx*gtXYdz - gtxxdzdz*gtXZ + 2*gtXZ*gtxzdxdz - gtxxdz*gtXZdz + 2*gtxzdx*gtXZdz)/2.;
    double const Gammat112dz=(gtXX*gtxxdydz + gtxxdy*gtXXdz + gtXYdz*gtyydx + gtXY*gtyydxdz + gtXZdz*(-gtxydz + gtxzdy + gtyzdx) + gtXZ*(-gtxydzdz + gtxzdydz + gtyzdxdz))/2.;
    double const Gammat113dz=(gtxxdz*gtXXdz + gtXX*gtxxdzdz + gtXYdz*(gtxydz - gtxzdy + gtyzdx) + gtXY*(gtxydzdz - gtxzdydz + gtyzdxdz) + gtXZdz*gtzzdx + gtXZ*gtzzdxdz)/2.;
    double const Gammat122dz=(2*gtXXdz*gtxydy + 2*gtXX*gtxydydz - gtXXdz*gtyydx - gtXX*gtyydxdz + gtXYdz*gtyydy + gtXY*gtyydydz - gtXZdz*gtyydz - gtXZ*gtyydzdz + 2*gtXZdz*gtyzdy + 2*gtXZ*gtyzdydz)/2.;
    double const Gammat123dz=(gtXYdz*gtyydz + gtXY*gtyydzdz + gtXXdz*(gtxydz + gtxzdy - gtyzdx) + gtXX*(gtxydzdz + gtxzdydz - gtyzdxdz) + gtXZdz*gtzzdy + gtXZ*gtzzdydz)/2.;
    double const Gammat133dz=(2*gtXXdz*gtxzdz + 2*gtXX*gtxzdzdz + 2*gtXYdz*gtyzdz + 2*gtXY*gtyzdzdz - gtXXdz*gtzzdx - gtXX*gtzzdxdz - gtXYdz*gtzzdy - gtXY*gtzzdydz + gtXZdz*gtzzdz + gtXZ*gtzzdzdz)/2.;
    double const Gammat211dz=(gtxxdxdz*gtXY + gtxxdx*gtXYdz - gtxxdydz*gtYY + 2*gtxydxdz*gtYY - gtxxdy*gtYYdz + 2*gtxydx*gtYYdz - gtxxdzdz*gtYZ + 2*gtxzdxdz*gtYZ - gtxxdz*gtYZdz + 2*gtxzdx*gtYZdz)/2.;
    double const Gammat212dz=(gtxxdydz*gtXY + gtxxdy*gtXYdz + gtYY*gtyydxdz + gtyydx*gtYYdz + gtYZ*(-gtxydzdz + gtxzdydz + gtyzdxdz) + (-gtxydz + gtxzdy + gtyzdx)*gtYZdz)/2.;
    double const Gammat213dz=(gtxxdzdz*gtXY + gtxxdz*gtXYdz + gtYYdz*(gtxydz - gtxzdy + gtyzdx) + gtYY*(gtxydzdz - gtxzdydz + gtyzdxdz) + gtYZdz*gtzzdx + gtYZ*gtzzdxdz)/2.;
    double const Gammat222dz=(2*gtXY*gtxydydz + 2*gtxydy*gtXYdz - gtXYdz*gtyydx - gtXY*gtyydxdz + gtYY*gtyydydz + gtyydy*gtYYdz - gtyydzdz*gtYZ + 2*gtYZ*gtyzdydz - gtyydz*gtYZdz + 2*gtyzdy*gtYZdz)/2.;
    double const Gammat223dz=(gtyydz*gtYYdz + gtYY*gtyydzdz + gtXYdz*(gtxydz + gtxzdy - gtyzdx) + gtXY*(gtxydzdz + gtxzdydz - gtyzdxdz) + gtYZdz*gtzzdy + gtYZ*gtzzdydz)/2.;
    double const Gammat233dz=(2*gtXYdz*gtxzdz + 2*gtXY*gtxzdzdz + 2*gtYYdz*gtyzdz + 2*gtYY*gtyzdzdz - gtXYdz*gtzzdx - gtXY*gtzzdxdz - gtYYdz*gtzzdy - gtYY*gtzzdydz + gtYZdz*gtzzdz + gtYZ*gtzzdzdz)/2.;
    double const Gammat311dz=(gtxxdxdz*gtXZ + gtxxdx*gtXZdz - gtxxdydz*gtYZ + 2*gtxydxdz*gtYZ - gtxxdy*gtYZdz + 2*gtxydx*gtYZdz - gtxxdzdz*gtZZ + 2*gtxzdxdz*gtZZ - gtxxdz*gtZZdz + 2*gtxzdx*gtZZdz)/2.;
    double const Gammat312dz=(gtxxdydz*gtXZ + gtxxdy*gtXZdz + gtyydxdz*gtYZ + gtyydx*gtYZdz + (-gtxydzdz + gtxzdydz + gtyzdxdz)*gtZZ + (-gtxydz + gtxzdy + gtyzdx)*gtZZdz)/2.;
    double const Gammat313dz=(gtxxdzdz*gtXZ + gtxxdz*gtXZdz + gtYZ*(gtxydzdz - gtxzdydz + gtyzdxdz) + (gtxydz - gtxzdy + gtyzdx)*gtYZdz + gtZZ*gtzzdxdz + gtzzdx*gtZZdz)/2.;
    double const Gammat322dz=(2*gtxydydz*gtXZ + 2*gtxydy*gtXZdz - gtXZdz*gtyydx - gtXZ*gtyydxdz + gtyydydz*gtYZ + gtyydy*gtYZdz - gtyydzdz*gtZZ + 2*gtyzdydz*gtZZ - gtyydz*gtZZdz + 2*gtyzdy*gtZZdz)/2.;
    double const Gammat323dz=(gtyydzdz*gtYZ + gtXZdz*(gtxydz + gtxzdy - gtyzdx) + gtXZ*(gtxydzdz + gtxzdydz - gtyzdxdz) + gtyydz*gtYZdz + gtZZ*gtzzdydz + gtzzdy*gtZZdz)/2.;
    double const Gammat333dz=(2*gtxzdz*gtXZdz + 2*gtXZ*gtxzdzdz + 2*gtyzdz*gtYZdz + 2*gtYZ*gtyzdzdz - gtXZdz*gtzzdx - gtXZ*gtzzdxdz - gtYZdz*gtzzdy - gtYZ*gtzzdydz + gtzzdz*gtZZdz + gtZZ*gtzzdzdz)/2.;

    // three ways to express the contracted conformal Christoffel symbols (GammaTCode in Mathematica notebook BSSN.nb)
    // 1. from the previous evolution step:
    double const GammatX=state(VEC(i,j,k),GAMMAX_,q);
    double const GammatY=state(VEC(i,j,k),GAMMAY_,q);
    double const GammatZ=state(VEC(i,j,k),GAMMAZ_,q);
    // 2. from the contraction of the confromal Christoffel symbol:
    //double const GammatX=Gammat111*gtXX + 2*Gammat112*gtXY + 2*Gammat113*gtXZ + Gammat122*gtYY + 2*Gammat123*gtYZ + Gammat133*gtZZ;
    //double const GammatY=Gammat211*gtXX + 2*Gammat212*gtXY + 2*Gammat213*gtXZ + Gammat222*gtYY + 2*Gammat223*gtYZ + Gammat233*gtZZ;
    //double const GammatZ=Gammat311*gtXX + 2*Gammat312*gtXY + 2*Gammat313*gtXZ + Gammat322*gtYY + 2*Gammat323*gtYZ + Gammat333*gtZZ;
    // 3. from the derivatives of the inverse metric (probably not useful because one needs to compute derivatives of inverse metric first):
    //double const GammatX=-gtXXdx - gtXYdy - gtXZdz;
    //double const GammatY=-gtXYdx - gtYYdy - gtYZdz;
    //double const GammatZ=-gtXZdx - gtYZdy - gtZZdz;

    // components of conformal Ricci tensor (RtddCode in Mathematica notebook BSSN.nb)
    double const Rtxx=-(Gammat112*Gammat211) + Gammat211dy + Gammat111*Gammat212 - Gammat212*Gammat212 - Gammat212dx + Gammat211*Gammat222 - Gammat113*Gammat311 + Gammat223*Gammat311 + Gammat311dz - 2*Gammat213*Gammat312 + Gammat111*Gammat313 - Gammat313*Gammat313 - Gammat313dx + Gammat211*Gammat323 + Gammat311*Gammat333;
    double const Rtxy=-Gammat111dy + Gammat112dx - Gammat122*Gammat211 + Gammat112*Gammat212 - Gammat123*Gammat311 + Gammat312dz + Gammat112*Gammat313 - Gammat313dy - Gammat213*Gammat322 + Gammat212*Gammat323 - Gammat313*Gammat323 + Gammat312*Gammat333;
    double const Rtxz=-Gammat111dz + Gammat113dx - Gammat123*Gammat211 + Gammat113*Gammat212 - Gammat212dz + Gammat213dy + Gammat213*Gammat222 - Gammat212*Gammat223 - Gammat133*Gammat311 - Gammat233*Gammat312 + Gammat113*Gammat313 + Gammat223*Gammat313;
    double const Rtyy=-(Gammat112*Gammat112) - Gammat112dy + Gammat111*Gammat122 + Gammat122dx - Gammat122*Gammat212 + Gammat112*Gammat222 - 2*Gammat123*Gammat312 + Gammat122*Gammat313 + Gammat113*Gammat322 - Gammat223*Gammat322 + Gammat322dz + Gammat222*Gammat323 - Gammat323*Gammat323 - Gammat323dy + Gammat322*Gammat333;
    double const Rtyz=-Gammat112dz + Gammat111*Gammat123 + Gammat123dx - Gammat122*Gammat213 - Gammat222dz + Gammat112*(-Gammat113 + Gammat223) + Gammat223dy - Gammat133*Gammat312 - Gammat233*Gammat322 + Gammat113*Gammat323 + Gammat223*Gammat323;
    double const Rtzz=-(Gammat113*Gammat113) - Gammat113dz + Gammat111*Gammat133 + Gammat133dx + Gammat133*Gammat212 - 2*Gammat123*Gammat213 - Gammat223*Gammat223 - Gammat223dz + Gammat112*Gammat233 + Gammat222*Gammat233 + Gammat233dy - Gammat133*Gammat313 - Gammat233*Gammat323 + Gammat113*Gammat333 + Gammat223*Gammat333;

    // components of the phi contribution to the relation between the Ricci tensor and the conformal Ricci tensor: Rdd=Rtdd+Rtphidd 
    double const Rtphixx=(-(Gammat111*phidx) + phidxdx - Gammat211*phidy - Gammat311*phidz + (gtxx*(-2 + phi)*(-9*GammatX*phidx + gtXX*phidxdx + 2*gtXY*phidxdy + 2*gtXZ*phidxdz - 9*GammatY*phidy + gtYY*phidydy + 2*gtYZ*phidydz - 9*GammatZ*phidz + gtZZ*phidzdz))/phi)/phi;
    double const Rtphixy=(-(Gammat112*phidx) + phidxdy - Gammat212*phidy - Gammat312*phidz + (gtxy*(-2 + phi)*(-9*GammatX*phidx + gtXX*phidxdx + 2*gtXY*phidxdy + 2*gtXZ*phidxdz - 9*GammatY*phidy + gtYY*phidydy + 2*gtYZ*phidydz - 9*GammatZ*phidz + gtZZ*phidzdz))/phi)/phi;
    double const Rtphixz=(-(Gammat113*phidx) + phidxdz - Gammat213*phidy - Gammat313*phidz + (gtxz*(-2 + phi)*(-9*GammatX*phidx + gtXX*phidxdx + 2*gtXY*phidxdy + 2*gtXZ*phidxdz - 9*GammatY*phidy + gtYY*phidydy + 2*gtYZ*phidydz - 9*GammatZ*phidz + gtZZ*phidzdz))/phi)/phi;
    double const Rtphiyy=(-(Gammat122*phidx) - Gammat222*phidy + phidydy - Gammat322*phidz + (gtyy*(-2 + phi)*(-9*GammatX*phidx + gtXX*phidxdx + 2*gtXY*phidxdy + 2*gtXZ*phidxdz - 9*GammatY*phidy + gtYY*phidydy + 2*gtYZ*phidydz - 9*GammatZ*phidz + gtZZ*phidzdz))/phi)/phi;
    double const Rtphiyz=(-(Gammat123*phidx) - Gammat223*phidy + phidydz - Gammat323*phidz + (gtyz*(-2 + phi)*(-9*GammatX*phidx + gtXX*phidxdx + 2*gtXY*phidxdy + 2*gtXZ*phidxdz - 9*GammatY*phidy + gtYY*phidydy + 2*gtYZ*phidydz - 9*GammatZ*phidz + gtZZ*phidzdz))/phi)/phi;
    double const Rtphizz=(-(Gammat133*phidx) - Gammat233*phidy - Gammat333*phidz + phidzdz + (gtzz*(-2 + phi)*(-9*GammatX*phidx + gtXX*phidxdx + 2*gtXY*phidxdy + 2*gtXZ*phidxdz - 9*GammatY*phidy + gtYY*phidydy + 2*gtYZ*phidydz - 9*GammatZ*phidz + gtZZ*phidzdz))/phi)/phi;
    
    // components of the Ricci tensor
    double const Rxx=Rtxx+Rtphixx;
    double const Rxy=Rtxy+Rtphixy;
    double const Rxz=Rtxz+Rtphixz;
    double const Ryy=Rtyy+Rtphiyy;
    double const Ryz=Rtyz+Rtphiyz;
    double const Rzz=Rtzz+Rtphizz;

    // BSSN eq.1: time derivatives of the conformal metric components (Eq1Code in Mathematica notebook BSSN.nb)
    double const gtxxdt=-2*alp*Atxx - (2*(-2*betaXdx + betaYdy + betaZdz)*gtxx)/3. + betaX*gtxxdx + betaY*gtxxdy + betaZ*gtxxdz + 2*betaYdx*gtxy + 2*betaZdx*gtxz;
    double const gtxydt=-2*alp*Atxy + betaXdy*gtxx + ((betaXdx + betaYdy - 2*betaZdz)*gtxy)/3. + betaX*gtxydx + betaY*gtxydy + betaZ*gtxydz + betaZdy*gtxz + betaYdx*gtyy + betaZdx*gtyz;
    double const gtxzdt=-2*alp*Atxz + betaXdz*gtxx + betaYdz*gtxy + (betaXdx*gtxz)/3. - (2*betaYdy*gtxz)/3. + (betaZdz*gtxz)/3. + betaX*gtxzdx + betaY*gtxzdy + betaZ*gtxzdz + betaYdx*gtyz + betaZdx*gtzz;
    double const gtyydt=-2*alp*Atyy + 2*betaXdy*gtxy - (2*(betaXdx - 2*betaYdy + betaZdz)*gtyy)/3. + betaX*gtyydx + betaY*gtyydy + betaZ*gtyydz + 2*betaZdy*gtyz;
    double const gtyzdt=-2*alp*Atyz + betaXdz*gtxy + betaXdy*gtxz + betaYdz*gtyy - (2*betaXdx*gtyz)/3. + (betaYdy*gtyz)/3. + (betaZdz*gtyz)/3. + betaX*gtyzdx + betaY*gtyzdy + betaZ*gtyzdz + betaZdy*gtzz;
    double const gtzzdt=-2*alp*Atzz + 2*betaXdz*gtxz + 2*betaYdz*gtyz + 2*betaZdz*gtzz - (2*(betaXdx + betaYdy + betaZdz)*gtzz)/3. + betaX*gtzzdx + betaY*gtzzdy + betaZ*gtzzdz;

    // BSSN eq.2: time derivatives of the conformal extrinsic curvature components (Eq2Code in Mathematica notebook BSSN.nb)
    double const Atxxdt=-2*alp*(Atxy*AtXy + Atxz*AtXz) + Atxxdx*betaX + 2*Atxx*betaXdx + Atxxdy*betaY + 2*Atxy*betaYdx + Atxxdz*betaZ + 2*Atxz*betaZdx - (2*Atxx*(betaXdx + betaYdy + betaZdz))/3. + alp*Atxx*(-2*AtXx + K) - (phi*phi*(2*alpdxdx - alpdydy - alpdzdz - 2*alpdx*Gammat111 + alpdx*Gammat122 + alpdx*Gammat133 - 2*alpdy*Gammat211 + alpdy*Gammat222 + alpdy*Gammat233 - 2*alpdz*Gammat311 + alpdz*Gammat322 + alpdz*Gammat333 - 2*alp*Rxx + alp*Ryy + alp*Rzz + 16*alp*M_PI*Sxx - 8*alp*M_PI*Syy - 8*alp*M_PI*Szz))/3.;
    double const Atxydt=Atxydx*betaX + Atxy*betaXdx + Atxx*betaXdy + Atxydy*betaY + Atyy*betaYdx + Atxy*betaYdy + Atxydz*betaZ + Atyz*betaZdx + Atxz*betaZdy - (2*Atxy*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*Atxx*AtXy - 2*Atxz*AtYz + Atxy*(-2*AtYy + K)) + phi*phi*(-alpdxdy + alpdx*Gammat112 + alpdy*Gammat212 + alpdz*Gammat312 + alp*Rxy - 8*alp*M_PI*Sxy);
    double const Atxzdt=Atxzdx*betaX + Atxz*betaXdx + Atxx*betaXdz + Atxzdy*betaY + Atyz*betaYdx + Atxy*betaYdz + Atxzdz*betaZ + Atzz*betaZdx + Atxz*betaZdz - (2*Atxz*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*Atxx*AtXz - 2*Atxy*AtYz + Atxz*(-2*AtZz + K)) + phi*phi*(-alpdxdz + alpdx*Gammat113 + alpdy*Gammat213 + alpdz*Gammat313 + alp*Rxz - 8*alp*M_PI*Sxz);
    double const Atyydt=Atyydx*betaX + 2*Atxy*betaXdy + Atyydy*betaY + 2*Atyy*betaYdy + Atyydz*betaZ + 2*Atyz*betaZdy - (2*Atyy*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*Atxy*AtXy - 2*Atyz*AtYz + Atyy*(-2*AtYy + K)) - (phi*phi*(-alpdxdx + 2*alpdydy - alpdzdz + alpdx*Gammat111 - 2*alpdx*Gammat122 + alpdx*Gammat133 + alpdy*Gammat211 - 2*alpdy*Gammat222 + alpdy*Gammat233 + alpdz*Gammat311 - 2*alpdz*Gammat322 + alpdz*Gammat333 + alp*(Rxx - 2*Ryy + Rzz - 8*M_PI*Sxx + 16*M_PI*Syy - 8*M_PI*Szz)))/3.;
    double const Atyzdt=Atyzdx*betaX + Atxz*betaXdy + Atxy*betaXdz + Atyzdy*betaY + Atyz*betaYdy + Atyy*betaYdz + Atyzdz*betaZ + Atzz*betaZdy + Atyz*betaZdz - (2*Atyz*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*Atxy*AtXz - 2*Atyy*AtYz + Atyz*(-2*AtZz + K)) + phi*phi*(-alpdydz + alpdx*Gammat123 + alpdy*Gammat223 + alpdz*Gammat323 + alp*Ryz - 8*alp*M_PI*Syz);
    double const Atzzdt=Atzzdx*betaX + 2*Atxz*betaXdz + Atzzdy*betaY + 2*Atyz*betaYdz + Atzzdz*betaZ + 2*Atzz*betaZdz - (2*Atzz*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*Atxz*AtXz - 2*Atyz*AtYz + Atzz*(-2*AtZz + K)) - (phi*phi*(-alpdxdx - alpdydy + 2*alpdzdz + alpdx*Gammat111 + alpdx*Gammat122 - 2*alpdx*Gammat133 + alpdy*Gammat211 + alpdy*Gammat222 - 2*alpdy*Gammat233 + alpdz*Gammat311 + alpdz*Gammat322 - 2*alpdz*Gammat333 + alp*(Rxx + Ryy - 2*(Rzz + 4*M_PI*(Sxx + Syy - 2*Szz)))))/3.;

    // BSSN eq.3: time derivative of the conformal factor (Eq3Code in Mathematica notebook BSSN.nb)
    double const phidt=-1/3.*((betaXdx + betaYdy + betaZdz)*phi) + (alp*K*phi)/3. + betaX*phidx + betaY*phidy + betaZ*phidz;

    // BSSN eq.4: time derivative of the extrinsic curvature (Eq4Code in Mathematica notebook BSSN.nb)
    double const Kdt=betaX*Kdx + betaY*Kdy + betaZ*Kdz - (-((-alpdxdx + alpdx*Gammat111 + alpdy*Gammat211 + alpdz*Gammat311)*gtXX) - 2*(-alpdxdy + alpdx*Gammat112 + alpdy*Gammat212 + alpdz*Gammat312)*gtXY - 2*(-alpdxdz + alpdx*Gammat113 + alpdy*Gammat213 + alpdz*Gammat313)*gtXZ - (-alpdydy + alpdx*Gammat122 + alpdy*Gammat222 + alpdz*Gammat322)*gtYY - 2*(-alpdydz + alpdx*Gammat123 + alpdy*Gammat223 + alpdz*Gammat323)*gtYZ - (-alpdzdz + alpdx*Gammat133 + alpdy*Gammat233 + alpdz*Gammat333)*gtZZ)*(phi*phi) + alp*(Atxx*AtXX + 2*Atxy*AtXY + 2*Atxz*AtXZ + Atyy*AtYY + 2*Atyz*AtYZ + Atzz*AtZZ + (K*K)/3. + 4*M_PI*(EE + S));
    
    // BSSN eq.5: time derivative of the extrinsic curvature (Eq4Code in Mathematica notebook BSSN.nb)
    double const GammatXdt=(-6*(alpdx*AtXX + alpdy*AtXY + alpdz*AtXZ) - 3*betaXdx*GammatX + 2*(betaXdx + betaYdy + betaZdz)*GammatX + 3*betaX*GammatXdx + 3*betaY*GammatXdy + 3*betaZ*GammatXdz - 3*betaXdy*GammatY - 3*betaXdz*GammatZ + 4*betaXdxdx*gtXX + betaYdxdy*gtXX + betaZdxdz*gtXX + 7*betaXdxdy*gtXY + betaYdydy*gtXY + betaZdydz*gtXY + 7*betaXdxdz*gtXZ + betaYdydz*gtXZ + betaZdzdz*gtXZ + 3*betaXdydy*gtYY + 6*betaXdydz*gtYZ + 3*betaXdzdz*gtZZ + 6*alp*(AtXX*Gammat111 + 2*AtXY*Gammat112 + 2*AtXZ*Gammat113 + AtYY*Gammat122 + 2*AtYZ*Gammat123 + AtZZ*Gammat133 - (2*(gtXX*Kdx + gtXY*Kdy + gtXZ*Kdz))/3. - (3*(AtXX*phidx + AtXY*phidy + AtXZ*phidz))/phi) - 48*alp*M_PI*(gtXX*Sx + gtXY*Sy + gtXZ*Sz))/3.;
    double const GammatYdt=(-6*(alpdx*AtXY + alpdy*AtYY + alpdz*AtYZ) - 3*betaYdx*GammatX - 3*betaYdy*GammatY + 2*(betaXdx + betaYdy + betaZdz)*GammatY + 3*betaX*GammatYdx + 3*betaY*GammatYdy + 3*betaZ*GammatYdz - 3*betaYdz*GammatZ + 3*betaYdxdx*gtXX + betaXdxdx*gtXY + 7*betaYdxdy*gtXY + betaZdxdz*gtXY + 6*betaYdxdz*gtXZ + betaXdxdy*gtYY + 4*betaYdydy*gtYY + betaZdydz*gtYY + betaXdxdz*gtYZ + 7*betaYdydz*gtYZ + betaZdzdz*gtYZ + 3*betaYdzdz*gtZZ + 6*alp*(AtXX*Gammat211 + 2*AtXY*Gammat212 + 2*AtXZ*Gammat213 + AtYY*Gammat222 + 2*AtYZ*Gammat223 + AtZZ*Gammat233 - (2*(gtXY*Kdx + gtYY*Kdy + gtYZ*Kdz))/3. - (3*(AtXY*phidx + AtYY*phidy + AtYZ*phidz))/phi) - 48*alp*M_PI*(gtXY*Sx + gtYY*Sy + gtYZ*Sz))/3.;
    double const GammatZdt=(-6*(alpdx*AtXZ + alpdy*AtYZ + alpdz*AtZZ) - 3*betaZdx*GammatX - 3*betaZdy*GammatY - 3*betaZdz*GammatZ + 2*(betaXdx + betaYdy + betaZdz)*GammatZ + 3*betaX*GammatZdx + 3*betaY*GammatZdy + 3*betaZ*GammatZdz + 3*betaZdxdx*gtXX + 6*betaZdxdy*gtXY + betaXdxdx*gtXZ + betaYdxdy*gtXZ + 7*betaZdxdz*gtXZ + 3*betaZdydy*gtYY + betaXdxdy*gtYZ + betaYdydy*gtYZ + 7*betaZdydz*gtYZ + betaXdxdz*gtZZ + betaYdydz*gtZZ + 4*betaZdzdz*gtZZ + 6*alp*(AtXX*Gammat311 + 2*AtXY*Gammat312 + 2*AtXZ*Gammat313 + AtYY*Gammat322 + 2*AtYZ*Gammat323 + AtZZ*Gammat333 - (2*(gtXZ*Kdx + gtYZ*Kdy + gtZZ*Kdz))/3. - (3*(AtXZ*phidx + AtYZ*phidy + AtZZ*phidz))/phi) - 48*alp*M_PI*(gtXZ*Sx + gtYZ*Sy + gtZZ*Sz))/3.;

    // 1+log slicing condition
    double const muL=2/alp;
    double const muS=1/(alp*alp);

    // gauge condition for the lapse function
    double const alpdt=alpdx*betaX + alpdy*betaY + alpdz*betaZ - alp*alp*K*muL;

    // gauge condition for the shift vector
    double eta = 1 ; //FIXME
    double const betaXdt=betaXdy*betaY + betaXdz*betaZ + betaX*(betaXdx - eta) + alp*alp*GammatX*muS;
    double const betaYdt=betaX*betaYdx + betaYdz*betaZ + betaY*(betaYdy - eta) + alp*alp*GammatY*muS;
    double const betaZdt=betaX*betaZdx + betaY*betaZdy + betaZ*betaZdz - betaZ*eta + alp*alp*GammatZ*muS;
    
    // returning the RHS of the EOM vector
    return { phidt, gtxxdt, gtxydt, gtxzdt, gtyydt, gtyzdt, gtzzdt,
             Atxxdt, Atxydt, Atxzdt, Atyydt, Atyzdt, Atzzdt,
             Kdt, GammatXdt, GammatYdt, GammatZdt };

}

#define INSTANTIATE_TEMPLATE(DER_ORD) \
template                     \
grace::bssn_state_t GRACE_HOST_DEVICE                                \
compute_bssn_rhs<DER_ORD>( VEC(int i, int j, int k), int q                    \
                , grace::var_array_t<GRACE_NSPACEDIM> const state    \
                , std::array<std::array<double,4>,4> const& Tmunu    \
                , std::array<double,GRACE_NSPACEDIM> const& idx)

INSTANTIATE_TEMPLATE(2) ; 
INSTANTIATE_TEMPLATE(4) ; 
#undef INSTANTIATE_TEMPLATE
}