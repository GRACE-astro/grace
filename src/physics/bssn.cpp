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

#include <grace/physics/bssn.hh>

#include <grace/utils/fd_utils.hh>

namespace grace {

template< size_t der_order >
bssn_state_t GRACE_HOST_DEVICE 
compute_bssn_rhs( grace::var_array_t<GRACE_NSPACEDIM> const state
                , std::array<std::array<double,4>,4> const& Tmunu
                , std::array<double,GRACE_NSPACEDIM> const& idx)
{

    // local definition of variables necessary to compute the r.h.s. of the BSSN equations

    // spatial compoents of the energy momentum tensor
    double const Sxx=Tmunu[1][1];
    double const Sxy=Tmunu[1][2];
    double const Syy=Tmunu[2][2];
    double const Sxz=Tmunu[1][3];
    double const Syz=Tmunu[2][3];
    double const Szz=Tmunu[3][3];
    
    // conformal (tilde) metric components
    double const gtxx=state(VEC(i,j,k),GTXX_,q);
    double const gtxy=state(VEC(i,j,k),GTXY_,q);
    double const gtyy=state(VEC(i,j,k),GTYY_,q);
    double const gtxz=state(VEC(i,j,k),GTXZ_,q);
    double const gtyz=state(VEC(i,j,k),GTYZ_,q);
    double const gtzz=state(VEC(i,j,k),GTZZ_,q);
    
    //inverse conformal (tilde) metric components
    double const gtXX=(gtyz*gtyz - gtyy*gtzz)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxx*(gtyz*gtyz) + gtxy*gtxy*gtzz - gtxx*gtyy*gtzz);
    double const gtXY=(-(gtxz*gtyz) + gtxy*gtzz)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz));
    double const gtXZ=(gtxz*gtyy - gtxy*gtyz)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxx*(gtyz*gtyz) + gtxy*gtxy*gtzz - gtxx*gtyy*gtzz);
    double const gtYY=(gtxz*gtxz - gtxx*gtzz)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxx*(gtyz*gtyz) + gtxy*gtxy*gtzz - gtxx*gtyy*gtzz);
    double const gtYZ=(-(gtxy*gtxz) + gtxx*gtyz)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz));
    double const gtZZ=(gtxy*gtxy - gtxx*gtyy)/(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxx*(gtyz*gtyz) + gtxy*gtxy*gtzz - gtxx*gtyy*gtzz);

    // first x-derivatives of the conformal (tilde) metric components
    double const gtxxdx=grace::fd_der<der_order,0>(state,GTXX_,VEC(i,j,k),q);
    double const gtxydx=grace::fd_der<der_order,0>(state,GTXY_,VEC(i,j,k),q);
    double const gtyydx=grace::fd_der<der_order,0>(state,GTYY_,VEC(i,j,k),q);
    double const gtxzdx=grace::fd_der<der_order,0>(state,GTXZ_,VEC(i,j,k),q);
    double const gtyzdx=grace::fd_der<der_order,0>(state,GTYZ_,VEC(i,j,k),q);
    double const gtzzdx=grace::fd_der<der_order,0>(state,GTZZ_,VEC(i,j,k),q);

    // first y-derivatives of the conformal (tilde) metric components
    double const gtxxdy=grace::fd_der<der_order,1>(state,GTXX_,VEC(i,j,k),q);
    double const gtxydy=grace::fd_der<der_order,1>(state,GTXY_,VEC(i,j,k),q);
    double const gtyydy=grace::fd_der<der_order,1>(state,GTYY_,VEC(i,j,k),q);
    double const gtxzdy=grace::fd_der<der_order,1>(state,GTXZ_,VEC(i,j,k),q);
    double const gtyzdy=grace::fd_der<der_order,1>(state,GTYZ_,VEC(i,j,k),q);
    double const gtzzdy=grace::fd_der<der_order,1>(state,GTZZ_,VEC(i,j,k),q);

    // first z-derivatives of the conformal (tilde) metric components
    double const gtxxdz=grace::fd_der<der_order,2>(state,GTXX_,VEC(i,j,k),q);
    double const gtxydz=grace::fd_der<der_order,2>(state,GTXY_,VEC(i,j,k),q);
    double const gtyydz=grace::fd_der<der_order,2>(state,GTYY_,VEC(i,j,k),q);
    double const gtxzdz=grace::fd_der<der_order,2>(state,GTXZ_,VEC(i,j,k),q);
    double const gtyzdz=grace::fd_der<der_order,2>(state,GTYZ_,VEC(i,j,k),q);
    double const gtzzdz=grace::fd_der<der_order,2>(state,GTZZ_,VEC(i,j,k),q);

    // second x-derivatives of the conformal (tilde) metric components
    double const gtxxdxdx=grace::fd_der<der_order,0,0>(state,GTXX_,VEC(i,j,k),q);
    double const gtxydxdx=grace::fd_der<der_order,0,0>(state,GTXY_,VEC(i,j,k),q);
    double const gtyydxdx=grace::fd_der<der_order,0,0>(state,GTYY_,VEC(i,j,k),q);
    double const gtxzdxdx=grace::fd_der<der_order,0,0>(state,GTXZ_,VEC(i,j,k),q);
    double const gtyzdxdx=grace::fd_der<der_order,0,0>(state,GTYZ_,VEC(i,j,k),q);
    double const gtzzdxdx=grace::fd_der<der_order,0,0>(state,GTZZ_,VEC(i,j,k),q);

    // second y-derivatives of the conformal (tilde) metric components
    double const gtxxdydy=grace::fd_der<der_order,1,1>(state,GTXX_,VEC(i,j,k),q);
    double const gtxydydy=grace::fd_der<der_order,1,1>(state,GTXY_,VEC(i,j,k),q);
    double const gtyydydy=grace::fd_der<der_order,1,1>(state,GTYY_,VEC(i,j,k),q);
    double const gtxzdydy=grace::fd_der<der_order,1,1>(state,GTXZ_,VEC(i,j,k),q);
    double const gtyzdydy=grace::fd_der<der_order,1,1>(state,GTYZ_,VEC(i,j,k),q);
    double const gtzzdydy=grace::fd_der<der_order,1,1>(state,GTZZ_,VEC(i,j,k),q);

    // second z-derivatives of the conformal (tilde) metric components
    double const gtxxdzdz=grace::fd_der<der_order,2,2>(state,GTXX_,VEC(i,j,k),q);
    double const gtxydzdz=grace::fd_der<der_order,2,2>(state,GTXY_,VEC(i,j,k),q);
    double const gtyydzdz=grace::fd_der<der_order,2,2>(state,GTYY_,VEC(i,j,k),q);
    double const gtxzdzdz=grace::fd_der<der_order,2,2>(state,GTXZ_,VEC(i,j,k),q);
    double const gtyzdzdz=grace::fd_der<der_order,2,2>(state,GTYZ_,VEC(i,j,k),q);
    double const gtzzdzdz=grace::fd_der<der_order,2,2>(state,GTZZ_,VEC(i,j,k),q);

    // x-y-derivatives of the conformal (tilde) metric components
    double const gtxxdxdy=grace::fd_der<der_order,0,1>(state,GTXX_,VEC(i,j,k),q);
    double const gtxydxdy=grace::fd_der<der_order,0,1>(state,GTXY_,VEC(i,j,k),q);
    double const gtyydxdy=grace::fd_der<der_order,0,1>(state,GTYY_,VEC(i,j,k),q);
    double const gtxzdxdy=grace::fd_der<der_order,0,1>(state,GTXZ_,VEC(i,j,k),q);
    double const gtyzdxdy=grace::fd_der<der_order,0,1>(state,GTYZ_,VEC(i,j,k),q);
    double const gtzzdxdy=grace::fd_der<der_order,0,1>(state,GTZZ_,VEC(i,j,k),q);

    // x-z-derivatives of the conformal (tilde) metric components
    double const gtxxdxdz=grace::fd_der<der_order,0,2>(state,GTXX_,VEC(i,j,k),q);
    double const gtxydxdz=grace::fd_der<der_order,0,2>(state,GTXY_,VEC(i,j,k),q);
    double const gtyydxdz=grace::fd_der<der_order,0,2>(state,GTYY_,VEC(i,j,k),q);
    double const gtxzdxdz=grace::fd_der<der_order,0,2>(state,GTXZ_,VEC(i,j,k),q);
    double const gtyzdxdz=grace::fd_der<der_order,0,2>(state,GTYZ_,VEC(i,j,k),q);
    double const gtzzdxdz=grace::fd_der<der_order,0,2>(state,GTZZ_,VEC(i,j,k),q);

    // y-z-derivatives of the conformal (tilde) metric components
    double const gtxxdydz=grace::fd_der<der_order,1,2>(state,GTXX_,VEC(i,j,k),q);
    double const gtxydydz=grace::fd_der<der_order,1,2>(state,GTXY_,VEC(i,j,k),q);
    double const gtyydydz=grace::fd_der<der_order,1,2>(state,GTYY_,VEC(i,j,k),q);
    double const gtxzdydz=grace::fd_der<der_order,1,2>(state,GTXZ_,VEC(i,j,k),q);
    double const gtyzdydz=grace::fd_der<der_order,1,2>(state,GTYZ_,VEC(i,j,k),q);
    double const gtzzdydz=grace::fd_der<der_order,1,2>(state,GTZZ_,VEC(i,j,k),q);

    // lapse function
    double const alp=state(ALP_);
    
    // first derivatives of the lapse function 
    double const alpdx=grace::fd_der<der_order,0>(state,ALP_);
    double const alpdy=grace::fd_der<der_order,1>(state,ALP_);
    double const alpdz=grace::fd_der<der_order,2>(state,ALP_);

    // second derivatives of the lapse function 
    double const alpdxdx=grace::fd_der<der_order,0,0>(state,ALP_);
    double const alpdxdy=grace::fd_der<der_order,0,1>(state,ALP_);
    double const alpdydy=grace::fd_der<der_order,1,1>(state,ALP_);
    double const alpdxdz=grace::fd_der<der_order,0,2>(state,ALP_);
    double const alpdydz=grace::fd_der<der_order,1,2>(state,ALP_);
    double const alpdzdz=grace::fd_der<der_order,2,2>(state,ALP_);

    // shift vector components (with upper indices)
    double const betax=state(VEC(i,j,k),BETAX_,q);
    double const betay=state(VEC(i,j,k),BETAY_,q);
    double const betaz=state(VEC(i,j,k),BETAZ_,q);

    // first derivatives of the shift vector components (with upper indices)
    double const betaxdx=grace::fd_der<der_order,0>(state,BETAX_,VEC(i,j,k),q);
    double const betaxdy=grace::fd_der<der_order,1>(state,BETAX_,VEC(i,j,k),q);
    double const betaxdz=grace::fd_der<der_order,2>(state,BETAX_,VEC(i,j,k),q);
    double const betaydx=grace::fd_der<der_order,0>(state,BETAY_,VEC(i,j,k),q);
    double const betaydy=grace::fd_der<der_order,1>(state,BETAY_,VEC(i,j,k),q);
    double const betaydz=grace::fd_der<der_order,2>(state,BETAY_,VEC(i,j,k),q);
    double const betazdx=grace::fd_der<der_order,0>(state,BETAZ_,VEC(i,j,k),q);
    double const betazdy=grace::fd_der<der_order,1>(state,BETAZ_,VEC(i,j,k),q);
    double const betazdz=grace::fd_der<der_order,2>(state,BETAZ_,VEC(i,j,k),q);

    // second derivatives of the shift vector components (with upper indices)
    // x-component
    double const betaxdxdx=grace::fd_der<der_order,0,0>(state,BETAX_,VEC(i,j,k),q);
    double const betaxdxdy=grace::fd_der<der_order,0,1>(state,BETAX_,VEC(i,j,k),q);
    double const betaxdxdz=grace::fd_der<der_order,0,2>(state,BETAX_,VEC(i,j,k),q);
    double const betaxdydy=grace::fd_der<der_order,1,1>(state,BETAX_,VEC(i,j,k),q);
    double const betaxdydz=grace::fd_der<der_order,1,2>(state,BETAX_,VEC(i,j,k),q);
    double const betaxdzdz=grace::fd_der<der_order,2,2>(state,BETAX_,VEC(i,j,k),q);
    // y-component
    double const betaydxdx=grace::fd_der<der_order,0,0>(state,BETAY_,VEC(i,j,k),q);
    double const betaydxdy=grace::fd_der<der_order,0,1>(state,BETAY_,VEC(i,j,k),q);
    double const betaydxdz=grace::fd_der<der_order,0,2>(state,BETAY_,VEC(i,j,k),q);
    double const betaydydy=grace::fd_der<der_order,1,1>(state,BETAY_,VEC(i,j,k),q);
    double const betaydydz=grace::fd_der<der_order,1,2>(state,BETAY_,VEC(i,j,k),q);
    double const betaydzdz=grace::fd_der<der_order,2,2>(state,BETAY_,VEC(i,j,k),q);
    // z-component
    double const betazdxdx=grace::fd_der<der_order,0,0>(state,BETAZ_,VEC(i,j,k),q);
    double const betazdxdy=grace::fd_der<der_order,0,1>(state,BETAZ_,VEC(i,j,k),q);
    double const betazdxdz=grace::fd_der<der_order,0,2>(state,BETAZ_,VEC(i,j,k),q);
    double const betazdydy=grace::fd_der<der_order,1,1>(state,BETAZ_,VEC(i,j,k),q);
    double const betazdydz=grace::fd_der<der_order,1,2>(state,BETAZ_,VEC(i,j,k),q);
    double const betazdzdz=grace::fd_der<der_order,2,2>(state,BETAZ_,VEC(i,j,k),q);

    // trace of the extrinsic curvature
    double const K=state(K_);

    // first derivatives of the extrinsic curvature trace 
    double const Kdx=grace::fd_der<der_order,0>(state,K_);
    double const Kdy=grace::fd_der<der_order,1>(state,K_);
    double const Kdz=grace::fd_der<der_order,2>(state,K_);

    // conformal (tilde) trace-free extrinsic curvature
    double const Atxx=state(VEC(i,j,k),ATXX_,q);
    double const Atxy=state(VEC(i,j,k),ATXY_,q);
    double const Atyy=state(VEC(i,j,k),ATYY_,q);
    double const Atxz=state(VEC(i,j,k),ATXZ_,q);
    double const Atyz=state(VEC(i,j,k),ATYZ_,q);
    double const Atzz=state(VEC(i,j,k),ATZZ_,q);

    // in order to reduce the number of terms in the BSSN equations it is usefull to 
    // to use implicit definitions of upper-lower and upper-upper index components
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
    double const Atxxdx=grace::fd_der<der_order,0>(state,ATXX_,VEC(i,j,k),q);
    double const Atxydx=grace::fd_der<der_order,0>(state,ATXY_,VEC(i,j,k),q);
    double const Atyydx=grace::fd_der<der_order,0>(state,ATYY_,VEC(i,j,k),q);
    double const Atxzdx=grace::fd_der<der_order,0>(state,ATXZ_,VEC(i,j,k),q);
    double const Atyzdx=grace::fd_der<der_order,0>(state,ATYZ_,VEC(i,j,k),q);
    double const Atzzdx=grace::fd_der<der_order,0>(state,ATZZ_,VEC(i,j,k),q);

    // first y-derivatives of the conformal (tilde) trace-free extrinsic curvature
    double const Atxxdy=grace::fd_der<der_order,1>(state,ATXX_,VEC(i,j,k),q);
    double const Atxydy=grace::fd_der<der_order,1>(state,ATXY_,VEC(i,j,k),q);
    double const Atyydy=grace::fd_der<der_order,1>(state,ATYY_,VEC(i,j,k),q);
    double const Atxzdy=grace::fd_der<der_order,1>(state,ATXZ_,VEC(i,j,k),q);
    double const Atyzdy=grace::fd_der<der_order,1>(state,ATYZ_,VEC(i,j,k),q);
    double const Atzzdy=grace::fd_der<der_order,1>(state,ATZZ_,VEC(i,j,k),q);

    // first z-derivatives of the conformal (tilde) trace-free extrinsic curvature
    double const Atxxdz=grace::fd_der<der_order,2>(state,ATXX_,VEC(i,j,k),q);
    double const Atxydz=grace::fd_der<der_order,2>(state,ATXY_,VEC(i,j,k),q);
    double const Atyydz=grace::fd_der<der_order,2>(state,ATYY_,VEC(i,j,k),q);
    double const Atxzdz=grace::fd_der<der_order,2>(state,ATXZ_,VEC(i,j,k),q);
    double const Atyzdz=grace::fd_der<der_order,2>(state,ATYZ_,VEC(i,j,k),q);
    double const Atzzdz=grace::fd_der<der_order,2>(state,ATZZ_,VEC(i,j,k),q);

    // first derivatives of the contracted conformal Christoffel symbol (check if GAMMA is actually GAMMAT)
    double const GammatXdx=grace::fd_der<der_order,0>(state,GAMMAX_,VEC(i,j,k),q);
    double const GammatXdy=grace::fd_der<der_order,1>(state,GAMMAX_,VEC(i,j,k),q);
    double const GammatXdz=grace::fd_der<der_order,2>(state,GAMMAX_,VEC(i,j,k),q);
    double const GammatYdx=grace::fd_der<der_order,0>(state,GAMMAY_,VEC(i,j,k),q);
    double const GammatYdy=grace::fd_der<der_order,1>(state,GAMMAY_,VEC(i,j,k),q);
    double const GammatYdz=grace::fd_der<der_order,2>(state,GAMMAY_,VEC(i,j,k),q);
    double const GammatZdx=grace::fd_der<der_order,0>(state,GAMMAZ_,VEC(i,j,k),q);
    double const GammatZdy=grace::fd_der<der_order,1>(state,GAMMAZ_,VEC(i,j,k),q);
    double const GammatZdz=grace::fd_der<der_order,2>(state,GAMMAZ_,VEC(i,j,k),q);

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

    // BSSN eq.1: time derivatives of the conformal metric components (Eq1Code in Mathematica notebook BSSN.nb)
    double const gtxxdt=-2*alp*Atxx - (2*(-2*betaxdx + betaydy + betazdz)*gtxx)/3. + betax*gtxxdx + betay*gtxxdy + betaz*gtxxdz + 2*betaydx*gtxy + 2*betazdx*gtxz;
    double const gtxydt=-2*alp*Atxy + betaxdy*gtxx + ((betaxdx + betaydy - 2*betazdz)*gtxy)/3. + betax*gtxydx + betay*gtxydy + betaz*gtxydz + betazdy*gtxz + betaydx*gtyy + betazdx*gtyz;
    double const gtxzdt=-2*alp*Atxz + betaxdz*gtxx + betaydz*gtxy + (betaxdx*gtxz)/3. - (2*betaydy*gtxz)/3. + (betazdz*gtxz)/3. + betax*gtxzdx + betay*gtxzdy + betaz*gtxzdz + betaydx*gtyz + betazdx*gtzz;
    double const gtyydt=-2*alp*Atyy + 2*betaxdy*gtxy - (2*(betaxdx - 2*betaydy + betazdz)*gtyy)/3. + betax*gtyydx + betay*gtyydy + betaz*gtyydz + 2*betazdy*gtyz;
    double const gtyzdt=-2*alp*Atyz + betaxdz*gtxy + betaxdy*gtxz + betaydz*gtyy - (2*betaxdx*gtyz)/3. + (betaydy*gtyz)/3. + (betazdz*gtyz)/3. + betax*gtyzdx + betay*gtyzdy + betaz*gtyzdz + betazdy*gtzz;
    double const gtzzdt=-2*alp*Atzz + 2*betaxdz*gtxz + 2*betaydz*gtyz + 2*betazdz*gtzz - (2*(betaxdx + betaydy + betazdz)*gtzz)/3. + betax*gtzzdx + betay*gtzzdy + betaz*gtzzdz;

    // BSSN eq.2: time derivatives of the conformal extrinsic curvature components (Eq2Code in Mathematica notebook BSSN.nb)
    double const Atxxdt=-2*alp*(Atxy*AtXy + Atxz*AtXz) + Atxxdx*betax + 2*Atxx*betaxdx + Atxxdy*betay + 2*Atxy*betaydx + Atxxdz*betaz + 2*Atxz*betazdx - (2*Atxx*(betaxdx + betaydy + betazdz))/3. + alp*Atxx*(-2*AtXx + K) - (phi*phi*(2*alpdxdx - alpdydy - alpdzdz - 2*alpdx*Gammat111 + alpdx*Gammat122 + alpdx*Gammat133 - 2*alpdy*Gammat211 + alpdy*Gammat222 + alpdy*Gammat233 - 2*alpdz*Gammat311 + alpdz*Gammat322 + alpdz*Gammat333 - 2*alp*Rxx + alp*Ryy + alp*Rzz + 16*alp*pi*Sxx - 8*alp*pi*Syy - 8*alp*pi*Szz))/3.;
    double const Atxydt=Atxydx*betax + Atxy*betaxdx + Atxx*betaxdy + Atxydy*betay + Atyy*betaydx + Atxy*betaydy + Atxydz*betaz + Atyz*betazdx + Atxz*betazdy - (2*Atxy*(betaxdx + betaydy + betazdz))/3. + alp*(-2*Atxx*AtXy - 2*Atxz*AtYz + Atxy*(-2*AtYy + K)) + phi*phi*(-alpdxdy + alpdx*Gammat112 + alpdy*Gammat212 + alpdz*Gammat312 + alp*Rxy - 8*alp*pi*Sxy);
    double const Atxzdt=Atxzdx*betax + Atxz*betaxdx + Atxx*betaxdz + Atxzdy*betay + Atyz*betaydx + Atxy*betaydz + Atxzdz*betaz + Atzz*betazdx + Atxz*betazdz - (2*Atxz*(betaxdx + betaydy + betazdz))/3. + alp*(-2*Atxx*AtXz - 2*Atxy*AtYz + Atxz*(-2*AtZz + K)) + phi*phi*(-alpdxdz + alpdx*Gammat113 + alpdy*Gammat213 + alpdz*Gammat313 + alp*Rxz - 8*alp*pi*Sxz);
    double const Atyydt=Atyydx*betax + 2*Atxy*betaxdy + Atyydy*betay + 2*Atyy*betaydy + Atyydz*betaz + 2*Atyz*betazdy - (2*Atyy*(betaxdx + betaydy + betazdz))/3. + alp*(-2*Atxy*AtXy - 2*Atyz*AtYz + Atyy*(-2*AtYy + K)) - (phi*phi*(-alpdxdx + 2*alpdydy - alpdzdz + alpdx*Gammat111 - 2*alpdx*Gammat122 + alpdx*Gammat133 + alpdy*Gammat211 - 2*alpdy*Gammat222 + alpdy*Gammat233 + alpdz*Gammat311 - 2*alpdz*Gammat322 + alpdz*Gammat333 + alp*(Rxx - 2*Ryy + Rzz - 8*pi*Sxx + 16*pi*Syy - 8*pi*Szz)))/3.;
    double const Atyzdt=Atyzdx*betax + Atxz*betaxdy + Atxy*betaxdz + Atyzdy*betay + Atyz*betaydy + Atyy*betaydz + Atyzdz*betaz + Atzz*betazdy + Atyz*betazdz - (2*Atyz*(betaxdx + betaydy + betazdz))/3. + alp*(-2*Atxy*AtXz - 2*Atyy*AtYz + Atyz*(-2*AtZz + K)) + phi*phi*(-alpdydz + alpdx*Gammat123 + alpdy*Gammat223 + alpdz*Gammat323 + alp*Ryz - 8*alp*pi*Syz);
    double const Atzzdt=Atzzdx*betax + 2*Atxz*betaxdz + Atzzdy*betay + 2*Atyz*betaydz + Atzzdz*betaz + 2*Atzz*betazdz - (2*Atzz*(betaxdx + betaydy + betazdz))/3. + alp*(-2*Atxz*AtXz - 2*Atyz*AtYz + Atzz*(-2*AtZz + K)) - (phi*phi*(-alpdxdx - alpdydy + 2*alpdzdz + alpdx*Gammat111 + alpdx*Gammat122 - 2*alpdx*Gammat133 + alpdy*Gammat211 + alpdy*Gammat222 - 2*alpdy*Gammat233 + alpdz*Gammat311 + alpdz*Gammat322 - 2*alpdz*Gammat333 + alp*(Rxx + Ryy - 2*(Rzz + 4*pi*(Sxx + Syy - 2*Szz)))))/3.;

    // BSSN eq.3: time derivative of the conformal factor (Eq3Code in Mathematica notebook BSSN.nb)
    double const phidt=-1/3.*((betaxdx + betaydy + betazdz)*phi) + (alp*K*phi)/3. + betax*phidx + betay*phidy + betaz*phidz;

    // BSSN eq.4: time derivative of the extrinsic curvature (Eq4Code in Mathematica notebook BSSN.nb)
    double const Kdt=betax*Kdx + betay*Kdy + betaz*Kdz - (-((-alpdxdx + alpdx*Gammat111 + alpdy*Gammat211 + alpdz*Gammat311)*gtXX) - 2*(-alpdxdy + alpdx*Gammat112 + alpdy*Gammat212 + alpdz*Gammat312)*gtXY - 2*(-alpdxdz + alpdx*Gammat113 + alpdy*Gammat213 + alpdz*Gammat313)*gtXZ - (-alpdydy + alpdx*Gammat122 + alpdy*Gammat222 + alpdz*Gammat322)*gtYY - 2*(-alpdydz + alpdx*Gammat123 + alpdy*Gammat223 + alpdz*Gammat323)*gtYZ - (-alpdzdz + alpdx*Gammat133 + alpdy*Gammat233 + alpdz*Gammat333)*gtZZ)*(phi*phi) + alp*(Atxx*AtXX + 2*Atxy*AtXY + 2*Atxz*AtXZ + Atyy*AtYY + 2*Atyz*AtYZ + Atzz*AtZZ + (K*K)/3. + 4*pi*(EE + S));
    
    // BSSN eq.5: time derivative of the extrinsic curvature (Eq4Code in Mathematica notebook BSSN.nb)
    double const GammatXdt=(-6*(alpdx*AtXX + alpdy*AtXY + alpdz*AtXZ) - 3*betaxdx*GammatX + 2*(betaxdx + betaydy + betazdz)*GammatX + 3*betax*GammatXdx + 3*betay*GammatXdy + 3*betaz*GammatXdz - 3*betaxdy*GammatY - 3*betaxdz*GammatZ + 4*betaxdxdx*gtXX + betaydxdy*gtXX + betazdxdz*gtXX + 7*betaxdxdy*gtXY + betaydydy*gtXY + betazdydz*gtXY + 7*betaxdxdz*gtXZ + betaydydz*gtXZ + betazdzdz*gtXZ + 3*betaxdydy*gtYY + 6*betaxdydz*gtYZ + 3*betaxdzdz*gtZZ + 6*alp*(AtXX*Gammat111 + 2*AtXY*Gammat112 + 2*AtXZ*Gammat113 + AtYY*Gammat122 + 2*AtYZ*Gammat123 + AtZZ*Gammat133 - (2*(gtXX*Kdx + gtXY*Kdy + gtXZ*Kdz))/3. - (3*(AtXX*phidx + AtXY*phidy + AtXZ*phidz))/phi) - 48*alp*pi*(gtXX*Sx + gtXY*Sy + gtXZ*Sz))/3.;
    double const GammatYdt=(-6*(alpdx*AtXY + alpdy*AtYY + alpdz*AtYZ) - 3*betaydx*GammatX - 3*betaydy*GammatY + 2*(betaxdx + betaydy + betazdz)*GammatY + 3*betax*GammatYdx + 3*betay*GammatYdy + 3*betaz*GammatYdz - 3*betaydz*GammatZ + 3*betaydxdx*gtXX + betaxdxdx*gtXY + 7*betaydxdy*gtXY + betazdxdz*gtXY + 6*betaydxdz*gtXZ + betaxdxdy*gtYY + 4*betaydydy*gtYY + betazdydz*gtYY + betaxdxdz*gtYZ + 7*betaydydz*gtYZ + betazdzdz*gtYZ + 3*betaydzdz*gtZZ + 6*alp*(AtXX*Gammat211 + 2*AtXY*Gammat212 + 2*AtXZ*Gammat213 + AtYY*Gammat222 + 2*AtYZ*Gammat223 + AtZZ*Gammat233 - (2*(gtXY*Kdx + gtYY*Kdy + gtYZ*Kdz))/3. - (3*(AtXY*phidx + AtYY*phidy + AtYZ*phidz))/phi) - 48*alp*pi*(gtXY*Sx + gtYY*Sy + gtYZ*Sz))/3.;
    double const GammatZdt=(-6*(alpdx*AtXZ + alpdy*AtYZ + alpdz*AtZZ) - 3*betazdx*GammatX - 3*betazdy*GammatY - 3*betazdz*GammatZ + 2*(betaxdx + betaydy + betazdz)*GammatZ + 3*betax*GammatZdx + 3*betay*GammatZdy + 3*betaz*GammatZdz + 3*betazdxdx*gtXX + 6*betazdxdy*gtXY + betaxdxdx*gtXZ + betaydxdy*gtXZ + 7*betazdxdz*gtXZ + 3*betazdydy*gtYY + betaxdxdy*gtYZ + betaydydy*gtYZ + 7*betazdydz*gtYZ + betaxdxdz*gtZZ + betaydydz*gtZZ + 4*betazdzdz*gtZZ + 6*alp*(AtXX*Gammat311 + 2*AtXY*Gammat312 + 2*AtXZ*Gammat313 + AtYY*Gammat322 + 2*AtYZ*Gammat323 + AtZZ*Gammat333 - (2*(gtXZ*Kdx + gtYZ*Kdy + gtZZ*Kdz))/3. - (3*(AtXZ*phidx + AtYZ*phidy + AtZZ*phidz))/phi) - 48*alp*pi*(gtXZ*Sx + gtYZ*Sy + gtZZ*Sz))/3.;

    /*
     * state --> contains all evolved variables 
     * e.g. if you need gtxx at i,j,k:
     * 
     *  gtxx := state(VEC(i,j,k), GTXX_, q) ; 
     * 
     * For computing derivatives:
     * 
     *  \partial_x gxx(i,j,k) := grace::fd_der<idir, der_order>(state, GTXX_, VEC(i,j,k), q) ; 
     * Where idir is 0, 1, 2 for X, Y, Z. 
     * 
     * double const betaxdy = grace::fd_der<1, der_order>(state,BETAX_, VEC(i,j,k), q) ; 
     * 
     * The BSSN vars indices are as follows:
     * 
     * GTXX_, GTXY_, GTXZ_, GTYY_, GTYZ_, GTZZ_,
     * PHI_, K_, BETAX_, BETAY_, BETAZ_, ALP_,
     * GAMMAX_,GAMMAY_,GAMMAZ_,
     * ATXX_, ATXY_, ATXZ_, ATYY_, ATYZ_, ATZZ_
     * 
     * Hydro quantities:
     * 
     * I will pass T_{\mu\nu} to this function:
     * 
     * Ttt = Tmunu[0][0]  ; // ALL INDICES LOW, INDICES 0,1,2,3
     * 
     */
}


}