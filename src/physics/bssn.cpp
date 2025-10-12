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

#include <grace/utils/fd_utils.hh>

#include <Kokkos_Core.hpp>

namespace grace {

template< size_t der_order >
bssn_state_t GRACE_HOST_DEVICE 
compute_bssn_rhs( VEC(int i, int j, int k), int q
                , grace::var_array_t const state
                , std::array<std::array<double,4>,4> const& Tmunu
                , std::array<double,GRACE_NSPACEDIM> const& idx
                , double const k1, double const eta )
{

    static constexpr const double pi = M_PI ; 


    // conformal (tilde) metric components
    double const gtxx = state(VEC(i,j,k),GTXX_+0,q);
    double const gtxy = state(VEC(i,j,k),GTXX_+1,q);
    double const gtxz = state(VEC(i,j,k),GTXX_+2,q);
    double const gtyy = state(VEC(i,j,k),GTXX_+3,q);
    double const gtyz = state(VEC(i,j,k),GTXX_+4,q);
    double const gtzz = state(VEC(i,j,k),GTXX_+5,q);

    // first derivatives of the conformal metric components
    double const gtxxdx = grace::fd_der<der_order,0>(state,GTXX_+0, VEC(i,j,k),q) * idx[0 ];
    double const gtxxdy = grace::fd_der<der_order,1>(state,GTXX_+0, VEC(i,j,k),q) * idx[1 ];
    double const gtxxdz = grace::fd_der<der_order,2>(state,GTXX_+0, VEC(i,j,k),q) * idx[2 ];
    double const gtxydx = grace::fd_der<der_order,0>(state,GTXX_+1, VEC(i,j,k),q) * idx[0 ];
    double const gtxydy = grace::fd_der<der_order,1>(state,GTXX_+1, VEC(i,j,k),q) * idx[1 ];
    double const gtxydz = grace::fd_der<der_order,2>(state,GTXX_+1, VEC(i,j,k),q) * idx[2 ];
    double const gtxzdx = grace::fd_der<der_order,0>(state,GTXX_+2, VEC(i,j,k),q) * idx[0 ];
    double const gtxzdy = grace::fd_der<der_order,1>(state,GTXX_+2, VEC(i,j,k),q) * idx[1 ];
    double const gtxzdz = grace::fd_der<der_order,2>(state,GTXX_+2, VEC(i,j,k),q) * idx[2 ];
    double const gtyydx = grace::fd_der<der_order,0>(state,GTXX_+3, VEC(i,j,k),q) * idx[0 ];
    double const gtyydy = grace::fd_der<der_order,1>(state,GTXX_+3, VEC(i,j,k),q) * idx[1 ];
    double const gtyydz = grace::fd_der<der_order,2>(state,GTXX_+3, VEC(i,j,k),q) * idx[2 ];
    double const gtyzdx = grace::fd_der<der_order,0>(state,GTXX_+4, VEC(i,j,k),q) * idx[0 ];
    double const gtyzdy = grace::fd_der<der_order,1>(state,GTXX_+4, VEC(i,j,k),q) * idx[1 ];
    double const gtyzdz = grace::fd_der<der_order,2>(state,GTXX_+4, VEC(i,j,k),q) * idx[2 ];
    double const gtzzdx = grace::fd_der<der_order,0>(state,GTXX_+5, VEC(i,j,k),q) * idx[0 ];
    double const gtzzdy = grace::fd_der<der_order,1>(state,GTXX_+5, VEC(i,j,k),q) * idx[1 ];
    double const gtzzdz = grace::fd_der<der_order,2>(state,GTXX_+5, VEC(i,j,k),q) * idx[2 ];

    // second derivatives of the conformal metric components
    double const gtxxdxdx = grace::fd_second_der<der_order,0>(state,GTXX_+0, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const gtxxdydy = grace::fd_second_der<der_order,1>(state,GTXX_+0, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const gtxxdzdz = grace::fd_second_der<der_order,2>(state,GTXX_+0, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);
    double const gtxydxdx = grace::fd_second_der<der_order,0>(state,GTXX_+1, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const gtxydydy = grace::fd_second_der<der_order,1>(state,GTXX_+1, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const gtxydzdz = grace::fd_second_der<der_order,2>(state,GTXX_+1, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);
    double const gtxzdxdx = grace::fd_second_der<der_order,0>(state,GTXX_+2, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const gtxzdydy = grace::fd_second_der<der_order,1>(state,GTXX_+2, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const gtxzdzdz = grace::fd_second_der<der_order,2>(state,GTXX_+2, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);
    double const gtyydxdx = grace::fd_second_der<der_order,0>(state,GTXX_+3, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const gtyydydy = grace::fd_second_der<der_order,1>(state,GTXX_+3, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const gtyydzdz = grace::fd_second_der<der_order,2>(state,GTXX_+3, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);
    double const gtyzdxdx = grace::fd_second_der<der_order,0>(state,GTXX_+4, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const gtyzdydy = grace::fd_second_der<der_order,1>(state,GTXX_+4, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const gtyzdzdz = grace::fd_second_der<der_order,2>(state,GTXX_+4, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);
    double const gtzzdxdx = grace::fd_second_der<der_order,0>(state,GTXX_+5, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const gtzzdydy = grace::fd_second_der<der_order,1>(state,GTXX_+5, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const gtzzdzdz = grace::fd_second_der<der_order,2>(state,GTXX_+5, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);

    // mixed second derivatives of the conformal metric components
    double const gtxxdxdy = grace::fd_der<der_order,0,1>(state,GTXX_+0, VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const gtxxdxdz = grace::fd_der<der_order,0,2>(state,GTXX_+0, VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const gtxxdydz = grace::fd_der<der_order,1,2>(state,GTXX_+0, VEC(i,j,k),q) * idx[1 ] * idx[2 ];
    double const gtxydxdy = grace::fd_der<der_order,0,1>(state,GTXX_+1, VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const gtxydxdz = grace::fd_der<der_order,0,2>(state,GTXX_+1, VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const gtxydydz = grace::fd_der<der_order,1,2>(state,GTXX_+1, VEC(i,j,k),q) * idx[1 ] * idx[2 ];
    double const gtxzdxdy = grace::fd_der<der_order,0,1>(state,GTXX_+2, VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const gtxzdxdz = grace::fd_der<der_order,0,2>(state,GTXX_+2, VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const gtxzdydz = grace::fd_der<der_order,1,2>(state,GTXX_+2, VEC(i,j,k),q) * idx[1 ] * idx[2 ];
    double const gtyydxdy = grace::fd_der<der_order,0,1>(state,GTXX_+3, VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const gtyydxdz = grace::fd_der<der_order,0,2>(state,GTXX_+3, VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const gtyydydz = grace::fd_der<der_order,1,2>(state,GTXX_+3, VEC(i,j,k),q) * idx[1 ] * idx[2 ];
    double const gtyzdxdy = grace::fd_der<der_order,0,1>(state,GTXX_+4, VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const gtyzdxdz = grace::fd_der<der_order,0,2>(state,GTXX_+4, VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const gtyzdydz = grace::fd_der<der_order,1,2>(state,GTXX_+4, VEC(i,j,k),q) * idx[1 ] * idx[2 ];
    double const gtzzdxdy = grace::fd_der<der_order,0,1>(state,GTXX_+5, VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const gtzzdxdz = grace::fd_der<der_order,0,2>(state,GTXX_+5, VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const gtzzdydz = grace::fd_der<der_order,1,2>(state,GTXX_+5, VEC(i,j,k),q) * idx[1 ] * idx[2 ];

    // inverse conformal metric components, assuming unit metric determinant
    double const gtXX=-(gtyz*gtyz) + gtyy*gtzz;
    double const gtXY=gtxz*gtyz - gtxy*gtzz;
    double const gtXZ=-(gtxz*gtyy) + gtxy*gtyz;
    double const gtYY=-(gtxz*gtxz) + gtxx*gtzz;
    double const gtYZ=gtxy*gtxz - gtxx*gtyz;
    double const gtZZ=-(gtxy*gtxy) + gtxx*gtyy;

    // first derivatives of inverse conformal metric components
    double const gtXXdx =-2*gtyz*gtyzdx + gtyydx*gtzz + gtyy*gtzzdx;
    double const gtXXdy =-2*gtyz*gtyzdy + gtyydy*gtzz + gtyy*gtzzdy;
    double const gtXXdz =-2*gtyz*gtyzdz + gtyydz*gtzz + gtyy*gtzzdz;
    double const gtXYdx =gtxzdx*gtyz + gtxz*gtyzdx - gtxydx*gtzz - gtxy*gtzzdx;
    double const gtXYdy =gtxzdy*gtyz + gtxz*gtyzdy - gtxydy*gtzz - gtxy*gtzzdy;
    double const gtXYdz =gtxzdz*gtyz + gtxz*gtyzdz - gtxydz*gtzz - gtxy*gtzzdz;
    double const gtXZdx =-(gtxzdx*gtyy) - gtxz*gtyydx + gtxydx*gtyz + gtxy*gtyzdx;
    double const gtXZdy =-(gtxzdy*gtyy) - gtxz*gtyydy + gtxydy*gtyz + gtxy*gtyzdy;
    double const gtXZdz =-(gtxzdz*gtyy) - gtxz*gtyydz + gtxydz*gtyz + gtxy*gtyzdz;
    double const gtYYdx =-2*gtxz*gtxzdx + gtxxdx*gtzz + gtxx*gtzzdx;
    double const gtYYdy =-2*gtxz*gtxzdy + gtxxdy*gtzz + gtxx*gtzzdy;
    double const gtYYdz =-2*gtxz*gtxzdz + gtxxdz*gtzz + gtxx*gtzzdz;
    double const gtYZdx =gtxydx*gtxz + gtxy*gtxzdx - gtxxdx*gtyz - gtxx*gtyzdx;
    double const gtYZdy =gtxydy*gtxz + gtxy*gtxzdy - gtxxdy*gtyz - gtxx*gtyzdy;
    double const gtYZdz =gtxydz*gtxz + gtxy*gtxzdz - gtxxdz*gtyz - gtxx*gtyzdz;
    double const gtZZdx =-2*gtxy*gtxydx + gtxxdx*gtyy + gtxx*gtyydx;
    double const gtZZdy =-2*gtxy*gtxydy + gtxxdy*gtyy + gtxx*gtyydy;
    double const gtZZdz =-2*gtxy*gtxydz + gtxxdz*gtyy + gtxx*gtyydz;

    // conformal factor, assuming gtdd=Exp[-4*phi]*gdd, i.e., psi=Exp[phi] in EG notation
    double const phi = state(VEC(i,j,k),PHI_,q);

    // first derivatives of the conformal factor
    double const phidx = grace::fd_der<der_order,0>(state,PHI_,VEC(i,j,k),q) * idx[0 ];
    double const phidy = grace::fd_der<der_order,1>(state,PHI_,VEC(i,j,k),q) * idx[1 ];
    double const phidz = grace::fd_der<der_order,2>(state,PHI_,VEC(i,j,k),q) * idx[2 ];

    // second derivatives of the conformal factor
    double const phidxdx = grace::fd_second_der<der_order,0>(state,PHI_,VEC(i,j,k),q) * math::int_pow<2>(idx[0 ]);
    double const phidydy = grace::fd_second_der<der_order,1>(state,PHI_,VEC(i,j,k),q) * math::int_pow<2>(idx[1 ]);
    double const phidzdz = grace::fd_second_der<der_order,2>(state,PHI_,VEC(i,j,k),q) * math::int_pow<2>(idx[2 ]);

    // mixed second derivatives of the conformal factor
    double const phidxdy = grace::fd_der<der_order,0,1>(state,PHI_,VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const phidxdz = grace::fd_der<der_order,0,2>(state,PHI_,VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const phidydz = grace::fd_der<der_order,1,2>(state,PHI_,VEC(i,j,k),q) * idx[1 ] * idx[2 ];

    // lapse function
    double const alp = state(VEC(i,j,k),ALP_,q);

    // first derivatives of the lapse function
    double const alpdx = grace::fd_der<der_order,0>(state,ALP_,VEC(i,j,k),q) * idx[0 ];
    double const alpdy = grace::fd_der<der_order,1>(state,ALP_,VEC(i,j,k),q) * idx[1 ];
    double const alpdz = grace::fd_der<der_order,2>(state,ALP_,VEC(i,j,k),q) * idx[2 ];

    // second derivatives of the lapse function
    double const alpdxdx = grace::fd_second_der<der_order,0>(state,ALP_,VEC(i,j,k),q) * math::int_pow<2>(idx[0 ]);
    double const alpdydy = grace::fd_second_der<der_order,1>(state,ALP_,VEC(i,j,k),q) * math::int_pow<2>(idx[1 ]);
    double const alpdzdz = grace::fd_second_der<der_order,2>(state,ALP_,VEC(i,j,k),q) * math::int_pow<2>(idx[2 ]);

    // second mixed derivatives of the lapse function
    double const alpdxdy = grace::fd_der<der_order,0,1>(state,ALP_,VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const alpdxdz = grace::fd_der<der_order,0,2>(state,ALP_,VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const alpdydz = grace::fd_der<der_order,1,2>(state,ALP_,VEC(i,j,k),q) * idx[1 ] * idx[2 ];

    // shift vector components (with upper indices)
    double const betaX = state(VEC(i,j,k),BETAX_+0,q);
    double const betaY = state(VEC(i,j,k),BETAX_+1,q);
    double const betaZ = state(VEC(i,j,k),BETAX_+2,q);

    // first derivatives of the shift vector components
    double const betaXdx = grace::fd_der<der_order,0>(state,BETAX_+0, VEC(i,j,k),q)* idx[0 ];
    double const betaXdy = grace::fd_der<der_order,1>(state,BETAX_+0, VEC(i,j,k),q)* idx[1 ];
    double const betaXdz = grace::fd_der<der_order,2>(state,BETAX_+0, VEC(i,j,k),q)* idx[2 ];
    double const betaYdx = grace::fd_der<der_order,0>(state,BETAX_+1, VEC(i,j,k),q)* idx[0 ];
    double const betaYdy = grace::fd_der<der_order,1>(state,BETAX_+1, VEC(i,j,k),q)* idx[1 ];
    double const betaYdz = grace::fd_der<der_order,2>(state,BETAX_+1, VEC(i,j,k),q)* idx[2 ];
    double const betaZdx = grace::fd_der<der_order,0>(state,BETAX_+2, VEC(i,j,k),q)* idx[0 ];
    double const betaZdy = grace::fd_der<der_order,1>(state,BETAX_+2, VEC(i,j,k),q)* idx[1 ];
    double const betaZdz = grace::fd_der<der_order,2>(state,BETAX_+2, VEC(i,j,k),q)* idx[2 ];

    // second derivatives of the shift vector components
    double const betaXdxdx = grace::fd_second_der<der_order,0>(state,BETAX_+0, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const betaXdydy = grace::fd_second_der<der_order,1>(state,BETAX_+0, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const betaXdzdz = grace::fd_second_der<der_order,2>(state,BETAX_+0, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);
    double const betaYdxdx = grace::fd_second_der<der_order,0>(state,BETAX_+1, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const betaYdydy = grace::fd_second_der<der_order,1>(state,BETAX_+1, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const betaYdzdz = grace::fd_second_der<der_order,2>(state,BETAX_+1, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);
    double const betaZdxdx = grace::fd_second_der<der_order,0>(state,BETAX_+2, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const betaZdydy = grace::fd_second_der<der_order,1>(state,BETAX_+2, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const betaZdzdz = grace::fd_second_der<der_order,2>(state,BETAX_+2, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);

    // second mixed derivatives of the shift vector components
    double const betaXdxdy = grace::fd_der<der_order,0,1>(state,BETAX_+0, VEC(i,j,k),q)* idx[0 ] * idx[1 ];
    double const betaXdxdz = grace::fd_der<der_order,0,2>(state,BETAX_+0, VEC(i,j,k),q)* idx[0 ] * idx[2 ];
    double const betaXdydz = grace::fd_der<der_order,1,2>(state,BETAX_+0, VEC(i,j,k),q)* idx[1 ] * idx[2 ];
    double const betaYdxdy = grace::fd_der<der_order,0,1>(state,BETAX_+1, VEC(i,j,k),q)* idx[0 ] * idx[1 ];
    double const betaYdxdz = grace::fd_der<der_order,0,2>(state,BETAX_+1, VEC(i,j,k),q)* idx[0 ] * idx[2 ];
    double const betaYdydz = grace::fd_der<der_order,1,2>(state,BETAX_+1, VEC(i,j,k),q)* idx[1 ] * idx[2 ];
    double const betaZdxdy = grace::fd_der<der_order,0,1>(state,BETAX_+2, VEC(i,j,k),q)* idx[0 ] * idx[1 ];
    double const betaZdxdz = grace::fd_der<der_order,0,2>(state,BETAX_+2, VEC(i,j,k),q)* idx[0 ] * idx[2 ];
    double const betaZdydz = grace::fd_der<der_order,1,2>(state,BETAX_+2, VEC(i,j,k),q)* idx[1 ] * idx[2 ];

    // components of the energy momentum tensor (with lower indices)
    double const Ttt = Tmunu[0][0];
    double const Ttx = Tmunu[0][1];
    double const Tty = Tmunu[0][2];
    double const Ttz = Tmunu[0][3];
    double const Txx = Tmunu[1][1];
    double const Txy = Tmunu[1][2];
    double const Txz = Tmunu[1][3];
    double const Tyy = Tmunu[2][2];
    double const Tyz = Tmunu[2][3];
    double const Tzz = Tmunu[3][3];

    // spatial compoents of the energy momentum tensor
    double const Sxx = Txx;
    double const Sxy = Txy;
    double const Sxz = Txz;
    double const Syy = Tyy;
    double const Syz = Tyz;
    double const Szz = Tzz;

    // momentum density components
    double const Sx = (-Ttx + betaX*Txx + betaY*Txy + betaZ*Txz)/alp;
    double const Sy = (-Tty + betaX*Txy + betaY*Tyy + betaZ*Tyz)/alp;
    double const Sz = (-Ttz + betaX*Txz + betaY*Tyz + betaZ*Tzz)/alp;

    // trace of the spatial energy momentum tensor
    double const S = (gtXX*Sxx + 2*gtXY*Sxy + 2*gtXZ*Sxz + gtYY*Syy + 2*gtYZ*Syz + gtZZ*Szz)*exp(-4.*phi);

    // energy density
    double const EE = (Ttt - 2*betaY*Tty - 2*betaZ*Ttz + betaX*betaX*Txx + 2*betaX*(-Ttx + betaY*Txy + betaZ*Txz) + betaY*betaY*Tyy + 2*betaY*betaZ*Tyz + betaZ*betaZ*Tzz)/(alp*alp);

    // trace of the extrinsic curvature
    double const K = state(VEC(i,j,k),K_,q);

    // first derivatives of the extrinsic curvature trace
    double const Kdx = grace::fd_der<der_order,0>(state,K_,VEC(i,j,k),q) * idx[0 ];
    double const Kdy = grace::fd_der<der_order,1>(state,K_,VEC(i,j,k),q) * idx[1 ];
    double const Kdz = grace::fd_der<der_order,2>(state,K_,VEC(i,j,k),q) * idx[2 ];

    // conformal trace-free extrinsic curvature
    double const Atxx = state(VEC(i,j,k),ATXX_+0,q);
    double const Atxy = state(VEC(i,j,k),ATXX_+1,q);
    double const Atxz = state(VEC(i,j,k),ATXX_+2,q);
    double const Atyy = state(VEC(i,j,k),ATXX_+3,q);
    double const Atyz = state(VEC(i,j,k),ATXX_+4,q);
    double const Atzz = state(VEC(i,j,k),ATXX_+5,q);

    // conformal trace-free extrinsic curvature, first index raised with conformal metric
    double const AtXx=Atxx*gtXX + Atxy*gtXY + Atxz*gtXZ;
    double const AtXy=Atxy*gtXX + Atyy*gtXY + Atyz*gtXZ;
    double const AtXz=Atxz*gtXX + Atyz*gtXY + Atzz*gtXZ;
    double const AtYy=Atxy*gtXY + Atyy*gtYY + Atyz*gtYZ;
    double const AtYz=Atxz*gtXY + Atyz*gtYY + Atzz*gtYZ;
    double const AtZz=Atxz*gtXZ + Atyz*gtYZ + Atzz*gtZZ;

    // conformal trace-free extrinsic curvature, both indices raised with conformal metric
    double const AtXX=AtXx*gtXX + AtXy*gtXY + AtXz*gtXZ;
    double const AtXY=AtXx*gtXY + AtXy*gtYY + AtXz*gtYZ;
    double const AtXZ=AtXx*gtXZ + AtXy*gtYZ + AtXz*gtZZ;
    double const AtYY=AtXy*gtXY + AtYy*gtYY + AtYz*gtYZ;
    double const AtYZ=AtXy*gtXZ + AtYy*gtYZ + AtYz*gtZZ;
    double const AtZZ=AtXz*gtXZ + AtYz*gtYZ + AtZz*gtZZ;

    // derivatives of the conformal (tilde) trace-free extrinsic curvature, can be replaced by momentum constraint
    double const Atxxdx = grace::fd_der<der_order,0>(state,ATXX_+0, VEC(i,j,k),q)* idx[0 ];
    double const Atxxdy = grace::fd_der<der_order,1>(state,ATXX_+0, VEC(i,j,k),q)* idx[1 ];
    double const Atxxdz = grace::fd_der<der_order,2>(state,ATXX_+0, VEC(i,j,k),q)* idx[2 ];
    double const Atxydx = grace::fd_der<der_order,0>(state,ATXX_+1, VEC(i,j,k),q)* idx[0 ];
    double const Atxydy = grace::fd_der<der_order,1>(state,ATXX_+1, VEC(i,j,k),q)* idx[1 ];
    double const Atxydz = grace::fd_der<der_order,2>(state,ATXX_+1, VEC(i,j,k),q)* idx[2 ];
    double const Atxzdx = grace::fd_der<der_order,0>(state,ATXX_+2, VEC(i,j,k),q)* idx[0 ];
    double const Atxzdy = grace::fd_der<der_order,1>(state,ATXX_+2, VEC(i,j,k),q)* idx[1 ];
    double const Atxzdz = grace::fd_der<der_order,2>(state,ATXX_+2, VEC(i,j,k),q)* idx[2 ];
    double const Atyydx = grace::fd_der<der_order,0>(state,ATXX_+3, VEC(i,j,k),q)* idx[0 ];
    double const Atyydy = grace::fd_der<der_order,1>(state,ATXX_+3, VEC(i,j,k),q)* idx[1 ];
    double const Atyydz = grace::fd_der<der_order,2>(state,ATXX_+3, VEC(i,j,k),q)* idx[2 ];
    double const Atyzdx = grace::fd_der<der_order,0>(state,ATXX_+4, VEC(i,j,k),q)* idx[0 ];
    double const Atyzdy = grace::fd_der<der_order,1>(state,ATXX_+4, VEC(i,j,k),q)* idx[1 ];
    double const Atyzdz = grace::fd_der<der_order,2>(state,ATXX_+4, VEC(i,j,k),q)* idx[2 ];
    double const Atzzdx = grace::fd_der<der_order,0>(state,ATXX_+5, VEC(i,j,k),q)* idx[0 ];
    double const Atzzdy = grace::fd_der<der_order,1>(state,ATXX_+5, VEC(i,j,k),q)* idx[1 ];
    double const Atzzdz = grace::fd_der<der_order,2>(state,ATXX_+5, VEC(i,j,k),q)* idx[2 ];

    // contracted conformal Christoffel symbols
    double const GammatX = state(VEC(i,j,k),GAMMAX_+0,q);
    double const GammatY = state(VEC(i,j,k),GAMMAX_+1,q);
    double const GammatZ = state(VEC(i,j,k),GAMMAX_+2,q);

    // derivatives of the contracted conformal Christoffel symbol
    double const GammatXdx = grace::fd_der<der_order,0>(state,GAMMAX_+0, VEC(i,j,k),q)* idx[0 ];
    double const GammatXdy = grace::fd_der<der_order,1>(state,GAMMAX_+0, VEC(i,j,k),q)* idx[1 ];
    double const GammatXdz = grace::fd_der<der_order,2>(state,GAMMAX_+0, VEC(i,j,k),q)* idx[2 ];
    double const GammatYdx = grace::fd_der<der_order,0>(state,GAMMAX_+1, VEC(i,j,k),q)* idx[0 ];
    double const GammatYdy = grace::fd_der<der_order,1>(state,GAMMAX_+1, VEC(i,j,k),q)* idx[1 ];
    double const GammatYdz = grace::fd_der<der_order,2>(state,GAMMAX_+1, VEC(i,j,k),q)* idx[2 ];
    double const GammatZdx = grace::fd_der<der_order,0>(state,GAMMAX_+2, VEC(i,j,k),q)* idx[0 ];
    double const GammatZdy = grace::fd_der<der_order,1>(state,GAMMAX_+2, VEC(i,j,k),q)* idx[1 ];
    double const GammatZdz = grace::fd_der<der_order,2>(state,GAMMAX_+2, VEC(i,j,k),q)* idx[2 ];

    // components of conformal Christoffel symbols
    double const GammatXxx=(gtXX*gtxxdx - gtxxdy*gtXY + 2*gtXY*gtxydx - gtxxdz*gtXZ + 2*gtXZ*gtxzdx)/2.;
    double const GammatXxy=(gtXX*(gtxxdx + gtxxdy - gtxydx) + gtXY*gtxydx + gtXZ*(-gtxydz + gtxzdx + gtxzdy))/2.;
    double const GammatXxz=(gtXX*(gtxxdx + gtxxdz - gtxzdx) + gtXZ*gtxzdx + gtXY*(gtxydx + gtxydz - gtxzdy))/2.;
    double const GammatXyy=(2*gtXX*gtxydy - gtXX*gtyydx + gtXY*gtyydy - gtXZ*gtyydz + 2*gtXZ*gtyzdy)/2.;
    double const GammatXyz=(gtXX*(gtxydy + gtxydz - gtyzdx) + gtXY*(gtyydy + gtyydz - gtyzdy) + gtXZ*gtyzdy)/2.;
    double const GammatXzz=(2*gtXX*gtxzdz + 2*gtXY*gtyzdz - gtXX*gtzzdx - gtXY*gtzzdy + gtXZ*gtzzdz)/2.;
    double const GammatYxx=(gtxxdx*gtXY - gtxxdy*gtYY + 2*gtxydx*gtYY - gtxxdz*gtYZ + 2*gtxzdx*gtYZ)/2.;
    double const GammatYxy=((gtxxdx + gtxxdy)*gtXY + gtxydx*(-gtXY + gtYY) + (-gtxydz + gtxzdx + gtxzdy)*gtYZ)/2.;
    double const GammatYxz=((gtxxdx + gtxxdz)*gtXY + (gtxydx + gtxydz - gtxzdy)*gtYY + gtxzdx*(-gtXY + gtYZ))/2.;
    double const GammatYyy=(2*gtXY*gtxydy - gtXY*gtyydx + gtYY*gtyydy - gtyydz*gtYZ + 2*gtYZ*gtyzdy)/2.;
    double const GammatYyz=(gtXY*(gtxydy + gtxydz - gtyzdx) + gtYY*(gtyydy + gtyydz - gtyzdy) + gtYZ*gtyzdy)/2.;
    double const GammatYzz=(2*gtXY*gtxzdz + 2*gtYY*gtyzdz - gtXY*gtzzdx - gtYY*gtzzdy + gtYZ*gtzzdz)/2.;
    double const GammatZxx=(gtxxdx*gtXZ - gtxxdy*gtYZ + 2*gtxydx*gtYZ - gtxxdz*gtZZ + 2*gtxzdx*gtZZ)/2.;
    double const GammatZxy=((gtxxdx + gtxxdy)*gtXZ + gtxydx*(-gtXZ + gtYZ) + (-gtxydz + gtxzdx + gtxzdy)*gtZZ)/2.;
    double const GammatZxz=((gtxxdx + gtxxdz)*gtXZ + (gtxydx + gtxydz - gtxzdy)*gtYZ + gtxzdx*(-gtXZ + gtZZ))/2.;
    double const GammatZyy=(2*gtxydy*gtXZ - gtXZ*gtyydx + gtyydy*gtYZ - gtyydz*gtZZ + 2*gtyzdy*gtZZ)/2.;
    double const GammatZyz=((gtyydy + gtyydz)*gtYZ + gtXZ*(gtxydy + gtxydz - gtyzdx) + gtyzdy*(-gtYZ + gtZZ))/2.;
    double const GammatZzz=(2*gtXZ*gtxzdz + 2*gtYZ*gtyzdz - gtXZ*gtzzdx - gtYZ*gtzzdy + gtZZ*gtzzdz)/2.;

    // second covariant derivative of lapse
    double const DiDjalpxx=alpdxdx - alpdy*GammatYxx - alpdz*GammatZxx + 2*alpdy*gtxx*gtXY*phidx + 2*alpdz*gtxx*gtXZ*phidx + 2*alpdy*gtxx*gtYY*phidy + 2*alpdz*gtxx*gtYZ*phidy + 2*alpdy*gtxx*gtYZ*phidz + 2*alpdz*gtxx*gtZZ*phidz + alpdx*(-GammatXxx - 4*phidx + 2*gtxx*gtXX*phidx + 2*gtxx*gtXY*phidy + 2*gtxx*gtXZ*phidz);
    double const DiDjalpxy=alpdxdy - alpdy*GammatYxy - alpdz*GammatZxy - 2*alpdy*phidx + 2*alpdy*gtxy*gtXY*phidx + 2*alpdz*gtxy*gtXZ*phidx + 2*alpdy*gtxy*gtYY*phidy + 2*alpdz*gtxy*gtYZ*phidy + 2*alpdy*gtxy*gtYZ*phidz + 2*alpdz*gtxy*gtZZ*phidz + alpdx*(-GammatXxy + 2*gtXX*gtxy*phidx - 2*phidy + 2*gtxy*gtXY*phidy + 2*gtxy*gtXZ*phidz);
    double const DiDjalpxz=alpdxdz - alpdy*GammatYxz - alpdz*GammatZxz - 2*alpdz*phidx + 2*alpdy*gtXY*gtxz*phidx + 2*alpdz*gtxz*gtXZ*phidx + 2*alpdy*gtxz*gtYY*phidy + 2*alpdz*gtxz*gtYZ*phidy + 2*alpdy*gtxz*gtYZ*phidz + 2*alpdz*gtxz*gtZZ*phidz + alpdx*(-GammatXxz + 2*gtXX*gtxz*phidx + 2*gtXY*gtxz*phidy - 2*phidz + 2*gtxz*gtXZ*phidz);
    double const DiDjalpyy=alpdydy - alpdx*GammatXyy - alpdy*GammatYyy - alpdz*GammatZyy + 2*alpdy*gtXY*gtyy*phidx + 2*alpdz*gtXZ*gtyy*phidx - 4*alpdy*phidy + 2*alpdy*gtyy*gtYY*phidy + 2*alpdz*gtyy*gtYZ*phidy + 2*alpdy*gtyy*gtYZ*phidz + 2*alpdz*gtyy*gtZZ*phidz + 2*alpdx*gtyy*(gtXX*phidx + gtXY*phidy + gtXZ*phidz);
    double const DiDjalpyz=alpdydz - alpdx*GammatXyz - alpdy*GammatYyz - alpdz*GammatZyz + 2*alpdy*gtXY*gtyz*phidx + 2*alpdz*gtXZ*gtyz*phidx - 2*alpdz*phidy + 2*alpdy*gtYY*gtyz*phidy + 2*alpdz*gtyz*gtYZ*phidy - 2*alpdy*phidz + 2*alpdy*gtyz*gtYZ*phidz + 2*alpdz*gtyz*gtZZ*phidz + 2*alpdx*gtyz*(gtXX*phidx + gtXY*phidy + gtXZ*phidz);
    double const DiDjalpzz=alpdzdz - alpdx*GammatXzz - alpdy*GammatYzz - alpdz*GammatZzz + 2*alpdy*gtXY*gtzz*phidx + 2*alpdz*gtXZ*gtzz*phidx + 2*alpdy*gtYY*gtzz*phidy + 2*alpdz*gtYZ*gtzz*phidy - 4*alpdz*phidz + 2*alpdy*gtYZ*gtzz*phidz + 2*alpdz*gtzz*gtZZ*phidz + 2*alpdx*gtzz*(gtXX*phidx + gtXY*phidy + gtXZ*phidz);

    // contracted second covariant derivative of lapse
    double const DDalp=alpdxdx*gtXX - alpdy*GammatYxx*gtXX - alpdz*GammatZxx*gtXX + 2*alpdxdy*gtXY - 2*alpdy*GammatYxy*gtXY - 2*alpdz*GammatZxy*gtXY + 2*alpdxdz*gtXZ - 2*alpdy*GammatYxz*gtXZ - 2*alpdz*GammatZxz*gtXZ + alpdydy*gtYY - alpdy*GammatYyy*gtYY - alpdz*GammatZyy*gtYY + 2*alpdydz*gtYZ - 2*alpdy*GammatYyz*gtYZ - 2*alpdz*GammatZyz*gtYZ + alpdzdz*gtZZ - alpdy*GammatYzz*gtZZ - alpdz*GammatZzz*gtZZ + 2*alpdy*gtXY*phidx + 2*alpdz*gtXZ*phidx + 2*alpdy*gtYY*phidy + 2*alpdz*gtYZ*phidy + 2*alpdy*gtYZ*phidz + 2*alpdz*gtZZ*phidz - alpdx*(GammatXxx*gtXX + 2*GammatXxy*gtXY + 2*GammatXxz*gtXZ + GammatXyy*gtYY + 2*GammatXyz*gtYZ + GammatXzz*gtZZ - 2*gtXX*phidx - 2*gtXY*phidy - 2*gtXZ*phidz);

    // components of conformal Ricci tensor
    double const Rt0xx=GammatXdx*gtxx - (gtXX*gtxxdxdx)/2. + GammatYdx*gtxy - gtxxdxdy*gtXY + GammatZdx*gtxz - gtxxdxdz*gtXZ - (gtxxdydy*gtYY)/2. - gtxxdydz*gtYZ - (gtxxdzdz*gtZZ)/2.;
    double const Rt0xy=(GammatXdy*gtxx + GammatXdx*gtxy + GammatYdy*gtxy - gtXX*gtxydxdx - 2*gtXY*gtxydxdy + GammatZdy*gtxz - 2*gtxydxdz*gtXZ + GammatYdx*gtyy - gtxydydy*gtYY + GammatZdx*gtyz - 2*gtxydydz*gtYZ - gtxydzdz*gtZZ)/2.;
    double const Rt0xz=(GammatXdz*gtxx + GammatYdz*gtxy + GammatXdx*gtxz + GammatZdz*gtxz - gtXX*gtxzdxdx - 2*gtXY*gtxzdxdy - 2*gtXZ*gtxzdxdz - gtxzdydy*gtYY + GammatYdx*gtyz - 2*gtxzdydz*gtYZ + GammatZdx*gtzz - gtxzdzdz*gtZZ)/2.;
    double const Rt0yy=GammatXdy*gtxy + GammatYdy*gtyy - (gtXX*gtyydxdx)/2. - gtXY*gtyydxdy - gtXZ*gtyydxdz - (gtYY*gtyydydy)/2. + GammatZdy*gtyz - gtyydydz*gtYZ - (gtyydzdz*gtZZ)/2.;
    double const Rt0yz=(GammatXdz*gtxy + GammatXdy*gtxz + GammatYdz*gtyy + GammatYdy*gtyz + GammatZdz*gtyz - gtXX*gtyzdxdx - 2*gtXY*gtyzdxdy - 2*gtXZ*gtyzdxdz - gtYY*gtyzdydy - 2*gtYZ*gtyzdydz + GammatZdy*gtzz - gtyzdzdz*gtZZ)/2.;
    double const Rt0zz=GammatXdz*gtxz + GammatYdz*gtyz + GammatZdz*gtzz - (gtXX*gtzzdxdx)/2. - gtXY*gtzzdxdy - gtXZ*gtzzdxdz - (gtYY*gtzzdydy)/2. - gtYZ*gtzzdydz - (gtZZ*gtzzdzdz)/2.;

    // components of the phi contribution to the relation between the Ricci tensor and the conformal Ricci tensor: Rdd=Rt0dd+Rtphidd
    double const Rtphixx=(-2*(GammatXxx*(-3 + gtxx*gtXX)*phidx + 2*GammatXxy*gtxx*gtXY*phidx + 2*GammatXxz*gtxx*gtXZ*phidx + GammatXyy*gtxx*gtYY*phidx + 2*GammatXyz*gtxx*gtYZ*phidx + GammatXzz*gtxx*gtZZ*phidx - 6*(phidx*phidx) + 2*gtxx*gtXX*(phidx*phidx) + 3*phidxdx - gtxx*gtXX*phidxdx - 2*gtxx*gtXY*phidxdy - 2*gtxx*gtXZ*phidxdz - 3*GammatYxx*phidy + GammatYxx*gtxx*gtXX*phidy + 2*GammatYxy*gtxx*gtXY*phidy + 2*GammatYxz*gtxx*gtXZ*phidy + GammatYyy*gtxx*gtYY*phidy + 2*GammatYyz*gtxx*gtYZ*phidy + GammatYzz*gtxx*gtZZ*phidy + 4*gtxx*gtXY*phidx*phidy + 2*gtxx*gtYY*(phidy*phidy) - gtxx*gtYY*phidydy - 2*gtxx*gtYZ*phidydz - 3*GammatZxx*phidz + GammatZxx*gtxx*gtXX*phidz + 2*GammatZxy*gtxx*gtXY*phidz + 2*GammatZxz*gtxx*gtXZ*phidz + GammatZyy*gtxx*gtYY*phidz + 2*GammatZyz*gtxx*gtYZ*phidz + GammatZzz*gtxx*gtZZ*phidz + 4*gtxx*gtXZ*phidx*phidz + 4*gtxx*gtYZ*phidy*phidz + 2*gtxx*gtZZ*(phidz*phidz) - gtxx*gtZZ*phidzdz))/3.;
    double const Rtphixy=(-2*(GammatXxx*gtXX*gtxy*phidx + GammatXxy*(-3 + 2*gtxy*gtXY)*phidx + 2*GammatXxz*gtxy*gtXZ*phidx + GammatXyy*gtxy*gtYY*phidx + 2*GammatXyz*gtxy*gtYZ*phidx + GammatXzz*gtxy*gtZZ*phidx + 2*gtXX*gtxy*(phidx*phidx) - gtXX*gtxy*phidxdx + 3*phidxdy - 2*gtxy*gtXY*phidxdy - 2*gtxy*gtXZ*phidxdz - 3*GammatYxy*phidy + GammatYxx*gtXX*gtxy*phidy + 2*GammatYxy*gtxy*gtXY*phidy + 2*GammatYxz*gtxy*gtXZ*phidy + GammatYyy*gtxy*gtYY*phidy + 2*GammatYyz*gtxy*gtYZ*phidy + GammatYzz*gtxy*gtZZ*phidy - 6*phidx*phidy + 4*gtxy*gtXY*phidx*phidy + 2*gtxy*gtYY*(phidy*phidy) - gtxy*gtYY*phidydy - 2*gtxy*gtYZ*phidydz - 3*GammatZxy*phidz + GammatZxx*gtXX*gtxy*phidz + 2*GammatZxy*gtxy*gtXY*phidz + 2*GammatZxz*gtxy*gtXZ*phidz + GammatZyy*gtxy*gtYY*phidz + 2*GammatZyz*gtxy*gtYZ*phidz + GammatZzz*gtxy*gtZZ*phidz + 4*gtxy*gtXZ*phidx*phidz + 4*gtxy*gtYZ*phidy*phidz + 2*gtxy*gtZZ*(phidz*phidz) - gtxy*gtZZ*phidzdz))/3.;
    double const Rtphixz=(-2*(GammatXxx*gtXX*gtxz*phidx + 2*GammatXxy*gtXY*gtxz*phidx + GammatXxz*(-3 + 2*gtxz*gtXZ)*phidx + GammatXyy*gtxz*gtYY*phidx + 2*GammatXyz*gtxz*gtYZ*phidx + GammatXzz*gtxz*gtZZ*phidx + 2*gtXX*gtxz*(phidx*phidx) - gtXX*gtxz*phidxdx - 2*gtXY*gtxz*phidxdy + 3*phidxdz - 2*gtxz*gtXZ*phidxdz - 3*GammatYxz*phidy + GammatYxx*gtXX*gtxz*phidy + 2*GammatYxy*gtXY*gtxz*phidy + 2*GammatYxz*gtxz*gtXZ*phidy + GammatYyy*gtxz*gtYY*phidy + 2*GammatYyz*gtxz*gtYZ*phidy + GammatYzz*gtxz*gtZZ*phidy + 4*gtXY*gtxz*phidx*phidy + 2*gtxz*gtYY*(phidy*phidy) - gtxz*gtYY*phidydy - 2*gtxz*gtYZ*phidydz - 3*GammatZxz*phidz + GammatZxx*gtXX*gtxz*phidz + 2*GammatZxy*gtXY*gtxz*phidz + 2*GammatZxz*gtxz*gtXZ*phidz + GammatZyy*gtxz*gtYY*phidz + 2*GammatZyz*gtxz*gtYZ*phidz + GammatZzz*gtxz*gtZZ*phidz - 6*phidx*phidz + 4*gtxz*gtXZ*phidx*phidz + 4*gtxz*gtYZ*phidy*phidz + 2*gtxz*gtZZ*(phidz*phidz) - gtxz*gtZZ*phidzdz))/3.;
    double const Rtphiyy=(-2*(GammatXxx*gtXX*gtyy*phidx + 2*GammatXxy*gtXY*gtyy*phidx + 2*GammatXxz*gtXZ*gtyy*phidx + GammatXyy*(-3 + gtyy*gtYY)*phidx + 2*GammatXyz*gtyy*gtYZ*phidx + GammatXzz*gtyy*gtZZ*phidx + 2*gtXX*gtyy*(phidx*phidx) - gtXX*gtyy*phidxdx - 2*gtXY*gtyy*phidxdy - 2*gtXZ*gtyy*phidxdz - 3*GammatYyy*phidy + GammatYxx*gtXX*gtyy*phidy + 2*GammatYxy*gtXY*gtyy*phidy + 2*GammatYxz*gtXZ*gtyy*phidy + GammatYyy*gtyy*gtYY*phidy + 2*GammatYyz*gtyy*gtYZ*phidy + GammatYzz*gtyy*gtZZ*phidy + 4*gtXY*gtyy*phidx*phidy - 6*(phidy*phidy) + 2*gtyy*gtYY*(phidy*phidy) + 3*phidydy - gtyy*gtYY*phidydy - 2*gtyy*gtYZ*phidydz - 3*GammatZyy*phidz + GammatZxx*gtXX*gtyy*phidz + 2*GammatZxy*gtXY*gtyy*phidz + 2*GammatZxz*gtXZ*gtyy*phidz + GammatZyy*gtyy*gtYY*phidz + 2*GammatZyz*gtyy*gtYZ*phidz + GammatZzz*gtyy*gtZZ*phidz + 4*gtXZ*gtyy*phidx*phidz + 4*gtyy*gtYZ*phidy*phidz + 2*gtyy*gtZZ*(phidz*phidz) - gtyy*gtZZ*phidzdz))/3.;
    double const Rtphiyz=(-2*(GammatXxx*gtXX*gtyz*phidx + 2*GammatXxy*gtXY*gtyz*phidx + 2*GammatXxz*gtXZ*gtyz*phidx + GammatXyy*gtYY*gtyz*phidx + GammatXyz*(-3 + 2*gtyz*gtYZ)*phidx + GammatXzz*gtyz*gtZZ*phidx + 2*gtXX*gtyz*(phidx*phidx) - gtXX*gtyz*phidxdx - 2*gtXY*gtyz*phidxdy - 2*gtXZ*gtyz*phidxdz - 3*GammatYyz*phidy + GammatYxx*gtXX*gtyz*phidy + 2*GammatYxy*gtXY*gtyz*phidy + 2*GammatYxz*gtXZ*gtyz*phidy + GammatYyy*gtYY*gtyz*phidy + 2*GammatYyz*gtyz*gtYZ*phidy + GammatYzz*gtyz*gtZZ*phidy + 4*gtXY*gtyz*phidx*phidy + 2*gtYY*gtyz*(phidy*phidy) - gtYY*gtyz*phidydy + 3*phidydz - 2*gtyz*gtYZ*phidydz - 3*GammatZyz*phidz + GammatZxx*gtXX*gtyz*phidz + 2*GammatZxy*gtXY*gtyz*phidz + 2*GammatZxz*gtXZ*gtyz*phidz + GammatZyy*gtYY*gtyz*phidz + 2*GammatZyz*gtyz*gtYZ*phidz + GammatZzz*gtyz*gtZZ*phidz + 4*gtXZ*gtyz*phidx*phidz - 6*phidy*phidz + 4*gtyz*gtYZ*phidy*phidz + 2*gtyz*gtZZ*(phidz*phidz) - gtyz*gtZZ*phidzdz))/3.;
    double const Rtphizz=(-2*(GammatXxx*gtXX*gtzz*phidx + 2*GammatXxy*gtXY*gtzz*phidx + 2*GammatXxz*gtXZ*gtzz*phidx + GammatXyy*gtYY*gtzz*phidx + 2*GammatXyz*gtYZ*gtzz*phidx + GammatXzz*(-3 + gtzz*gtZZ)*phidx + 2*gtXX*gtzz*(phidx*phidx) - gtXX*gtzz*phidxdx - 2*gtXY*gtzz*phidxdy - 2*gtXZ*gtzz*phidxdz - 3*GammatYzz*phidy + GammatYxx*gtXX*gtzz*phidy + 2*GammatYxy*gtXY*gtzz*phidy + 2*GammatYxz*gtXZ*gtzz*phidy + GammatYyy*gtYY*gtzz*phidy + 2*GammatYyz*gtYZ*gtzz*phidy + GammatYzz*gtzz*gtZZ*phidy + 4*gtXY*gtzz*phidx*phidy + 2*gtYY*gtzz*(phidy*phidy) - gtYY*gtzz*phidydy - 2*gtYZ*gtzz*phidydz - 3*GammatZzz*phidz + GammatZxx*gtXX*gtzz*phidz + 2*GammatZxy*gtXY*gtzz*phidz + 2*GammatZxz*gtXZ*gtzz*phidz + GammatZyy*gtYY*gtzz*phidz + 2*GammatZyz*gtYZ*gtzz*phidz + GammatZzz*gtzz*gtZZ*phidz + 4*gtXZ*gtzz*phidx*phidz + 4*gtYZ*gtzz*phidy*phidz - 6*(phidz*phidz) + 2*gtzz*gtZZ*(phidz*phidz) + 3*phidzdz - gtzz*gtZZ*phidzdz))/3.;

    // components of the part quadratic in first derivatives Qdd
    double const Qxx=-(GammatXxx*GammatXxx) - 2*GammatXxy*GammatYxx - GammatYxy*GammatYxy - 2*GammatXxz*GammatZxx - 2*GammatYxz*GammatZxy - GammatZxz*GammatZxz - gtxxdx*gtXXdx - gtxxdy*gtXYdx - gtxydx*gtXYdx - gtxxdz*gtXZdx - gtxzdx*gtXZdx - gtxydy*gtYYdx - gtxydz*gtYZdx - gtxzdy*gtYZdx - gtxzdz*gtZZdx;
    double const Qxy=(-2*GammatXxx*GammatXxy - 2*GammatXyy*GammatYxx - 2*GammatXxy*GammatYxy - 2*GammatYxy*GammatYyy - 2*GammatXyz*GammatZxx - 2*GammatXxz*GammatZxy - 2*GammatYyz*GammatZxy - 2*GammatYxz*GammatZyy - 2*GammatZxz*GammatZyz - gtxxdx*gtXXdy - gtXXdx*gtxydx - gtXYdx*gtxydy - gtxxdy*gtXYdy - gtxydx*gtXYdy - gtxydz*gtXZdx - gtxxdz*gtXZdy - gtxzdx*gtXZdy - gtXYdx*gtyydx - gtYYdx*gtyydy - gtxydy*gtYYdy - gtXZdx*gtyzdx - gtyydz*gtYZdx - gtYZdx*gtyzdy - gtxydz*gtYZdy - gtxzdy*gtYZdy - gtyzdz*gtZZdx - gtxzdz*gtZZdy)/2.;
    double const Qxz=(-2*GammatXxx*GammatXxz - 2*GammatXyz*GammatYxx - 2*GammatXxy*GammatYxz - 2*GammatYxy*GammatYyz - 2*GammatXzz*GammatZxx - 2*GammatYzz*GammatZxy - 2*GammatXxz*GammatZxz - 2*GammatYxz*GammatZyz - 2*GammatZxz*GammatZzz - gtxxdx*gtXXdz - gtxxdy*gtXYdz - gtxydx*gtXYdz - gtXXdx*gtxzdx - gtXYdx*gtxzdy - gtXZdx*gtxzdz - gtxxdz*gtXZdz - gtxzdx*gtXZdz - gtxydy*gtYYdz - gtXYdx*gtyzdx - gtYYdx*gtyzdy - gtYZdx*gtyzdz - gtxydz*gtYZdz - gtxzdy*gtYZdz - gtXZdx*gtzzdx - gtYZdx*gtzzdy - gtZZdx*gtzzdz - gtxzdz*gtZZdz)/2.;
    double const Qyy=-(GammatXxy*GammatXxy) - 2*GammatXyy*GammatYxy - GammatYyy*GammatYyy - 2*GammatXyz*GammatZxy - 2*GammatYyz*GammatZyy - GammatZyz*GammatZyz - gtXXdy*gtxydx - gtxydy*gtXYdy - gtxydz*gtXZdy - gtXYdy*gtyydx - gtyydy*gtYYdy - gtXZdy*gtyzdx - gtyydz*gtYZdy - gtyzdy*gtYZdy - gtyzdz*gtZZdy;
    double const Qyz=(-2*GammatXxy*GammatXxz - 2*GammatXyy*GammatYxz - 2*GammatYyy*GammatYyz - 2*GammatXzz*GammatZxy - 2*GammatXyz*(GammatYxy + GammatZxz) - 2*GammatYzz*GammatZyy - 2*GammatYyz*GammatZyz - 2*GammatZyz*GammatZzz - gtXXdz*gtxydx - gtxydy*gtXYdz - gtXXdy*gtxzdx - gtXYdy*gtxzdy - gtXZdy*gtxzdz - gtxydz*gtXZdz - gtXYdz*gtyydx - gtyydy*gtYYdz - gtXYdy*gtyzdx - gtXZdz*gtyzdx - gtYYdy*gtyzdy - gtYZdy*gtyzdz - gtyydz*gtYZdz - gtyzdy*gtYZdz - gtXZdy*gtzzdx - gtYZdy*gtzzdy - gtZZdy*gtzzdz - gtyzdz*gtZZdz)/2.;
    double const Qzz=-(GammatXxz*GammatXxz) - 2*GammatXyz*GammatYxz - GammatYyz*GammatYyz - 2*GammatXzz*GammatZxz - 2*GammatYzz*GammatZyz - GammatZzz*GammatZzz - gtXXdz*gtxzdx - gtXYdz*gtxzdy - gtxzdz*gtXZdz - gtXYdz*gtyzdx - gtYYdz*gtyzdy - gtyzdz*gtYZdz - gtXZdz*gtzzdx - gtYZdz*gtzzdy - gtzzdz*gtZZdz;

    // trace of the conformal Ricci
    double const Rt = GammatXdx + GammatYdy + GammatZdz + (-2*(GammatXxx*GammatXxx)*gtXX - 2*(GammatYxy*GammatYxy)*gtXX - 4*GammatXxz*GammatZxx*gtXX - 4*GammatYxz*GammatZxy*gtXX - 2*(GammatZxz*GammatZxz)*gtXX - gtXX*gtxxdx*gtXXdx - 4*GammatXyy*GammatYxx*gtXY - 4*GammatYxy*GammatYyy*gtXY - 4*GammatXyz*GammatZxx*gtXY - 4*GammatXxz*GammatZxy*gtXY - 4*GammatYyz*GammatZxy*gtXY - 4*GammatYxz*GammatZyy*gtXY - 4*GammatZxz*GammatZyz*gtXY + gtXXdx*gtxxdy*gtXY - gtxxdx*gtXXdy*gtXY - 2*gtXXdx*gtXY*gtxydx - 2*gtXX*gtxxdy*gtXYdx - 2*gtxxdy*gtXY*gtXYdy - 4*GammatXyz*GammatYxx*gtXZ - 4*GammatYxy*GammatYyz*gtXZ - 4*GammatXzz*GammatZxx*gtXZ - 4*GammatYzz*GammatZxy*gtXZ - 4*GammatXxz*GammatZxz*gtXZ - 4*GammatYxz*GammatZyz*gtXZ - 4*GammatZxz*GammatZzz*gtXZ + gtXXdx*gtxxdz*gtXZ - gtxxdx*gtXXdz*gtXZ + 2*gtXYdx*gtxydz*gtXZ - 2*gtxxdy*gtXYdz*gtXZ - 4*GammatXxx*(GammatXxy*gtXY + GammatXxz*gtXZ) - 2*gtXXdx*gtXZ*gtxzdx - 2*gtXX*gtxxdz*gtXZdx - 2*gtXY*gtxydz*gtXZdx - 2*gtXYdx*gtXZ*gtxzdy + 2*gtXY*gtXZdx*gtxzdy - 2*gtxxdz*gtXY*gtXZdy - 2*gtxxdz*gtXZ*gtXZdz - 2*(GammatXxy*GammatXxy)*gtYY - 4*GammatXyy*GammatYxy*gtYY - 2*(GammatYyy*GammatYyy)*gtYY - 4*GammatXyz*GammatZxy*gtYY - 4*GammatYyz*GammatZyy*gtYY - 2*(GammatZyz*GammatZyz)*gtYY + gtxxdy*gtXXdy*gtYY - 2*gtXXdy*gtxydx*gtYY - 2*gtxydz*gtXZdy*gtYY + 2*gtxzdy*gtXZdy*gtYY - 2*gtXY*gtXYdx*gtyydx - 2*gtXYdy*gtYY*gtyydx - 2*gtXX*gtxydy*gtYYdx + gtXX*gtyydx*gtYYdx - gtXY*gtYYdx*gtyydy - 2*gtXY*gtxydy*gtYYdy + gtXY*gtyydx*gtYYdy - gtYY*gtyydy*gtYYdy + gtXZ*gtYYdx*gtyydz - 2*gtxydy*gtXZ*gtYYdz + gtXZ*gtyydx*gtYYdz - 4*GammatXyz*GammatYxy*gtYZ - 4*GammatXyy*GammatYxz*gtYZ - 4*GammatYyy*GammatYyz*gtYZ - 4*GammatXzz*GammatZxy*gtYZ - 4*GammatXyz*GammatZxz*gtYZ - 4*GammatYzz*GammatZyy*gtYZ - 4*GammatYyz*GammatZyz*gtYZ - 4*GammatZyz*GammatZzz*gtYZ + gtXXdy*gtxxdz*gtYZ + gtxxdy*gtXXdz*gtYZ - 2*gtXXdz*gtxydx*gtYZ + 2*gtXYdy*gtxydz*gtYZ - 2*gtXXdy*gtxzdx*gtYZ - 2*gtXYdy*gtxzdy*gtYZ - 2*gtxydz*gtXZdz*gtYZ + 2*gtxzdy*gtXZdz*gtYZ - 2*gtXYdz*gtyydx*gtYZ + gtYYdy*gtyydz*gtYZ - gtyydy*gtYYdz*gtYZ - 4*GammatXxy*(GammatYxx*gtXX + GammatYxy*gtXY + GammatYxz*gtXZ + GammatXxz*gtYZ) - 2*gtXYdx*gtXZ*gtyzdx - 2*gtXY*gtXZdx*gtyzdx - 2*gtXZdy*gtYY*gtyzdx - 2*gtXYdy*gtYZ*gtyzdx - 2*gtXZdz*gtYZ*gtyzdx - 2*gtXX*gtxydz*gtYZdx - 2*gtXX*gtxzdy*gtYZdx - 2*gtXY*gtyydz*gtYZdx + 2*gtXX*gtyzdx*gtYZdx - 2*gtXZ*gtYYdx*gtyzdy - 2*gtYYdy*gtYZ*gtyzdy - 2*gtXY*gtxydz*gtYZdy - 2*gtXY*gtxzdy*gtYZdy - 2*gtYY*gtyydz*gtYZdy + 2*gtXY*gtyzdx*gtYZdy - 2*gtxydz*gtXZ*gtYZdz - 2*gtXZ*gtxzdy*gtYZdz - 2*gtyydz*gtYZ*gtYZdz + 2*gtXZ*gtyzdx*gtYZdz - 2*(GammatXxz*GammatXxz)*gtZZ - 4*GammatXyz*GammatYxz*gtZZ - 2*(GammatYyz*GammatYyz)*gtZZ - 4*GammatXzz*GammatZxz*gtZZ - 4*GammatYzz*GammatZyz*gtZZ - 2*(GammatZzz*GammatZzz)*gtZZ + gtxxdz*gtXXdz*gtZZ + 2*gtxydz*gtXYdz*gtZZ - 2*gtXXdz*gtxzdx*gtZZ - 2*gtXYdz*gtxzdy*gtZZ + gtyydz*gtYYdz*gtZZ - 2*gtXYdz*gtyzdx*gtZZ - 2*gtYYdz*gtyzdy*gtZZ - 2*gtXZ*gtXZdx*gtzzdx - 2*gtXZdy*gtYZ*gtzzdx - 2*gtXZdz*gtZZ*gtzzdx - 2*gtXX*gtxzdz*gtZZdx - 2*gtXY*gtyzdz*gtZZdx + gtXX*gtzzdx*gtZZdx - 2*gtXZ*gtYZdx*gtzzdy - 2*gtYZ*gtYZdy*gtzzdy - 2*gtYZdz*gtZZ*gtzzdy + gtXY*gtZZdx*gtzzdy - 2*gtXY*gtxzdz*gtZZdy - 2*gtYY*gtyzdz*gtZZdy + gtXY*gtzzdx*gtZZdy + gtYY*gtzzdy*gtZZdy - gtXZ*gtZZdx*gtzzdz - gtYZ*gtZZdy*gtzzdz - 2*gtXZ*gtxzdz*gtZZdz - 2*gtYZ*gtyzdz*gtZZdz + gtXZ*gtzzdx*gtZZdz + gtYZ*gtzzdy*gtZZdz - gtZZ*gtzzdz*gtZZdz)/2.;

    // trace-free Ricci tensor
    double const RTFxx=Qxx - (gtxx*Rt)/3. + Rt0xx + Rtphixx;
    double const RTFxy=Qxy - (gtxy*Rt)/3. + Rt0xy + Rtphixy;
    double const RTFxz=Qxz - (gtxz*Rt)/3. + Rt0xz + Rtphixz;
    double const RTFyy=Qyy - (gtyy*Rt)/3. + Rt0yy + Rtphiyy;
    double const RTFyz=Qyz - (gtyz*Rt)/3. + Rt0yz + Rtphiyz;
    double const RTFzz=Qzz - (gtzz*Rt)/3. + Rt0zz + Rtphizz;

    // output array
    bssn_state_t res;

    // calculation of BSSN RHS starts here

    // BSSN equation for conformal metric d/dt gtij
    std::array<double,6> const betakgammatddk {
        betaX * fd_der_upwind<der_order,0>(state,GTXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,GTXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,GTXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ,
        betaX * fd_der_upwind<der_order,0>(state,GTXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,GTXY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,GTXY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ,
        betaX * fd_der_upwind<der_order,0>(state,GTXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,GTXZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,GTXZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ,
        betaX * fd_der_upwind<der_order,0>(state,GTXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,GTYY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,GTYY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ,
        betaX * fd_der_upwind<der_order,0>(state,GTXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,GTYZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,GTYZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ,
        betaX * fd_der_upwind<der_order,0>(state,GTXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,GTZZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,GTZZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) 
    } ; 
    res[GTXXL+0] = -2*alp*Atxx - (2*(-2*betaXdx + betaYdy + betaZdz)*gtxx)/3. + betakgammatddk[0] + 2*betaYdx*gtxy + 2*betaZdx*gtxz;
    res[GTXXL+1] = -2*alp*Atxy + betaXdy*gtxx + ((betaXdx + betaYdy - 2*betaZdz)*gtxy)/3. + betakgammatddk[1] + betaZdy*gtxz + betaYdx*gtyy + betaZdx*gtyz;
    res[GTXXL+2] = -2*alp*Atxz + betaXdz*gtxx + betaYdz*gtxy + (betaXdx*gtxz)/3. - (2*betaYdy*gtxz)/3. + (betaZdz*gtxz)/3. + betakgammatddk[2] + betaYdx*gtyz + betaZdx*gtzz;
    res[GTXXL+3] = -2*alp*Atyy + 2*betaXdy*gtxy - (2*(betaXdx - 2*betaYdy + betaZdz)*gtyy)/3. + betakgammatddk[3] + 2*betaZdy*gtyz;
    res[GTXXL+4] = -2*alp*Atyz + betaXdz*gtxy + betaXdy*gtxz + betaYdz*gtyy - (2*betaXdx*gtyz)/3. + (betaYdy*gtyz)/3. + (betaZdz*gtyz)/3. + betakgammatddk[4] + betaZdy*gtzz;
    res[GTXXL+5] = -2*alp*Atzz + 2*betaXdz*gtxz + 2*betaYdz*gtyz + 2*betaZdz*gtzz - (2*(betaXdx + betaYdy + betaZdz)*gtzz)/3. + betakgammatddk[5];

    // BSSN equation for conformal traceless extrinsic curvature d/dt Atij, note that Dti beta^j =d/dxi beta^j because of unit determinant of conformal metric 
    std::array<double,6> const betakAtddk {
        betaX * fd_der_upwind<der_order,0>(state,ATXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,ATXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,ATXX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ,
        betaX * fd_der_upwind<der_order,0>(state,ATXY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,ATXY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,ATXY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ,
        betaX * fd_der_upwind<der_order,0>(state,ATXZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,ATXZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,ATXZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ,
        betaX * fd_der_upwind<der_order,0>(state,ATYY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,ATYY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,ATYY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ,
        betaX * fd_der_upwind<der_order,0>(state,ATYZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,ATYZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,ATYZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ,
        betaX * fd_der_upwind<der_order,0>(state,ATZZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaY * fd_der_upwind<der_order,1>(state,ATZZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) + betaZ * fd_der_upwind<der_order,2>(state,ATZZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) 
    } ; 
    res[ATXXL+0] = betakAtddk[0] + 2*Atxx*betaXdx + 2*Atxy*betaYdx + 2*Atxz*betaZdx - (2*Atxx*(betaXdx + betaYdy + betaZdz))/3. - alp*(2*(Atxx*Atxx)*gtXX + 4*Atxx*Atxy*gtXY + 4*Atxx*Atxz*gtXZ + 2*(Atxy*Atxy)*gtYY + 4*Atxy*Atxz*gtYZ + 2*(Atxz*Atxz)*gtZZ - Atxx*K) + (exp(-4.*phi)*(-3*DiDjalpxx + DDalp*gtxx + alp*(3*RTFxx - 24*pi*Sxx + 8*gtxx*pi*S*exp(4.*phi))))/3.;
    res[ATXXL+1] = betakAtddk[1] + Atxy*betaXdx + Atxx*betaXdy + Atyy*betaYdx + Atxy*betaYdy  + Atyz*betaZdx + Atxz*betaZdy - (2*Atxy*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*(Atxy*Atxy*gtXY + Atxy*Atxz*gtXZ + Atxx*(Atxy*gtXX + Atyy*gtXY + Atyz*gtXZ) + Atxy*Atyy*gtYY + Atxz*Atyy*gtYZ + Atxy*Atyz*gtYZ + Atxz*Atyz*gtZZ) + Atxy*K) + (exp(-4.*phi)*(-3*DiDjalpxy + DDalp*gtxy + alp*(3*RTFxy - 24*pi*Sxy + 8*gtxy*pi*S*exp(4.*phi))))/3.;
    res[ATXXL+2] = betakAtddk[2] + Atxz*betaXdx + Atxx*betaXdz + Atyz*betaYdx + Atxy*betaYdz + Atzz*betaZdx + Atxz*betaZdz - (2*Atxz*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*(Atxx*(Atxz*gtXX + Atyz*gtXY + Atzz*gtXZ) + Atxy*(Atxz*gtXY + Atyz*gtYY + Atzz*gtYZ) + Atxz*(Atxz*gtXZ + Atyz*gtYZ + Atzz*gtZZ)) + Atxz*K) + (exp(-4.*phi)*(-3*DiDjalpxz + DDalp*gtxz + alp*(3*RTFxz - 24*pi*Sxz + 8*gtxz*pi*S*exp(4.*phi))))/3.;
    res[ATXXL+3] = betakAtddk[3] + 2*Atxy*betaXdy + 2*Atyy*betaYdy +  2*Atyz*betaZdy - (2*Atyy*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*(Atxy*Atxy*gtXX + 2*Atxy*(Atyy*gtXY + Atyz*gtXZ) + Atyy*Atyy*gtYY + 2*Atyy*Atyz*gtYZ + Atyz*Atyz*gtZZ) + Atyy*K) + (exp(-4.*phi)*(-3*DiDjalpyy + DDalp*gtyy + alp*(3*RTFyy - 24*pi*Syy + 8*gtyy*pi*S*exp(4.*phi))))/3.;
    res[ATXXL+4] = betakAtddk[4] + Atxz*betaXdy + Atxy*betaXdz + Atyz*betaYdy + Atyy*betaYdz +  Atzz*betaZdy + Atyz*betaZdz - (2*Atyz*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*(Atxz*Atyy*gtXY + Atxz*Atyz*gtXZ + Atxy*(Atxz*gtXX + Atyz*gtXY + Atzz*gtXZ) + Atyy*Atyz*gtYY + Atyz*Atyz*gtYZ + Atyy*Atzz*gtYZ + Atyz*Atzz*gtZZ) + Atyz*K) + (exp(-4.*phi)*(-3*DiDjalpyz + DDalp*gtyz + alp*(3*RTFyz - 24*pi*Syz + 8*gtyz*pi*S*exp(4.*phi))))/3.;
    res[ATXXL+5] = betakAtddk[5] + 2*Atxz*betaXdz + 2*Atyz*betaYdz + 2*Atzz*betaZdz - (2*Atzz*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*(Atxz*Atxz*gtXX + 2*Atxz*(Atyz*gtXY + Atzz*gtXZ) + Atyz*Atyz*gtYY + 2*Atyz*Atzz*gtYZ + Atzz*Atzz*gtZZ) + Atzz*K) + (exp(-4.*phi)*(-3*DiDjalpzz + DDalp*gtzz + alp*(3*RTFzz - 24*pi*Szz + 8*gtzz*pi*S*exp(4.*phi))))/3.;

    // BSSN equation for conformal factor d/dt phi
    double const betakdphidk = betaX * fd_der_upwind<der_order,0>(state,PHI_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                             + betaY * fd_der_upwind<der_order,1>(state,PHI_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                             + betaZ * fd_der_upwind<der_order,2>(state,PHI_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ; 
    res[PHIL] = (betaXdx + betaYdy + betaZdz - alp*K)/6. + betakdphidk;

    // BSSN equation for conformal extrinsic curvature trace d/dt K
    double const betakdKdk   = betaX * fd_der_upwind<der_order,0>(state,K_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                             + betaY * fd_der_upwind<der_order,1>(state,K_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                             + betaZ * fd_der_upwind<der_order,2>(state,K_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ; 
    res[KL] = betakdKdk + alp*(Atxx*AtXX + 2*Atxy*AtXY + 2*Atxz*AtXZ + Atyy*AtYY + 2*Atyz*AtYZ + Atzz*AtZZ + (K*K)/3. + 4*pi*(EE + S)) + (-(alpdxdx*gtXX) + alpdy*GammatYxx*gtXX + alpdz*GammatZxx*gtXX - 2*alpdxdy*gtXY + 2*alpdy*GammatYxy*gtXY + 2*alpdz*GammatZxy*gtXY - 2*alpdxdz*gtXZ + 2*alpdy*GammatYxz*gtXZ + 2*alpdz*GammatZxz*gtXZ - alpdydy*gtYY + alpdy*GammatYyy*gtYY + alpdz*GammatZyy*gtYY - 2*alpdydz*gtYZ + 2*alpdy*GammatYyz*gtYZ + 2*alpdz*GammatZyz*gtYZ - alpdzdz*gtZZ + alpdy*GammatYzz*gtZZ + alpdz*GammatZzz*gtZZ - 2*alpdy*gtXY*phidx - 2*alpdz*gtXZ*phidx - 2*alpdy*gtYY*phidy - 2*alpdz*gtYZ*phidy - 2*alpdy*gtYZ*phidz - 2*alpdz*gtZZ*phidz + alpdx*(GammatXxx*gtXX + 2*GammatXxy*gtXY + 2*GammatXxz*gtXZ + GammatXyy*gtYY + 2*GammatXyz*gtYZ + GammatXzz*gtZZ - 2*gtXX*phidx - 2*gtXY*phidy - 2*gtXZ*phidz))*exp(-4.*phi);

    // BSSN equation for contracted conformal Christoffels d/dt Gammatu
    double const betakdGammaXdk = betaX * fd_der_upwind<der_order,0>(state,GAMMAX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                                + betaY * fd_der_upwind<der_order,1>(state,GAMMAX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                                + betaZ * fd_der_upwind<der_order,2>(state,GAMMAX_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ; 
    double const betakdGammaYdk = betaX * fd_der_upwind<der_order,0>(state,GAMMAY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                                + betaY * fd_der_upwind<der_order,1>(state,GAMMAY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                                + betaZ * fd_der_upwind<der_order,2>(state,GAMMAY_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ; 
    double const betakdGammaZdk = betaX * fd_der_upwind<der_order,0>(state,GAMMAZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                                + betaY * fd_der_upwind<der_order,1>(state,GAMMAZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                                + betaZ * fd_der_upwind<der_order,2>(state,GAMMAZ_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ; 
    res[GAMMAXL+0] = (-6*alpdx*AtXX - 6*alpdy*AtXY - 6*alpdz*AtXZ - betaXdx*GammatX + 2*betaYdy*GammatX + 2*betaZdz*GammatX + 3*betakdGammaXdk - 3*betaXdy*GammatY - 3*betaXdz*GammatZ + 4*betaXdxdx*gtXX + betaYdxdy*gtXX + betaZdxdz*gtXX + 7*betaXdxdy*gtXY + betaYdydy*gtXY + betaZdydz*gtXY + 7*betaXdxdz*gtXZ + betaYdydz*gtXZ + betaZdzdz*gtXZ + 3*betaXdydy*gtYY + 6*betaXdydz*gtYZ + 3*betaXdzdz*gtZZ + 2*alp*(6*AtXZ*GammatXxz + 3*AtYY*GammatXyy + 6*AtYZ*GammatXyz + 3*AtZZ*GammatXzz - 2*gtXX*Kdx - 2*gtXY*Kdy - 2*gtXZ*Kdz + 3*AtXX*(GammatXxx + 6*phidx) + 6*AtXY*(GammatXxy + 3*phidy) + 18*AtXZ*phidz - 24*gtXX*pi*Sx - 24*gtXY*pi*Sy - 24*gtXZ*pi*Sz))/3.;
    res[GAMMAXL+1] = (-6*alpdx*AtXY - 6*alpdy*AtYY - 6*alpdz*AtYZ - 3*betaYdx*GammatX + 2*betaXdx*GammatY - betaYdy*GammatY + 2*betaZdz*GammatY + 3*betakdGammaYdk - 3*betaYdz*GammatZ + 3*betaYdxdx*gtXX + betaXdxdx*gtXY + 7*betaYdxdy*gtXY + betaZdxdz*gtXY + 6*betaYdxdz*gtXZ + betaXdxdy*gtYY + 4*betaYdydy*gtYY + betaZdydz*gtYY + betaXdxdz*gtYZ + 7*betaYdydz*gtYZ + betaZdzdz*gtYZ + 3*betaYdzdz*gtZZ + 2*alp*(3*AtXX*GammatYxx + 6*AtXZ*GammatYxz + 3*AtYY*GammatYyy + 6*AtYZ*GammatYyz + 3*AtZZ*GammatYzz - 2*gtXY*Kdx - 2*gtYY*Kdy - 2*gtYZ*Kdz + 6*AtXY*(GammatYxy + 3*phidx) + 18*AtYY*phidy + 18*AtYZ*phidz - 24*gtXY*pi*Sx - 24*gtYY*pi*Sy - 24*gtYZ*pi*Sz))/3.;
    res[GAMMAXL+2] = (-6*alpdx*AtXZ - 6*alpdy*AtYZ - 6*alpdz*AtZZ - 3*betaZdx*GammatX - 3*betaZdy*GammatY + 2*betaXdx*GammatZ + 2*betaYdy*GammatZ - betaZdz*GammatZ + 3*betakdGammaZdk+ 3*betaZdxdx*gtXX + 6*betaZdxdy*gtXY + betaXdxdx*gtXZ + betaYdxdy*gtXZ + 7*betaZdxdz*gtXZ + 3*betaZdydy*gtYY + betaXdxdy*gtYZ + betaYdydy*gtYZ + 7*betaZdydz*gtYZ + betaXdxdz*gtZZ + betaYdydz*gtZZ + 4*betaZdzdz*gtZZ + 2*alp*(3*AtXX*GammatZxx + 6*AtXY*GammatZxy + 6*AtXZ*GammatZxz + 3*AtYY*GammatZyy + 6*AtYZ*GammatZyz + 3*AtZZ*GammatZzz - 2*gtXZ*Kdx - 2*gtYZ*Kdy - 2*gtZZ*Kdz + 18*AtXZ*phidx + 18*AtYZ*phidy + 18*AtZZ*phidz - 24*gtXZ*pi*Sx - 24*gtYZ*pi*Sy - 24*gtZZ*pi*Sz))/3.;

    double const GammatXdt = res[GAMMAXL+0] ;
    double const GammatYdt = res[GAMMAXL+1] ;
    double const GammatZdt = res[GAMMAXL+2] ;

    /* 1 + log slicing condition */
    double const betakdalpdk = betaX * fd_der_upwind<der_order,0>(state,ALP_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                             + betaY * fd_der_upwind<der_order,1>(state,ALP_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ})
                             + betaZ * fd_der_upwind<der_order,2>(state,ALP_,VEC(i,j,k),q,{-betaX,-betaY,-betaZ}) ;
    res[ALPL] = betakdalpdk - 2*alp*K;

    /* Gamma driver */
    double const BX = state(VEC(i,j,k),BBX_+0,q);
    double const BY = state(VEC(i,j,k),BBX_+1,q);
    double const BZ = state(VEC(i,j,k),BBX_+2,q);

    res[BETAXL+0] = BX*k1;
    res[BETAXL+1] = BY*k1;
    res[BETAXL+2] = BZ*k1;

    res[BBXL+0] = -(BX*eta) + GammatXdt;
    res[BBXL+1] = -(BY*eta) + GammatYdt;
    res[BBXL+2] = -(BZ*eta) + GammatZdt;

    return std::move(res);


}

template< size_t der_order >
std::array<double,4> GRACE_HOST_DEVICE 
compute_bssn_constraint_violations(
      VEC(int i, int j, int k), int q
    , grace::var_array_t const state
    , std::array<std::array<double,4>,4> const& Tdd
    , std::array<double,GRACE_NSPACEDIM> const& idx
)
{

    double const pi = M_PI ; 
    
    double const gtxx = state(VEC(i,j,k),GTXX_+0,q);
    double const gtxy = state(VEC(i,j,k),GTXX_+1,q);
    double const gtxz = state(VEC(i,j,k),GTXX_+2,q);
    double const gtyy = state(VEC(i,j,k),GTXX_+3,q);
    double const gtyz = state(VEC(i,j,k),GTXX_+4,q);
    double const gtzz = state(VEC(i,j,k),GTXX_+5,q);
    double const gtXX=-(gtyz*gtyz) + gtyy*gtzz;
    double const gtXY=gtxz*gtyz - gtxy*gtzz;
    double const gtXZ=-(gtxz*gtyy) + gtxy*gtyz;
    double const gtYY=-(gtxz*gtxz) + gtxx*gtzz;
    double const gtYZ=gtxy*gtxz - gtxx*gtyz;
    double const gtZZ=-(gtxy*gtxy) + gtxx*gtyy;

    double const alp = state(VEC(i,j,k),ALP_,q);
    double const betaX = state(VEC(i,j,k),BETAX_+0,q);
    double const betaY = state(VEC(i,j,k),BETAX_+1,q);
    double const betaZ = state(VEC(i,j,k),BETAX_+2,q);

    double const Atxx = state(VEC(i,j,k),ATXX_+0,q);
    double const Atxy = state(VEC(i,j,k),ATXX_+1,q);
    double const Atxz = state(VEC(i,j,k),ATXX_+2,q);
    double const Atyy = state(VEC(i,j,k),ATXX_+3,q);
    double const Atyz = state(VEC(i,j,k),ATXX_+4,q);
    double const Atzz = state(VEC(i,j,k),ATXX_+5,q);
    double const AtXX=Atxx*(gtXX*gtXX) + 2*Atxy*gtXX*gtXY + Atyy*(gtXY*gtXY) + 2*Atxz*gtXX*gtXZ + 2*Atyz*gtXY*gtXZ + Atzz*(gtXZ*gtXZ);
    double const AtXY=Atxx*gtXX*gtXY + Atxy*(gtXY*gtXY) + Atxz*gtXY*gtXZ + Atxy*gtXX*gtYY + Atyy*gtXY*gtYY + Atyz*gtXZ*gtYY + Atxz*gtXX*gtYZ + Atyz*gtXY*gtYZ + Atzz*gtXZ*gtYZ;
    double const AtXZ=Atxx*gtXX*gtXZ + Atxy*gtXY*gtXZ + Atxz*(gtXZ*gtXZ) + Atxy*gtXX*gtYZ + Atyy*gtXY*gtYZ + Atyz*gtXZ*gtYZ + Atxz*gtXX*gtZZ + Atyz*gtXY*gtZZ + Atzz*gtXZ*gtZZ;
    double const AtYY=Atxx*(gtXY*gtXY) + 2*Atxy*gtXY*gtYY + Atyy*(gtYY*gtYY) + 2*Atxz*gtXY*gtYZ + 2*Atyz*gtYY*gtYZ + Atzz*(gtYZ*gtYZ);
    double const AtYZ=Atxx*gtXY*gtXZ + Atxy*gtXZ*gtYY + Atxy*gtXY*gtYZ + Atxz*gtXZ*gtYZ + Atyy*gtYY*gtYZ + Atyz*(gtYZ*gtYZ) + Atxz*gtXY*gtZZ + Atyz*gtYY*gtZZ + Atzz*gtYZ*gtZZ;
    double const AtZZ=Atxx*(gtXZ*gtXZ) + 2*Atxy*gtXZ*gtYZ + Atyy*(gtYZ*gtYZ) + 2*Atxz*gtXZ*gtZZ + 2*Atyz*gtYZ*gtZZ + Atzz*(gtZZ*gtZZ);
    double const Atxxdx = grace::fd_der<der_order,0>(state,ATXX_+0, VEC(i,j,k),q) * idx[0 ];
    double const Atxxdy = grace::fd_der<der_order,1>(state,ATXX_+0, VEC(i,j,k),q) * idx[1 ];
    double const Atxxdz = grace::fd_der<der_order,2>(state,ATXX_+0, VEC(i,j,k),q) * idx[2 ];
    double const Atxydx = grace::fd_der<der_order,0>(state,ATXX_+1, VEC(i,j,k),q) * idx[0 ];
    double const Atxydy = grace::fd_der<der_order,1>(state,ATXX_+1, VEC(i,j,k),q) * idx[1 ];
    double const Atxydz = grace::fd_der<der_order,2>(state,ATXX_+1, VEC(i,j,k),q) * idx[2 ];
    double const Atxzdx = grace::fd_der<der_order,0>(state,ATXX_+2, VEC(i,j,k),q) * idx[0 ];
    double const Atxzdy = grace::fd_der<der_order,1>(state,ATXX_+2, VEC(i,j,k),q) * idx[1 ];
    double const Atxzdz = grace::fd_der<der_order,2>(state,ATXX_+2, VEC(i,j,k),q) * idx[2 ];
    double const Atyydx = grace::fd_der<der_order,0>(state,ATXX_+3, VEC(i,j,k),q) * idx[0 ];
    double const Atyydy = grace::fd_der<der_order,1>(state,ATXX_+3, VEC(i,j,k),q) * idx[1 ];
    double const Atyydz = grace::fd_der<der_order,2>(state,ATXX_+3, VEC(i,j,k),q) * idx[2 ];
    double const Atyzdx = grace::fd_der<der_order,0>(state,ATXX_+4, VEC(i,j,k),q) * idx[0 ];
    double const Atyzdy = grace::fd_der<der_order,1>(state,ATXX_+4, VEC(i,j,k),q) * idx[1 ];
    double const Atyzdz = grace::fd_der<der_order,2>(state,ATXX_+4, VEC(i,j,k),q) * idx[2 ];
    double const Atzzdx = grace::fd_der<der_order,0>(state,ATXX_+5, VEC(i,j,k),q) * idx[0 ];
    double const Atzzdy = grace::fd_der<der_order,1>(state,ATXX_+5, VEC(i,j,k),q) * idx[1 ];
    double const Atzzdz = grace::fd_der<der_order,2>(state,ATXX_+5, VEC(i,j,k),q) * idx[2 ];

    double const phi = state(VEC(i,j,k),PHI_,q);
    double const phidx = grace::fd_der<der_order,0>(state,PHI_,VEC(i,j,k),q) * idx[0 ];
    double const phidy = grace::fd_der<der_order,1>(state,PHI_,VEC(i,j,k),q) * idx[1 ];
    double const phidz = grace::fd_der<der_order,2>(state,PHI_,VEC(i,j,k),q) * idx[2 ];
    double const phidxdx = grace::fd_second_der<der_order,0>(state,PHI_,VEC(i,j,k),q) * math::int_pow<2>(idx[0 ]);
    double const phidydy = grace::fd_second_der<der_order,1>(state,PHI_,VEC(i,j,k),q) * math::int_pow<2>(idx[1 ]);
    double const phidzdz = grace::fd_second_der<der_order,2>(state,PHI_,VEC(i,j,k),q) * math::int_pow<2>(idx[2 ]);
    double const phidxdy = grace::fd_der<der_order,0,1>(state,PHI_,VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const phidxdz = grace::fd_der<der_order,0,2>(state,PHI_,VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const phidydz = grace::fd_der<der_order,1,2>(state,PHI_,VEC(i,j,k),q) * idx[1 ] * idx[2 ];

    double const K = state(VEC(i,j,k),K_,q);
    double const Kdx = grace::fd_der<der_order,0>(state,K_,VEC(i,j,k),q) * idx[0 ];
    double const Kdy = grace::fd_der<der_order,1>(state,K_,VEC(i,j,k),q) * idx[1 ];
    double const Kdz = grace::fd_der<der_order,2>(state,K_,VEC(i,j,k),q) * idx[2 ];

    double const GammatX = state(VEC(i,j,k),GAMMAX_+0,q);
    double const GammatY = state(VEC(i,j,k),GAMMAX_+1,q);
    double const GammatZ = state(VEC(i,j,k),GAMMAX_+2,q);
    double const GammatXdx = grace::fd_der<der_order,0>(state,GAMMAX_+0, VEC(i,j,k),q)* idx[0 ];
    double const GammatXdy = grace::fd_der<der_order,1>(state,GAMMAX_+0, VEC(i,j,k),q)* idx[1 ];
    double const GammatXdz = grace::fd_der<der_order,2>(state,GAMMAX_+0, VEC(i,j,k),q)* idx[2 ];
    double const GammatYdx = grace::fd_der<der_order,0>(state,GAMMAX_+1, VEC(i,j,k),q)* idx[0 ];
    double const GammatYdy = grace::fd_der<der_order,1>(state,GAMMAX_+1, VEC(i,j,k),q)* idx[1 ];
    double const GammatYdz = grace::fd_der<der_order,2>(state,GAMMAX_+1, VEC(i,j,k),q)* idx[2 ];
    double const GammatZdx = grace::fd_der<der_order,0>(state,GAMMAX_+2, VEC(i,j,k),q)* idx[0 ];
    double const GammatZdy = grace::fd_der<der_order,1>(state,GAMMAX_+2, VEC(i,j,k),q)* idx[1 ];
    double const GammatZdz = grace::fd_der<der_order,2>(state,GAMMAX_+2, VEC(i,j,k),q)* idx[2 ];

    double const gtxxdx = grace::fd_der<der_order,0>(state,GTXX_+0, VEC(i,j,k),q) * idx[0 ];
    double const gtxxdy = grace::fd_der<der_order,1>(state,GTXX_+0, VEC(i,j,k),q) * idx[1 ];
    double const gtxxdz = grace::fd_der<der_order,2>(state,GTXX_+0, VEC(i,j,k),q) * idx[2 ];
    double const gtxydx = grace::fd_der<der_order,0>(state,GTXX_+1, VEC(i,j,k),q) * idx[0 ];
    double const gtxydy = grace::fd_der<der_order,1>(state,GTXX_+1, VEC(i,j,k),q) * idx[1 ];
    double const gtxydz = grace::fd_der<der_order,2>(state,GTXX_+1, VEC(i,j,k),q) * idx[2 ];
    double const gtxzdx = grace::fd_der<der_order,0>(state,GTXX_+2, VEC(i,j,k),q) * idx[0 ];
    double const gtxzdy = grace::fd_der<der_order,1>(state,GTXX_+2, VEC(i,j,k),q) * idx[1 ];
    double const gtxzdz = grace::fd_der<der_order,2>(state,GTXX_+2, VEC(i,j,k),q) * idx[2 ];
    double const gtyydx = grace::fd_der<der_order,0>(state,GTXX_+3, VEC(i,j,k),q) * idx[0 ];
    double const gtyydy = grace::fd_der<der_order,1>(state,GTXX_+3, VEC(i,j,k),q) * idx[1 ];
    double const gtyydz = grace::fd_der<der_order,2>(state,GTXX_+3, VEC(i,j,k),q) * idx[2 ];
    double const gtyzdx = grace::fd_der<der_order,0>(state,GTXX_+4, VEC(i,j,k),q) * idx[0 ];
    double const gtyzdy = grace::fd_der<der_order,1>(state,GTXX_+4, VEC(i,j,k),q) * idx[1 ];
    double const gtyzdz = grace::fd_der<der_order,2>(state,GTXX_+4, VEC(i,j,k),q) * idx[2 ];
    double const gtzzdx = grace::fd_der<der_order,0>(state,GTXX_+5, VEC(i,j,k),q) * idx[0 ];
    double const gtzzdy = grace::fd_der<der_order,1>(state,GTXX_+5, VEC(i,j,k),q) * idx[1 ];
    double const gtzzdz = grace::fd_der<der_order,2>(state,GTXX_+5, VEC(i,j,k),q) * idx[2 ];
    double const gtXXdx = -(gtXX*gtXX*gtxxdx) - 2*gtXX*(gtXY*gtxydx + gtXZ*gtxzdx) - gtXY*(gtXY*gtyydx + 2*gtXZ*gtyzdx) - gtXZ*gtXZ*gtzzdx;
    double const gtXXdy = -(gtXX*gtXX*gtxxdy) - 2*gtXX*(gtXY*gtxydy + gtXZ*gtxzdy) - gtXY*(gtXY*gtyydy + 2*gtXZ*gtyzdy) - gtXZ*gtXZ*gtzzdy;
    double const gtXXdz = -(gtXX*gtXX*gtxxdz) - 2*gtXX*(gtXY*gtxydz + gtXZ*gtxzdz) - gtXY*(gtXY*gtyydz + 2*gtXZ*gtyzdz) - gtXZ*gtXZ*gtzzdz;
    double const gtXYdx = -(gtXY*gtXY*gtxydx) - gtXX*(gtxxdx*gtXY + gtxydx*gtYY + gtxzdx*gtYZ) - gtXY*(gtXZ*gtxzdx + gtYY*gtyydx + gtYZ*gtyzdx) - gtXZ*(gtYY*gtyzdx + gtYZ*gtzzdx);
    double const gtXYdy = -(gtXY*gtXY*gtxydy) - gtXX*(gtxxdy*gtXY + gtxydy*gtYY + gtxzdy*gtYZ) - gtXY*(gtXZ*gtxzdy + gtYY*gtyydy + gtYZ*gtyzdy) - gtXZ*(gtYY*gtyzdy + gtYZ*gtzzdy);
    double const gtXYdz = -(gtXY*gtXY*gtxydz) - gtXX*(gtxxdz*gtXY + gtxydz*gtYY + gtxzdz*gtYZ) - gtXY*(gtXZ*gtxzdz + gtYY*gtyydz + gtYZ*gtyzdz) - gtXZ*(gtYY*gtyzdz + gtYZ*gtzzdz);
    double const gtXZdx = -(gtXX*(gtxxdx*gtXZ + gtxydx*gtYZ + gtxzdx*gtZZ)) - gtXY*(gtxydx*gtXZ + gtyydx*gtYZ + gtyzdx*gtZZ) - gtXZ*(gtXZ*gtxzdx + gtYZ*gtyzdx + gtZZ*gtzzdx);
    double const gtXZdy = -(gtXX*(gtxxdy*gtXZ + gtxydy*gtYZ + gtxzdy*gtZZ)) - gtXY*(gtxydy*gtXZ + gtyydy*gtYZ + gtyzdy*gtZZ) - gtXZ*(gtXZ*gtxzdy + gtYZ*gtyzdy + gtZZ*gtzzdy);
    double const gtXZdz = -(gtXX*(gtxxdz*gtXZ + gtxydz*gtYZ + gtxzdz*gtZZ)) - gtXY*(gtxydz*gtXZ + gtyydz*gtYZ + gtyzdz*gtZZ) - gtXZ*(gtXZ*gtxzdz + gtYZ*gtyzdz + gtZZ*gtzzdz);
    double const gtYYdx = -(gtxxdx*(gtXY*gtXY)) - 2*gtXY*(gtxydx*gtYY + gtxzdx*gtYZ) - gtYY*(gtYY*gtyydx + 2*gtYZ*gtyzdx) - gtYZ*gtYZ*gtzzdx;
    double const gtYYdy = -(gtxxdy*(gtXY*gtXY)) - 2*gtXY*(gtxydy*gtYY + gtxzdy*gtYZ) - gtYY*(gtYY*gtyydy + 2*gtYZ*gtyzdy) - gtYZ*gtYZ*gtzzdy;
    double const gtYYdz = -(gtxxdz*(gtXY*gtXY)) - 2*gtXY*(gtxydz*gtYY + gtxzdz*gtYZ) - gtYY*(gtYY*gtyydz + 2*gtYZ*gtyzdz) - gtYZ*gtYZ*gtzzdz;
    double const gtYZdx = -(gtYY*gtyydx*gtYZ) - gtXZ*(gtxydx*gtYY + gtxzdx*gtYZ) - gtYZ*gtYZ*gtyzdx - gtYY*gtyzdx*gtZZ - gtXY*(gtxxdx*gtXZ + gtxydx*gtYZ + gtxzdx*gtZZ) - gtYZ*gtZZ*gtzzdx;
    double const gtYZdy = -(gtYY*gtyydy*gtYZ) - gtXZ*(gtxydy*gtYY + gtxzdy*gtYZ) - gtYZ*gtYZ*gtyzdy - gtYY*gtyzdy*gtZZ - gtXY*(gtxxdy*gtXZ + gtxydy*gtYZ + gtxzdy*gtZZ) - gtYZ*gtZZ*gtzzdy;
    double const gtYZdz = -(gtYY*gtyydz*gtYZ) - gtXZ*(gtxydz*gtYY + gtxzdz*gtYZ) - gtYZ*gtYZ*gtyzdz - gtYY*gtyzdz*gtZZ - gtXY*(gtxxdz*gtXZ + gtxydz*gtYZ + gtxzdz*gtZZ) - gtYZ*gtZZ*gtzzdz;
    double const gtZZdx = -(gtxxdx*(gtXZ*gtXZ)) - 2*gtXZ*(gtxydx*gtYZ + gtxzdx*gtZZ) - gtYZ*(gtyydx*gtYZ + 2*gtyzdx*gtZZ) - gtZZ*gtZZ*gtzzdx;
    double const gtZZdy = -(gtxxdy*(gtXZ*gtXZ)) - 2*gtXZ*(gtxydy*gtYZ + gtxzdy*gtZZ) - gtYZ*(gtyydy*gtYZ + 2*gtyzdy*gtZZ) - gtZZ*gtZZ*gtzzdy;
    double const gtZZdz = -(gtxxdz*(gtXZ*gtXZ)) - 2*gtXZ*(gtxydz*gtYZ + gtxzdz*gtZZ) - gtYZ*(gtyydz*gtYZ + 2*gtyzdz*gtZZ) - gtZZ*gtZZ*gtzzdz;

    double const GammatXxx=(gtXX*gtxxdx - gtxxdy*gtXY + 2*gtXY*gtxydx - gtxxdz*gtXZ + 2*gtXZ*gtxzdx)/2.;
    double const GammatXxy=(gtXX*(gtxxdx + gtxxdy - gtxydx) + gtXY*gtxydx + gtXZ*(-gtxydz + gtxzdx + gtxzdy))/2.;
    double const GammatXxz=(gtXX*(gtxxdx + gtxxdz - gtxzdx) + gtXZ*gtxzdx + gtXY*(gtxydx + gtxydz - gtxzdy))/2.;
    double const GammatXyy=(2*gtXX*gtxydy - gtXX*gtyydx + gtXY*gtyydy - gtXZ*gtyydz + 2*gtXZ*gtyzdy)/2.;
    double const GammatXyz=(gtXX*(gtxydy + gtxydz - gtyzdx) + gtXY*(gtyydy + gtyydz - gtyzdy) + gtXZ*gtyzdy)/2.;
    double const GammatXzz=(2*gtXX*gtxzdz + 2*gtXY*gtyzdz - gtXX*gtzzdx - gtXY*gtzzdy + gtXZ*gtzzdz)/2.;
    double const GammatYxx=(gtxxdx*gtXY - gtxxdy*gtYY + 2*gtxydx*gtYY - gtxxdz*gtYZ + 2*gtxzdx*gtYZ)/2.;
    double const GammatYxy=((gtxxdx + gtxxdy)*gtXY + gtxydx*(-gtXY + gtYY) + (-gtxydz + gtxzdx + gtxzdy)*gtYZ)/2.;
    double const GammatYxz=((gtxxdx + gtxxdz)*gtXY + (gtxydx + gtxydz - gtxzdy)*gtYY + gtxzdx*(-gtXY + gtYZ))/2.;
    double const GammatYyy=(2*gtXY*gtxydy - gtXY*gtyydx + gtYY*gtyydy - gtyydz*gtYZ + 2*gtYZ*gtyzdy)/2.;
    double const GammatYyz=(gtXY*(gtxydy + gtxydz - gtyzdx) + gtYY*(gtyydy + gtyydz - gtyzdy) + gtYZ*gtyzdy)/2.;
    double const GammatYzz=(2*gtXY*gtxzdz + 2*gtYY*gtyzdz - gtXY*gtzzdx - gtYY*gtzzdy + gtYZ*gtzzdz)/2.;
    double const GammatZxx=(gtxxdx*gtXZ - gtxxdy*gtYZ + 2*gtxydx*gtYZ - gtxxdz*gtZZ + 2*gtxzdx*gtZZ)/2.;
    double const GammatZxy=((gtxxdx + gtxxdy)*gtXZ + gtxydx*(-gtXZ + gtYZ) + (-gtxydz + gtxzdx + gtxzdy)*gtZZ)/2.;
    double const GammatZxz=((gtxxdx + gtxxdz)*gtXZ + (gtxydx + gtxydz - gtxzdy)*gtYZ + gtxzdx*(-gtXZ + gtZZ))/2.;
    double const GammatZyy=(2*gtxydy*gtXZ - gtXZ*gtyydx + gtyydy*gtYZ - gtyydz*gtZZ + 2*gtyzdy*gtZZ)/2.;
    double const GammatZyz=((gtyydy + gtyydz)*gtYZ + gtXZ*(gtxydy + gtxydz - gtyzdx) + gtyzdy*(-gtYZ + gtZZ))/2.;
    double const GammatZzz=(2*gtXZ*gtxzdz + 2*gtYZ*gtyzdz - gtXZ*gtzzdx - gtYZ*gtzzdy + gtZZ*gtzzdz)/2.;

    double const Ttt= Tdd[0][0];
    double const Ttx= Tdd[0][1];
    double const Tty= Tdd[0][2];
    double const Ttz= Tdd[0][3];
    double const Txx= Tdd[1][1];
    double const Txy= Tdd[1][2];
    double const Txz= Tdd[1][3];
    double const Tyy= Tdd[2][2];
    double const Tyz= Tdd[2][3];
    double const Tzz= Tdd[3][3];
    double const EE = (Ttt - 2*betaZ*Ttz + betaX*betaX*Txx + 2*betaX*(-Ttx + betaY*Txy + betaZ*Txz) + betaY*(-2*Tty + betaY*Tyy + 2*betaZ*Tyz) + betaZ*betaZ*Tzz)/(alp*alp);
    double const Sx = Ttx/alp - (betaX*Txx)/alp - (betaY*Txy)/alp - (betaZ*Txz)/alp;
    double const Sy = Tty/alp - (betaX*Txy)/alp - (betaY*Tyy)/alp - (betaZ*Tyz)/alp;
    double const Sz = Ttz/alp - (betaX*Txz)/alp - (betaY*Tyz)/alp - (betaZ*Tzz)/alp;



    std::array<double,4> res;
    int ww=0;

    /* Hamiltonian constraint */
    res[ww] = 3*Atxx*AtXX + 6*Atxy*AtXY + 6*Atxz*AtXZ + 3*Atyy*AtYY + 6*Atyz*AtYZ + 3*Atzz*AtZZ - 3*(GammatXdx + GammatYdy + GammatZdz + GammatXxx*GammatXxx*gtXX + GammatYxy*GammatYxy*gtXX + 2*GammatXxz*GammatZxx*gtXX + 2*GammatYxz*GammatZxy*gtXX + GammatZxz*GammatZxz*gtXX + (3*gtXX*gtxxdx*gtXXdx)/2. + 2*GammatXyy*GammatYxx*gtXY + 2*GammatYxy*GammatYyy*gtXY + 2*GammatXyz*GammatZxx*gtXY + 2*GammatXxz*GammatZxy*gtXY + 2*GammatYyz*GammatZxy*gtXY + 2*GammatYxz*GammatZyy*gtXY + 2*GammatZxz*GammatZyz*gtXY + (gtXXdx*gtxxdy*gtXY)/2. + (3*gtxxdx*gtXXdy*gtXY)/2. + gtXXdx*gtXY*gtxydx + gtXX*gtxxdy*gtXYdx + 2*gtXX*gtxydx*gtXYdx + 2*gtXY*gtXYdx*gtxydy + gtxxdy*gtXY*gtXYdy + 2*gtXY*gtxydx*gtXYdy + 2*GammatXyz*GammatYxx*gtXZ + 2*GammatYxy*GammatYyz*gtXZ + 2*GammatXzz*GammatZxx*gtXZ + 2*GammatYzz*GammatZxy*gtXZ + 2*GammatXxz*GammatZxz*gtXZ + 2*GammatYxz*GammatZyz*gtXZ + 2*GammatZxz*GammatZzz*gtXZ + (gtXXdx*gtxxdz*gtXZ)/2. + (3*gtxxdx*gtXXdz*gtXZ)/2. + gtXYdx*gtxydz*gtXZ + gtxxdy*gtXYdz*gtXZ + 2*gtxydx*gtXYdz*gtXZ + 2*GammatXxx*(GammatXxy*gtXY + GammatXxz*gtXZ) + gtXXdx*gtXZ*gtxzdx + gtXX*gtxxdz*gtXZdx + gtXY*gtxydz*gtXZdx + 2*gtXX*gtxzdx*gtXZdx + gtXYdx*gtXZ*gtxzdy + gtXY*gtXZdx*gtxzdy + gtxxdz*gtXY*gtXZdy + 2*gtXY*gtxzdx*gtXZdy + 2*gtXZ*gtXZdx*gtxzdz + gtxxdz*gtXZ*gtXZdz + 2*gtXZ*gtxzdx*gtXZdz + GammatXxy*GammatXxy*gtYY + 2*GammatXyy*GammatYxy*gtYY + GammatYyy*GammatYyy*gtYY + 2*GammatXyz*GammatZxy*gtYY + 2*GammatYyz*GammatZyy*gtYY + GammatZyz*GammatZyz*gtYY + (gtxxdy*gtXXdy*gtYY)/2. + gtXXdy*gtxydx*gtYY + 2*gtxydy*gtXYdy*gtYY + gtxydz*gtXZdy*gtYY + gtxzdy*gtXZdy*gtYY + gtXY*gtXYdx*gtyydx + gtXYdy*gtYY*gtyydx + gtXX*gtxydy*gtYYdx + (gtXX*gtyydx*gtYYdx)/2. + (3*gtXY*gtYYdx*gtyydy)/2. + gtXY*gtxydy*gtYYdy + (gtXY*gtyydx*gtYYdy)/2. + (3*gtYY*gtyydy*gtYYdy)/2. + (gtXZ*gtYYdx*gtyydz)/2. + gtxydy*gtXZ*gtYYdz + (gtXZ*gtyydx*gtYYdz)/2. + 2*GammatXyz*GammatYxy*gtYZ + 2*GammatXyy*GammatYxz*gtYZ + 2*GammatYyy*GammatYyz*gtYZ + 2*GammatXzz*GammatZxy*gtYZ + 2*GammatXyz*GammatZxz*gtYZ + 2*GammatYzz*GammatZyy*gtYZ + 2*GammatYyz*GammatZyz*gtYZ + 2*GammatZyz*GammatZzz*gtYZ + (gtXXdy*gtxxdz*gtYZ)/2. + (gtxxdy*gtXXdz*gtYZ)/2. + gtXXdz*gtxydx*gtYZ + gtXYdy*gtxydz*gtYZ + 2*gtxydy*gtXYdz*gtYZ + gtXXdy*gtxzdx*gtYZ + gtXYdy*gtxzdy*gtYZ + 2*gtXZdy*gtxzdz*gtYZ + gtxydz*gtXZdz*gtYZ + gtxzdy*gtXZdz*gtYZ + gtXYdz*gtyydx*gtYZ + (gtYYdy*gtyydz*gtYZ)/2. + (3*gtyydy*gtYYdz*gtYZ)/2. + 2*GammatXxy*(GammatYxx*gtXX + GammatYxy*gtXY + GammatYxz*gtXZ + GammatXxz*gtYZ) + gtXYdx*gtXZ*gtyzdx + gtXY*gtXZdx*gtyzdx + gtXZdy*gtYY*gtyzdx + gtXYdy*gtYZ*gtyzdx + gtXZdz*gtYZ*gtyzdx + gtXX*gtxydz*gtYZdx + gtXX*gtxzdy*gtYZdx + gtXY*gtyydz*gtYZdx + gtXX*gtyzdx*gtYZdx + gtXZ*gtYYdx*gtyzdy + gtYYdy*gtYZ*gtyzdy + 2*gtXY*gtYZdx*gtyzdy + gtXY*gtxydz*gtYZdy + gtXY*gtxzdy*gtYZdy + gtYY*gtyydz*gtYZdy + gtXY*gtyzdx*gtYZdy + 2*gtYY*gtyzdy*gtYZdy + 2*gtXZ*gtYZdx*gtyzdz + 2*gtYZ*gtYZdy*gtyzdz + gtxydz*gtXZ*gtYZdz + gtXZ*gtxzdy*gtYZdz + gtyydz*gtYZ*gtYZdz + gtXZ*gtyzdx*gtYZdz + 2*gtYZ*gtyzdy*gtYZdz + GammatXxz*GammatXxz*gtZZ + 2*GammatXyz*GammatYxz*gtZZ + GammatYyz*GammatYyz*gtZZ + 2*GammatXzz*GammatZxz*gtZZ + 2*GammatYzz*GammatZyz*gtZZ + GammatZzz*GammatZzz*gtZZ + (gtxxdz*gtXXdz*gtZZ)/2. + gtxydz*gtXYdz*gtZZ + gtXXdz*gtxzdx*gtZZ + gtXYdz*gtxzdy*gtZZ + 2*gtxzdz*gtXZdz*gtZZ + (gtyydz*gtYYdz*gtZZ)/2. + gtXYdz*gtyzdx*gtZZ + gtYYdz*gtyzdy*gtZZ + 2*gtyzdz*gtYZdz*gtZZ + gtXZ*gtXZdx*gtzzdx + gtXZdy*gtYZ*gtzzdx + gtXZdz*gtZZ*gtzzdx + gtXX*gtxzdz*gtZZdx + gtXY*gtyzdz*gtZZdx + (gtXX*gtzzdx*gtZZdx)/2. + gtXZ*gtYZdx*gtzzdy + gtYZ*gtYZdy*gtzzdy + gtYZdz*gtZZ*gtzzdy + (gtXY*gtZZdx*gtzzdy)/2. + gtXY*gtxzdz*gtZZdy + gtYY*gtyzdz*gtZZdy + (gtXY*gtzzdx*gtZZdy)/2. + (gtYY*gtzzdy*gtZZdy)/2. + (3*gtXZ*gtZZdx*gtzzdz)/2. + (3*gtYZ*gtZZdy*gtzzdz)/2. + gtXZ*gtxzdz*gtZZdz + gtYZ*gtyzdz*gtZZdz + (gtXZ*gtzzdx*gtZZdz)/2. + (gtYZ*gtzzdy*gtZZdz)/2. + (3*gtZZ*gtzzdz*gtZZdz)/2.) - 2*(K*K) - 24*gtXX*(GammatXxx*phidx - phidx*phidx - phidxdx + GammatYxx*phidy + GammatZxx*phidz) - 48*gtXY*(GammatXxy*phidx - phidxdy + GammatYxy*phidy - phidx*phidy + GammatZxy*phidz) - 24*gtYY*(GammatXyy*phidx + GammatYyy*phidy - phidy*phidy - phidydy + GammatZyy*phidz) - 48*gtXZ*(GammatXxz*phidx - phidxdz + GammatYxz*phidy + GammatZxz*phidz - phidx*phidz) - 48*gtYZ*(GammatXyz*phidx + GammatYyz*phidy - phidydz + GammatZyz*phidz - phidy*phidz) - 24*gtZZ*(GammatXzz*phidx + GammatYzz*phidy + GammatZzz*phidz - phidz*phidz - phidzdz) + 48*EE*pi; ww++;

    /* Momentum constraint */
    res[ww] = Atxxdx*gtXX - 2*Atxz*GammatZxx*gtXX + Atxxdy*gtXY + Atxydx*gtXY - Atyy*GammatYxx*gtXY - Atyz*GammatZxx*gtXY - 3*Atxz*GammatZxy*gtXY + Atxxdz*gtXZ + Atxzdx*gtXZ - Atxz*GammatXxx*gtXZ - Atyz*GammatYxx*gtXZ - Atzz*GammatZxx*gtXZ - 3*Atxz*GammatZxz*gtXZ + Atxydy*gtYY - Atyy*GammatYxy*gtYY - Atyz*GammatZxy*gtYY - Atxz*GammatZyy*gtYY + Atxydz*gtYZ + Atxzdy*gtYZ - Atxz*GammatXxy*gtYZ - Atyz*GammatYxy*gtYZ - Atyy*GammatYxz*gtYZ - Atzz*GammatZxy*gtYZ - Atyz*GammatZxz*gtYZ - 2*Atxz*GammatZyz*gtYZ + Atxzdz*gtZZ - Atxz*GammatXxz*gtZZ - Atyz*GammatYxz*gtZZ - Atzz*GammatZxz*gtZZ - Atxz*GammatZzz*gtZZ - (2*Kdx)/3. + 6*Atxz*gtXZ*phidx + 6*Atxz*gtYZ*phidy + 6*Atxz*gtZZ*phidz - Atxx*(2*GammatXxx*gtXX + 3*GammatXxy*gtXY + 3*GammatXxz*gtXZ + GammatXyy*gtYY + 2*GammatXyz*gtYZ + GammatXzz*gtZZ - 6*gtXX*phidx - 6*gtXY*phidy - 6*gtXZ*phidz) - Atxy*(2*GammatYxx*gtXX + GammatXxx*gtXY + 3*GammatYxy*gtXY + 3*GammatYxz*gtXZ + GammatXxy*gtYY + GammatYyy*gtYY + GammatXxz*gtYZ + 2*GammatYyz*gtYZ + GammatYzz*gtZZ - 6*gtXY*phidx - 6*gtYY*phidy - 6*gtYZ*phidz) - 8*pi*Sx; ww++;
    res[ww] = Atxydx*gtXX - Atyy*GammatYxx*gtXX - Atyz*GammatZxx*gtXX - Atxz*GammatZxy*gtXX + Atxydy*gtXY + Atyydx*gtXY - 3*Atyy*GammatYxy*gtXY - 3*Atyz*GammatZxy*gtXY - Atxz*GammatZyy*gtXY + Atxydz*gtXZ + Atyzdx*gtXZ - Atxz*GammatXxy*gtXZ - Atyz*GammatYxy*gtXZ - 2*Atyy*GammatYxz*gtXZ - Atzz*GammatZxy*gtXZ - 2*Atyz*GammatZxz*gtXZ - Atxz*GammatZyz*gtXZ - Atxx*(GammatXxy*gtXX + GammatXyy*gtXY + GammatXyz*gtXZ) + Atyydy*gtYY - 2*Atyy*GammatYyy*gtYY - 2*Atyz*GammatZyy*gtYY + Atyydz*gtYZ + Atyzdy*gtYZ - Atxz*GammatXyy*gtYZ - Atyz*GammatYyy*gtYZ - 3*Atyy*GammatYyz*gtYZ - Atzz*GammatZyy*gtYZ - 3*Atyz*GammatZyz*gtYZ + Atyzdz*gtZZ - Atxz*GammatXyz*gtZZ - Atyz*GammatYyz*gtZZ - Atyy*GammatYzz*gtZZ - Atzz*GammatZyz*gtZZ - Atyz*GammatZzz*gtZZ - (2*Kdy)/3. + 6*Atyy*gtXY*phidx + 6*Atyz*gtXZ*phidx + 6*Atyy*gtYY*phidy + 6*Atyz*gtYZ*phidy + 6*Atyy*gtYZ*phidz + 6*Atyz*gtZZ*phidz - Atxy*(GammatXxx*gtXX + GammatYxy*gtXX + 3*GammatXxy*gtXY + GammatYyy*gtXY + 2*GammatXxz*gtXZ + GammatYyz*gtXZ + 2*GammatXyy*gtYY + 3*GammatXyz*gtYZ + GammatXzz*gtZZ - 6*gtXX*phidx - 6*gtXY*phidy - 6*gtXZ*phidz) - 8*pi*Sy; ww++;
    res[ww] = Atxzdx*gtXX - Atyz*GammatYxx*gtXX - Atxy*GammatYxz*gtXX - Atzz*GammatZxx*gtXX + Atxzdy*gtXY + Atyzdx*gtXY - Atxy*GammatXxz*gtXY - 2*Atyz*GammatYxy*gtXY - Atyy*GammatYxz*gtXY - Atxy*GammatYyz*gtXY - 2*Atzz*GammatZxy*gtXY - Atyz*GammatZxz*gtXY + Atxzdz*gtXZ + Atzzdx*gtXZ - 3*Atyz*GammatYxz*gtXZ - Atxy*GammatYzz*gtXZ - 3*Atzz*GammatZxz*gtXZ - Atxx*(GammatXxz*gtXX + GammatXyz*gtXY + GammatXzz*gtXZ) + Atyzdy*gtYY - Atxy*GammatXyz*gtYY - Atyz*GammatYyy*gtYY - Atyy*GammatYyz*gtYY - Atzz*GammatZyy*gtYY - Atyz*GammatZyz*gtYY + Atyzdz*gtYZ + Atzzdy*gtYZ - Atxy*GammatXzz*gtYZ - 3*Atyz*GammatYyz*gtYZ - Atyy*GammatYzz*gtYZ - 3*Atzz*GammatZyz*gtYZ - Atyz*GammatZzz*gtYZ + Atzzdz*gtZZ - 2*Atyz*GammatYzz*gtZZ - 2*Atzz*GammatZzz*gtZZ - (2*Kdz)/3. + 6*Atyz*gtXY*phidx + 6*Atzz*gtXZ*phidx + 6*Atyz*gtYY*phidy + 6*Atzz*gtYZ*phidy + 6*Atyz*gtYZ*phidz + 6*Atzz*gtZZ*phidz - Atxz*(GammatXxx*gtXX + GammatZxz*gtXX + 2*GammatXxy*gtXY + GammatZyz*gtXY + 3*GammatXxz*gtXZ + GammatZzz*gtXZ + GammatXyy*gtYY + 3*GammatXyz*gtYZ + 2*GammatXzz*gtZZ - 6*gtXX*phidx - 6*gtXY*phidy - 6*gtXZ*phidz) - 8*pi*Sz; ww++;

    /* All done! */
    return std::move(res);
}


#define INSTANTIATE_TEMPLATE(DER_ORD)                                \
template                                                             \
grace::bssn_state_t GRACE_HOST_DEVICE                                \
compute_bssn_rhs<DER_ORD>( VEC(int , int , int ), int                \
                , grace::var_array_t const          \
                , std::array<std::array<double,4>,4> const&          \
                , std::array<double,GRACE_NSPACEDIM> const&          \
                , double const k1, double const eta );               \
template                                                             \
std::array<double,4> GRACE_HOST_DEVICE                               \
compute_bssn_constraint_violations<DER_ORD>(                         \
                  VEC(int , int , int ), int                         \
                , grace::var_array_t  const          \
                , std::array<std::array<double,4>,4> const&          \
                , std::array<double,GRACE_NSPACEDIM> const& )

INSTANTIATE_TEMPLATE(2) ; 
INSTANTIATE_TEMPLATE(4) ; 
#undef INSTANTIATE_TEMPLATE
}