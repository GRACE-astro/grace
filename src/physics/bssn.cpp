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
// #include <grace/physics/tetrads.hh>

#include <grace/utils/numerics/fd_utils.hh>

#include <Kokkos_Core.hpp>

namespace grace {

template< size_t der_order >
bssn_state_t GRACE_HOST_DEVICE 
compute_bssn_rhs( VEC(int i, int j, int k), int q
                , grace::var_array_t<GRACE_NSPACEDIM> const state
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
    double const Sxx = 0;//Txx;
    double const Sxy = 0;//Txy;
    double const Sxz = 0;//Txz;
    double const Syy = 0;//Tyy;
    double const Syz = 0;//Tyz;
    double const Szz = 0;//Tzz;

    // momentum density components
    double const Sx = 0; //-((-Ttx + betaX*Txx + betaY*Txy + betaZ*Txz)/alp);
    double const Sy = 0; //-((-Tty + betaX*Txy + betaY*Tyy + betaZ*Tyz)/alp);
    double const Sz = 0; //-((-Ttz + betaX*Txz + betaY*Tyz + betaZ*Tzz)/alp);

    // trace of the spatial energy momentum tensor
    double const S = 0 ; // (gtXX*Sxx + 2*gtXY*Sxy + 2*gtXZ*Sxz + gtYY*Syy + 2*gtYZ*Syz + gtZZ*Szz)*exp(-4.*phi);

    // energy density
    double const EE = 0 ; // (Ttt - 2*betaY*Tty - 2*betaZ*Ttz + betaX*betaX*Txx + 2*betaX*(-Ttx + betaY*Txy + betaZ*Txz) + betaY*betaY*Tyy + 2*betaY*betaZ*Tyz + betaZ*betaZ*Tzz)/(alp*alp);

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
    double const DDalp=alpdxdx*gtXX - alpdy*GammatYxx*gtXX - alpdz*GammatZxx*gtXX + 2*alpdxdy*gtXY - 2*alpdy*GammatYxy*gtXY - 2*alpdz*GammatZxy*gtXY + 2*alpdxdz*gtXZ - 2*alpdy*GammatYxz*gtXZ - 2*alpdz*GammatZxz*gtXZ + alpdydy*gtYY - alpdy*GammatYyy*gtYY - alpdz*GammatZyy*gtYY + 2*alpdydz*gtYZ - 2*alpdy*GammatYyz*gtYZ - 2*alpdz*GammatZyz*gtYZ + alpdzdz*gtZZ - alpdy*GammatYzz*gtZZ - alpdz*GammatZzz*gtZZ + 4*alpdy*gtXY*phidx + 4*alpdz*gtXZ*phidx + 4*alpdy*gtYY*phidy + 4*alpdz*gtYZ*phidy + 4*alpdy*gtYZ*phidz + 4*alpdz*gtZZ*phidz - alpdx*(GammatXxx*gtXX + 2*GammatXxy*gtXY + 2*GammatXxz*gtXZ + GammatXyy*gtYY + 2*GammatXyz*gtYZ + GammatXzz*gtZZ - 4*gtXX*phidx - 4*gtXY*phidy - 4*gtXZ*phidz);

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
    double const Rt = GammatXdx + GammatYdy + GammatZdz + GammatXxx*GammatXxx*gtXX + GammatYxy*GammatYxy*gtXX + 2*GammatXxz*GammatZxx*gtXX + 2*GammatYxz*GammatZxy*gtXX + GammatZxz*GammatZxz*gtXX + (3*gtXX*gtxxdx*gtXXdx)/2. + 2*GammatXyy*GammatYxx*gtXY + 2*GammatYxy*GammatYyy*gtXY + 2*GammatXyz*GammatZxx*gtXY + 2*GammatXxz*GammatZxy*gtXY + 2*GammatYyz*GammatZxy*gtXY + 2*GammatYxz*GammatZyy*gtXY + 2*GammatZxz*GammatZyz*gtXY + (gtXXdx*gtxxdy*gtXY)/2. + (3*gtxxdx*gtXXdy*gtXY)/2. + gtXXdx*gtXY*gtxydx + gtXX*gtxxdy*gtXYdx + 2*gtXX*gtxydx*gtXYdx + 2*gtXY*gtXYdx*gtxydy + gtxxdy*gtXY*gtXYdy + 2*gtXY*gtxydx*gtXYdy + 2*GammatXyz*GammatYxx*gtXZ + 2*GammatYxy*GammatYyz*gtXZ + 2*GammatXzz*GammatZxx*gtXZ + 2*GammatYzz*GammatZxy*gtXZ + 2*GammatXxz*GammatZxz*gtXZ + 2*GammatYxz*GammatZyz*gtXZ + 2*GammatZxz*GammatZzz*gtXZ + (gtXXdx*gtxxdz*gtXZ)/2. + (3*gtxxdx*gtXXdz*gtXZ)/2. + gtXYdx*gtxydz*gtXZ + gtxxdy*gtXYdz*gtXZ + 2*gtxydx*gtXYdz*gtXZ + 2*GammatXxx*(GammatXxy*gtXY + GammatXxz*gtXZ) + gtXXdx*gtXZ*gtxzdx + gtXX*gtxxdz*gtXZdx + gtXY*gtxydz*gtXZdx + 2*gtXX*gtxzdx*gtXZdx + gtXYdx*gtXZ*gtxzdy + gtXY*gtXZdx*gtxzdy + gtxxdz*gtXY*gtXZdy + 2*gtXY*gtxzdx*gtXZdy + 2*gtXZ*gtXZdx*gtxzdz + gtxxdz*gtXZ*gtXZdz + 2*gtXZ*gtxzdx*gtXZdz + GammatXxy*GammatXxy*gtYY + 2*GammatXyy*GammatYxy*gtYY + GammatYyy*GammatYyy*gtYY + 2*GammatXyz*GammatZxy*gtYY + 2*GammatYyz*GammatZyy*gtYY + GammatZyz*GammatZyz*gtYY + (gtxxdy*gtXXdy*gtYY)/2. + gtXXdy*gtxydx*gtYY + 2*gtxydy*gtXYdy*gtYY + gtxydz*gtXZdy*gtYY + gtxzdy*gtXZdy*gtYY + gtXY*gtXYdx*gtyydx + gtXYdy*gtYY*gtyydx + gtXX*gtxydy*gtYYdx + (gtXX*gtyydx*gtYYdx)/2. + (3*gtXY*gtYYdx*gtyydy)/2. + gtXY*gtxydy*gtYYdy + (gtXY*gtyydx*gtYYdy)/2. + (3*gtYY*gtyydy*gtYYdy)/2. + (gtXZ*gtYYdx*gtyydz)/2. + gtxydy*gtXZ*gtYYdz + (gtXZ*gtyydx*gtYYdz)/2. + 2*GammatXyz*GammatYxy*gtYZ + 2*GammatXyy*GammatYxz*gtYZ + 2*GammatYyy*GammatYyz*gtYZ + 2*GammatXzz*GammatZxy*gtYZ + 2*GammatXyz*GammatZxz*gtYZ + 2*GammatYzz*GammatZyy*gtYZ + 2*GammatYyz*GammatZyz*gtYZ + 2*GammatZyz*GammatZzz*gtYZ + (gtXXdy*gtxxdz*gtYZ)/2. + (gtxxdy*gtXXdz*gtYZ)/2. + gtXXdz*gtxydx*gtYZ + gtXYdy*gtxydz*gtYZ + 2*gtxydy*gtXYdz*gtYZ + gtXXdy*gtxzdx*gtYZ + gtXYdy*gtxzdy*gtYZ + 2*gtXZdy*gtxzdz*gtYZ + gtxydz*gtXZdz*gtYZ + gtxzdy*gtXZdz*gtYZ + gtXYdz*gtyydx*gtYZ + (gtYYdy*gtyydz*gtYZ)/2. + (3*gtyydy*gtYYdz*gtYZ)/2. + 2*GammatXxy*(GammatYxx*gtXX + GammatYxy*gtXY + GammatYxz*gtXZ + GammatXxz*gtYZ) + gtXYdx*gtXZ*gtyzdx + gtXY*gtXZdx*gtyzdx + gtXZdy*gtYY*gtyzdx + gtXYdy*gtYZ*gtyzdx + gtXZdz*gtYZ*gtyzdx + gtXX*gtxydz*gtYZdx + gtXX*gtxzdy*gtYZdx + gtXY*gtyydz*gtYZdx + gtXX*gtyzdx*gtYZdx + gtXZ*gtYYdx*gtyzdy + gtYYdy*gtYZ*gtyzdy + 2*gtXY*gtYZdx*gtyzdy + gtXY*gtxydz*gtYZdy + gtXY*gtxzdy*gtYZdy + gtYY*gtyydz*gtYZdy + gtXY*gtyzdx*gtYZdy + 2*gtYY*gtyzdy*gtYZdy + 2*gtXZ*gtYZdx*gtyzdz + 2*gtYZ*gtYZdy*gtyzdz + gtxydz*gtXZ*gtYZdz + gtXZ*gtxzdy*gtYZdz + gtyydz*gtYZ*gtYZdz + gtXZ*gtyzdx*gtYZdz + 2*gtYZ*gtyzdy*gtYZdz + GammatXxz*GammatXxz*gtZZ + 2*GammatXyz*GammatYxz*gtZZ + GammatYyz*GammatYyz*gtZZ + 2*GammatXzz*GammatZxz*gtZZ + 2*GammatYzz*GammatZyz*gtZZ + GammatZzz*GammatZzz*gtZZ + (gtxxdz*gtXXdz*gtZZ)/2. + gtxydz*gtXYdz*gtZZ + gtXXdz*gtxzdx*gtZZ + gtXYdz*gtxzdy*gtZZ + 2*gtxzdz*gtXZdz*gtZZ + (gtyydz*gtYYdz*gtZZ)/2. + gtXYdz*gtyzdx*gtZZ + gtYYdz*gtyzdy*gtZZ + 2*gtyzdz*gtYZdz*gtZZ + gtXZ*gtXZdx*gtzzdx + gtXZdy*gtYZ*gtzzdx + gtXZdz*gtZZ*gtzzdx + gtXX*gtxzdz*gtZZdx + gtXY*gtyzdz*gtZZdx + (gtXX*gtzzdx*gtZZdx)/2. + gtXZ*gtYZdx*gtzzdy + gtYZ*gtYZdy*gtzzdy + gtYZdz*gtZZ*gtzzdy + (gtXY*gtZZdx*gtzzdy)/2. + gtXY*gtxzdz*gtZZdy + gtYY*gtyzdz*gtZZdy + (gtXY*gtzzdx*gtZZdy)/2. + (gtYY*gtzzdy*gtZZdy)/2. + (3*gtXZ*gtZZdx*gtzzdz)/2. + (3*gtYZ*gtZZdy*gtzzdz)/2. + gtXZ*gtxzdz*gtZZdz + gtYZ*gtyzdz*gtZZdz + (gtXZ*gtzzdx*gtZZdz)/2. + (gtYZ*gtzzdy*gtZZdz)/2. + (3*gtZZ*gtzzdz*gtZZdz)/2.;

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
    res[GTXXL+0] = -2*alp*Atxx - (2*(-2*betaXdx + betaYdy + betaZdz)*gtxx)/3. + betaX*gtxxdx + betaY*gtxxdy + betaZ*gtxxdz + 2*betaYdx*gtxy + 2*betaZdx*gtxz;
    res[GTXXL+1] = -2*alp*Atxy + betaXdy*gtxx + ((betaXdx + betaYdy - 2*betaZdz)*gtxy)/3. + betaX*gtxydx + betaY*gtxydy + betaZ*gtxydz + betaZdy*gtxz + betaYdx*gtyy + betaZdx*gtyz;
    res[GTXXL+2] = -2*alp*Atxz + betaXdz*gtxx + betaYdz*gtxy + (betaXdx*gtxz)/3. - (2*betaYdy*gtxz)/3. + (betaZdz*gtxz)/3. + betaX*gtxzdx + betaY*gtxzdy + betaZ*gtxzdz + betaYdx*gtyz + betaZdx*gtzz;
    res[GTXXL+3] = -2*alp*Atyy + 2*betaXdy*gtxy - (2*(betaXdx - 2*betaYdy + betaZdz)*gtyy)/3. + betaX*gtyydx + betaY*gtyydy + betaZ*gtyydz + 2*betaZdy*gtyz;
    res[GTXXL+4] = -2*alp*Atyz + betaXdz*gtxy + betaXdy*gtxz + betaYdz*gtyy - (2*betaXdx*gtyz)/3. + (betaYdy*gtyz)/3. + (betaZdz*gtyz)/3. + betaX*gtyzdx + betaY*gtyzdy + betaZ*gtyzdz + betaZdy*gtzz;
    res[GTXXL+5] = -2*alp*Atzz + 2*betaXdz*gtxz + 2*betaYdz*gtyz + 2*betaZdz*gtzz - (2*(betaXdx + betaYdy + betaZdz)*gtzz)/3. + betaX*gtzzdx + betaY*gtzzdy + betaZ*gtzzdz;

    // BSSN equation for conformal traceless extrinsic curvature d/dt Atij, note that Dti beta^j =d/dxi beta^j because of unit determinant of conformal metric 
    res[ATXXL+0] = Atxxdx*betaX + 2*Atxx*betaXdx + Atxxdy*betaY + 2*Atxy*betaYdx + Atxxdz*betaZ + 2*Atxz*betaZdx - (2*Atxx*(betaXdx + betaYdy + betaZdz))/3. - alp*(2*(Atxx*Atxx)*gtXX + 4*Atxx*Atxy*gtXY + 4*Atxx*Atxz*gtXZ + 2*(Atxy*Atxy)*gtYY + 4*Atxy*Atxz*gtYZ + 2*(Atxz*Atxz)*gtZZ - Atxx*K) + ((-3*DiDjalpxx + DDalp*gtxx + 3*alp*RTFxx + 8*alp*pi*((-3 + gtxx*gtXX)*Sxx + gtxx*(2*gtXY*Sxy + 2*gtXZ*Sxz + gtYY*Syy + 2*gtYZ*Syz + gtZZ*Szz)))*exp(-4.*phi))/3.;
    res[ATXXL+1] = Atxydx*betaX + Atxy*betaXdx + Atxx*betaXdy + Atxydy*betaY + Atyy*betaYdx + Atxy*betaYdy + Atxydz*betaZ + Atyz*betaZdx + Atxz*betaZdy - (2*Atxy*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*(Atxy*Atxy*gtXY + Atxy*Atxz*gtXZ + Atxx*(Atxy*gtXX + Atyy*gtXY + Atyz*gtXZ) + Atxy*Atyy*gtYY + Atxz*Atyy*gtYZ + Atxy*Atyz*gtYZ + Atxz*Atyz*gtZZ) + Atxy*K) + (-DiDjalpxy + (DDalp*gtxy)/3. + alp*RTFxy - 8*alp*pi*(Sxy - (gtxy*(gtXX*Sxx + 2*gtXY*Sxy + 2*gtXZ*Sxz + gtYY*Syy + 2*gtYZ*Syz + gtZZ*Szz))/3.))*exp(-4.*phi);
    res[ATXXL+2] = Atxzdx*betaX + Atxz*betaXdx + Atxx*betaXdz + Atxzdy*betaY + Atyz*betaYdx + Atxy*betaYdz + Atxzdz*betaZ + Atzz*betaZdx + Atxz*betaZdz - (2*Atxz*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*(Atxx*(Atxz*gtXX + Atyz*gtXY + Atzz*gtXZ) + Atxy*(Atxz*gtXY + Atyz*gtYY + Atzz*gtYZ) + Atxz*(Atxz*gtXZ + Atyz*gtYZ + Atzz*gtZZ)) + Atxz*K) + (-DiDjalpxz + (DDalp*gtxz)/3. + alp*RTFxz - 8*alp*pi*(Sxz - (gtxz*(gtXX*Sxx + 2*gtXY*Sxy + 2*gtXZ*Sxz + gtYY*Syy + 2*gtYZ*Syz + gtZZ*Szz))/3.))*exp(-4.*phi);
    res[ATXXL+3] = Atyydx*betaX + 2*Atxy*betaXdy + Atyydy*betaY + 2*Atyy*betaYdy + Atyydz*betaZ + 2*Atyz*betaZdy - (2*Atyy*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*(Atxy*Atxy*gtXX + 2*Atxy*(Atyy*gtXY + Atyz*gtXZ) + Atyy*Atyy*gtYY + 2*Atyy*Atyz*gtYZ + Atyz*Atyz*gtZZ) + Atyy*K) + (-DiDjalpyy + (DDalp*gtyy)/3. + alp*RTFyy - 8*alp*pi*(Syy - (gtyy*(gtXX*Sxx + 2*gtXY*Sxy + 2*gtXZ*Sxz + gtYY*Syy + 2*gtYZ*Syz + gtZZ*Szz))/3.))*exp(-4.*phi);
    res[ATXXL+4] = Atyzdx*betaX + Atxz*betaXdy + Atxy*betaXdz + Atyzdy*betaY + Atyz*betaYdy + Atyy*betaYdz + Atyzdz*betaZ + Atzz*betaZdy + Atyz*betaZdz - (2*Atyz*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*(Atxz*Atyy*gtXY + Atxz*Atyz*gtXZ + Atxy*(Atxz*gtXX + Atyz*gtXY + Atzz*gtXZ) + Atyy*Atyz*gtYY + Atyz*Atyz*gtYZ + Atyy*Atzz*gtYZ + Atyz*Atzz*gtZZ) + Atyz*K) + (-DiDjalpyz + (DDalp*gtyz)/3. + alp*RTFyz - 8*alp*pi*(Syz - (gtyz*(gtXX*Sxx + 2*gtXY*Sxy + 2*gtXZ*Sxz + gtYY*Syy + 2*gtYZ*Syz + gtZZ*Szz))/3.))*exp(-4.*phi);
    res[ATXXL+5] = Atzzdx*betaX + 2*Atxz*betaXdz + Atzzdy*betaY + 2*Atyz*betaYdz + Atzzdz*betaZ + 2*Atzz*betaZdz - (2*Atzz*(betaXdx + betaYdy + betaZdz))/3. + alp*(-2*(Atxz*Atxz*gtXX + 2*Atxz*(Atyz*gtXY + Atzz*gtXZ) + Atyz*Atyz*gtYY + 2*Atyz*Atzz*gtYZ + Atzz*Atzz*gtZZ) + Atzz*K) + (-DiDjalpzz + (DDalp*gtzz)/3. + alp*RTFzz - 8*alp*pi*(Szz - (gtzz*(gtXX*Sxx + 2*gtXY*Sxy + 2*gtXZ*Sxz + gtYY*Syy + 2*gtYZ*Syz + gtZZ*Szz))/3.))*exp(-4.*phi);

    // BSSN equation for conformal factor d/dt phi
    res[PHIL] = (betaXdx + betaYdy + betaZdz - alp*K + 6*betaX*phidx + 6*betaY*phidy + 6*betaZ*phidz)/6.;

    // BSSN equation for conformal extrinsic curvature trace d/dt K
    res[KL] = betaX*Kdx + betaY*Kdy + betaZ*Kdz + alp*(Atxx*AtXX + 2*Atxy*AtXY + 2*Atxz*AtXZ + Atyy*AtYY + 2*Atyz*AtYZ + Atzz*AtZZ + (K*K)/3. + 4*pi*(EE + S)) + (-(alpdxdx*gtXX) + alpdy*GammatYxx*gtXX + alpdz*GammatZxx*gtXX - 2*alpdxdy*gtXY + 2*alpdy*GammatYxy*gtXY + 2*alpdz*GammatZxy*gtXY - 2*alpdxdz*gtXZ + 2*alpdy*GammatYxz*gtXZ + 2*alpdz*GammatZxz*gtXZ - alpdydy*gtYY + alpdy*GammatYyy*gtYY + alpdz*GammatZyy*gtYY - 2*alpdydz*gtYZ + 2*alpdy*GammatYyz*gtYZ + 2*alpdz*GammatZyz*gtYZ - alpdzdz*gtZZ + alpdy*GammatYzz*gtZZ + alpdz*GammatZzz*gtZZ - 4*alpdy*gtXY*phidx - 4*alpdz*gtXZ*phidx - 4*alpdy*gtYY*phidy - 4*alpdz*gtYZ*phidy - 4*alpdy*gtYZ*phidz - 4*alpdz*gtZZ*phidz + alpdx*(GammatXxx*gtXX + 2*GammatXxy*gtXY + 2*GammatXxz*gtXZ + GammatXyy*gtYY + 2*GammatXyz*gtYZ + GammatXzz*gtZZ - 4*gtXX*phidx - 4*gtXY*phidy - 4*gtXZ*phidz))*exp(-4.*phi);

    // BSSN equation for contracted conformal Christoffels d/dt Gammatu
    res[GAMMAXL+0] = (-6*alpdx*AtXX - 6*alpdy*AtXY - 6*alpdz*AtXZ - betaXdx*GammatX + 2*betaYdy*GammatX + 2*betaZdz*GammatX + 3*betaX*GammatXdx + 3*betaY*GammatXdy + 3*betaZ*GammatXdz - 3*betaXdy*GammatY - 3*betaXdz*GammatZ + 4*betaXdxdx*gtXX + betaYdxdy*gtXX + betaZdxdz*gtXX + 7*betaXdxdy*gtXY + betaYdydy*gtXY + betaZdydz*gtXY + 7*betaXdxdz*gtXZ + betaYdydz*gtXZ + betaZdzdz*gtXZ + 3*betaXdydy*gtYY + 6*betaXdydz*gtYZ + 3*betaXdzdz*gtZZ + 2*alp*(6*AtXZ*GammatXxz + 3*AtYY*GammatXyy + 6*AtYZ*GammatXyz + 3*AtZZ*GammatXzz - 2*gtXX*Kdx - 2*gtXY*Kdy - 2*gtXZ*Kdz + 3*AtXX*(GammatXxx + 6*phidx) + 6*AtXY*(GammatXxy + 3*phidy) + 18*AtXZ*phidz - 24*gtXX*pi*Sx - 24*gtXY*pi*Sy - 24*gtXZ*pi*Sz))/3.;
    res[GAMMAXL+1] = (-6*alpdx*AtXY - 6*alpdy*AtYY - 6*alpdz*AtYZ - 3*betaYdx*GammatX + 2*betaXdx*GammatY - betaYdy*GammatY + 2*betaZdz*GammatY + 3*betaX*GammatYdx + 3*betaY*GammatYdy + 3*betaZ*GammatYdz - 3*betaYdz*GammatZ + 3*betaYdxdx*gtXX + betaXdxdx*gtXY + 7*betaYdxdy*gtXY + betaZdxdz*gtXY + 6*betaYdxdz*gtXZ + betaXdxdy*gtYY + 4*betaYdydy*gtYY + betaZdydz*gtYY + betaXdxdz*gtYZ + 7*betaYdydz*gtYZ + betaZdzdz*gtYZ + 3*betaYdzdz*gtZZ + 2*alp*(3*AtXX*GammatYxx + 6*AtXZ*GammatYxz + 3*AtYY*GammatYyy + 6*AtYZ*GammatYyz + 3*AtZZ*GammatYzz - 2*gtXY*Kdx - 2*gtYY*Kdy - 2*gtYZ*Kdz + 6*AtXY*(GammatYxy + 3*phidx) + 18*AtYY*phidy + 18*AtYZ*phidz - 24*gtXY*pi*Sx - 24*gtYY*pi*Sy - 24*gtYZ*pi*Sz))/3.;
    res[GAMMAXL+2] = (-6*alpdx*AtXZ - 6*alpdy*AtYZ - 6*alpdz*AtZZ - 3*betaZdx*GammatX - 3*betaZdy*GammatY + 2*betaXdx*GammatZ + 2*betaYdy*GammatZ - betaZdz*GammatZ + 3*betaX*GammatZdx + 3*betaY*GammatZdy + 3*betaZ*GammatZdz + 3*betaZdxdx*gtXX + 6*betaZdxdy*gtXY + betaXdxdx*gtXZ + betaYdxdy*gtXZ + 7*betaZdxdz*gtXZ + 3*betaZdydy*gtYY + betaXdxdy*gtYZ + betaYdydy*gtYZ + 7*betaZdydz*gtYZ + betaXdxdz*gtZZ + betaYdydz*gtZZ + 4*betaZdzdz*gtZZ + 2*alp*(3*AtXX*GammatZxx + 6*AtXY*GammatZxy + 6*AtXZ*GammatZxz + 3*AtYY*GammatZyy + 6*AtYZ*GammatZyz + 3*AtZZ*GammatZzz - 2*gtXZ*Kdx - 2*gtYZ*Kdy - 2*gtZZ*Kdz + 18*AtXZ*phidx + 18*AtYZ*phidy + 18*AtZZ*phidz - 24*gtXZ*pi*Sx - 24*gtYZ*pi*Sy - 24*gtZZ*pi*Sz))/3.;

    double const GammatXdt = res[GAMMAXL+0] ;
    double const GammatYdt = res[GAMMAXL+1] ;
    double const GammatZdt = res[GAMMAXL+2] ;

    /* 1 + log slicing condition */
    res[ALPL] = alpdx*betaX + alpdy*betaY + alpdz*betaZ - 2*alp*K;

    /* Gamma driver */
    double const BX = state(VEC(i,j,k),BX_+0,q);
    double const BY = state(VEC(i,j,k),BX_+1,q);
    double const BZ = state(VEC(i,j,k),BX_+2,q);

    res[BETAXL+0] = BX*k1;
    res[BETAXL+1] = BY*k1;
    res[BETAXL+2] = BZ*k1;

    res[BXL+0] = -(BX*eta) + GammatXdt;
    res[BXL+1] = -(BY*eta) + GammatYdt;
    res[BXL+2] = -(BZ*eta) + GammatZdt;

    return std::move(res);
}

template< size_t der_order >
std::array<double,4> GRACE_HOST_DEVICE 
compute_bssn_constraint_violations(
      VEC(int i, int j, int k), int q
    , grace::var_array_t<GRACE_NSPACEDIM> const state
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

/**
 * @brief Return the psi4 pseudo-scalar
 * 
 * @tparam der_order 
 * @param q 
 * @param state 
 * @param idx 
 * @return std::array<double,2> real and imaginary part of the psi4 
 */
template< size_t der_order >
std::array<double,2> GRACE_HOST_DEVICE 
compute_psi4(
      VEC(int i, int j, int k), int q
    , grace::coord_array_t<GRACE_NSPACEDIM> const pcoords
    , grace::var_array_t<GRACE_NSPACEDIM> const state
    , std::array<double,GRACE_NSPACEDIM> const& idx
)
{
    double const x1=pcoords(VEC(i,j,k),0,q);
    double const x2=pcoords(VEC(i,j,k),1,q);
    double const x3=pcoords(VEC(i,j,k),2,q);
    
    double const n0 = 1.0/Kokkos::sqrt(2.0);
    
    // recovering the state variables and their derivatives 
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

    // real part of n vector
    double const nXre = (-(gtyy*x1*(x1*x1 + x2*x2)) + x2*(gtxy*(x1*x1 + x2*x2) - gtyz*x1*x3 + gtxz*x2*x3))/(sqrt(2)*(gtyy*(x1*x1) + x2*(-2*gtxy*x1 + gtxx*x2))*sqrt(((-(gtxy*gtxy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2))) - 2*gtxy*x3*(gtyz*x1*(x1*x1 + x2*x2) + gtxz*x2*(x1*x1 + x2*x2) + gtzz*x1*x2*x3) + x3*((-(gtyz*gtyz) + gtyy*gtzz)*(x1*x1)*x3 - gtxz*gtxz*(x2*x2)*x3 + 2*gtxz*x1*(gtyy*(x1*x1 + x2*x2) + gtyz*x2*x3)) + gtxx*(gtyy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x2*x3*(2*gtyz*(x1*x1 + x2*x2) + gtzz*x2*x3)))*exp(4.*phi))/(gtyy*(x1*x1) + x2*(-2*gtxy*x1 + gtxx*x2))));
    double const nYre = (gtxy*x1*(x1*x1 + x2*x2) - gtxx*x2*(x1*x1 + x2*x2) + x1*(gtyz*x1 - gtxz*x2)*x3)/(sqrt(2)*(gtyy*(x1*x1) + x2*(-2*gtxy*x1 + gtxx*x2))*sqrt(((-(gtxy*gtxy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2))) - 2*gtxy*x3*(gtyz*x1*(x1*x1 + x2*x2) + gtxz*x2*(x1*x1 + x2*x2) + gtzz*x1*x2*x3) + x3*((-(gtyz*gtyz) + gtyy*gtzz)*(x1*x1)*x3 - gtxz*gtxz*(x2*x2)*x3 + 2*gtxz*x1*(gtyy*(x1*x1 + x2*x2) + gtyz*x2*x3)) + gtxx*(gtyy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x2*x3*(2*gtyz*(x1*x1 + x2*x2) + gtzz*x2*x3)))*exp(4.*phi))/(gtyy*(x1*x1) + x2*(-2*gtxy*x1 + gtxx*x2))));
    double const nZre = -(x3/(sqrt(2)*sqrt(((-(gtxy*gtxy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2))) - 2*gtxy*x3*(gtyz*x1*(x1*x1 + x2*x2) + gtxz*x2*(x1*x1 + x2*x2) + gtzz*x1*x2*x3) + x3*((-(gtyz*gtyz) + gtyy*gtzz)*(x1*x1)*x3 - gtxz*gtxz*(x2*x2)*x3 + 2*gtxz*x1*(gtyy*(x1*x1 + x2*x2) + gtyz*x2*x3)) + gtxx*(gtyy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x2*x3*(2*gtyz*(x1*x1 + x2*x2) + gtzz*x2*x3)))*exp(4.*phi))/(gtyy*(x1*x1) + x2*(-2*gtxy*x1 + gtxx*x2)))));
    // imaginary part of n vector
    double const nXim = 0;
    double const nYim = 0;
    double const nZim = 0;
    // real part of m vector
    double const mXre = (sqrt(-(gtxz*gtxz*gtyy) + 2*gtxy*gtxz*gtyz - gtxx*(gtyz*gtyz) - gtxy*gtxy*gtzz + gtxx*gtyy*gtzz)*((-(gtyz*gtyz) + gtyy*gtzz)*x1*x3 + gtxz*(gtyy*(x1*x1 + x2*x2) + gtyz*x2*x3) - gtxy*(gtyz*(x1*x1 + x2*x2) + gtzz*x2*x3))*sqrt(-(((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*((gtZZ*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x3*(-2*gtXZ*x1*(x1*x1 + x2*x2) - 2*gtYZ*x2*(x1*x1 + x2*x2) + (gtXX*(x1*x1) + 2*gtXY*x1*x2 + gtYY*(x2*x2))*x3))*(gtZZ*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x3*(-2*gtXZ*x1*(x1*x1 + x2*x2) - 2*gtYZ*x2*(x1*x1 + x2*x2) + (gtXX*(x1*x1) + 2*gtXY*x1*x2 + gtYY*(x2*x2))*x3))))/(gtxy*gtxy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + 2*gtxy*x3*(gtyz*x1*(x1*x1 + x2*x2) + gtxz*x2*(x1*x1 + x2*x2) + gtzz*x1*x2*x3) + x3*((gtyz*gtyz - gtyy*gtzz)*(x1*x1)*x3 + gtxz*gtxz*(x2*x2)*x3 - 2*gtxz*x1*(gtyy*(x1*x1 + x2*x2) + gtyz*x2*x3)) - gtxx*(gtyy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x2*x3*(2*gtyz*(x1*x1 + x2*x2) + gtzz*x2*x3)))))*exp(-2.*phi))/(sqrt(2)*((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)))*(gtZZ*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x3*(-2*gtXZ*x1*(x1*x1 + x2*x2) - 2*gtYZ*x2*(x1*x1 + x2*x2) + (gtXX*(x1*x1) + 2*gtXY*x1*x2 + gtYY*(x2*x2))*x3)));
    double const mYre = -((sqrt(-(gtxz*gtxz*gtyy) + 2*gtxy*gtxz*gtyz - gtxx*(gtyz*gtyz) - gtxy*gtxy*gtzz + gtxx*gtyy*gtzz)*(gtxz*(-(gtyz*x1) + gtxz*x2)*x3 + gtxy*(gtxz*(x1*x1 + x2*x2) + gtzz*x1*x3) - gtxx*(gtyz*(x1*x1 + x2*x2) + gtzz*x2*x3))*sqrt(-(((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*((gtZZ*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x3*(-2*gtXZ*x1*(x1*x1 + x2*x2) - 2*gtYZ*x2*(x1*x1 + x2*x2) + (gtXX*(x1*x1) + 2*gtXY*x1*x2 + gtYY*(x2*x2))*x3))*(gtZZ*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x3*(-2*gtXZ*x1*(x1*x1 + x2*x2) - 2*gtYZ*x2*(x1*x1 + x2*x2) + (gtXX*(x1*x1) + 2*gtXY*x1*x2 + gtYY*(x2*x2))*x3))))/(gtxy*gtxy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + 2*gtxy*x3*(gtyz*x1*(x1*x1 + x2*x2) + gtxz*x2*(x1*x1 + x2*x2) + gtzz*x1*x2*x3) + x3*((gtyz*gtyz - gtyy*gtzz)*(x1*x1)*x3 + gtxz*gtxz*(x2*x2)*x3 - 2*gtxz*x1*(gtyy*(x1*x1 + x2*x2) + gtyz*x2*x3)) - gtxx*(gtyy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x2*x3*(2*gtyz*(x1*x1 + x2*x2) + gtzz*x2*x3)))))*exp(-2.*phi))/(sqrt(2)*((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)))*(gtZZ*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x3*(-2*gtXZ*x1*(x1*x1 + x2*x2) - 2*gtYZ*x2*(x1*x1 + x2*x2) + (gtXX*(x1*x1) + 2*gtXY*x1*x2 + gtYY*(x2*x2))*x3))));
    double const mZre = -((sqrt(-(gtxz*gtxz*gtyy) + 2*gtxy*gtxz*gtyz - gtxx*(gtyz*gtyz) - gtxy*gtxy*gtzz + gtxx*gtyy*gtzz)*(-(gtxy*gtxy*(x1*x1 + x2*x2)) + gtxx*gtyy*(x1*x1 + x2*x2) + gtxz*gtyy*x1*x3 + gtxx*gtyz*x2*x3 - gtxy*(gtyz*x1 + gtxz*x2)*x3)*sqrt(-(((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*((gtZZ*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x3*(-2*gtXZ*x1*(x1*x1 + x2*x2) - 2*gtYZ*x2*(x1*x1 + x2*x2) + (gtXX*(x1*x1) + 2*gtXY*x1*x2 + gtYY*(x2*x2))*x3))*(gtZZ*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x3*(-2*gtXZ*x1*(x1*x1 + x2*x2) - 2*gtYZ*x2*(x1*x1 + x2*x2) + (gtXX*(x1*x1) + 2*gtXY*x1*x2 + gtYY*(x2*x2))*x3))))/(gtxy*gtxy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + 2*gtxy*x3*(gtyz*x1*(x1*x1 + x2*x2) + gtxz*x2*(x1*x1 + x2*x2) + gtzz*x1*x2*x3) + x3*((gtyz*gtyz - gtyy*gtzz)*(x1*x1)*x3 + gtxz*gtxz*(x2*x2)*x3 - 2*gtxz*x1*(gtyy*(x1*x1 + x2*x2) + gtyz*x2*x3)) - gtxx*(gtyy*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x2*x3*(2*gtyz*(x1*x1 + x2*x2) + gtzz*x2*x3)))))*exp(-2.*phi))/(sqrt(2)*((gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz))*(gtxz*gtxz*gtyy - 2*gtxy*gtxz*gtyz + gtxy*gtxy*gtzz + gtxx*(gtyz*gtyz - gtyy*gtzz)))*(gtZZ*((x1*x1 + x2*x2)*(x1*x1 + x2*x2)) + x3*(-2*gtXZ*x1*(x1*x1 + x2*x2) - 2*gtYZ*x2*(x1*x1 + x2*x2) + (gtXX*(x1*x1) + 2*gtXY*x1*x2 + gtYY*(x2*x2))*x3))));
    // imaginary part of m vector
    double const mXim = (x2*exp(-2.*phi))/(sqrt(2)*sqrt(gtyy*(x1*x1) - 2*gtxy*x1*x2 + gtxx*(x2*x2)));
    double const mYim = -((x1*exp(-2.*phi))/(sqrt(2)*sqrt(gtyy*(x1*x1) - 2*gtxy*x1*x2 + gtxx*(x2*x2))));
    double const mZim = 0;

    // extrinsic curvature Kdd built from traceles conformal Atdd
    double const Kxx=((3*Atxx + gtxx*K)*exp(4.*phi))/3.;
    double const Kxy=((3*Atxy + gtxy*K)*exp(4.*phi))/3.;
    double const Kxz=((3*Atxz + gtxz*K)*exp(4.*phi))/3.;
    double const Kyy=((3*Atyy + gtyy*K)*exp(4.*phi))/3.;
    double const Kyz=((3*Atyz + gtyz*K)*exp(4.*phi))/3.;
    double const Kzz=((3*Atzz + gtzz*K)*exp(4.*phi))/3.;

    // extrinsic curvature Kud with first index raised
    double const KXx=(gtXX*Kxx + gtXY*Kxy + gtXZ*Kxz)*exp(-4.*phi);
    double const KXy=(gtXX*Kxy + gtXY*Kyy + gtXZ*Kyz)*exp(-4.*phi);
    double const KXz=(gtXX*Kxz + gtXY*Kyz + gtXZ*Kzz)*exp(-4.*phi);
    double const KYy=(gtXY*Kxy + gtYY*Kyy + gtYZ*Kyz)*exp(-4.*phi);
    double const KYz=(gtXY*Kxz + gtYY*Kyz + gtYZ*Kzz)*exp(-4.*phi);
    double const KZz=(gtXZ*Kxz + gtYZ*Kyz + gtZZ*Kzz)*exp(-4.*phi);

    // derivative of Kdd
    double const Kxxdx=((3*Atxxdx + gtxxdx*K + gtxx*Kdx + 12*Atxx*phidx + 4*gtxx*K*phidx)*exp(4.*phi))/3.;
    double const Kxxdy=((3*Atxxdy + gtxxdy*K + gtxx*Kdy + 12*Atxx*phidy + 4*gtxx*K*phidy)*exp(4.*phi))/3.;
    double const Kxxdz=((3*Atxxdz + gtxxdz*K + gtxx*Kdz + 12*Atxx*phidz + 4*gtxx*K*phidz)*exp(4.*phi))/3.;
    double const Kxydx=((3*Atxydx + gtxydx*K + gtxy*Kdx + 12*Atxy*phidx + 4*gtxy*K*phidx)*exp(4.*phi))/3.;
    double const Kxydy=((3*Atxydy + gtxydy*K + gtxy*Kdy + 12*Atxy*phidy + 4*gtxy*K*phidy)*exp(4.*phi))/3.;
    double const Kxydz=((3*Atxydz + gtxydz*K + gtxy*Kdz + 12*Atxy*phidz + 4*gtxy*K*phidz)*exp(4.*phi))/3.;
    double const Kxzdx=((3*Atxzdx + gtxzdx*K + gtxz*Kdx + 12*Atxz*phidx + 4*gtxz*K*phidx)*exp(4.*phi))/3.;
    double const Kxzdy=((3*Atxzdy + gtxzdy*K + gtxz*Kdy + 12*Atxz*phidy + 4*gtxz*K*phidy)*exp(4.*phi))/3.;
    double const Kxzdz=((3*Atxzdz + gtxzdz*K + gtxz*Kdz + 12*Atxz*phidz + 4*gtxz*K*phidz)*exp(4.*phi))/3.;
    double const Kyydx=((3*Atyydx + gtyydx*K + gtyy*Kdx + 12*Atyy*phidx + 4*gtyy*K*phidx)*exp(4.*phi))/3.;
    double const Kyydy=((3*Atyydy + gtyydy*K + gtyy*Kdy + 12*Atyy*phidy + 4*gtyy*K*phidy)*exp(4.*phi))/3.;
    double const Kyydz=((3*Atyydz + gtyydz*K + gtyy*Kdz + 12*Atyy*phidz + 4*gtyy*K*phidz)*exp(4.*phi))/3.;
    double const Kyzdx=((3*Atyzdx + gtyzdx*K + gtyz*Kdx + 12*Atyz*phidx + 4*gtyz*K*phidx)*exp(4.*phi))/3.;
    double const Kyzdy=((3*Atyzdy + gtyzdy*K + gtyz*Kdy + 12*Atyz*phidy + 4*gtyz*K*phidy)*exp(4.*phi))/3.;
    double const Kyzdz=((3*Atyzdz + gtyzdz*K + gtyz*Kdz + 12*Atyz*phidz + 4*gtyz*K*phidz)*exp(4.*phi))/3.;
    double const Kzzdx=((3*Atzzdx + gtzzdx*K + gtzz*Kdx + 12*Atzz*phidx + 4*gtzz*K*phidx)*exp(4.*phi))/3.;
    double const Kzzdy=((3*Atzzdy + gtzzdy*K + gtzz*Kdy + 12*Atzz*phidy + 4*gtzz*K*phidy)*exp(4.*phi))/3.;
    double const Kzzdz=((3*Atzzdz + gtzzdz*K + gtzz*Kdz + 12*Atzz*phidz + 4*gtzz*K*phidz)*exp(4.*phi))/3.;
    // transformation matrix from conformal to original Christoffels: \
    C=Gamma-Gammat
    double const CXxx=-2*(-2*phidx + gtxx*gtXX*phidx + gtxx*gtXY*phidy + gtxx*gtXZ*phidz);
    double const CXxy=-2*(gtXX*gtxy*phidx + (-1 + gtxy*gtXY)*phidy + gtxy*gtXZ*phidz);
    double const CXxz=-2*(gtXX*gtxz*phidx + gtXY*gtxz*phidy + (-1 + gtxz*gtXZ)*phidz);
    double const CXyy=-2*gtyy*(gtXX*phidx + gtXY*phidy + gtXZ*phidz);
    double const CXyz=-2*gtyz*(gtXX*phidx + gtXY*phidy + gtXZ*phidz);
    double const CXzz=-2*gtzz*(gtXX*phidx + gtXY*phidy + gtXZ*phidz);
    double const CYxx=-2*gtxx*(gtXY*phidx + gtYY*phidy + gtYZ*phidz);
    double const CYxy=-2*(-phidx + gtxy*gtXY*phidx + gtxy*gtYY*phidy + gtxy*gtYZ*phidz);
    double const CYxz=-2*gtxz*(gtXY*phidx + gtYY*phidy + gtYZ*phidz);
    double const CYyy=-2*(gtXY*gtyy*phidx + (-2 + gtyy*gtYY)*phidy + gtyy*gtYZ*phidz);
    double const CYyz=-2*(gtXY*gtyz*phidx + gtYY*gtyz*phidy + (-1 + gtyz*gtYZ)*phidz);
    double const CYzz=-2*gtzz*(gtXY*phidx + gtYY*phidy + gtYZ*phidz);
    double const CZxx=-2*gtxx*(gtXZ*phidx + gtYZ*phidy + gtZZ*phidz);
    double const CZxy=-2*gtxy*(gtXZ*phidx + gtYZ*phidy + gtZZ*phidz);
    double const CZxz=-2*(-phidx + gtxz*gtXZ*phidx + gtxz*gtYZ*phidy + gtxz*gtZZ*phidz);
    double const CZyy=-2*gtyy*(gtXZ*phidx + gtYZ*phidy + gtZZ*phidz);
    double const CZyz=-2*(gtXZ*gtyz*phidx + (-1 + gtyz*gtYZ)*phidy + gtyz*gtZZ*phidz);
    double const CZzz=-2*(gtXZ*gtzz*phidx + gtYZ*gtzz*phidy + (-2 + gtzz*gtZZ)*phidz);

    // original Christoffels: Gamma=Gammat+C
    double const GammaXxx=CXxx + GammatXxx;
    double const GammaXxy=CXxy + GammatXxy;
    double const GammaXxz=CXxz + GammatXxz;
    double const GammaXyy=CXyy + GammatXyy;
    double const GammaXyz=CXyz + GammatXyz;
    double const GammaXzz=CXzz + GammatXzz;
    double const GammaYxx=CYxx + GammatYxx;
    double const GammaYxy=CYxy + GammatYxy;
    double const GammaYxz=CYxz + GammatYxz;
    double const GammaYyy=CYyy + GammatYyy;
    double const GammaYyz=CYyz + GammatYyz;
    double const GammaYzz=CYzz + GammatYzz;
    double const GammaZxx=CZxx + GammatZxx;
    double const GammaZxy=CZxy + GammatZxy;
    double const GammaZxz=CZxz + GammatZxz;
    double const GammaZyy=CZyy + GammatZyy;
    double const GammaZyz=CZyz + GammatZyz;
    double const GammaZzz=CZzz + GammatZzz;

    // linearly independent components of Riemann
    double const Rxyxy=(-2*gtXYdy*(gtyydx + 4*gtyy*phidx) + 8*gtXY*(gtyydx + 4*gtyy*phidx)*phidy - 2*gtXXdy*(gtxxdy + 4*gtxx*phidy) + 8*gtXX*phidy*(gtxxdy + 4*gtxx*phidy) + 8*gtXX*phidx*(-2*gtxydy + gtyydx + 4*gtyy*phidx - 8*gtxy*phidy) + 2*gtXXdx*(2*gtxydy - gtyydx - 4*gtyy*phidx + 8*gtxy*phidy) + 2*gtXYdx*(gtyydy + 4*gtyy*phidy) - 8*gtXY*phidx*(gtyydy + 4*gtyy*phidy) + 2*gtXX*(2*gtxydxdy - gtyydxdx + 8*gtxydy*phidx - 8*gtyydx*phidx - 16*gtyy*(phidx*phidx) - 4*gtyy*phidxdx + 8*gtxy*phidxdy + 8*gtxydx*phidy + 32*gtxy*phidx*phidy) - 2*gtXX*(gtxxdydy + 8*gtxxdy*phidy + 4*gtxx*(4*(phidy*phidy) + phidydy)) + 8*gtXZ*phidy*(-gtxydz + gtxzdy + gtyzdx + 4*gtyz*phidx + 4*gtxz*phidy - 4*gtxy*phidz) + 2*gtXZdy*(gtxydz - gtxzdy - gtyzdx - 4*gtyz*phidx - 4*gtxz*phidy + 4*gtxy*phidz) - (-(gtxydz*gtXZ) + gtXZ*gtxzdy + gtXY*gtyydx + gtXZ*gtyzdx + 4*gtXY*gtyy*phidx + 4*gtXZ*gtyz*phidx + 4*gtxz*gtXZ*phidy + gtXX*(gtxxdy + 4*gtxx*phidy) - 4*gtxy*gtXZ*phidz)*(-(gtxydz*gtXZ) + gtXZ*gtxzdy + gtXY*gtyydx + gtXZ*gtyzdx + 4*gtXY*gtyy*phidx + 4*gtXZ*gtyz*phidx + 4*gtxz*gtXZ*phidy + gtXX*(gtxxdy + 4*gtxx*phidy) - 4*gtxy*gtXZ*phidz) - 2*gtXZdx*(gtyydz - 2*gtyzdy - 8*gtyz*phidy + 4*gtyy*phidz) + 8*gtXZ*phidx*(gtyydz - 2*gtyzdy - 8*gtyz*phidy + 4*gtyy*phidz) + 2*gtXZ*(gtxydydz - gtxzdydy - gtyzdxdy - 4*gtyzdy*phidx - 4*gtyz*phidxdy + 4*gtxydz*phidy - 8*gtxzdy*phidy - 4*gtyzdx*phidy - 16*gtyz*phidx*phidy - 16*gtxz*(phidy*phidy) - 4*gtxz*phidydy + 4*gtxy*phidydz + 4*gtxydy*phidz + 16*gtxy*phidy*phidz) + (2*gtxydy*gtXZ + gtyydy*gtYZ - gtyydz*gtZZ + 2*gtyzdy*gtZZ + 4*gtyy*gtYZ*phidy + 8*gtyz*gtZZ*phidy - gtXZ*(gtyydx + 4*gtyy*phidx - 8*gtxy*phidy) - 4*gtyy*gtZZ*phidz)*(gtXZ*(gtzzdx + 4*gtzz*phidx) + gtXX*(gtxxdz + 4*gtxx*phidz) + gtXY*(gtxydz - gtxzdy + gtyzdx + 4*gtyz*phidx - 4*gtxz*phidy + 4*gtxy*phidz)) - (gtxxdy*gtXZ + gtyydx*gtYZ - gtxydz*gtZZ + gtxzdy*gtZZ + gtyzdx*gtZZ + 4*gtyy*gtYZ*phidx + 4*gtyz*gtZZ*phidx + 4*gtxx*gtXZ*phidy + 4*gtxz*gtZZ*phidy - 4*gtxy*gtZZ*phidz)*(gtXZ*(gtzzdy + 4*gtzz*phidy) + gtXX*(gtxydz + gtxzdy - gtyzdx - 4*gtyz*phidx + 4*gtxz*phidy + 4*gtxy*phidz) + gtXY*(gtyydz + 4*gtyy*phidz)) - (gtxxdy*gtXY - gtxydz*gtYZ + gtxzdy*gtYZ + gtYZ*gtyzdx + 4*gtyz*gtYZ*phidx + gtYY*(gtyydx + 4*gtyy*phidx) + 4*gtxx*gtXY*phidy + 4*gtxz*gtYZ*phidy - 4*gtxy*gtYZ*phidz)*(gtXX*(2*gtxydy - gtyydx - 4*gtyy*phidx + 8*gtxy*phidy) + gtXY*(gtyydy + 4*gtyy*phidy) - gtXZ*(gtyydz - 2*gtyzdy - 8*gtyz*phidy + 4*gtyy*phidz)) - (-(gtxxdy*gtXY) + 2*gtXY*gtxydx - gtxxdz*gtXZ + 2*gtXZ*gtxzdx + 8*gtxy*gtXY*phidx + 8*gtxz*gtXZ*phidx + gtXX*(gtxxdx + 4*gtxx*phidx) - 4*gtxx*gtXY*phidy - 4*gtxx*gtXZ*phidz)*(gtXX*(-2*gtxydy + gtyydx + 4*gtyy*phidx - 8*gtxy*phidy) - gtXY*(gtyydy + 4*gtyy*phidy) + gtXZ*(gtyydz - 2*gtyzdy - 8*gtyz*phidy + 4*gtyy*phidz)) - (-(gtxydz*gtXZ) + gtXZ*gtxzdy + gtXY*gtyydx + gtXZ*gtyzdx + 4*gtXY*gtyy*phidx + 4*gtXZ*gtyz*phidx + 4*gtxz*gtXZ*phidy + gtXX*(gtxxdy + 4*gtxx*phidy) - 4*gtxy*gtXZ*phidz)*(gtXY*(-2*gtxydy + gtyydx + 4*gtyy*phidx - 8*gtxy*phidy) - gtYY*(gtyydy + 4*gtyy*phidy) + gtYZ*(gtyydz - 2*gtyzdy - 8*gtyz*phidy + 4*gtyy*phidz)) - 2*gtXZ*(gtyydxdz - 2*gtyzdxdy + 4*(gtyydz*phidx - 2*gtyzdy*phidx - 2*gtyz*phidxdy + gtyy*phidxdz - 2*gtyzdx*phidy - 8*gtyz*phidx*phidy + gtyydx*phidz + 4*gtyy*phidx*phidz)))/4.;
    double const Rxzxz=(-2*gtXZdz*(gtzzdx + 4*gtzz*phidx) + 8*gtXZ*(gtzzdx + 4*gtzz*phidx)*phidz - 2*gtXXdz*(gtxxdz + 4*gtxx*phidz) + 8*gtXX*phidz*(gtxxdz + 4*gtxx*phidz) - 2*gtXYdz*(gtxydz - gtxzdy + gtyzdx + 4*gtyz*phidx - 4*gtxz*phidy + 4*gtxy*phidz) + 8*gtXY*phidz*(gtxydz - gtxzdy + gtyzdx + 4*gtyz*phidx - 4*gtxz*phidy + 4*gtxy*phidz) + 8*gtXX*phidx*(-2*gtxzdz + gtzzdx + 4*gtzz*phidx - 8*gtxz*phidz) + 2*gtXXdx*(2*gtxzdz - gtzzdx - 4*gtzz*phidx + 8*gtxz*phidz) + 2*gtXYdx*(2*gtyzdz - gtzzdy - 4*gtzz*phidy + 8*gtyz*phidz) - 8*gtXY*phidx*(2*gtyzdz - gtzzdy - 4*gtzz*phidy + 8*gtyz*phidz) + 2*gtXZdx*(gtzzdz + 4*gtzz*phidz) - 8*gtXZ*phidx*(gtzzdz + 4*gtzz*phidz) + 2*gtXX*(2*gtxzdxdz - gtzzdxdx + 8*gtxzdz*phidx - 8*gtzzdx*phidx - 16*gtzz*(phidx*phidx) - 4*gtzz*phidxdx + 8*gtxz*phidxdz + 8*gtxzdx*phidz + 32*gtxz*phidx*phidz) + 2*gtXY*(2*gtyzdxdz - gtzzdxdy + 8*gtyzdz*phidx - 4*gtzzdy*phidx - 4*gtzz*phidxdy + 8*gtyz*phidxdz - 4*gtzzdx*phidy - 16*gtzz*phidx*phidy + 8*gtyzdx*phidz + 32*gtyz*phidx*phidz) - (gtXZ*(gtzzdx + 4*gtzz*phidx) + gtXX*(gtxxdz + 4*gtxx*phidz) + gtXY*(gtxydz - gtxzdy + gtyzdx + 4*gtyz*phidx - 4*gtxz*phidy + 4*gtxy*phidz))*(gtXZ*(gtzzdx + 4*gtzz*phidx) + gtXX*(gtxxdz + 4*gtxx*phidz) + gtXY*(gtxydz - gtxzdy + gtyzdx + 4*gtyz*phidx - 4*gtxz*phidy + 4*gtxy*phidz)) - (gtxxdz*gtXY + gtxydz*gtYY - gtxzdy*gtYY + gtYY*gtyzdx + gtYZ*gtzzdx + 4*gtYY*gtyz*phidx + 4*gtYZ*gtzz*phidx - 4*gtxz*gtYY*phidy + 4*gtxx*gtXY*phidz + 4*gtxy*gtYY*phidz)*(gtXZ*(gtzzdy + 4*gtzz*phidy) + gtXX*(gtxydz + gtxzdy - gtyzdx - 4*gtyz*phidx + 4*gtxz*phidy + 4*gtxy*phidz) + gtXY*(gtyydz + 4*gtyy*phidz)) + (gtxxdy*gtXY - 2*gtXY*gtxydx + gtxxdz*gtXZ - 2*gtXZ*gtxzdx - 8*gtxy*gtXY*phidx - 8*gtxz*gtXZ*phidx - gtXX*(gtxxdx + 4*gtxx*phidx) + 4*gtxx*gtXY*phidy + 4*gtxx*gtXZ*phidz)*(gtXX*(-2*gtxzdz + gtzzdx + 4*gtzz*phidx - 8*gtxz*phidz) + gtXY*(-2*gtyzdz + gtzzdy + 4*gtzz*phidy - 8*gtyz*phidz) - gtXZ*(gtzzdz + 4*gtzz*phidz)) - (gtxxdz*gtXZ + gtxydz*gtYZ - gtxzdy*gtYZ + gtYZ*gtyzdx + gtZZ*gtzzdx + 4*gtyz*gtYZ*phidx + 4*gtzz*gtZZ*phidx - 4*gtxz*gtYZ*phidy + 4*gtxx*gtXZ*phidz + 4*gtxy*gtYZ*phidz)*(gtXX*(2*gtxzdz - gtzzdx - 4*gtzz*phidx + 8*gtxz*phidz) + gtXY*(2*gtyzdz - gtzzdy - 4*gtzz*phidy + 8*gtyz*phidz) + gtXZ*(gtzzdz + 4*gtzz*phidz)) - (-(gtxydz*gtXZ) + gtXZ*gtxzdy + gtXY*gtyydx + gtXZ*gtyzdx + 4*gtXY*gtyy*phidx + 4*gtXZ*gtyz*phidx + 4*gtxz*gtXZ*phidy + gtXX*(gtxxdy + 4*gtxx*phidy) - 4*gtxy*gtXZ*phidz)*(gtXY*(-2*gtxzdz + gtzzdx + 4*gtzz*phidx - 8*gtxz*phidz) + gtYY*(-2*gtyzdz + gtzzdy + 4*gtzz*phidy - 8*gtyz*phidz) - gtYZ*(gtzzdz + 4*gtzz*phidz)) - (gtXZ*(gtzzdx + 4*gtzz*phidx) + gtXX*(gtxxdz + 4*gtxx*phidz) + gtXY*(gtxydz - gtxzdy + gtyzdx + 4*gtyz*phidx - 4*gtxz*phidy + 4*gtxy*phidz))*(gtXZ*(-2*gtxzdz + gtzzdx + 4*gtzz*phidx - 8*gtxz*phidz) + gtYZ*(-2*gtyzdz + gtzzdy + 4*gtzz*phidy - 8*gtyz*phidz) - gtZZ*(gtzzdz + 4*gtzz*phidz)) - 2*gtXY*(gtxydzdz - gtxzdydz + gtyzdxdz + 4*gtyzdz*phidx + 4*gtyz*phidxdz - 4*gtxzdz*phidy - 4*gtxz*phidydz + 8*gtxydz*phidz - 4*gtxzdy*phidz + 4*gtyzdx*phidz + 16*gtyz*phidx*phidz - 16*gtxz*phidy*phidz + 16*gtxy*(phidz*phidz) + 4*gtxy*phidzdz) - 2*gtXX*(gtxxdzdz + 8*gtxxdz*phidz + 4*gtxx*(4*(phidz*phidz) + phidzdz)))/4.;
    double const Ryzyz=(-2*gtxydz*gtXYdz - 2*gtXYdz*gtxzdy + 4*gtXYdy*gtxzdz + 2*gtXZ*gtxzdz*gtYY*gtyydz - gtYY*gtYY*(gtyydz*gtyydz) - 2*gtyydz*gtYYdz - 2*gtYY*gtyydzdz + 2*gtXYdz*gtyzdx + 4*gtYY*gtyzdydz - 2*gtxydz*gtXZ*gtYY*gtyzdz - 2*gtXZ*gtxzdy*gtYY*gtyzdz + 2*(gtYY*gtYY)*gtyydy*gtyzdz + 4*gtYYdy*gtyzdz - 2*gtYY*gtyydz*gtYZ*gtyzdz + 2*gtXZ*gtYY*gtyzdx*gtyzdz + 4*gtYY*gtYZ*gtyzdy*gtyzdz - 2*gtXYdy*gtzzdx - gtXZ*gtYY*gtyydz*gtzzdx + 2*gtXZ*gtxzdy*gtYY*gtzzdy - gtYY*gtYY*gtyydy*gtzzdy - 2*gtYYdy*gtzzdy + 2*gtXZ*gtxzdz*gtYZ*gtzzdy - gtYY*gtyydz*gtYZ*gtzzdy - 2*gtXZ*gtYY*gtyzdx*gtzzdy - 2*gtYY*gtYZ*gtyzdy*gtzzdy + 2*(gtYZ*gtYZ)*gtyzdz*gtzzdy - 2*gtYZdz*gtzzdy - 2*gtYY*gtyzdz*gtZZ*gtzzdy - 2*gtXZ*gtYZ*gtzzdx*gtzzdy - 2*(gtYZ*gtYZ)*(gtzzdy*gtzzdy) + gtYY*gtZZ*(gtzzdy*gtzzdy) - 2*gtYY*gtzzdydy + gtXZ*gtYY*gtyydx*gtzzdz - 2*gtxydz*gtXZ*gtYZ*gtzzdz + gtYY*gtyydy*gtYZ*gtzzdz - 2*gtyydz*(gtYZ*gtYZ)*gtzzdz + 2*gtXZ*gtYZ*gtyzdx*gtzzdz + 2*(gtYZ*gtYZ)*gtyzdy*gtzzdz + 2*gtYZdy*gtzzdz + gtYY*gtyydz*gtZZ*gtzzdz + 8*gtXYdz*gtyz*phidx + 8*gtXZ*gtYY*gtyz*gtyzdz*phidx - 8*gtXYdy*gtzz*phidx - 4*gtXZ*gtYY*gtyydz*gtzz*phidx - 8*gtXZ*gtYY*gtyz*gtzzdy*phidx - 8*gtXZ*gtYZ*gtzz*gtzzdy*phidx + 4*gtXZ*gtyy*gtYY*gtzzdz*phidx + 8*gtXZ*gtyz*gtYZ*gtzzdz*phidx - 8*gtXYdz*gtxz*phidy - 8*gtxz*gtXZ*gtYY*gtyzdz*phidy + 8*gtyy*(gtYY*gtYY)*gtyzdz*phidy + 16*gtYY*gtyz*gtYZ*gtyzdz*phidy + 8*gtXZ*gtxzdy*gtYY*gtzz*phidy - 4*(gtYY*gtYY)*gtyydy*gtzz*phidy - 8*gtYYdy*gtzz*phidy + 8*gtXZ*gtxzdz*gtYZ*gtzz*phidy - 4*gtYY*gtyydz*gtYZ*gtzz*phidy - 8*gtXZ*gtYY*gtyzdx*gtzz*phidy - 8*gtYY*gtYZ*gtyzdy*gtzz*phidy + 8*(gtYZ*gtYZ)*gtyzdz*gtzz*phidy - 8*gtYZdz*gtzz*phidy - 8*gtYY*gtyzdz*gtzz*gtZZ*phidy - 8*gtXZ*gtYZ*gtzz*gtzzdx*phidy - 8*gtYY*gtzzdy*phidy + 8*gtxz*gtXZ*gtYY*gtzzdy*phidy - 4*gtyy*(gtYY*gtYY)*gtzzdy*phidy - 8*gtYY*gtyz*gtYZ*gtzzdy*phidy - 16*(gtYZ*gtYZ)*gtzz*gtzzdy*phidy + 8*gtYY*gtzz*gtZZ*gtzzdy*phidy - 8*gtYZ*gtzzdz*phidy + 4*gtyy*gtYY*gtYZ*gtzzdz*phidy + 8*gtyz*(gtYZ*gtYZ)*gtzzdz*phidy - 32*gtXZ*gtYY*gtyz*gtzz*phidx*phidy - 32*gtXZ*gtYZ*(gtzz*gtzz)*phidx*phidy + 32*gtxz*gtXZ*gtYY*gtzz*(phidy*phidy) - 16*gtyy*(gtYY*gtYY)*gtzz*(phidy*phidy) - 32*gtYY*gtyz*gtYZ*gtzz*(phidy*phidy) - 32*(gtYZ*gtYZ)*(gtzz*gtzz)*(phidy*phidy) + 16*gtYY*(gtzz*gtzz)*gtZZ*(phidy*phidy) - 8*gtYY*gtzz*phidydy + 16*gtYY*gtyz*phidydz - 8*gtxy*gtXYdz*phidz + 16*gtXYdy*gtxz*phidz + 8*gtXZ*gtxzdz*gtyy*gtYY*phidz - 8*gtYY*gtyydz*phidz + 8*gtxz*gtXZ*gtYY*gtyydz*phidz - 8*gtyy*(gtYY*gtYY)*gtyydz*phidz - 8*gtyy*gtYYdz*phidz - 8*gtxydz*gtXZ*gtYY*gtyz*phidz - 8*gtXZ*gtxzdy*gtYY*gtyz*phidz + 8*(gtYY*gtYY)*gtyydy*gtyz*phidz + 16*gtYYdy*gtyz*phidz - 8*gtYY*gtyydz*gtyz*gtYZ*phidz + 8*gtXZ*gtYY*gtyz*gtyzdx*phidz + 16*gtYY*gtyzdy*phidz + 16*gtYY*gtyz*gtYZ*gtyzdy*phidz - 8*gtxy*gtXZ*gtYY*gtyzdz*phidz - 8*gtyy*gtYY*gtYZ*gtyzdz*phidz + 4*gtXZ*gtYY*gtyydx*gtzz*phidz - 8*gtxydz*gtXZ*gtYZ*gtzz*phidz + 4*gtYY*gtyydy*gtYZ*gtzz*phidz - 8*gtyydz*(gtYZ*gtYZ)*gtzz*phidz + 8*gtXZ*gtYZ*gtyzdx*gtzz*phidz + 8*(gtYZ*gtYZ)*gtyzdy*gtzz*phidz + 8*gtYZdy*gtzz*phidz + 4*gtYY*gtyydz*gtzz*gtZZ*phidz - 4*gtXZ*gtyy*gtYY*gtzzdx*phidz + 8*gtYZ*gtzzdy*phidz + 8*gtxz*gtXZ*gtYZ*gtzzdy*phidz - 4*gtyy*gtYY*gtYZ*gtzzdy*phidz + 8*gtyz*(gtYZ*gtYZ)*gtzzdy*phidz - 8*gtYY*gtyz*gtZZ*gtzzdy*phidz - 8*gtxy*gtXZ*gtYZ*gtzzdz*phidz - 8*gtyy*(gtYZ*gtYZ)*gtzzdz*phidz + 4*gtyy*gtYY*gtZZ*gtzzdz*phidz + 32*gtXZ*gtYY*(gtyz*gtyz)*phidx*phidz + 32*gtXZ*gtyz*gtYZ*gtzz*phidx*phidz - 32*gtxz*gtXZ*gtYY*gtyz*phidy*phidz + 32*gtyy*(gtYY*gtYY)*gtyz*phidy*phidz + 64*gtYY*(gtyz*gtyz)*gtYZ*phidy*phidz + 32*gtxz*gtXZ*gtYZ*gtzz*phidy*phidz + 64*gtyz*(gtYZ*gtYZ)*gtzz*phidy*phidz - 32*gtYY*gtyz*gtzz*gtZZ*phidy*phidz + 32*gtxz*gtXZ*gtyy*gtYY*(phidz*phidz) - 16*(gtyy*gtyy)*(gtYY*gtYY)*(phidz*phidz) - 32*gtxy*gtXZ*gtYY*gtyz*(phidz*phidz) - 32*gtyy*gtYY*gtyz*gtYZ*(phidz*phidz) - 32*gtxy*gtXZ*gtYZ*gtzz*(phidz*phidz) - 32*gtyy*(gtYZ*gtYZ)*gtzz*(phidz*phidz) + 16*gtyy*gtYY*gtzz*gtZZ*(phidz*phidz) - gtXY*gtXY*(gtxydz*gtxydz + gtxzdy*gtxzdy - 4*gtxydy*gtxzdz + 2*gtxzdz*gtyydx + gtxxdz*gtyydz + gtyzdx*gtyzdx - 2*gtxxdy*gtyzdz + 2*gtxydy*gtzzdx - gtyydx*gtzzdx + gtxxdy*gtzzdy + 8*gtxzdz*gtyy*phidx + 8*gtyz*gtyzdx*phidx + 8*gtxydy*gtzz*phidx - 4*gtyydx*gtzz*phidx - 4*gtyy*gtzzdx*phidx + 16*(gtyz*gtyz)*(phidx*phidx) - 16*gtyy*gtzz*(phidx*phidx) - 16*gtxy*gtxzdz*phidy - 8*gtxz*gtyzdx*phidy - 8*gtxx*gtyzdz*phidy + 4*gtxxdy*gtzz*phidy + 8*gtxy*gtzzdx*phidy + 4*gtxx*gtzzdy*phidy - 32*gtxz*gtyz*phidx*phidy + 32*gtxy*gtzz*phidx*phidy + 16*(gtxz*gtxz)*(phidy*phidy) + 16*gtxx*gtzz*(phidy*phidy) - 16*gtxydy*gtxz*phidz + 4*gtxxdz*gtyy*phidz + 8*gtxz*gtyydx*phidz + 4*gtxx*gtyydz*phidz - 8*gtxxdy*gtyz*phidz - 8*gtxy*gtyzdx*phidz + 32*gtxz*gtyy*phidx*phidz - 32*gtxy*gtyz*phidx*phidz - 32*gtxy*gtxz*phidy*phidz - 32*gtxx*gtyz*phidy*phidz + 16*(gtxy*gtxy)*(phidz*phidz) + 16*gtxx*gtyy*(phidz*phidz) - 2*gtxzdy*(gtyzdx + 4*gtyz*phidx - 4*gtxz*phidy - 4*gtxy*phidz) + 2*gtxydz*(gtxzdy - gtyzdx - 4*gtyz*phidx + 4*gtxz*phidy + 4*gtxy*phidz)) - gtXX*(gtxydz*gtxydz*gtYY - gtxzdy*gtxzdy*gtYY - 2*gtxzdz*gtYY*gtyydx + 2*gtxydz*gtxzdz*gtYZ - 2*gtxzdy*gtxzdz*gtYZ + 2*gtxzdy*gtYY*gtyzdx - 2*gtxzdz*gtYZ*gtyzdx - gtYY*(gtyzdx*gtyzdx) + gtYY*gtyydx*gtzzdx + 2*gtxzdy*gtYZ*gtzzdx - 8*gtxzdz*gtyy*gtYY*phidx + 8*gtxzdy*gtYY*gtyz*phidx - 8*gtxzdz*gtyz*gtYZ*phidx - 8*gtYY*gtyz*gtyzdx*phidx + 4*gtYY*gtyydx*gtzz*phidx + 8*gtxzdy*gtYZ*gtzz*phidx + 4*gtyy*gtYY*gtzzdx*phidx - 16*gtYY*(gtyz*gtyz)*(phidx*phidx) + 16*gtyy*gtYY*gtzz*(phidx*phidx) - 8*gtxx*gtXY*gtxzdz*phidy - 8*gtxz*gtxzdy*gtYY*phidy - 8*gtxz*gtxzdz*gtYZ*phidy + 8*gtxz*gtYY*gtyzdx*phidy + 4*gtxx*gtXY*gtzzdx*phidy + 8*gtxz*gtYZ*gtzzdx*phidy + 32*gtxz*gtYY*gtyz*phidx*phidy + 16*gtxx*gtXY*gtzz*phidx*phidy + 32*gtxz*gtYZ*gtzz*phidx*phidy - 16*(gtxz*gtxz)*gtYY*(phidy*phidy) + 4*gtxx*gtXY*gtxydz*phidz + 4*gtxx*gtXY*gtxzdy*phidz + 8*gtxy*gtxydz*gtYY*phidz - 8*gtxz*gtYY*gtyydx*phidz + 8*gtxydz*gtxz*gtYZ*phidz - 8*gtxz*gtxzdy*gtYZ*phidz + 8*gtxy*gtxzdz*gtYZ*phidz - 4*gtxx*gtXY*gtyzdx*phidz - 8*gtxz*gtYZ*gtyzdx*phidz - 32*gtxz*gtyy*gtYY*phidx*phidz - 16*gtxx*gtXY*gtyz*phidx*phidz - 32*gtxz*gtyz*gtYZ*phidx*phidz - 16*gtxx*gtXY*gtxz*phidy*phidz - 32*(gtxz*gtxz)*gtYZ*phidy*phidz + 16*gtxx*gtxy*gtXY*(phidz*phidz) + 16*(gtxy*gtxy)*gtYY*(phidz*phidz) + 32*gtxy*gtxz*gtYZ*(phidz*phidz) + gtxxdz*gtXY*(gtxydz + gtxzdy - gtyzdx - 4*gtyz*phidx + 4*gtxz*phidy + 4*gtxy*phidz) + gtxxdy*gtXY*(-2*gtxzdz + gtzzdx + 4*gtzz*phidx - 8*gtxz*phidz)) - 8*gtyy*gtYY*phidzdz + gtXY*(-2*gtxydzdz + 2*gtxzdydz + 2*gtxzdz*gtYY*gtyydy - 3*gtxydz*gtYY*gtyydz - gtxzdy*gtYY*gtyydz - 4*gtxzdz*gtyydz*gtYZ + gtYY*gtyydz*gtyzdx + 2*gtyzdxdz + 4*gtxzdz*gtYZ*gtyzdy + 4*gtxydy*gtYY*gtyzdz + 4*gtxzdy*gtYZ*gtyzdz - gtYY*gtyydy*gtzzdx + gtyydz*gtYZ*gtzzdx - 2*gtYZ*gtyzdy*gtzzdx - 2*gtzzdxdy - gtxxdz*gtXZ*gtzzdy - 2*gtxydy*gtYY*gtzzdy - 2*gtxydz*gtYZ*gtzzdy - 4*gtxzdy*gtYZ*gtzzdy + 2*gtYZ*gtyzdx*gtzzdy - 2*gtxzdz*gtZZ*gtzzdy + gtZZ*gtzzdx*gtzzdy + gtxxdy*gtXZ*gtzzdz + 2*gtxydy*gtYZ*gtzzdz - gtyydx*gtYZ*gtzzdz + gtxydz*gtZZ*gtzzdz + gtxzdy*gtZZ*gtzzdz - gtyzdx*gtZZ*gtzzdz + 4*gtYY*gtyydz*gtyz*phidx + 8*gtyzdz*phidx - 4*gtYY*gtyydy*gtzz*phidx + 4*gtyydz*gtYZ*gtzz*phidx - 8*gtYZ*gtyzdy*gtzz*phidx - 8*gtzzdy*phidx + 8*gtyz*gtYZ*gtzzdy*phidx + 4*gtzz*gtZZ*gtzzdy*phidx - 4*gtyy*gtYZ*gtzzdz*phidx - 4*gtyz*gtZZ*gtzzdz*phidx - 8*gtzz*phidxdy + 8*gtyz*phidxdz - 8*gtxzdz*phidy + 8*gtxzdz*gtyy*gtYY*phidy - 4*gtxz*gtYY*gtyydz*phidy + 16*gtxzdz*gtyz*gtYZ*phidy + 16*gtxy*gtYY*gtyzdz*phidy + 16*gtxz*gtYZ*gtyzdz*phidy - 4*gtxxdz*gtXZ*gtzz*phidy - 8*gtxydy*gtYY*gtzz*phidy - 8*gtxydz*gtYZ*gtzz*phidy - 16*gtxzdy*gtYZ*gtzz*phidy + 8*gtYZ*gtyzdx*gtzz*phidy - 8*gtxzdz*gtzz*gtZZ*phidy - 4*gtyy*gtYY*gtzzdx*phidy - 8*gtyz*gtYZ*gtzzdx*phidy + 4*gtzz*gtZZ*gtzzdx*phidy - 8*gtxy*gtYY*gtzzdy*phidy - 16*gtxz*gtYZ*gtzzdy*phidy + 4*gtxx*gtXZ*gtzzdz*phidy + 8*gtxy*gtYZ*gtzzdz*phidy + 4*gtxz*gtZZ*gtzzdz*phidy - 16*gtyy*gtYY*gtzz*phidx*phidy + 16*(gtzz*gtzz)*gtZZ*phidx*phidy - 32*gtxy*gtYY*gtzz*(phidy*phidy) - 64*gtxz*gtYZ*gtzz*(phidy*phidy) + 8*gtxz*phidydz - 8*gtxydz*phidz + 16*gtxzdy*phidz - 12*gtxydz*gtyy*gtYY*phidz - 4*gtxzdy*gtyy*gtYY*phidz + 8*gtxz*gtYY*gtyydy*phidz - 12*gtxy*gtYY*gtyydz*phidz + 16*gtxydy*gtYY*gtyz*phidz - 16*gtxzdz*gtyy*gtYZ*phidz - 16*gtxz*gtyydz*gtYZ*phidz + 16*gtxzdy*gtyz*gtYZ*phidz + 4*gtyy*gtYY*gtyzdx*phidz + 16*gtxz*gtYZ*gtyzdy*phidz + 4*gtxxdy*gtXZ*gtzz*phidz + 8*gtxydy*gtYZ*gtzz*phidz - 4*gtyydx*gtYZ*gtzz*phidz + 4*gtxydz*gtzz*gtZZ*phidz + 4*gtxzdy*gtzz*gtZZ*phidz - 4*gtyzdx*gtzz*gtZZ*phidz + 4*gtyy*gtYZ*gtzzdx*phidz - 4*gtxx*gtXZ*gtzzdy*phidz - 8*gtxy*gtYZ*gtzzdy*phidz - 8*gtxz*gtZZ*gtzzdy*phidz + 4*gtxy*gtZZ*gtzzdz*phidz + 16*gtyy*gtYY*gtyz*phidx*phidz - 16*gtyz*gtzz*gtZZ*phidx*phidz + 16*gtxz*gtyy*gtYY*phidy*phidz + 64*gtxy*gtYY*gtyz*phidy*phidz + 128*gtxz*gtyz*gtYZ*phidy*phidz - 16*gtxz*gtzz*gtZZ*phidy*phidz - 48*gtxy*gtyy*gtYY*(phidz*phidz) - 64*gtxz*gtyy*gtYZ*(phidz*phidz) + 16*gtxy*gtzz*gtZZ*(phidz*phidz) - 8*gtxy*phidzdz))/4.;
    double const Rxyxz=(2*gtXXdx*gtxydz + gtXY*(gtxydz*gtxydz)*gtXZ + 2*gtxydzdz*gtXZ + 2*gtXXdx*gtxzdy - 2*gtXY*gtxydz*gtXZ*gtxzdy + gtXY*gtXZ*(gtxzdy*gtxzdy) - 2*gtXZ*gtxzdydz + 2*gtxydz*gtXZdz - 2*gtxzdy*gtXZdz - 2*gtXYdz*gtyydx + 2*(gtXY*gtXY)*gtxzdy*gtyydx + 2*(gtXY*gtXY)*gtxydx*gtyydz + 2*gtXYdx*gtyydz - gtxxdz*gtXY*gtXZ*gtyydz + 2*gtXY*gtXZ*gtxzdx*gtyydz - gtxydz*gtXZ*gtYY*gtyydz + gtXZ*gtxzdy*gtYY*gtyydz + 2*gtXY*gtxydz*gtyydz*gtYZ - 2*gtXY*gtxzdy*gtyydz*gtYZ - 2*gtXXdx*gtyzdx + 2*gtXY*gtxydz*gtXZ*gtyzdx + 2*gtXY*gtXZ*gtxzdy*gtyzdx - 2*gtXZdz*gtyzdx - 2*(gtXY*gtXY)*gtyydx*gtyzdx + gtXZ*gtYY*gtyydz*gtyzdx - 3*gtXY*gtXZ*(gtyzdx*gtyzdx) - 2*gtXZ*gtyzdxdz - 2*gtXY*gtyydx*gtYZ*gtyzdz + 2*gtXY*gtxydz*gtyzdz*gtZZ - 2*gtXY*gtxzdy*gtyzdz*gtZZ - 2*gtXY*gtyzdx*gtyzdz*gtZZ + 2*gtxydz*(gtXZ*gtXZ)*gtzzdx - gtXY*gtXZ*gtyydx*gtzzdx + gtXZ*gtyydz*gtYZ*gtzzdx - 2*(gtXZ*gtXZ)*gtyzdx*gtzzdx + 2*gtXZ*gtzzdxdy + 2*gtXY*gtxydx*gtXZ*gtzzdy - gtxxdz*(gtXZ*gtXZ)*gtzzdy + 2*(gtXZ*gtXZ)*gtxzdx*gtzzdy + 2*gtXZdx*gtzzdy - gtXZ*gtYY*gtyydx*gtzzdy + 2*gtXY*gtyydx*gtYZ*gtzzdy + 2*gtXY*gtyzdx*gtZZ*gtzzdy + gtXZ*gtZZ*gtzzdx*gtzzdy - gtXZ*gtyydx*gtYZ*gtzzdz + gtxydz*gtXZ*gtZZ*gtzzdz - gtXZ*gtxzdy*gtZZ*gtzzdz - gtXZ*gtyzdx*gtZZ*gtzzdz - 8*gtXYdz*gtyy*phidx + 8*(gtXY*gtXY)*gtxzdy*gtyy*phidx - 8*gtXY*gtyydz*phidx + 8*gtxy*(gtXY*gtXY)*gtyydz*phidx + 8*gtXY*gtxz*gtXZ*gtyydz*phidx - 8*gtXXdx*gtyz*phidx + 8*gtXY*gtxydz*gtXZ*gtyz*phidx + 8*gtXY*gtXZ*gtxzdy*gtyz*phidx - 8*gtXZdz*gtyz*phidx - 8*(gtXY*gtXY)*gtyydx*gtyz*phidx + 4*gtXZ*gtYY*gtyydz*gtyz*phidx - 8*(gtXY*gtXY)*gtyy*gtyzdx*phidx - 24*gtXY*gtXZ*gtyz*gtyzdx*phidx - 8*gtXZ*gtyzdz*phidx - 8*gtXY*gtyy*gtYZ*gtyzdz*phidx + 8*gtxydz*(gtXZ*gtXZ)*gtzz*phidx - 4*gtXY*gtXZ*gtyydx*gtzz*phidx + 4*gtXZ*gtyydz*gtYZ*gtzz*phidx - 8*(gtXZ*gtXZ)*gtyzdx*gtzz*phidx - 8*gtXY*gtyz*gtyzdz*gtZZ*phidx - 4*gtXY*gtXZ*gtyy*gtzzdx*phidx - 8*(gtXZ*gtXZ)*gtyz*gtzzdx*phidx + 8*gtxy*gtXY*gtXZ*gtzzdy*phidx + 8*gtxz*(gtXZ*gtXZ)*gtzzdy*phidx - 4*gtXZ*gtyy*gtYY*gtzzdy*phidx + 8*gtXY*gtyy*gtYZ*gtzzdy*phidx + 8*gtXY*gtyz*gtZZ*gtzzdy*phidx + 4*gtXZ*gtzz*gtZZ*gtzzdy*phidx - 4*gtXZ*gtyy*gtYZ*gtzzdz*phidx - 4*gtXZ*gtyz*gtZZ*gtzzdz*phidx - 32*(gtXY*gtXY)*gtyy*gtyz*(phidx*phidx) - 48*gtXY*gtXZ*(gtyz*gtyz)*(phidx*phidx) - 16*gtXY*gtXZ*gtyy*gtzz*(phidx*phidx) - 32*(gtXZ*gtXZ)*gtyz*gtzz*(phidx*phidx) + 8*gtXZ*gtzz*phidxdy - 8*gtXZ*gtyz*phidxdz - 8*gtxx*gtXXdz*phidy + 8*gtXXdx*gtxz*phidy - 8*gtXY*gtxydz*gtxz*gtXZ*phidy + 8*gtXY*gtxz*gtXZ*gtxzdy*phidy - 8*gtXZ*gtxzdz*phidy - 8*gtxz*gtXZdz*phidy + 8*(gtXY*gtXY)*gtxz*gtyydx*phidy - 8*gtxx*(gtXY*gtXY)*gtyydz*phidy + 4*gtxz*gtXZ*gtYY*gtyydz*phidy - 8*gtXY*gtxz*gtyydz*gtYZ*phidy + 8*gtXY*gtxz*gtXZ*gtyzdx*phidy - 8*gtxx*gtXY*gtXZ*gtyzdz*phidy + 8*gtXY*gtxydx*gtXZ*gtzz*phidy - 4*gtxxdz*(gtXZ*gtXZ)*gtzz*phidy + 8*(gtXZ*gtXZ)*gtxzdx*gtzz*phidy + 8*gtXZdx*gtzz*phidy - 4*gtXZ*gtYY*gtyydx*gtzz*phidy + 8*gtXY*gtyydx*gtYZ*gtzz*phidy - 8*gtXY*gtxz*gtyzdz*gtZZ*phidy + 8*gtXY*gtyzdx*gtzz*gtZZ*phidy + 8*gtXZ*gtzzdx*phidy + 4*gtXZ*gtzz*gtZZ*gtzzdx*phidy - 4*gtxx*gtXY*gtXZ*gtzzdy*phidy - 4*gtxx*(gtXZ*gtXZ)*gtzzdz*phidy - 4*gtxz*gtXZ*gtZZ*gtzzdz*phidy + 32*(gtXY*gtXY)*gtxz*gtyy*phidx*phidy + 32*gtXY*gtxz*gtXZ*gtyz*phidx*phidy + 32*gtxy*gtXY*gtXZ*gtzz*phidx*phidy + 32*gtxz*(gtXZ*gtXZ)*gtzz*phidx*phidy - 16*gtXZ*gtyy*gtYY*gtzz*phidx*phidy + 32*gtXY*gtyy*gtYZ*gtzz*phidx*phidy + 32*gtXY*gtyz*gtzz*gtZZ*phidx*phidy + 16*gtXZ*(gtzz*gtzz)*gtZZ*phidx*phidy + 16*gtXY*(gtxz*gtxz)*gtXZ*(phidy*phidy) - 16*gtxx*gtXY*gtXZ*gtzz*(phidy*phidy) - 8*gtxz*gtXZ*phidydz + 8*gtXXdx*gtxy*phidz + 8*gtxydz*gtXZ*phidz + 8*gtxy*gtXY*gtxydz*gtXZ*phidz - 8*gtxy*gtXY*gtXZ*gtxzdy*phidz + 8*gtxy*gtXZdz*phidz + 8*(gtXY*gtXY)*gtxydx*gtyy*phidz + 8*gtXYdx*gtyy*phidz - 4*gtxxdz*gtXY*gtXZ*gtyy*phidz + 8*gtXY*gtXZ*gtxzdx*gtyy*phidz - 4*gtxydz*gtXZ*gtyy*gtYY*phidz + 4*gtXZ*gtxzdy*gtyy*gtYY*phidz + 8*gtXY*gtyydx*phidz - 4*gtxx*gtXY*gtXZ*gtyydz*phidz - 4*gtxy*gtXZ*gtYY*gtyydz*phidz + 8*gtXY*gtxydz*gtyy*gtYZ*phidz - 8*gtXY*gtxzdy*gtyy*gtYZ*phidz + 8*gtxy*gtXY*gtyydz*gtYZ*phidz - 8*gtXY*gtyydx*gtyz*gtYZ*phidz + 8*gtxy*gtXY*gtXZ*gtyzdx*phidz + 4*gtXZ*gtyy*gtYY*gtyzdx*phidz - 4*gtXZ*gtyydx*gtYZ*gtzz*phidz + 8*gtXY*gtxydz*gtyz*gtZZ*phidz - 8*gtXY*gtxzdy*gtyz*gtZZ*phidz - 8*gtXY*gtyz*gtyzdx*gtZZ*phidz + 8*gtxy*gtXY*gtyzdz*gtZZ*phidz + 4*gtxydz*gtXZ*gtzz*gtZZ*phidz - 4*gtXZ*gtxzdy*gtzz*gtZZ*phidz - 4*gtXZ*gtyzdx*gtzz*gtZZ*phidz + 8*gtxy*(gtXZ*gtXZ)*gtzzdx*phidz + 4*gtXZ*gtyy*gtYZ*gtzzdx*phidz - 4*gtxx*(gtXZ*gtXZ)*gtzzdy*phidz + 4*gtxy*gtXZ*gtZZ*gtzzdz*phidz + 32*gtxy*(gtXY*gtXY)*gtyy*phidx*phidz + 32*gtXY*gtxz*gtXZ*gtyy*phidx*phidz + 32*gtxy*gtXY*gtXZ*gtyz*phidx*phidz + 16*gtXZ*gtyy*gtYY*gtyz*phidx*phidz - 32*gtXY*gtyy*gtyz*gtYZ*phidx*phidz + 32*gtxy*(gtXZ*gtXZ)*gtzz*phidx*phidz - 32*gtXY*(gtyz*gtyz)*gtZZ*phidx*phidz - 16*gtXZ*gtyz*gtzz*gtZZ*phidx*phidz - 32*gtxy*gtXY*gtxz*gtXZ*phidy*phidz - 32*gtxx*(gtXY*gtXY)*gtyy*phidy*phidz + 16*gtxz*gtXZ*gtyy*gtYY*phidy*phidz - 32*gtxx*gtXY*gtXZ*gtyz*phidy*phidz - 32*gtXY*gtxz*gtyy*gtYZ*phidy*phidz - 32*gtxx*(gtXZ*gtXZ)*gtzz*phidy*phidz - 32*gtXY*gtxz*gtyz*gtZZ*phidy*phidz - 16*gtxz*gtXZ*gtzz*gtZZ*phidy*phidz + 16*(gtxy*gtxy)*gtXY*gtXZ*(phidz*phidz) - 16*gtxx*gtXY*gtXZ*gtyy*(phidz*phidz) - 16*gtxy*gtXZ*gtyy*gtYY*(phidz*phidz) + 32*gtxy*gtXY*gtyy*gtYZ*(phidz*phidz) + 32*gtxy*gtXY*gtyz*gtZZ*(phidz*phidz) + 16*gtxy*gtXZ*gtzz*gtZZ*(phidz*phidz) + gtXX*(-2*gtxxdydz + 2*gtxydxdz - 2*gtxxdy*gtXY*gtxydz + 2*gtXY*gtxydx*gtxydz + gtxxdz*gtxydz*gtXZ + 2*gtxydz*gtXZ*gtxzdx + 2*gtxzdxdy + 2*gtXY*gtxydx*gtxzdy - gtxxdz*gtXZ*gtxzdy + 2*gtXZ*gtxzdx*gtxzdy - 2*gtxxdy*gtXZ*gtxzdz - gtxxdz*gtXY*gtyydx - gtxydz*gtYY*gtyydx - gtxzdy*gtYY*gtyydx + gtxxdx*gtXY*gtyydz + gtxxdy*gtYY*gtyydz + gtxydz*gtxydz*gtYZ - gtxzdy*gtxzdy*gtYZ - 2*gtxzdz*gtyydx*gtYZ + gtxxdz*gtyydz*gtYZ - 2*gtXY*gtxydx*gtyzdx - gtxxdz*gtXZ*gtyzdx - 2*gtXZ*gtxzdx*gtyzdx + gtYY*gtyydx*gtyzdx - 2*gtxydz*gtYZ*gtyzdx + gtYZ*(gtyzdx*gtyzdx) - 2*gtyzdxdx + 2*gtxydz*gtxzdz*gtZZ - 2*gtxzdy*gtxzdz*gtZZ - 2*gtxzdz*gtyzdx*gtZZ + gtyydx*gtYZ*gtzzdx - gtxydz*gtZZ*gtzzdx + gtxzdy*gtZZ*gtzzdx + gtyzdx*gtZZ*gtzzdx + gtxxdx*gtXZ*gtzzdy + gtxxdy*gtYZ*gtzzdy + gtxxdz*gtZZ*gtzzdy + 8*gtxy*gtXY*gtxydz*phidx + 8*gtxydz*gtxz*gtXZ*phidx + 8*gtxy*gtXY*gtxzdy*phidx + 8*gtxz*gtXZ*gtxzdy*phidx - 4*gtxxdz*gtXY*gtyy*phidx - 4*gtxydz*gtyy*gtYY*phidx - 4*gtxzdy*gtyy*gtYY*phidx + 4*gtxx*gtXY*gtyydz*phidx - 8*gtXY*gtxydx*gtyz*phidx - 4*gtxxdz*gtXZ*gtyz*phidx - 8*gtXZ*gtxzdx*gtyz*phidx + 4*gtYY*gtyydx*gtyz*phidx - 8*gtxzdz*gtyy*gtYZ*phidx - 8*gtxydz*gtyz*gtYZ*phidx - 8*gtyzdx*phidx - 8*gtxy*gtXY*gtyzdx*phidx - 8*gtxz*gtXZ*gtyzdx*phidx + 4*gtyy*gtYY*gtyzdx*phidx + 8*gtyz*gtYZ*gtyzdx*phidx + 4*gtyydx*gtYZ*gtzz*phidx - 8*gtxzdz*gtyz*gtZZ*phidx - 4*gtxydz*gtzz*gtZZ*phidx + 4*gtxzdy*gtzz*gtZZ*phidx + 4*gtyzdx*gtzz*gtZZ*phidx + 4*gtyy*gtYZ*gtzzdx*phidx + 4*gtyz*gtZZ*gtzzdx*phidx + 4*gtxx*gtXZ*gtzzdy*phidx - 32*gtxy*gtXY*gtyz*(phidx*phidx) - 32*gtxz*gtXZ*gtyz*(phidx*phidx) + 16*gtyy*gtYY*gtyz*(phidx*phidx) + 16*(gtyz*gtyz)*gtYZ*(phidx*phidx) + 16*gtyy*gtYZ*gtzz*(phidx*phidx) + 16*gtyz*gtzz*gtZZ*(phidx*phidx) - 8*gtyz*phidxdx + 8*gtxz*phidxdy + 8*gtxy*phidxdz - 8*gtxxdz*phidy - 8*gtxx*gtXY*gtxydz*phidy + 8*gtXY*gtxydx*gtxz*phidy - 4*gtxxdz*gtxz*gtXZ*phidy + 8*gtxzdx*phidy + 8*gtxz*gtXZ*gtxzdx*phidy - 8*gtxx*gtXZ*gtxzdz*phidy - 4*gtxz*gtYY*gtyydx*phidy + 4*gtxx*gtYY*gtyydz*phidy - 8*gtxz*gtxzdy*gtYZ*phidy + 4*gtxxdx*gtXZ*gtzz*phidy + 4*gtxxdy*gtYZ*gtzz*phidy - 8*gtxz*gtxzdz*gtZZ*phidy + 4*gtxxdz*gtzz*gtZZ*phidy + 4*gtxz*gtZZ*gtzzdx*phidy + 4*gtxx*gtYZ*gtzzdy*phidy + 32*gtxy*gtXY*gtxz*phidx*phidy + 32*(gtxz*gtxz)*gtXZ*phidx*phidy - 16*gtxz*gtyy*gtYY*phidx*phidy + 16*gtxx*gtXZ*gtzz*phidx*phidy + 16*gtxz*gtzz*gtZZ*phidx*phidy - 16*(gtxz*gtxz)*gtYZ*(phidy*phidy) + 16*gtxx*gtYZ*gtzz*(phidy*phidy) - 8*gtxx*phidydz - 8*gtxxdy*gtxy*gtXY*phidz + 8*gtxydx*phidz + 8*gtxy*gtXY*gtxydx*phidz + 4*gtxxdz*gtxy*gtXZ*phidz + 4*gtxx*gtxydz*gtXZ*phidz - 8*gtxxdy*gtxz*gtXZ*phidz + 8*gtxy*gtXZ*gtxzdx*phidz - 4*gtxx*gtXZ*gtxzdy*phidz + 4*gtxxdx*gtXY*gtyy*phidz + 4*gtxxdy*gtyy*gtYY*phidz - 4*gtxx*gtXY*gtyydx*phidz - 4*gtxy*gtYY*gtyydx*phidz + 8*gtxy*gtxydz*gtYZ*phidz + 4*gtxxdz*gtyy*gtYZ*phidz - 8*gtxz*gtyydx*gtYZ*phidz + 4*gtxx*gtyydz*gtYZ*phidz - 4*gtxx*gtXZ*gtyzdx*phidz - 8*gtxy*gtYZ*gtyzdx*phidz + 8*gtxydz*gtxz*gtZZ*phidz - 8*gtxz*gtxzdy*gtZZ*phidz + 8*gtxy*gtxzdz*gtZZ*phidz - 8*gtxz*gtyzdx*gtZZ*phidz - 4*gtxy*gtZZ*gtzzdx*phidz + 4*gtxx*gtZZ*gtzzdy*phidz + 32*(gtxy*gtxy)*gtXY*phidx*phidz + 32*gtxy*gtxz*gtXZ*phidx*phidz - 16*gtxy*gtyy*gtYY*phidx*phidz - 16*gtxx*gtXZ*gtyz*phidx*phidz - 32*gtxz*gtyy*gtYZ*phidx*phidz - 32*gtxy*gtyz*gtYZ*phidx*phidz - 32*gtxz*gtyz*gtZZ*phidx*phidz - 16*gtxy*gtzz*gtZZ*phidx*phidz - 32*gtxx*gtxy*gtXY*phidy*phidz - 48*gtxx*gtxz*gtXZ*phidy*phidz + 16*gtxx*gtyy*gtYY*phidy*phidz - 32*(gtxz*gtxz)*gtZZ*phidy*phidz + 16*gtxx*gtzz*gtZZ*phidy*phidz + 16*gtxx*gtxy*gtXZ*(phidz*phidz) + 16*(gtxy*gtxy)*gtYZ*(phidz*phidz) + 16*gtxx*gtyy*gtYZ*(phidz*phidz) + 32*gtxy*gtxz*gtZZ*(phidz*phidz)) - gtxxdy*(2*gtXXdz + 2*(gtXY*gtXY)*(gtyydz + 4*gtyy*phidz) + gtXY*gtXZ*(2*gtyzdz + gtzzdy + 4*gtzz*phidy + 8*gtyz*phidz) + gtXZ*gtXZ*(gtzzdz + 4*gtzz*phidz)) + gtXX*gtXX*(-(gtxxdy*(gtxxdz + 4*gtxx*phidz)) + gtxxdx*(gtxydz + gtxzdy - gtyzdx - 4*gtyz*phidx + 4*gtxz*phidy + 4*gtxy*phidz) - 4*gtxx*(-(gtxydz*phidx) - gtxzdy*phidx + gtyzdx*phidx + 4*gtyz*(phidx*phidx) + gtxxdz*phidy - 4*gtxz*phidx*phidy - 4*gtxy*phidx*phidz + 4*gtxx*phidy*phidz)) + 8*gtxy*gtXZ*phidzdz)/4.;
    double const Rxyyz=(2*gtXXdy*gtxydz + 2*gtXXdy*gtxzdy - 2*gtXYdz*gtyydy + 2*(gtXY*gtXY)*gtxzdy*gtyydy - 2*(gtXY*gtXY)*gtxydy*gtyydz + 2*gtXYdy*gtyydz + 2*gtXZdz*gtyydz + 2*(gtXY*gtXY)*gtyydx*gtyydz - gtXZ*gtYY*(gtyydz*gtyydz) + 2*gtXZ*gtyydzdz + 2*gtXY*(gtyydz*gtyydz)*gtYZ - 2*gtXXdy*gtyzdx - 2*(gtXY*gtXY)*gtyydy*gtyzdx + 2*gtXY*gtXZ*gtyydz*gtyzdx + 4*gtXY*gtXZ*gtxzdy*gtyzdy - 4*gtXZdz*gtyzdy + 2*gtXZ*gtYY*gtyydz*gtyzdy - 2*gtXY*gtyydz*gtYZ*gtyzdy - 4*gtXY*gtXZ*gtyzdx*gtyzdy - 4*gtXZ*gtyzdydz - 4*gtXY*gtxydy*gtXZ*gtyzdz + 2*gtXY*gtXZ*gtyydx*gtyzdz - 2*gtXY*gtyydy*gtYZ*gtyzdz + 2*gtXY*gtyydz*gtyzdz*gtZZ - 4*gtXY*gtyzdy*gtyzdz*gtZZ - gtXY*gtXZ*gtyydy*gtzzdx + gtXZ*gtXZ*gtyydz*gtzzdx - 2*(gtXZ*gtXZ)*gtyzdy*gtzzdx + 2*(gtXZ*gtXZ)*gtxzdy*gtzzdy + 2*gtXZdy*gtzzdy + gtXY*gtXZ*gtyydx*gtzzdy - gtXZ*gtYY*gtyydy*gtzzdy + 2*gtXY*gtyydy*gtYZ*gtzzdy + gtXZ*gtyydz*gtYZ*gtzzdy + 2*gtXY*gtyzdy*gtZZ*gtzzdy + gtXZ*gtZZ*(gtzzdy*gtzzdy) + 2*gtXZ*gtzzdydy - 2*gtxydy*(gtXZ*gtXZ)*gtzzdz + gtXZ*gtXZ*gtyydx*gtzzdz - gtXZ*gtyydy*gtYZ*gtzzdz + gtXZ*gtyydz*gtZZ*gtzzdz - 2*gtXZ*gtyzdy*gtZZ*gtzzdz + 8*(gtXY*gtXY)*gtyy*gtyydz*phidx - 8*gtXXdy*gtyz*phidx - 8*(gtXY*gtXY)*gtyydy*gtyz*phidx + 8*gtXY*gtXZ*gtyydz*gtyz*phidx - 16*gtXY*gtXZ*gtyz*gtyzdy*phidx + 8*gtXY*gtXZ*gtyy*gtyzdz*phidx - 4*gtXY*gtXZ*gtyydy*gtzz*phidx + 4*(gtXZ*gtXZ)*gtyydz*gtzz*phidx - 8*(gtXZ*gtXZ)*gtyzdy*gtzz*phidx + 4*gtXY*gtXZ*gtyy*gtzzdy*phidx + 4*(gtXZ*gtXZ)*gtyy*gtzzdz*phidx + 8*gtXXdy*gtxz*phidy - 8*gtXYdz*gtyy*phidy + 8*(gtXY*gtXY)*gtxzdy*gtyy*phidy + 8*(gtXY*gtXY)*gtxz*gtyydy*phidy - 8*gtXY*gtyydz*phidy - 8*gtxy*(gtXY*gtXY)*gtyydz*phidy + 16*gtXY*gtXZ*gtxzdy*gtyz*phidy - 16*gtXZdz*gtyz*phidy + 8*gtXZ*gtYY*gtyydz*gtyz*phidy - 8*gtXY*gtyydz*gtyz*gtYZ*phidy - 8*(gtXY*gtXY)*gtyy*gtyzdx*phidy - 16*gtXY*gtXZ*gtyz*gtyzdx*phidy + 16*gtXY*gtxz*gtXZ*gtyzdy*phidy - 16*gtXZ*gtyzdz*phidy - 16*gtxy*gtXY*gtXZ*gtyzdz*phidy - 8*gtXY*gtyy*gtYZ*gtyzdz*phidy + 8*(gtXZ*gtXZ)*gtxzdy*gtzz*phidy + 8*gtXZdy*gtzz*phidy + 4*gtXY*gtXZ*gtyydx*gtzz*phidy - 4*gtXZ*gtYY*gtyydy*gtzz*phidy + 8*gtXY*gtyydy*gtYZ*gtzz*phidy + 4*gtXZ*gtyydz*gtYZ*gtzz*phidy - 16*gtXY*gtyz*gtyzdz*gtZZ*phidy + 8*gtXY*gtyzdy*gtzz*gtZZ*phidy - 4*gtXY*gtXZ*gtyy*gtzzdx*phidy - 8*(gtXZ*gtXZ)*gtyz*gtzzdx*phidy + 8*gtXZ*gtzzdy*phidy + 8*gtxz*(gtXZ*gtXZ)*gtzzdy*phidy - 4*gtXZ*gtyy*gtYY*gtzzdy*phidy + 8*gtXY*gtyy*gtYZ*gtzzdy*phidy + 8*gtXY*gtyz*gtZZ*gtzzdy*phidy + 8*gtXZ*gtzz*gtZZ*gtzzdy*phidy - 8*gtxy*(gtXZ*gtXZ)*gtzzdz*phidy - 4*gtXZ*gtyy*gtYZ*gtzzdz*phidy - 8*gtXZ*gtyz*gtZZ*gtzzdz*phidy - 32*(gtXY*gtXY)*gtyy*gtyz*phidx*phidy - 64*gtXY*gtXZ*(gtyz*gtyz)*phidx*phidy - 32*(gtXZ*gtXZ)*gtyz*gtzz*phidx*phidy + 32*(gtXY*gtXY)*gtxz*gtyy*(phidy*phidy) + 64*gtXY*gtxz*gtXZ*gtyz*(phidy*phidy) + 32*gtxz*(gtXZ*gtXZ)*gtzz*(phidy*phidy) - 16*gtXZ*gtyy*gtYY*gtzz*(phidy*phidy) + 32*gtXY*gtyy*gtYZ*gtzz*(phidy*phidy) + 32*gtXY*gtyz*gtzz*gtZZ*(phidy*phidy) + 16*gtXZ*(gtzz*gtzz)*gtZZ*(phidy*phidy) + 2*gtXXdz*(-2*gtxydy + gtyydx + 4*gtyy*phidx - 8*gtxy*phidy) + 8*gtXZ*gtzz*phidydy - 16*gtXZ*gtyz*phidydz + 8*gtXXdy*gtxy*phidz - 8*(gtXY*gtXY)*gtxydy*gtyy*phidz + 8*gtXYdy*gtyy*phidz + 8*gtXZdz*gtyy*phidz + 8*(gtXY*gtXY)*gtyy*gtyydx*phidz + 8*gtXY*gtyydy*phidz + 8*gtXZ*gtyydz*phidz - 8*gtXZ*gtyy*gtYY*gtyydz*phidz - 16*gtXY*gtxydy*gtXZ*gtyz*phidz + 8*gtXY*gtXZ*gtyydx*gtyz*phidz + 16*gtXY*gtyy*gtyydz*gtYZ*phidz - 8*gtXY*gtyydy*gtyz*gtYZ*phidz + 8*gtXY*gtXZ*gtyy*gtyzdx*phidz + 8*gtXZ*gtyy*gtYY*gtyzdy*phidz - 8*gtXY*gtyy*gtYZ*gtyzdy*phidz - 8*gtxydy*(gtXZ*gtXZ)*gtzz*phidz + 4*(gtXZ*gtXZ)*gtyydx*gtzz*phidz - 4*gtXZ*gtyydy*gtYZ*gtzz*phidz + 8*gtXY*gtyydz*gtyz*gtZZ*phidz - 16*gtXY*gtyz*gtyzdy*gtZZ*phidz + 8*gtXY*gtyy*gtyzdz*gtZZ*phidz + 4*gtXZ*gtyydz*gtzz*gtZZ*phidz - 8*gtXZ*gtyzdy*gtzz*gtZZ*phidz + 4*(gtXZ*gtXZ)*gtyy*gtzzdx*phidz + 4*gtXZ*gtyy*gtYZ*gtzzdy*phidz + 4*gtXZ*gtyy*gtZZ*gtzzdz*phidz + 32*(gtXY*gtXY)*(gtyy*gtyy)*phidx*phidz + 64*gtXY*gtXZ*gtyy*gtyz*phidx*phidz + 32*(gtXZ*gtXZ)*gtyy*gtzz*phidx*phidz - 32*gtxy*(gtXY*gtXY)*gtyy*phidy*phidz - 64*gtxy*gtXY*gtXZ*gtyz*phidy*phidz + 32*gtXZ*gtyy*gtYY*gtyz*phidy*phidz - 64*gtXY*gtyy*gtyz*gtYZ*phidy*phidz - 32*gtxy*(gtXZ*gtXZ)*gtzz*phidy*phidz - 64*gtXY*(gtyz*gtyz)*gtZZ*phidy*phidz - 32*gtXZ*gtyz*gtzz*gtZZ*phidy*phidz - 16*gtXZ*(gtyy*gtyy)*gtYY*(phidz*phidz) + 32*gtXY*(gtyy*gtyy)*gtYZ*(phidz*phidz) + 32*gtXY*gtyy*gtyz*gtZZ*(phidz*phidz) + 16*gtXZ*gtyy*gtzz*gtZZ*(phidz*phidz) + gtXX*gtXX*(gtxxdz*(-2*gtxydy + gtyydx + 4*gtyy*phidx - 8*gtxy*phidy) + gtxxdy*(gtxydz + gtxzdy - gtyzdx - 4*gtyz*phidx + 4*gtxz*phidy + 4*gtxy*phidz) + 4*gtxx*(gtxydz*phidy + gtxzdy*phidy - gtyzdx*phidy - 4*gtyz*phidx*phidy + 4*gtxz*(phidy*phidy) - 2*gtxydy*phidz + gtyydx*phidz + 4*gtyy*phidx*phidz - 4*gtxy*phidy*phidz)) + gtXX*(-2*gtxydydz + 2*gtxydz*gtXZ*gtxzdy + 2*gtXZ*(gtxzdy*gtxzdy) + 2*gtxzdydy - 4*gtxydy*gtXZ*gtxzdz + 2*gtXZ*gtxzdz*gtyydx + 2*gtyydxdz - gtxydz*gtYY*gtyydy - gtxzdy*gtYY*gtyydy + gtxxdz*gtXZ*gtyydz + 2*gtxydy*gtYY*gtyydz - gtYY*gtyydx*gtyydz - 2*gtxzdz*gtyydy*gtYZ + 2*gtxydz*gtyydz*gtYZ + 2*gtxzdy*gtyydz*gtYZ - 2*gtXZ*gtxzdy*gtyzdx + gtYY*gtyydy*gtyzdx - 2*gtyydz*gtYZ*gtyzdx - 2*gtyzdxdy - 2*gtxxdz*gtXZ*gtyzdy - 2*gtxydz*gtYZ*gtyzdy - 2*gtxzdy*gtYZ*gtyzdy + 2*gtYZ*gtyzdx*gtyzdy + 2*gtxzdz*gtyydz*gtZZ - 4*gtxzdz*gtyzdy*gtZZ + gtyydy*gtYZ*gtzzdx - gtyydz*gtZZ*gtzzdx + 2*gtyzdy*gtZZ*gtzzdx + gtxxdy*gtXZ*gtzzdy + 2*gtxydy*gtYZ*gtzzdy - gtyydx*gtYZ*gtzzdy + gtxydz*gtZZ*gtzzdy + gtxzdy*gtZZ*gtzzdy - gtyzdx*gtZZ*gtzzdy + 8*gtXZ*gtxzdz*gtyy*phidx + 8*gtyydz*phidx - 4*gtyy*gtYY*gtyydz*phidx - 8*gtXZ*gtxzdy*gtyz*phidx + 4*gtYY*gtyydy*gtyz*phidx - 8*gtyydz*gtyz*gtYZ*phidx - 8*gtyzdy*phidx + 8*gtyz*gtYZ*gtyzdy*phidx + 4*gtyydy*gtYZ*gtzz*phidx - 4*gtyydz*gtzz*gtZZ*phidx + 8*gtyzdy*gtzz*gtZZ*phidx - 4*gtyy*gtYZ*gtzzdy*phidx - 4*gtyz*gtZZ*gtzzdy*phidx - 8*gtyz*phidxdy + 8*gtyy*phidxdz - 16*gtxydz*phidy + 8*gtxydz*gtxz*gtXZ*phidy + 8*gtxzdy*phidy + 16*gtxz*gtXZ*gtxzdy*phidy - 16*gtxy*gtXZ*gtxzdz*phidy - 4*gtxydz*gtyy*gtYY*phidy - 4*gtxzdy*gtyy*gtYY*phidy - 4*gtxz*gtYY*gtyydy*phidy + 8*gtxy*gtYY*gtyydz*phidy - 8*gtxxdz*gtXZ*gtyz*phidy - 8*gtxzdz*gtyy*gtYZ*phidy + 8*gtxz*gtyydz*gtYZ*phidy - 8*gtxydz*gtyz*gtYZ*phidy - 8*gtxzdy*gtyz*gtYZ*phidy - 8*gtxz*gtXZ*gtyzdx*phidy + 4*gtyy*gtYY*gtyzdx*phidy + 8*gtyz*gtYZ*gtyzdx*phidy - 8*gtxz*gtYZ*gtyzdy*phidy + 4*gtxxdy*gtXZ*gtzz*phidy + 8*gtxydy*gtYZ*gtzz*phidy - 4*gtyydx*gtYZ*gtzz*phidy - 16*gtxzdz*gtyz*gtZZ*phidy + 4*gtxydz*gtzz*gtZZ*phidy + 4*gtxzdy*gtzz*gtZZ*phidy - 4*gtyzdx*gtzz*gtZZ*phidy + 4*gtyy*gtYZ*gtzzdx*phidy + 8*gtyz*gtZZ*gtzzdx*phidy + 4*gtxx*gtXZ*gtzzdy*phidy + 8*gtxy*gtYZ*gtzzdy*phidy + 4*gtxz*gtZZ*gtzzdy*phidy - 32*gtxz*gtXZ*gtyz*phidx*phidy + 16*gtyy*gtYY*gtyz*phidx*phidy + 32*(gtyz*gtyz)*gtYZ*phidx*phidy + 16*gtyz*gtzz*gtZZ*phidx*phidy + 32*(gtxz*gtxz)*gtXZ*(phidy*phidy) - 16*gtxz*gtyy*gtYY*(phidy*phidy) - 32*gtxz*gtyz*gtYZ*(phidy*phidy) + 16*gtxx*gtXZ*gtzz*(phidy*phidy) + 32*gtxy*gtYZ*gtzz*(phidy*phidy) + 16*gtxz*gtzz*gtZZ*(phidy*phidy) + 8*gtxz*phidydy - 8*gtxy*phidydz + 8*gtxydy*phidz - 16*gtxydy*gtxz*gtXZ*phidz + 8*gtxy*gtXZ*gtxzdy*phidz + 4*gtxxdz*gtXZ*gtyy*phidz + 8*gtxydy*gtyy*gtYY*phidz + 8*gtxz*gtXZ*gtyydx*phidz - 4*gtyy*gtYY*gtyydx*phidz - 4*gtxy*gtYY*gtyydy*phidz + 4*gtxx*gtXZ*gtyydz*phidz + 8*gtxydz*gtyy*gtYZ*phidz + 8*gtxzdy*gtyy*gtYZ*phidz - 8*gtxz*gtyydy*gtYZ*phidz + 8*gtxy*gtyydz*gtYZ*phidz - 8*gtyy*gtYZ*gtyzdx*phidz - 8*gtxx*gtXZ*gtyzdy*phidz - 8*gtxy*gtYZ*gtyzdy*phidz + 8*gtxzdz*gtyy*gtZZ*phidz + 8*gtxz*gtyydz*gtZZ*phidz - 16*gtxz*gtyzdy*gtZZ*phidz - 4*gtyy*gtZZ*gtzzdx*phidz + 4*gtxy*gtZZ*gtzzdy*phidz + 32*gtxz*gtXZ*gtyy*phidx*phidz - 16*(gtyy*gtyy)*gtYY*phidx*phidz - 32*gtyy*gtyz*gtYZ*phidx*phidz - 16*gtyy*gtzz*gtZZ*phidx*phidz - 32*gtxy*gtxz*gtXZ*phidy*phidz + 16*gtxy*gtyy*gtYY*phidy*phidz - 32*gtxx*gtXZ*gtyz*phidy*phidz - 32*gtxy*gtyz*gtYZ*phidy*phidz - 64*gtxz*gtyz*gtZZ*phidy*phidz + 16*gtxy*gtzz*gtZZ*phidy*phidz + 16*gtxx*gtXZ*gtyy*(phidz*phidz) + 32*gtxy*gtyy*gtYZ*(phidz*phidz) + 32*gtxz*gtyy*gtZZ*(phidz*phidz) + gtXY*(-(gtxxdz*gtyydy) + gtxxdy*gtyydz + 8*gtxy*gtxzdy*phidy - 4*gtxxdz*gtyy*phidy + 4*gtxx*gtyydz*phidy - 8*gtxy*gtyzdx*phidy - 32*gtxy*gtyz*phidx*phidy + 32*gtxy*gtxz*(phidy*phidy) + 2*gtxydz*(gtyydx + 4*gtyy*phidx - 4*gtxy*phidy) + 4*gtxxdy*gtyy*phidz + 8*gtxy*gtyydx*phidz - 4*gtxx*gtyydy*phidz + 32*gtxy*gtyy*phidx*phidz - 32*(gtxy*gtxy)*phidy*phidz - 2*gtxydy*(gtxydz - gtxzdy + gtyzdx + 4*gtyz*phidx - 4*gtxz*phidy + 4*gtxy*phidz))) + 8*gtXZ*gtyy*phidzdz)/4.;
    double const Rxzyz=(4*gtXXdy*gtxzdz + 2*(gtXY*gtXY)*gtxzdz*gtyydy - 2*(gtXY*gtXY)*gtxydz*gtyydz - 2*gtXYdz*gtyydz - gtXY*gtYY*(gtyydz*gtyydz) - 2*gtXY*gtyydzdz + 4*gtXY*gtXZ*gtxzdz*gtyzdy + 4*gtXY*gtyzdydz + 4*gtXYdy*gtyzdz - 4*gtXY*gtxydz*gtXZ*gtyzdz + 2*(gtXY*gtXY)*gtyydx*gtyzdz + 2*gtXY*gtYY*gtyydy*gtyzdz - 2*gtXZ*gtYY*gtyydz*gtyzdz + 4*gtXY*gtXZ*gtyzdx*gtyzdz + 4*gtXZ*gtYY*gtyzdy*gtyzdz - 2*gtXXdy*gtzzdx - gtXY*gtXY*gtyydy*gtzzdx - gtXY*gtXZ*gtyydz*gtzzdx - 2*gtXY*gtXZ*gtyzdy*gtzzdx - 2*gtXYdy*gtzzdy + 2*(gtXZ*gtXZ)*gtxzdz*gtzzdy - 2*gtXZdz*gtzzdy - gtXY*gtXY*gtyydx*gtzzdy - gtXY*gtYY*gtyydy*gtzzdy - gtXY*gtyydz*gtYZ*gtzzdy - 2*gtXY*gtXZ*gtyzdx*gtzzdy - 2*gtXZ*gtYY*gtyzdy*gtzzdy + 2*gtXZ*gtYZ*gtyzdz*gtzzdy - 2*gtXY*gtyzdz*gtZZ*gtzzdy - 2*(gtXZ*gtXZ)*gtzzdx*gtzzdy - 2*gtXZ*gtYZ*(gtzzdy*gtzzdy) + gtXY*gtZZ*(gtzzdy*gtzzdy) - 2*gtXY*gtzzdydy - 2*gtxydz*(gtXZ*gtXZ)*gtzzdz + 2*gtXZdy*gtzzdz + gtXY*gtXZ*gtyydx*gtzzdz + gtXY*gtyydy*gtYZ*gtzzdz - 2*gtXZ*gtyydz*gtYZ*gtzzdz + 2*(gtXZ*gtXZ)*gtyzdx*gtzzdz + 2*gtXZ*gtYZ*gtyzdy*gtzzdz + gtXY*gtyydz*gtZZ*gtzzdz + 8*(gtXY*gtXY)*gtyy*gtyzdz*phidx + 16*gtXY*gtXZ*gtyz*gtyzdz*phidx - 8*gtXXdy*gtzz*phidx - 4*(gtXY*gtXY)*gtyydy*gtzz*phidx - 4*gtXY*gtXZ*gtyydz*gtzz*phidx - 8*gtXY*gtXZ*gtyzdy*gtzz*phidx - 4*(gtXY*gtXY)*gtyy*gtzzdy*phidx - 8*gtXY*gtXZ*gtyz*gtzzdy*phidx - 8*(gtXZ*gtXZ)*gtzz*gtzzdy*phidx + 4*gtXY*gtXZ*gtyy*gtzzdz*phidx + 8*(gtXZ*gtXZ)*gtyz*gtzzdz*phidx + 8*(gtXY*gtXY)*gtxzdz*gtyy*phidy + 16*gtXY*gtXZ*gtxzdz*gtyz*phidy + 8*gtXY*gtyy*gtYY*gtyzdz*phidy + 16*gtXZ*gtYY*gtyz*gtyzdz*phidy - 8*gtXYdy*gtzz*phidy + 8*(gtXZ*gtXZ)*gtxzdz*gtzz*phidy - 8*gtXZdz*gtzz*phidy - 4*(gtXY*gtXY)*gtyydx*gtzz*phidy - 4*gtXY*gtYY*gtyydy*gtzz*phidy - 4*gtXY*gtyydz*gtYZ*gtzz*phidy - 8*gtXY*gtXZ*gtyzdx*gtzz*phidy - 8*gtXZ*gtYY*gtyzdy*gtzz*phidy + 8*gtXZ*gtYZ*gtyzdz*gtzz*phidy - 8*gtXY*gtyzdz*gtzz*gtZZ*phidy - 4*(gtXY*gtXY)*gtyy*gtzzdx*phidy - 8*gtXY*gtXZ*gtyz*gtzzdx*phidy - 8*(gtXZ*gtXZ)*gtzz*gtzzdx*phidy - 8*gtXY*gtzzdy*phidy - 4*gtXY*gtyy*gtYY*gtzzdy*phidy - 8*gtXZ*gtYY*gtyz*gtzzdy*phidy - 16*gtXZ*gtYZ*gtzz*gtzzdy*phidy + 8*gtXY*gtzz*gtZZ*gtzzdy*phidy - 8*gtXZ*gtzzdz*phidy + 4*gtXY*gtyy*gtYZ*gtzzdz*phidy + 8*gtXZ*gtyz*gtYZ*gtzzdz*phidy - 32*(gtXY*gtXY)*gtyy*gtzz*phidx*phidy - 64*gtXY*gtXZ*gtyz*gtzz*phidx*phidy - 32*(gtXZ*gtXZ)*(gtzz*gtzz)*phidx*phidy - 16*gtXY*gtyy*gtYY*gtzz*(phidy*phidy) - 32*gtXZ*gtYY*gtyz*gtzz*(phidy*phidy) - 32*gtXZ*gtYZ*(gtzz*gtzz)*(phidy*phidy) + 16*gtXY*(gtzz*gtzz)*gtZZ*(phidy*phidy) - 8*gtXY*gtzz*phidydy + 16*gtXY*gtyz*phidydz + 16*gtXXdy*gtxz*phidz - 8*(gtXY*gtXY)*gtxydz*gtyy*phidz - 8*gtXYdz*gtyy*phidz + 8*(gtXY*gtXY)*gtxz*gtyydy*phidz - 8*gtXY*gtyydz*phidz - 8*gtxy*(gtXY*gtXY)*gtyydz*phidz - 8*gtXY*gtyy*gtYY*gtyydz*phidz + 16*gtXYdy*gtyz*phidz - 16*gtXY*gtxydz*gtXZ*gtyz*phidz + 8*(gtXY*gtXY)*gtyydx*gtyz*phidz + 8*gtXY*gtYY*gtyydy*gtyz*phidz - 8*gtXZ*gtYY*gtyydz*gtyz*phidz + 16*gtXY*gtXZ*gtyz*gtyzdx*phidz + 16*gtXY*gtyzdy*phidz + 16*gtXY*gtxz*gtXZ*gtyzdy*phidz + 16*gtXZ*gtYY*gtyz*gtyzdy*phidz - 16*gtxy*gtXY*gtXZ*gtyzdz*phidz - 8*gtXZ*gtyy*gtYY*gtyzdz*phidz - 8*gtxydz*(gtXZ*gtXZ)*gtzz*phidz + 8*gtXZdy*gtzz*phidz + 4*gtXY*gtXZ*gtyydx*gtzz*phidz + 4*gtXY*gtyydy*gtYZ*gtzz*phidz - 8*gtXZ*gtyydz*gtYZ*gtzz*phidz + 8*(gtXZ*gtXZ)*gtyzdx*gtzz*phidz + 8*gtXZ*gtYZ*gtyzdy*gtzz*phidz + 4*gtXY*gtyydz*gtzz*gtZZ*phidz - 4*gtXY*gtXZ*gtyy*gtzzdx*phidz + 8*gtXZ*gtzzdy*phidz + 8*gtxz*(gtXZ*gtXZ)*gtzzdy*phidz - 4*gtXY*gtyy*gtYZ*gtzzdy*phidz + 8*gtXZ*gtyz*gtYZ*gtzzdy*phidz - 8*gtXY*gtyz*gtZZ*gtzzdy*phidz - 8*gtxy*(gtXZ*gtXZ)*gtzzdz*phidz - 8*gtXZ*gtyy*gtYZ*gtzzdz*phidz + 4*gtXY*gtyy*gtZZ*gtzzdz*phidz + 32*(gtXY*gtXY)*gtyy*gtyz*phidx*phidz + 64*gtXY*gtXZ*(gtyz*gtyz)*phidx*phidz + 32*(gtXZ*gtXZ)*gtyz*gtzz*phidx*phidz + 32*(gtXY*gtXY)*gtxz*gtyy*phidy*phidz + 64*gtXY*gtxz*gtXZ*gtyz*phidy*phidz + 32*gtXY*gtyy*gtYY*gtyz*phidy*phidz + 64*gtXZ*gtYY*(gtyz*gtyz)*phidy*phidz + 32*gtxz*(gtXZ*gtXZ)*gtzz*phidy*phidz + 64*gtXZ*gtyz*gtYZ*gtzz*phidy*phidz - 32*gtXY*gtyz*gtzz*gtZZ*phidy*phidz - 32*gtxy*(gtXY*gtXY)*gtyy*(phidz*phidz) - 16*gtXY*(gtyy*gtyy)*gtYY*(phidz*phidz) - 64*gtxy*gtXY*gtXZ*gtyz*(phidz*phidz) - 32*gtXZ*gtyy*gtYY*gtyz*(phidz*phidz) - 32*gtxy*(gtXZ*gtXZ)*gtzz*(phidz*phidz) - 32*gtXZ*gtyy*gtYZ*gtzz*(phidz*phidz) + 16*gtXY*gtyy*gtzz*gtZZ*(phidz*phidz) - 2*gtXXdz*(gtxydz + gtxzdy - gtyzdx - 4*gtyz*phidx + 4*gtxz*phidy + 4*gtxy*phidz) - gtXX*gtXX*(gtxxdz*(gtxydz + gtxzdy - gtyzdx - 4*gtyz*phidx + 4*gtxz*phidy + 4*gtxy*phidz) + gtxxdy*(-2*gtxzdz + gtzzdx + 4*gtzz*phidx - 8*gtxz*phidz) + 4*gtxx*(-2*gtxzdz*phidy + gtzzdx*phidy + 4*gtzz*phidx*phidy + gtxydz*phidz + gtxzdy*phidz - gtyzdx*phidz - 4*gtyz*phidx*phidz - 4*gtxz*phidy*phidz + 4*gtxy*(phidz*phidz))) - 8*gtXY*gtyy*phidzdz - gtXX*(2*gtxydzdz - 2*gtxzdydz + 2*gtxydz*gtXZ*gtxzdz - 2*gtXZ*gtxzdy*gtxzdz + gtxydz*gtYY*gtyydz + gtxzdy*gtYY*gtyydz + 2*gtxzdz*gtyydz*gtYZ - 2*gtXZ*gtxzdz*gtyzdx - gtYY*gtyydz*gtyzdx - 2*gtyzdxdz - 4*gtxydy*gtYY*gtyzdz + 2*gtYY*gtyydx*gtyzdz - 2*gtxydz*gtYZ*gtyzdz - 2*gtxzdy*gtYZ*gtyzdz + 2*gtYZ*gtyzdx*gtyzdz + 2*gtXZ*gtxzdy*gtzzdx - gtyydz*gtYZ*gtzzdx + 2*gtzzdxdy + gtxxdz*gtXZ*gtzzdy + 2*gtxydy*gtYY*gtzzdy - gtYY*gtyydx*gtzzdy + 2*gtxydz*gtYZ*gtzzdy + 2*gtxzdy*gtYZ*gtzzdy - 2*gtYZ*gtyzdx*gtzzdy + 2*gtxzdz*gtZZ*gtzzdy - gtZZ*gtzzdx*gtzzdy - gtxxdy*gtXZ*gtzzdz - 2*gtxydy*gtYZ*gtzzdz + gtyydx*gtYZ*gtzzdz - gtxydz*gtZZ*gtzzdz - gtxzdy*gtZZ*gtzzdz + gtyzdx*gtZZ*gtzzdz - 8*gtXZ*gtxzdz*gtyz*phidx - 4*gtYY*gtyydz*gtyz*phidx - 8*gtyzdz*phidx + 8*gtyy*gtYY*gtyzdz*phidx + 8*gtyz*gtYZ*gtyzdz*phidx + 8*gtXZ*gtxzdy*gtzz*phidx - 4*gtyydz*gtYZ*gtzz*phidx + 8*gtzzdy*phidx - 4*gtyy*gtYY*gtzzdy*phidx - 8*gtyz*gtYZ*gtzzdy*phidx - 4*gtzz*gtZZ*gtzzdy*phidx + 4*gtyy*gtYZ*gtzzdz*phidx + 4*gtyz*gtZZ*gtzzdz*phidx + 8*gtzz*phidxdy - 8*gtyz*phidxdz + 8*gtxzdz*phidy - 8*gtxz*gtXZ*gtxzdz*phidy + 4*gtxz*gtYY*gtyydz*phidy - 16*gtxy*gtYY*gtyzdz*phidy - 8*gtxz*gtYZ*gtyzdz*phidy + 4*gtxxdz*gtXZ*gtzz*phidy + 8*gtxydy*gtYY*gtzz*phidy - 4*gtYY*gtyydx*gtzz*phidy + 8*gtxydz*gtYZ*gtzz*phidy + 8*gtxzdy*gtYZ*gtzz*phidy - 8*gtYZ*gtyzdx*gtzz*phidy + 8*gtxzdz*gtzz*gtZZ*phidy + 8*gtxz*gtXZ*gtzzdx*phidy - 4*gtzz*gtZZ*gtzzdx*phidy + 8*gtxy*gtYY*gtzzdy*phidy + 8*gtxz*gtYZ*gtzzdy*phidy - 4*gtxx*gtXZ*gtzzdz*phidy - 8*gtxy*gtYZ*gtzzdz*phidy - 4*gtxz*gtZZ*gtzzdz*phidy + 32*gtxz*gtXZ*gtzz*phidx*phidy - 16*gtyy*gtYY*gtzz*phidx*phidy - 32*gtyz*gtYZ*gtzz*phidx*phidy - 16*(gtzz*gtzz)*gtZZ*phidx*phidy + 32*gtxy*gtYY*gtzz*(phidy*phidy) + 32*gtxz*gtYZ*gtzz*(phidy*phidy) - 8*gtxz*phidydz + 8*gtxydz*phidz + 8*gtxydz*gtxz*gtXZ*phidz - 16*gtxzdy*phidz - 8*gtxz*gtXZ*gtxzdy*phidz + 8*gtxy*gtXZ*gtxzdz*phidz + 4*gtxydz*gtyy*gtYY*phidz + 4*gtxzdy*gtyy*gtYY*phidz + 4*gtxy*gtYY*gtyydz*phidz - 16*gtxydy*gtYY*gtyz*phidz + 8*gtYY*gtyydx*gtyz*phidz + 8*gtxzdz*gtyy*gtYZ*phidz + 8*gtxz*gtyydz*gtYZ*phidz - 8*gtxydz*gtyz*gtYZ*phidz - 8*gtxzdy*gtyz*gtYZ*phidz - 8*gtxz*gtXZ*gtyzdx*phidz - 4*gtyy*gtYY*gtyzdx*phidz + 8*gtyz*gtYZ*gtyzdx*phidz - 8*gtxy*gtYZ*gtyzdz*phidz - 4*gtxxdy*gtXZ*gtzz*phidz - 8*gtxydy*gtYZ*gtzz*phidz + 4*gtyydx*gtYZ*gtzz*phidz - 4*gtxydz*gtzz*gtZZ*phidz - 4*gtxzdy*gtzz*gtZZ*phidz + 4*gtyzdx*gtzz*gtZZ*phidz - 4*gtyy*gtYZ*gtzzdx*phidz + 4*gtxx*gtXZ*gtzzdy*phidz + 8*gtxy*gtYZ*gtzzdy*phidz + 8*gtxz*gtZZ*gtzzdy*phidz - 4*gtxy*gtZZ*gtzzdz*phidz - 32*gtxz*gtXZ*gtyz*phidx*phidz + 16*gtyy*gtYY*gtyz*phidx*phidz + 32*(gtyz*gtyz)*gtYZ*phidx*phidz + 16*gtyz*gtzz*gtZZ*phidx*phidz - 32*(gtxz*gtxz)*gtXZ*phidy*phidz + 16*gtxz*gtyy*gtYY*phidy*phidz - 64*gtxy*gtYY*gtyz*phidy*phidz - 32*gtxz*gtyz*gtYZ*phidy*phidz + 16*gtxz*gtzz*gtZZ*phidy*phidz + 32*gtxy*gtxz*gtXZ*(phidz*phidz) + 16*gtxy*gtyy*gtYY*(phidz*phidz) + 32*gtxz*gtyy*gtYZ*(phidz*phidz) - 32*gtxy*gtyz*gtYZ*(phidz*phidz) - 16*gtxy*gtzz*gtZZ*(phidz*phidz) + gtXY*(2*(gtxydz*gtxydz) + gtxxdz*gtyydz - 2*gtxxdy*gtyzdz + gtxxdy*gtzzdy - 16*gtxy*gtxzdz*phidy - 8*gtxx*gtyzdz*phidy + 4*gtxxdy*gtzz*phidy + 8*gtxy*gtzzdx*phidy + 4*gtxx*gtzzdy*phidy + 32*gtxy*gtzz*phidx*phidy + 16*gtxx*gtzz*(phidy*phidy) + 8*gtxy*gtxzdy*phidz + 4*gtxxdz*gtyy*phidz + 4*gtxx*gtyydz*phidz - 8*gtxxdy*gtyz*phidz - 8*gtxy*gtyzdx*phidz - 32*gtxy*gtyz*phidx*phidz - 32*gtxy*gtxz*phidy*phidz - 32*gtxx*gtyz*phidy*phidz + 32*(gtxy*gtxy)*(phidz*phidz) + 16*gtxx*gtyy*(phidz*phidz) + 2*gtxydz*(gtxzdy - gtyzdx - 4*gtyz*phidx + 4*gtxz*phidy + 8*gtxy*phidz) + 2*gtxydy*(-2*gtxzdz + gtzzdx + 4*gtzz*phidx - 8*gtxz*phidz)) + 8*gtxy*phidzdz))/4.;


    // linearly independent components of Ricci tensor
    double const Rxx=(gtYY*Rxyxy + 2*gtYZ*Rxyxz + gtZZ*Rxzxz)*exp(-4.*phi);
    double const Rxy=(-(gtXY*Rxyxy) - gtXZ*Rxyxz + gtYZ*Rxyyz + gtZZ*Rxzyz)*exp(-4.*phi);
    double const Rxz=-((gtXY*Rxyxz + gtYY*Rxyyz + gtXZ*Rxzxz + gtYZ*Rxzyz)*exp(-4.*phi));
    double const Ryy=(gtXX*Rxyxy - 2*gtXZ*Rxyyz + gtZZ*Ryzyz)*exp(-4.*phi);
    double const Ryz=(gtXX*Rxyxz + gtXY*Rxyyz - gtXZ*Rxzyz - gtYZ*Ryzyz)*exp(-4.*phi);
    double const Rzz=(gtXX*Rxzxz + 2*gtXY*Rxzyz + gtYY*Ryzyz)*exp(-4.*phi);

    // real part of psi4
    double const psi4Re= K*(Kxx*(-(mXim*mXim) + mXre*mXre) - 2*Kxy*mXim*mYim - Kyy*(mYim*mYim) + 2*Kxy*mXre*mYre + Kyy*(mYre*mYre) - 2*Kxz*mXim*mZim - 2*Kyz*mYim*mZim - Kzz*(mZim*mZim) + 2*Kxz*mXre*mZre + 2*Kyz*mYre*mZre + Kzz*(mZre*mZre))*(n0*n0) + (Kxz*KXz*(mXim*mXim) - Kxz*KXz*(mXre*mXre) + KXy*Kyy*mXim*mYim + KXz*Kyz*mXim*mYim + Kxz*KYz*mXim*mYim + Kyy*KYy*(mYim*mYim) + Kyz*KYz*(mYim*mYim) - KXy*Kyy*mXre*mYre - KXz*Kyz*mXre*mYre - Kxz*KYz*mXre*mYre - Kyy*KYy*(mYre*mYre) - Kyz*KYz*(mYre*mYre) + KXx*Kxz*mXim*mZim + KXy*Kyz*mXim*mZim + KXz*Kzz*mXim*mZim + Kxz*KZz*mXim*mZim + KXy*Kxz*mYim*mZim + KYy*Kyz*mYim*mZim + Kyy*KYz*mYim*mZim + KYz*Kzz*mYim*mZim + Kyz*KZz*mYim*mZim + Kxz*KXz*(mZim*mZim) + Kyz*KYz*(mZim*mZim) + Kzz*KZz*(mZim*mZim) - KXx*Kxz*mXre*mZre - KXy*Kyz*mXre*mZre - KXz*Kzz*mXre*mZre - Kxz*KZz*mXre*mZre - KXy*Kxz*mYre*mZre - KYy*Kyz*mYre*mZre - Kyy*KYz*mYre*mZre - KYz*Kzz*mYre*mZre - Kyz*KZz*mYre*mZre - Kxz*KXz*(mZre*mZre) - Kyz*KYz*(mZre*mZre) - Kzz*KZz*(mZre*mZre) + Kxx*(KXx*(mXim*mXim - mXre*mXre) + KXy*mXim*mYim - KXy*mXre*mYre + KXz*mXim*mZim - KXz*mXre*mZre) + Kxy*(KXx*mXim*mYim + KYy*mXim*mYim - KXx*mXre*mYre - KYy*mXre*mYre + KXy*(mXim*mXim - mXre*mXre + mYim*mYim - mYre*mYre) + KYz*mXim*mZim + KXz*mYim*mZim - KYz*mXre*mZre - KXz*mYre*mZre))*(n0*n0) + Kxz*Kxz*(mZim*mZim)*(nXre*nXre) - Kxz*Kxz*(mZre*mZre)*(nXre*nXre) + 2*Kxz*Kyy*mYim*mZim*nXre*nYre + 2*Kxz*Kyz*(mZim*mZim)*nXre*nYre - 2*Kxz*Kyy*mYre*mZre*nXre*nYre - 2*Kxz*Kyz*(mZre*mZre)*nXre*nYre - 2*Kxz*Kyy*mXim*mZim*(nYre*nYre) + Kyz*Kyz*(mZim*mZim)*(nYre*nYre) - Kyy*Kzz*(mZim*mZim)*(nYre*nYre) + 2*Kxz*Kyy*mXre*mZre*(nYre*nYre) - Kyz*Kyz*(mZre*mZre)*(nYre*nYre) + Kyy*Kzz*(mZre*mZre)*(nYre*nYre) + Kxy*Kxy*(mYim*mYim*(nXre*nXre) - mYre*mYre*(nXre*nXre) - 2*mXim*mYim*nXre*nYre + 2*mXre*mYre*nXre*nYre + (mXim*mXim - mXre*mXre)*(nYre*nYre)) - 2*Kxz*Kyy*(mYim*mYim)*nXre*nZre + 2*Kxz*Kyy*(mYre*mYre)*nXre*nZre - 2*(Kxz*Kxz)*mXim*mZim*nXre*nZre - 2*Kxz*Kyz*mYim*mZim*nXre*nZre + 2*(Kxz*Kxz)*mXre*mZre*nXre*nZre + 2*Kxz*Kyz*mYre*mZre*nXre*nZre + 2*Kxz*Kyy*mXim*mYim*nYre*nZre - 2*Kxz*Kyy*mXre*mYre*nYre*nZre - 2*Kxz*Kyz*mXim*mZim*nYre*nZre - 2*(Kyz*Kyz)*mYim*mZim*nYre*nZre + 2*Kyy*Kzz*mYim*mZim*nYre*nZre + 2*Kxz*Kyz*mXre*mZre*nYre*nZre + 2*(Kyz*Kyz)*mYre*mZre*nYre*nZre - 2*Kyy*Kzz*mYre*mZre*nYre*nZre + Kxz*Kxz*(mXim*mXim)*(nZre*nZre) - Kxz*Kxz*(mXre*mXre)*(nZre*nZre) + 2*Kxz*Kyz*mXim*mYim*(nZre*nZre) + Kyz*Kyz*(mYim*mYim)*(nZre*nZre) - Kyy*Kzz*(mYim*mYim)*(nZre*nZre) - 2*Kxz*Kyz*mXre*mYre*(nZre*nZre) - Kyz*Kyz*(mYre*mYre)*(nZre*nZre) + Kyy*Kzz*(mYre*mYre)*(nZre*nZre) - 2*n0*(-(Kxydy*(mYim*mYim)*nXre) + Kyydx*(mYim*mYim)*nXre + Kxydy*(mYre*mYre)*nXre - Kyydx*(mYre*mYre)*nXre - Kxxdz*mXim*mZim*nXre + Kxzdx*mXim*mZim*nXre - Kxydz*mYim*mZim*nXre - Kxzdy*mYim*mZim*nXre + 2*Kyzdx*mYim*mZim*nXre - Kxzdz*(mZim*mZim)*nXre + Kzzdx*(mZim*mZim)*nXre + Kxxdz*mXre*mZre*nXre - Kxzdx*mXre*mZre*nXre + Kxydz*mYre*mZre*nXre + Kxzdy*mYre*mZre*nXre - 2*Kyzdx*mYre*mZre*nXre + Kxzdz*(mZre*mZre)*nXre - Kzzdx*(mZre*mZre)*nXre + Kxydy*mXim*mYim*nYre - Kyydx*mXim*mYim*nYre - Kxydy*mXre*mYre*nYre + Kyydx*mXre*mYre*nYre - Kxydz*mXim*mZim*nYre + 2*Kxzdy*mXim*mZim*nYre - Kyzdx*mXim*mZim*nYre - Kyydz*mYim*mZim*nYre + Kyzdy*mYim*mZim*nYre - Kyzdz*(mZim*mZim)*nYre + Kzzdy*(mZim*mZim)*nYre + Kxydz*mXre*mZre*nYre - 2*Kxzdy*mXre*mZre*nYre + Kyzdx*mXre*mZre*nYre + Kyydz*mYre*mZre*nYre - Kyzdy*mYre*mZre*nYre + Kyzdz*(mZre*mZre)*nYre - Kzzdy*(mZre*mZre)*nYre + Kxydx*(mXim*mYim*nXre - mXre*mYre*nXre - mXim*mXim*nYre + mXre*mXre*nYre) + Kxxdy*(-(mXim*mYim*nXre) + mXim*mXim*nYre + mXre*(mYre*nXre - mXre*nYre)) + Kxxdz*(mXim*mXim)*nZre - Kxzdx*(mXim*mXim)*nZre - Kxxdz*(mXre*mXre)*nZre + Kxzdx*(mXre*mXre)*nZre + 2*Kxydz*mXim*mYim*nZre - Kxzdy*mXim*mYim*nZre - Kyzdx*mXim*mYim*nZre + Kyydz*(mYim*mYim)*nZre - Kyzdy*(mYim*mYim)*nZre - 2*Kxydz*mXre*mYre*nZre + Kxzdy*mXre*mYre*nZre + Kyzdx*mXre*mYre*nZre - Kyydz*(mYre*mYre)*nZre + Kyzdy*(mYre*mYre)*nZre + Kxzdz*mXim*mZim*nZre - Kzzdx*mXim*mZim*nZre + Kyzdz*mYim*mZim*nZre - Kzzdy*mYim*mZim*nZre - Kxzdz*mXre*mZre*nZre + Kzzdx*mXre*mZre*nZre - Kyzdz*mYre*mZre*nZre + Kzzdy*mYre*mZre*nZre) + 2*Kxy*(Kzz*(-(mZim*mZim*nXre*nYre) + mZre*mZre*nXre*nYre + mZim*(mYim*nXre + mXim*nYre)*nZre - mZre*(mYre*nXre + mXre*nYre)*nZre + (-(mXim*mYim) + mXre*mYre)*(nZre*nZre)) + Kyz*((mXim*mZim - mXre*mZre)*(nYre*nYre) + mYim*mYim*nXre*nZre - mYre*mYre*nXre*nZre - mYim*nYre*(mZim*nXre + mXim*nZre) + mYre*nYre*(mZre*nXre + mXre*nZre)) + Kxz*(mYim*nXre*(mZim*nXre - mXim*nZre) + mYre*nXre*(-(mZre*nXre) + mXre*nZre) + nYre*(-(mXim*mZim*nXre) + mXre*mZre*nXre + mXim*mXim*nZre - mXre*mXre*nZre))) + Kxx*(Kyy*(-(mYim*mYim*(nXre*nXre)) + mYre*mYre*(nXre*nXre) + 2*mXim*mYim*nXre*nYre - 2*mXre*mYre*nXre*nYre + (-(mXim*mXim) + mXre*mXre)*(nYre*nYre)) + Kzz*(-(mZim*mZim*(nXre*nXre)) + mZre*mZre*(nXre*nXre) + 2*mXim*mZim*nXre*nZre - 2*mXre*mZre*nXre*nZre + (-(mXim*mXim) + mXre*mXre)*(nZre*nZre)) + 2*Kyz*(mYim*nXre*(-(mZim*nXre) + mXim*nZre) + mYre*nXre*(mZre*nXre - mXre*nZre) + nYre*(mXim*mZim*nXre - mXre*mZre*nXre - mXim*mXim*nZre + mXre*mXre*nZre))) - 2*n0*(GammaYxy*Kxy*mXim*mYim*nXre + GammaZxy*Kxz*mXim*mYim*nXre - GammaYxx*Kyy*mXim*mYim*nXre - GammaZxx*Kyz*mXim*mYim*nXre + GammaXyy*Kxx*(mYim*mYim)*nXre + GammaYyy*Kxy*(mYim*mYim)*nXre + GammaZyy*Kxz*(mYim*mYim)*nXre - GammaYxy*Kyy*(mYim*mYim)*nXre - GammaZxy*Kyz*(mYim*mYim)*nXre - GammaYxy*Kxy*mXre*mYre*nXre - GammaZxy*Kxz*mXre*mYre*nXre + GammaYxx*Kyy*mXre*mYre*nXre + GammaZxx*Kyz*mXre*mYre*nXre - GammaXyy*Kxx*(mYre*mYre)*nXre - GammaYyy*Kxy*(mYre*mYre)*nXre - GammaZyy*Kxz*(mYre*mYre)*nXre + GammaYxy*Kyy*(mYre*mYre)*nXre + GammaZxy*Kyz*(mYre*mYre)*nXre + GammaXxz*Kxx*mXim*mZim*nXre + GammaYxz*Kxy*mXim*mZim*nXre + GammaZxz*Kxz*mXim*mZim*nXre - GammaYxx*Kyz*mXim*mZim*nXre - GammaZxx*Kzz*mXim*mZim*nXre + 2*GammaXyz*Kxx*mYim*mZim*nXre - GammaXxz*Kxy*mYim*mZim*nXre + 2*GammaYyz*Kxy*mYim*mZim*nXre + 2*GammaZyz*Kxz*mYim*mZim*nXre - GammaYxz*Kyy*mYim*mZim*nXre - GammaYxy*Kyz*mYim*mZim*nXre - GammaZxz*Kyz*mYim*mZim*nXre - GammaZxy*Kzz*mYim*mZim*nXre + GammaXzz*Kxx*(mZim*mZim)*nXre + GammaYzz*Kxy*(mZim*mZim)*nXre - GammaXxz*Kxz*(mZim*mZim)*nXre + GammaZzz*Kxz*(mZim*mZim)*nXre - GammaYxz*Kyz*(mZim*mZim)*nXre - GammaZxz*Kzz*(mZim*mZim)*nXre - GammaXxz*Kxx*mXre*mZre*nXre - GammaYxz*Kxy*mXre*mZre*nXre - GammaZxz*Kxz*mXre*mZre*nXre + GammaYxx*Kyz*mXre*mZre*nXre + GammaZxx*Kzz*mXre*mZre*nXre - 2*GammaXyz*Kxx*mYre*mZre*nXre + GammaXxz*Kxy*mYre*mZre*nXre - 2*GammaYyz*Kxy*mYre*mZre*nXre - 2*GammaZyz*Kxz*mYre*mZre*nXre + GammaYxz*Kyy*mYre*mZre*nXre + GammaYxy*Kyz*mYre*mZre*nXre + GammaZxz*Kyz*mYre*mZre*nXre + GammaZxy*Kzz*mYre*mZre*nXre - GammaXzz*Kxx*(mZre*mZre)*nXre - GammaYzz*Kxy*(mZre*mZre)*nXre + GammaXxz*Kxz*(mZre*mZre)*nXre - GammaZzz*Kxz*(mZre*mZre)*nXre + GammaYxz*Kyz*(mZre*mZre)*nXre + GammaZxz*Kzz*(mZre*mZre)*nXre - GammaYxy*Kxy*(mXim*mXim)*nYre - GammaZxy*Kxz*(mXim*mXim)*nYre + GammaYxx*Kyy*(mXim*mXim)*nYre + GammaZxx*Kyz*(mXim*mXim)*nYre + GammaYxy*Kxy*(mXre*mXre)*nYre + GammaZxy*Kxz*(mXre*mXre)*nYre - GammaYxx*Kyy*(mXre*mXre)*nYre - GammaZxx*Kyz*(mXre*mXre)*nYre - GammaXyy*Kxx*mXim*mYim*nYre - GammaYyy*Kxy*mXim*mYim*nYre - GammaZyy*Kxz*mXim*mYim*nYre + GammaYxy*Kyy*mXim*mYim*nYre + GammaZxy*Kyz*mXim*mYim*nYre + GammaXyy*Kxx*mXre*mYre*nYre + GammaYyy*Kxy*mXre*mYre*nYre + GammaZyy*Kxz*mXre*mYre*nYre - GammaYxy*Kyy*mXre*mYre*nYre - GammaZxy*Kyz*mXre*mYre*nYre - GammaXyz*Kxx*mXim*mZim*nYre + 2*GammaXxz*Kxy*mXim*mZim*nYre - GammaYyz*Kxy*mXim*mZim*nYre - GammaZyz*Kxz*mXim*mZim*nYre + 2*GammaYxz*Kyy*mXim*mZim*nYre - GammaYxy*Kyz*mXim*mZim*nYre + 2*GammaZxz*Kyz*mXim*mZim*nYre - GammaZxy*Kzz*mXim*mZim*nYre + GammaXyz*Kxy*mYim*mZim*nYre - GammaXyy*Kxz*mYim*mZim*nYre + GammaYyz*Kyy*mYim*mZim*nYre - GammaYyy*Kyz*mYim*mZim*nYre + GammaZyz*Kyz*mYim*mZim*nYre - GammaZyy*Kzz*mYim*mZim*nYre + GammaXzz*Kxy*(mZim*mZim)*nYre - GammaXyz*Kxz*(mZim*mZim)*nYre + GammaYzz*Kyy*(mZim*mZim)*nYre - GammaYyz*Kyz*(mZim*mZim)*nYre + GammaZzz*Kyz*(mZim*mZim)*nYre - GammaZyz*Kzz*(mZim*mZim)*nYre + GammaXyz*Kxx*mXre*mZre*nYre - 2*GammaXxz*Kxy*mXre*mZre*nYre + GammaYyz*Kxy*mXre*mZre*nYre + GammaZyz*Kxz*mXre*mZre*nYre - 2*GammaYxz*Kyy*mXre*mZre*nYre + GammaYxy*Kyz*mXre*mZre*nYre - 2*GammaZxz*Kyz*mXre*mZre*nYre + GammaZxy*Kzz*mXre*mZre*nYre - GammaXyz*Kxy*mYre*mZre*nYre + GammaXyy*Kxz*mYre*mZre*nYre - GammaYyz*Kyy*mYre*mZre*nYre + GammaYyy*Kyz*mYre*mZre*nYre - GammaZyz*Kyz*mYre*mZre*nYre + GammaZyy*Kzz*mYre*mZre*nYre - GammaXzz*Kxy*(mZre*mZre)*nYre + GammaXyz*Kxz*(mZre*mZre)*nYre - GammaYzz*Kyy*(mZre*mZre)*nYre + GammaYyz*Kyz*(mZre*mZre)*nYre - GammaZzz*Kyz*(mZre*mZre)*nYre + GammaZyz*Kzz*(mZre*mZre)*nYre + GammaXxx*Kxy*(-(mXim*mYim*nXre) + mXim*mXim*nYre + mXre*(mYre*nXre - mXre*nYre)) - GammaXxz*Kxx*(mXim*mXim)*nZre - GammaYxz*Kxy*(mXim*mXim)*nZre - GammaZxz*Kxz*(mXim*mXim)*nZre + GammaYxx*Kyz*(mXim*mXim)*nZre + GammaZxx*Kzz*(mXim*mXim)*nZre + GammaXxz*Kxx*(mXre*mXre)*nZre + GammaYxz*Kxy*(mXre*mXre)*nZre + GammaZxz*Kxz*(mXre*mXre)*nZre - GammaYxx*Kyz*(mXre*mXre)*nZre - GammaZxx*Kzz*(mXre*mXre)*nZre - GammaXyz*Kxx*mXim*mYim*nZre - GammaXxz*Kxy*mXim*mYim*nZre - GammaYyz*Kxy*mXim*mYim*nZre - GammaZyz*Kxz*mXim*mYim*nZre - GammaYxz*Kyy*mXim*mYim*nZre + 2*GammaYxy*Kyz*mXim*mYim*nZre - GammaZxz*Kyz*mXim*mYim*nZre + 2*GammaZxy*Kzz*mXim*mYim*nZre - GammaXyz*Kxy*(mYim*mYim)*nZre + GammaXyy*Kxz*(mYim*mYim)*nZre - GammaYyz*Kyy*(mYim*mYim)*nZre + GammaYyy*Kyz*(mYim*mYim)*nZre - GammaZyz*Kyz*(mYim*mYim)*nZre + GammaZyy*Kzz*(mYim*mYim)*nZre + GammaXyz*Kxx*mXre*mYre*nZre + GammaXxz*Kxy*mXre*mYre*nZre + GammaYyz*Kxy*mXre*mYre*nZre + GammaZyz*Kxz*mXre*mYre*nZre + GammaYxz*Kyy*mXre*mYre*nZre - 2*GammaYxy*Kyz*mXre*mYre*nZre + GammaZxz*Kyz*mXre*mYre*nZre - 2*GammaZxy*Kzz*mXre*mYre*nZre + GammaXyz*Kxy*(mYre*mYre)*nZre - GammaXyy*Kxz*(mYre*mYre)*nZre + GammaYyz*Kyy*(mYre*mYre)*nZre - GammaYyy*Kyz*(mYre*mYre)*nZre + GammaZyz*Kyz*(mYre*mYre)*nZre - GammaZyy*Kzz*(mYre*mYre)*nZre - GammaXzz*Kxx*mXim*mZim*nZre - GammaYzz*Kxy*mXim*mZim*nZre + GammaXxz*Kxz*mXim*mZim*nZre - GammaZzz*Kxz*mXim*mZim*nZre + GammaYxz*Kyz*mXim*mZim*nZre + GammaZxz*Kzz*mXim*mZim*nZre - GammaXzz*Kxy*mYim*mZim*nZre + GammaXyz*Kxz*mYim*mZim*nZre - GammaYzz*Kyy*mYim*mZim*nZre + GammaYyz*Kyz*mYim*mZim*nZre - GammaZzz*Kyz*mYim*mZim*nZre + GammaZyz*Kzz*mYim*mZim*nZre + GammaXzz*Kxx*mXre*mZre*nZre + GammaYzz*Kxy*mXre*mZre*nZre - GammaXxz*Kxz*mXre*mZre*nZre + GammaZzz*Kxz*mXre*mZre*nZre - GammaYxz*Kyz*mXre*mZre*nZre - GammaZxz*Kzz*mXre*mZre*nZre + GammaXzz*Kxy*mYre*mZre*nZre - GammaXyz*Kxz*mYre*mZre*nZre + GammaYzz*Kyy*mYre*mZre*nZre - GammaYyz*Kyz*mYre*mZre*nZre + GammaZzz*Kyz*mYre*mZre*nZre - GammaZyz*Kzz*mYre*mZre*nZre + GammaXxx*Kxz*(-(mXim*mZim*nXre) + mXim*mXim*nZre + mXre*(mZre*nXre - mXre*nZre)) + GammaXxy*(Kxy*(-(mYim*mYim*nXre) + mYre*mYre*nXre + mXim*mYim*nYre - mXre*mYre*nYre) + Kxx*(mXim*mYim*nXre - mXim*mXim*nYre + mXre*(-(mYre*nXre) + mXre*nYre)) + Kxz*(-(mYim*mZim*nXre) + mYre*mZre*nXre - mXim*mZim*nYre + mXre*mZre*nYre + 2*mXim*mYim*nZre - 2*mXre*mYre*nZre))) - mXim*mXim*(nYre*nYre)*Rxyxy + mXre*mXre*(nYre*nYre)*Rxyxy + 2*mXim*mZim*nXre*nYre*Rxyxz - 2*mXre*mZre*nXre*nYre*Rxyxz - 2*(mXim*mXim)*nYre*nZre*Rxyxz + 2*(mXre*mXre)*nYre*nZre*Rxyxz + 2*mXim*mZim*(nYre*nYre)*Rxyyz - 2*mXre*mZre*(nYre*nYre)*Rxyyz - mZim*mZim*(nXre*nXre)*Rxzxz + mZre*mZre*(nXre*nXre)*Rxzxz + 2*mXim*mZim*nXre*nZre*Rxzxz - 2*mXre*mZre*nXre*nZre*Rxzxz - mXim*mXim*(nZre*nZre)*Rxzxz + mXre*mXre*(nZre*nZre)*Rxzxz - 2*(mZim*mZim)*nXre*nYre*Rxzyz + 2*(mZre*mZre)*nXre*nYre*Rxzyz + 2*mXim*mZim*nYre*nZre*Rxzyz - 2*mXre*mZre*nYre*nZre*Rxzyz - mZim*mZim*(nYre*nYre)*Ryzyz + mZre*mZre*(nYre*nYre)*Ryzyz - mYim*mYim*(nXre*nXre*Rxyxy - 2*nXre*nZre*Rxyyz + nZre*nZre*Ryzyz) + mYre*mYre*(nXre*nXre*Rxyxy - 2*nXre*nZre*Rxyyz + nZre*nZre*Ryzyz) + (n0*n0*(-(gtXX*(mYim*mYim)*Rxyxy) + gtXX*(mYre*mYre)*Rxyxy - 2*gtYZ*(mXim*mXim)*Rxyxz + 2*gtYZ*(mXre*mXre)*Rxyxz + 2*gtXZ*mXim*mYim*Rxyxz - 2*gtXZ*mXre*mYre*Rxyxz - 2*gtXX*mYim*mZim*Rxyxz + 2*gtXX*mYre*mZre*Rxyxz - 2*gtYZ*mXim*mYim*Rxyyz + 2*gtXZ*(mYim*mYim)*Rxyyz + 2*gtYZ*mXre*mYre*Rxyyz - 2*gtXZ*(mYre*mYre)*Rxyyz - gtZZ*(mXim*mXim)*Rxzxz + gtZZ*(mXre*mXre)*Rxzxz + 2*gtXZ*mXim*mZim*Rxzxz - gtXX*(mZim*mZim)*Rxzxz - 2*gtXZ*mXre*mZre*Rxzxz + gtXX*(mZre*mZre)*Rxzxz - 2*gtZZ*mXim*mYim*Rxzyz + 2*gtZZ*mXre*mYre*Rxzyz + 2*gtYZ*mXim*mZim*Rxzyz + 2*gtXZ*mYim*mZim*Rxzyz - 2*gtYZ*mXre*mZre*Rxzyz - 2*gtXZ*mYre*mZre*Rxzyz + 2*gtXY*(mXim*mYim*Rxyxy - mXre*mYre*Rxyxy + mXim*mZim*Rxyxz - mXre*mZre*Rxyxz - mYim*mZim*Rxyyz + mYre*mZre*Rxyyz - mZim*mZim*Rxzyz + mZre*mZre*Rxzyz) - gtZZ*(mYim*mYim)*Ryzyz + gtZZ*(mYre*mYre)*Ryzyz + 2*gtYZ*mYim*mZim*Ryzyz - 2*gtYZ*mYre*mZre*Ryzyz + gtYY*(-(mXim*mXim*Rxyxy) + mXre*mXre*Rxyxy + 2*mXim*mZim*Rxyyz - 2*mXre*mZre*Rxyyz - mZim*mZim*Ryzyz + mZre*mZre*Ryzyz)))/exp(4.*phi) + 2*mYim*(mXim*nXre*(nYre*Rxyxy + nZre*Rxyxz) - mXim*nZre*(nYre*Rxyyz + nZre*Rxzyz) + mZim*(-(nXre*nXre*Rxyxz) - nXre*nYre*Rxyyz + nXre*nZre*Rxzyz + nYre*nZre*Ryzyz)) - 2*mYre*(mXre*nXre*(nYre*Rxyxy + nZre*Rxyxz) - mXre*nZre*(nYre*Rxyyz + nZre*Rxzyz) + mZre*(-(nXre*nXre*Rxyxz) - nXre*nYre*Rxyyz + nXre*nZre*Rxzyz + nYre*nZre*Ryzyz));
    // imaginary part of psi4
    double const psi4Im= 2*K*(Kxx*mXim*mXre + Kxy*mXre*mYim + Kxy*mXim*mYre + Kyy*mYim*mYre + Kxz*mXre*mZim + Kyz*mYre*mZim + Kxz*mXim*mZre + Kyz*mYim*mZre + Kzz*mZim*mZre)*(n0*n0) - (2*Kxz*KXz*mXim*mXre + KXy*Kyy*mXre*mYim + KXz*Kyz*mXre*mYim + Kxz*KYz*mXre*mYim + KXy*Kyy*mXim*mYre + KXz*Kyz*mXim*mYre + Kxz*KYz*mXim*mYre + 2*Kyy*KYy*mYim*mYre + 2*Kyz*KYz*mYim*mYre + KXx*Kxz*mXre*mZim + KXy*Kyz*mXre*mZim + KXz*Kzz*mXre*mZim + Kxz*KZz*mXre*mZim + KXy*Kxz*mYre*mZim + KYy*Kyz*mYre*mZim + Kyy*KYz*mYre*mZim + KYz*Kzz*mYre*mZim + Kyz*KZz*mYre*mZim + KXx*Kxz*mXim*mZre + KXy*Kyz*mXim*mZre + KXz*Kzz*mXim*mZre + Kxz*KZz*mXim*mZre + KXy*Kxz*mYim*mZre + KYy*Kyz*mYim*mZre + Kyy*KYz*mYim*mZre + KYz*Kzz*mYim*mZre + Kyz*KZz*mYim*mZre + 2*Kxz*KXz*mZim*mZre + 2*Kyz*KYz*mZim*mZre + 2*Kzz*KZz*mZim*mZre + Kxx*(2*KXx*mXim*mXre + KXy*mXre*mYim + KXy*mXim*mYre + KXz*mXre*mZim + KXz*mXim*mZre) + Kxy*(2*KXy*mXim*mXre + KXx*mXre*mYim + KYy*mXre*mYim + KXx*mXim*mYre + KYy*mXim*mYre + 2*KXy*mYim*mYre + KYz*mXre*mZim + KXz*mYre*mZim + KYz*mXim*mZre + KXz*mYim*mZre))*(n0*n0) - 2*n0*(2*Kxydy*mYim*mYre*nXre - 2*Kyydx*mYim*mYre*nXre + Kxxdz*mXre*mZim*nXre - Kxzdx*mXre*mZim*nXre + Kxydz*mYre*mZim*nXre + Kxzdy*mYre*mZim*nXre - 2*Kyzdx*mYre*mZim*nXre + Kxxdz*mXim*mZre*nXre - Kxzdx*mXim*mZre*nXre + Kxydz*mYim*mZre*nXre + Kxzdy*mYim*mZre*nXre - 2*Kyzdx*mYim*mZre*nXre + 2*Kxzdz*mZim*mZre*nXre - 2*Kzzdx*mZim*mZre*nXre - Kxydy*mXre*mYim*nYre + Kyydx*mXre*mYim*nYre - Kxydy*mXim*mYre*nYre + Kyydx*mXim*mYre*nYre + Kxydz*mXre*mZim*nYre - 2*Kxzdy*mXre*mZim*nYre + Kyzdx*mXre*mZim*nYre + Kyydz*mYre*mZim*nYre - Kyzdy*mYre*mZim*nYre + Kxydz*mXim*mZre*nYre - 2*Kxzdy*mXim*mZre*nYre + Kyzdx*mXim*mZre*nYre + Kyydz*mYim*mZre*nYre - Kyzdy*mYim*mZre*nYre + 2*Kyzdz*mZim*mZre*nYre - 2*Kzzdy*mZim*mZre*nYre + Kxxdy*(mXre*mYim*nXre + mXim*mYre*nXre - 2*mXim*mXre*nYre) - Kxydx*(mXre*mYim*nXre + mXim*mYre*nXre - 2*mXim*mXre*nYre) - 2*Kxxdz*mXim*mXre*nZre + 2*Kxzdx*mXim*mXre*nZre - 2*Kxydz*mXre*mYim*nZre + Kxzdy*mXre*mYim*nZre + Kyzdx*mXre*mYim*nZre - 2*Kxydz*mXim*mYre*nZre + Kxzdy*mXim*mYre*nZre + Kyzdx*mXim*mYre*nZre - 2*Kyydz*mYim*mYre*nZre + 2*Kyzdy*mYim*mYre*nZre - Kxzdz*mXre*mZim*nZre + Kzzdx*mXre*mZim*nZre - Kyzdz*mYre*mZim*nZre + Kzzdy*mYre*mZim*nZre - Kxzdz*mXim*mZre*nZre + Kzzdx*mXim*mZre*nZre - Kyzdz*mYim*mZre*nZre + Kzzdy*mYim*mZre*nZre) + 2*n0*(GammaYxy*Kxy*mXre*mYim*nXre + GammaZxy*Kxz*mXre*mYim*nXre - GammaYxx*Kyy*mXre*mYim*nXre - GammaZxx*Kyz*mXre*mYim*nXre + GammaYxy*Kxy*mXim*mYre*nXre + GammaZxy*Kxz*mXim*mYre*nXre - GammaYxx*Kyy*mXim*mYre*nXre - GammaZxx*Kyz*mXim*mYre*nXre + 2*GammaXyy*Kxx*mYim*mYre*nXre + 2*GammaYyy*Kxy*mYim*mYre*nXre + 2*GammaZyy*Kxz*mYim*mYre*nXre - 2*GammaYxy*Kyy*mYim*mYre*nXre - 2*GammaZxy*Kyz*mYim*mYre*nXre + GammaXxz*Kxx*mXre*mZim*nXre + GammaYxz*Kxy*mXre*mZim*nXre + GammaZxz*Kxz*mXre*mZim*nXre - GammaYxx*Kyz*mXre*mZim*nXre - GammaZxx*Kzz*mXre*mZim*nXre + 2*GammaXyz*Kxx*mYre*mZim*nXre - GammaXxz*Kxy*mYre*mZim*nXre + 2*GammaYyz*Kxy*mYre*mZim*nXre + 2*GammaZyz*Kxz*mYre*mZim*nXre - GammaYxz*Kyy*mYre*mZim*nXre - GammaYxy*Kyz*mYre*mZim*nXre - GammaZxz*Kyz*mYre*mZim*nXre - GammaZxy*Kzz*mYre*mZim*nXre + GammaXxz*Kxx*mXim*mZre*nXre + GammaYxz*Kxy*mXim*mZre*nXre + GammaZxz*Kxz*mXim*mZre*nXre - GammaYxx*Kyz*mXim*mZre*nXre - GammaZxx*Kzz*mXim*mZre*nXre + 2*GammaXyz*Kxx*mYim*mZre*nXre - GammaXxz*Kxy*mYim*mZre*nXre + 2*GammaYyz*Kxy*mYim*mZre*nXre + 2*GammaZyz*Kxz*mYim*mZre*nXre - GammaYxz*Kyy*mYim*mZre*nXre - GammaYxy*Kyz*mYim*mZre*nXre - GammaZxz*Kyz*mYim*mZre*nXre - GammaZxy*Kzz*mYim*mZre*nXre + 2*GammaXzz*Kxx*mZim*mZre*nXre + 2*GammaYzz*Kxy*mZim*mZre*nXre - 2*GammaXxz*Kxz*mZim*mZre*nXre + 2*GammaZzz*Kxz*mZim*mZre*nXre - 2*GammaYxz*Kyz*mZim*mZre*nXre - 2*GammaZxz*Kzz*mZim*mZre*nXre - 2*GammaYxy*Kxy*mXim*mXre*nYre - 2*GammaZxy*Kxz*mXim*mXre*nYre + 2*GammaYxx*Kyy*mXim*mXre*nYre + 2*GammaZxx*Kyz*mXim*mXre*nYre - GammaXyy*Kxx*mXre*mYim*nYre - GammaYyy*Kxy*mXre*mYim*nYre - GammaZyy*Kxz*mXre*mYim*nYre + GammaYxy*Kyy*mXre*mYim*nYre + GammaZxy*Kyz*mXre*mYim*nYre - GammaXyy*Kxx*mXim*mYre*nYre - GammaYyy*Kxy*mXim*mYre*nYre - GammaZyy*Kxz*mXim*mYre*nYre + GammaYxy*Kyy*mXim*mYre*nYre + GammaZxy*Kyz*mXim*mYre*nYre - GammaXyz*Kxx*mXre*mZim*nYre + 2*GammaXxz*Kxy*mXre*mZim*nYre - GammaYyz*Kxy*mXre*mZim*nYre - GammaZyz*Kxz*mXre*mZim*nYre + 2*GammaYxz*Kyy*mXre*mZim*nYre - GammaYxy*Kyz*mXre*mZim*nYre + 2*GammaZxz*Kyz*mXre*mZim*nYre - GammaZxy*Kzz*mXre*mZim*nYre + GammaXyz*Kxy*mYre*mZim*nYre - GammaXyy*Kxz*mYre*mZim*nYre + GammaYyz*Kyy*mYre*mZim*nYre - GammaYyy*Kyz*mYre*mZim*nYre + GammaZyz*Kyz*mYre*mZim*nYre - GammaZyy*Kzz*mYre*mZim*nYre - GammaXyz*Kxx*mXim*mZre*nYre + 2*GammaXxz*Kxy*mXim*mZre*nYre - GammaYyz*Kxy*mXim*mZre*nYre - GammaZyz*Kxz*mXim*mZre*nYre + 2*GammaYxz*Kyy*mXim*mZre*nYre - GammaYxy*Kyz*mXim*mZre*nYre + 2*GammaZxz*Kyz*mXim*mZre*nYre - GammaZxy*Kzz*mXim*mZre*nYre + GammaXyz*Kxy*mYim*mZre*nYre - GammaXyy*Kxz*mYim*mZre*nYre + GammaYyz*Kyy*mYim*mZre*nYre - GammaYyy*Kyz*mYim*mZre*nYre + GammaZyz*Kyz*mYim*mZre*nYre - GammaZyy*Kzz*mYim*mZre*nYre + 2*GammaXzz*Kxy*mZim*mZre*nYre - 2*GammaXyz*Kxz*mZim*mZre*nYre + 2*GammaYzz*Kyy*mZim*mZre*nYre - 2*GammaYyz*Kyz*mZim*mZre*nYre + 2*GammaZzz*Kyz*mZim*mZre*nYre - 2*GammaZyz*Kzz*mZim*mZre*nYre - 2*GammaXxz*Kxx*mXim*mXre*nZre - 2*GammaYxz*Kxy*mXim*mXre*nZre - 2*GammaZxz*Kxz*mXim*mXre*nZre + 2*GammaYxx*Kyz*mXim*mXre*nZre + 2*GammaZxx*Kzz*mXim*mXre*nZre - GammaXyz*Kxx*mXre*mYim*nZre - GammaXxz*Kxy*mXre*mYim*nZre - GammaYyz*Kxy*mXre*mYim*nZre - GammaZyz*Kxz*mXre*mYim*nZre - GammaYxz*Kyy*mXre*mYim*nZre + 2*GammaYxy*Kyz*mXre*mYim*nZre - GammaZxz*Kyz*mXre*mYim*nZre + 2*GammaZxy*Kzz*mXre*mYim*nZre - GammaXyz*Kxx*mXim*mYre*nZre - GammaXxz*Kxy*mXim*mYre*nZre - GammaYyz*Kxy*mXim*mYre*nZre - GammaZyz*Kxz*mXim*mYre*nZre - GammaYxz*Kyy*mXim*mYre*nZre + 2*GammaYxy*Kyz*mXim*mYre*nZre - GammaZxz*Kyz*mXim*mYre*nZre + 2*GammaZxy*Kzz*mXim*mYre*nZre - 2*GammaXyz*Kxy*mYim*mYre*nZre + 2*GammaXyy*Kxz*mYim*mYre*nZre - 2*GammaYyz*Kyy*mYim*mYre*nZre + 2*GammaYyy*Kyz*mYim*mYre*nZre - 2*GammaZyz*Kyz*mYim*mYre*nZre + 2*GammaZyy*Kzz*mYim*mYre*nZre - GammaXzz*Kxx*mXre*mZim*nZre - GammaYzz*Kxy*mXre*mZim*nZre + GammaXxz*Kxz*mXre*mZim*nZre - GammaZzz*Kxz*mXre*mZim*nZre + GammaYxz*Kyz*mXre*mZim*nZre + GammaZxz*Kzz*mXre*mZim*nZre - GammaXzz*Kxy*mYre*mZim*nZre + GammaXyz*Kxz*mYre*mZim*nZre - GammaYzz*Kyy*mYre*mZim*nZre + GammaYyz*Kyz*mYre*mZim*nZre - GammaZzz*Kyz*mYre*mZim*nZre + GammaZyz*Kzz*mYre*mZim*nZre - GammaXzz*Kxx*mXim*mZre*nZre - GammaYzz*Kxy*mXim*mZre*nZre + GammaXxz*Kxz*mXim*mZre*nZre - GammaZzz*Kxz*mXim*mZre*nZre + GammaYxz*Kyz*mXim*mZre*nZre + GammaZxz*Kzz*mXim*mZre*nZre - GammaXzz*Kxy*mYim*mZre*nZre + GammaXyz*Kxz*mYim*mZre*nZre - GammaYzz*Kyy*mYim*mZre*nZre + GammaYyz*Kyz*mYim*mZre*nZre - GammaZzz*Kyz*mYim*mZre*nZre + GammaZyz*Kzz*mYim*mZre*nZre - GammaXxx*(Kxy*(mXre*mYim*nXre + mXim*mYre*nXre - 2*mXim*mXre*nYre) + Kxz*(mXre*mZim*nXre + mXim*mZre*nXre - 2*mXim*mXre*nZre)) + GammaXxy*(Kxx*(mXre*mYim*nXre + mXim*mYre*nXre - 2*mXim*mXre*nYre) + Kxy*(-2*mYim*mYre*nXre + mXre*mYim*nYre + mXim*mYre*nYre) - Kxz*(mYre*mZim*nXre + mYim*mZre*nXre + mXre*mZim*nYre + mXim*mZre*nYre - 2*mXre*mYim*nZre - 2*mXim*mYre*nZre))) + 2*(-(Kxz*Kxz*mZim*mZre*(nXre*nXre)) - Kxz*Kyy*mYre*mZim*nXre*nYre - Kxz*Kyy*mYim*mZre*nXre*nYre - 2*Kxz*Kyz*mZim*mZre*nXre*nYre + Kxz*Kyy*mXre*mZim*(nYre*nYre) + Kxz*Kyy*mXim*mZre*(nYre*nYre) - Kyz*Kyz*mZim*mZre*(nYre*nYre) + Kyy*Kzz*mZim*mZre*(nYre*nYre) - Kxy*Kxy*(mYim*nXre - mXim*nYre)*(mYre*nXre - mXre*nYre) + 2*Kxz*Kyy*mYim*mYre*nXre*nZre + Kxz*Kxz*mXre*mZim*nXre*nZre + Kxz*Kyz*mYre*mZim*nXre*nZre + Kxz*Kxz*mXim*mZre*nXre*nZre + Kxz*Kyz*mYim*mZre*nXre*nZre - Kxz*Kyy*mXre*mYim*nYre*nZre - Kxz*Kyy*mXim*mYre*nYre*nZre + Kxz*Kyz*mXre*mZim*nYre*nZre + Kyz*Kyz*mYre*mZim*nYre*nZre - Kyy*Kzz*mYre*mZim*nYre*nZre + Kxz*Kyz*mXim*mZre*nYre*nZre + Kyz*Kyz*mYim*mZre*nYre*nZre - Kyy*Kzz*mYim*mZre*nYre*nZre - Kxz*Kxz*mXim*mXre*(nZre*nZre) - Kxz*Kyz*mXre*mYim*(nZre*nZre) - Kxz*Kyz*mXim*mYre*(nZre*nZre) - Kyz*Kyz*mYim*mYre*(nZre*nZre) + Kyy*Kzz*mYim*mYre*(nZre*nZre) + Kxx*(Kyy*(mYim*nXre - mXim*nYre)*(mYre*nXre - mXre*nYre) + Kzz*(mZim*nXre - mXim*nZre)*(mZre*nXre - mXre*nZre) + Kyz*(mYre*nXre*(mZim*nXre - mXim*nZre) + mYim*nXre*(mZre*nXre - mXre*nZre) - nYre*(mXre*mZim*nXre + mXim*mZre*nXre - 2*mXim*mXre*nZre))) + Kxy*(2*Kzz*mZim*mZre*nXre*nYre - Kzz*mZim*(mYre*nXre + mXre*nYre)*nZre + Kzz*nZre*(-(mYim*mZre*nXre) - mXim*mZre*nYre + mXre*mYim*nZre + mXim*mYre*nZre) + Kyz*mYre*(mZim*nXre*nYre - 2*mYim*nXre*nZre + mXim*nYre*nZre) + Kyz*nYre*(-((mXre*mZim + mXim*mZre)*nYre) + mYim*(mZre*nXre + mXre*nZre)) + Kxz*(mYre*nXre*(-(mZim*nXre) + mXim*nZre) + mYim*nXre*(-(mZre*nXre) + mXre*nZre) + nYre*(mXre*mZim*nXre + mXim*mZre*nXre - 2*mXim*mXre*nZre)))) - 2*mXim*(mYre*nXre*(nYre*Rxyxy + nZre*Rxyxz) - mXre*(nYre*nYre*Rxyxy + 2*nYre*nZre*Rxyxz + nZre*nZre*Rxzxz) - mYre*nZre*(nYre*Rxyyz + nZre*Rxzyz) + mZre*(nXre*nYre*Rxyxz + nYre*nYre*Rxyyz + nXre*nZre*Rxzxz + nYre*nZre*Rxzyz)) + (2*(n0*n0)*(gtXX*mYim*mYre*Rxyxy + 2*gtYZ*mXim*mXre*Rxyxz - gtXZ*mXre*mYim*Rxyxz - gtXZ*mXim*mYre*Rxyxz + gtXX*mYre*mZim*Rxyxz + gtXX*mYim*mZre*Rxyxz + gtYZ*mXre*mYim*Rxyyz + gtYZ*mXim*mYre*Rxyyz - 2*gtXZ*mYim*mYre*Rxyyz + gtZZ*mXim*mXre*Rxzxz - gtXZ*mXre*mZim*Rxzxz - gtXZ*mXim*mZre*Rxzxz + gtXX*mZim*mZre*Rxzxz + gtZZ*mXre*mYim*Rxzyz + gtZZ*mXim*mYre*Rxzyz - gtYZ*mXre*mZim*Rxzyz - gtXZ*mYre*mZim*Rxzyz - gtYZ*mXim*mZre*Rxzyz - gtXZ*mYim*mZre*Rxzyz - gtXY*(mXre*mYim*Rxyxy + mXim*mYre*Rxyxy + mXre*mZim*Rxyxz + mXim*mZre*Rxyxz - mYre*mZim*Rxyyz - mYim*mZre*Rxyyz - 2*mZim*mZre*Rxzyz) + gtZZ*mYim*mYre*Ryzyz - gtYZ*mYre*mZim*Ryzyz - gtYZ*mYim*mZre*Ryzyz + gtYY*(mXim*mXre*Rxyxy - mXre*mZim*Rxyyz - mXim*mZre*Rxyyz + mZim*mZre*Ryzyz)))/exp(4.*phi) - 2*mZim*(mXre*(nXre*nYre*Rxyxz + nYre*nYre*Rxyyz + nXre*nZre*Rxzxz + nYre*nZre*Rxzyz) - mZre*(nXre*nXre*Rxzxz + 2*nXre*nYre*Rxzyz + nYre*nYre*Ryzyz) + mYre*(-(nXre*nXre*Rxyxz) - nXre*nYre*Rxyyz + nXre*nZre*Rxzyz + nYre*nZre*Ryzyz)) + 2*mYim*(-(mXre*nXre*(nYre*Rxyxy + nZre*Rxyxz)) + mXre*nZre*(nYre*Rxyyz + nZre*Rxzyz) + mZre*(nXre*nXre*Rxyxz + nXre*nYre*Rxyyz - nXre*nZre*Rxzyz - nYre*nZre*Ryzyz) + mYre*(nXre*nXre*Rxyxy - 2*nXre*nZre*Rxyyz + nZre*nZre*Ryzyz));

    std::array<double,2> psi4{psi4Re,psi4Im};
    return std::move(psi4);
}


#define INSTANTIATE_TEMPLATE(DER_ORD)                                \
template                                                             \
grace::bssn_state_t GRACE_HOST_DEVICE                                \
compute_bssn_rhs<DER_ORD>( VEC(int , int , int ), int                \
                , grace::var_array_t<GRACE_NSPACEDIM> const          \
                , std::array<std::array<double,4>,4> const&          \
                , std::array<double,GRACE_NSPACEDIM> const&          \
                , double const k1, double const eta );               \
template                                                             \
std::array<double,4> GRACE_HOST_DEVICE                               \
compute_bssn_constraint_violations<DER_ORD>(                         \
                  VEC(int , int , int ), int                         \
                , grace::var_array_t<GRACE_NSPACEDIM> const          \
                , std::array<std::array<double,4>,4> const&          \
                , std::array<double,GRACE_NSPACEDIM> const& );       \
template                                                             \
std::array<double,2> GRACE_HOST_DEVICE                               \
compute_psi4<DER_ORD>(                                               \
      VEC(int , int , int ), int                                     \
    , grace::coord_array_t<GRACE_NSPACEDIM> const                    \
    , grace::var_array_t<GRACE_NSPACEDIM> const                      \
    , std::array<double,GRACE_NSPACEDIM> const& );

INSTANTIATE_TEMPLATE(2) ; 
INSTANTIATE_TEMPLATE(4) ; 
#undef INSTANTIATE_TEMPLATE
}