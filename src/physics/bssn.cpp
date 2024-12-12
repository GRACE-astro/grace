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
                , std::array<std::array<double,4>,4> const& Tdd
                , std::array<double,GRACE_NSPACEDIM> const& idx
                , double const k1, double const eta )
{


    static constexpr const double pi = M_PI ; 

    double const gtxx = state(VEC(i,j,k),GTXX_+0,q);
    double const gtxy = state(VEC(i,j,k),GTXX_+1,q);
    double const gtxz = state(VEC(i,j,k),GTXX_+2,q);
    double const gtyy = state(VEC(i,j,k),GTXX_+3,q);
    double const gtyz = state(VEC(i,j,k),GTXX_+4,q);
    double const gtzz = state(VEC(i,j,k),GTXX_+5,q);

    double const Atxx = state(VEC(i,j,k),ATXX_+0,q);
    double const Atxy = state(VEC(i,j,k),ATXX_+1,q);
    double const Atxz = state(VEC(i,j,k),ATXX_+2,q);
    double const Atyy = state(VEC(i,j,k),ATXX_+3,q);
    double const Atyz = state(VEC(i,j,k),ATXX_+4,q);
    double const Atzz = state(VEC(i,j,k),ATXX_+5,q);

    double const K = state(VEC(i,j,k),K_,q);

    double const phi = state(VEC(i,j,k),PHI_,q);

    double const alp = state(VEC(i,j,k),ALP_,q);

    double const betaX = state(VEC(i,j,k),BETAX_+0,q);
    double const betaY = state(VEC(i,j,k),BETAX_+1,q);
    double const betaZ = state(VEC(i,j,k),BETAX_+2,q);
    double const betax = (betaX*gtxx + betaY*gtxy + betaZ*gtxz)*exp(4.*phi);
    double const betay = (betaX*gtxy + betaY*gtyy + betaZ*gtyz)*exp(4.*phi);
    double const betaz = (betaX*gtxz + betaY*gtyz + betaZ*gtzz)*exp(4.*phi);

    double const BX = state(VEC(i,j,k),BX_+0,q);
    double const BY = state(VEC(i,j,k),BX_+1,q);
    double const BZ = state(VEC(i,j,k),BX_+2,q);

    double const GammatX = state(VEC(i,j,k),GAMMAX_+0,q);
    double const GammatY = state(VEC(i,j,k),GAMMAX_+1,q);
    double const GammatZ = state(VEC(i,j,k),GAMMAX_+2,q);

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

    double const gtXX=-(gtyz*gtyz) + gtyy*gtzz;
    double const gtXY=gtxz*gtyz - gtxy*gtzz;
    double const gtXZ=-(gtxz*gtyy) + gtxy*gtyz;
    double const gtYY=-(gtxz*gtxz) + gtxx*gtzz;
    double const gtYZ=gtxy*gtxz - gtxx*gtyz;
    double const gtZZ=-(gtxy*gtxy) + gtxx*gtyy;

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

    double const phidx = grace::fd_der<der_order,0>(state,PHI_,VEC(i,j,k),q) * idx[0 ];
    double const phidy = grace::fd_der<der_order,1>(state,PHI_,VEC(i,j,k),q) * idx[1 ];
    double const phidz = grace::fd_der<der_order,2>(state,PHI_,VEC(i,j,k),q) * idx[2 ];
    double const phidxdx = grace::fd_second_der<der_order,0>(state,PHI_,VEC(i,j,k),q) * math::int_pow<2>(idx[0 ]);
    double const phidydy = grace::fd_second_der<der_order,1>(state,PHI_,VEC(i,j,k),q) * math::int_pow<2>(idx[1 ]);
    double const phidzdz = grace::fd_second_der<der_order,2>(state,PHI_,VEC(i,j,k),q) * math::int_pow<2>(idx[2 ]);
    double const phidxdy = grace::fd_der<der_order,0,1>(state,PHI_,VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const phidxdz = grace::fd_der<der_order,0,2>(state,PHI_,VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const phidydz = grace::fd_der<der_order,1,2>(state,PHI_,VEC(i,j,k),q) * idx[1 ] * idx[2 ];

    double const Kdx = grace::fd_der<der_order,0>(state,K_,VEC(i,j,k),q) * idx[0 ];
    double const Kdy = grace::fd_der<der_order,1>(state,K_,VEC(i,j,k),q) * idx[1 ];
    double const Kdz = grace::fd_der<der_order,2>(state,K_,VEC(i,j,k),q) * idx[2 ];

    double const alpdx = grace::fd_der<der_order,0>(state,ALP_,VEC(i,j,k),q) * idx[0 ];
    double const alpdy = grace::fd_der<der_order,1>(state,ALP_,VEC(i,j,k),q) * idx[1 ];
    double const alpdz = grace::fd_der<der_order,2>(state,ALP_,VEC(i,j,k),q) * idx[2 ];
    double const alpdxdx = grace::fd_second_der<der_order,0>(state,ALP_,VEC(i,j,k),q) * math::int_pow<2>(idx[0 ]);
    double const alpdydy = grace::fd_second_der<der_order,1>(state,ALP_,VEC(i,j,k),q) * math::int_pow<2>(idx[1 ]);
    double const alpdzdz = grace::fd_second_der<der_order,2>(state,ALP_,VEC(i,j,k),q) * math::int_pow<2>(idx[2 ]);
    double const alpdxdy = grace::fd_der<der_order,0,1>(state,ALP_,VEC(i,j,k),q) * idx[0 ] * idx[1 ];
    double const alpdxdz = grace::fd_der<der_order,0,2>(state,ALP_,VEC(i,j,k),q) * idx[0 ] * idx[2 ];
    double const alpdydz = grace::fd_der<der_order,1,2>(state,ALP_,VEC(i,j,k),q) * idx[1 ] * idx[2 ];

    double const betaXdx = grace::fd_der<der_order,0>(state,BETAX_+0, VEC(i,j,k),q)* idx[0 ];
    double const betaXdy = grace::fd_der<der_order,1>(state,BETAX_+0, VEC(i,j,k),q)* idx[1 ];
    double const betaXdz = grace::fd_der<der_order,2>(state,BETAX_+0, VEC(i,j,k),q)* idx[2 ];
    double const betaYdx = grace::fd_der<der_order,0>(state,BETAX_+1, VEC(i,j,k),q)* idx[0 ];
    double const betaYdy = grace::fd_der<der_order,1>(state,BETAX_+1, VEC(i,j,k),q)* idx[1 ];
    double const betaYdz = grace::fd_der<der_order,2>(state,BETAX_+1, VEC(i,j,k),q)* idx[2 ];
    double const betaZdx = grace::fd_der<der_order,0>(state,BETAX_+2, VEC(i,j,k),q)* idx[0 ];
    double const betaZdy = grace::fd_der<der_order,1>(state,BETAX_+2, VEC(i,j,k),q)* idx[1 ];
    double const betaZdz = grace::fd_der<der_order,2>(state,BETAX_+2, VEC(i,j,k),q)* idx[2 ];
    double const betaXdxdx = grace::fd_second_der<der_order,0>(state,BETAX_+0, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const betaXdydy = grace::fd_second_der<der_order,1>(state,BETAX_+0, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const betaXdzdz = grace::fd_second_der<der_order,2>(state,BETAX_+0, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);
    double const betaYdxdx = grace::fd_second_der<der_order,0>(state,BETAX_+1, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const betaYdydy = grace::fd_second_der<der_order,1>(state,BETAX_+1, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const betaYdzdz = grace::fd_second_der<der_order,2>(state,BETAX_+1, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);
    double const betaZdxdx = grace::fd_second_der<der_order,0>(state,BETAX_+2, VEC(i,j,k),q)* math::int_pow<2>(idx[0 ]);
    double const betaZdydy = grace::fd_second_der<der_order,1>(state,BETAX_+2, VEC(i,j,k),q)* math::int_pow<2>(idx[1 ]);
    double const betaZdzdz = grace::fd_second_der<der_order,2>(state,BETAX_+2, VEC(i,j,k),q)* math::int_pow<2>(idx[2 ]);
    double const betaXdxdy = grace::fd_der<der_order,0,1>(state,BETAX_+0, VEC(i,j,k),q)* idx[0 ] * idx[1 ];
    double const betaXdxdz = grace::fd_der<der_order,0,2>(state,BETAX_+0, VEC(i,j,k),q)* idx[0 ] * idx[2 ];
    double const betaXdydz = grace::fd_der<der_order,1,2>(state,BETAX_+0, VEC(i,j,k),q)* idx[1 ] * idx[2 ];
    double const betaYdxdy = grace::fd_der<der_order,0,1>(state,BETAX_+1, VEC(i,j,k),q)* idx[0 ] * idx[1 ];
    double const betaYdxdz = grace::fd_der<der_order,0,2>(state,BETAX_+1, VEC(i,j,k),q)* idx[0 ] * idx[2 ];
    double const betaYdydz = grace::fd_der<der_order,1,2>(state,BETAX_+1, VEC(i,j,k),q)* idx[1 ] * idx[2 ];
    double const betaZdxdy = grace::fd_der<der_order,0,1>(state,BETAX_+2, VEC(i,j,k),q)* idx[0 ] * idx[1 ];
    double const betaZdxdz = grace::fd_der<der_order,0,2>(state,BETAX_+2, VEC(i,j,k),q)* idx[0 ] * idx[2 ];
    double const betaZdydz = grace::fd_der<der_order,1,2>(state,BETAX_+2, VEC(i,j,k),q)* idx[1 ] * idx[2 ];

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

    double const Edens = (Ttt - 2*betaZ*Ttz + betaX*betaX*Txx + 2*betaX*(-Ttx + betaY*Txy + betaZ*Txz) + betaY*(-2*Tty + betaY*Tyy + 2*betaZ*Tyz) + betaZ*betaZ*Tzz)/(alp*alp);
    double const Stt= betaX*betaX*Txx + 2*betaX*(betaY*Txy + betaZ*Txz) + betaY*betaY*Tyy + 2*betaY*betaZ*Tyz + betaZ*betaZ*Tzz;
    double const Stx= betaX*Txx + betaY*Txy + betaZ*Txz;
    double const Sty= betaX*Txy + betaY*Tyy + betaZ*Tyz;
    double const Stz= betaX*Txz + betaY*Tyz + betaZ*Tzz;
    double const Sxx= Txx;
    double const Sxy= Txy;
    double const Sxz= Txz;
    double const Syy= Tyy;
    double const Syz= Tyz;
    double const Szz= Tzz;
    double const Sx= -((-Ttx + betaX*Txx + betaY*Txy + betaZ*Txz)/alp);
    double const Sy= -((-Tty + betaX*Txy + betaY*Tyy + betaZ*Tyz)/alp);
    double const Sz= -((-Ttz + betaX*Txz + betaY*Tyz + betaZ*Tzz)/alp);
    double const S = (gtXX*Sxx + 2*gtXY*Sxy + 2*gtXZ*Sxz + gtYY*Syy + 2*gtYZ*Syz + gtZZ*Szz)*exp(-4.*phi);


    bssn_state_t res ;


    /* Calculation of BSSN RHS starts here */


    double const Dtibetai = betaXdx + betaYdy + betaZdz + betaZ*GammatXxz + betaZ*GammatYyz + betaX*(GammatXxx + GammatYxy + GammatZxz) + betaY*(GammatXxy + GammatYyy + GammatZyz) + betaZ*GammatZzz;
    double const RddTFxx = -(GammatXxx*GammatXxx) - 2*GammatXxy*GammatYxx - GammatYxy*GammatYxy - 2*GammatXxz*GammatZxx - 2*GammatYxz*GammatZxy - GammatZxz*GammatZxz + GammatXdx*gtxx - gtxxdx*gtXXdx - (gtXX*gtxxdxdx)/2. + GammatYdx*gtxy - gtxxdxdy*gtXY - (gtxxdy + gtxydx)*gtXYdx + GammatZdx*gtxz - gtxxdxdz*gtXZ - (gtxxdz + gtxzdx)*gtXZdx - (gtxxdydy*gtYY)/2. - gtxydy*gtYYdx - gtxxdydz*gtYZ - (gtxydz + gtxzdy)*gtYZdx - (gtxxdzdz*gtZZ)/2. - gtxzdz*gtZZdx + (gtxx*(-2*(GammatXdx + GammatYdy + GammatZdz) + 2*(GammatXxx*GammatXxx)*gtXX + 2*(GammatYxy*GammatYxy)*gtXX + 4*GammatXxz*GammatZxx*gtXX + 4*GammatYxz*GammatZxy*gtXX + 2*(GammatZxz*GammatZxz)*gtXX + gtXX*gtxxdx*gtXXdx + 4*GammatXyy*GammatYxx*gtXY + 4*GammatYxy*GammatYyy*gtXY + 4*GammatXyz*GammatZxx*gtXY + 4*GammatXxz*GammatZxy*gtXY + 4*GammatYyz*GammatZxy*gtXY + 4*GammatYxz*GammatZyy*gtXY + 4*GammatZxz*GammatZyz*gtXY - gtXXdx*gtxxdy*gtXY + gtxxdx*gtXXdy*gtXY + 2*gtXXdx*gtXY*gtxydx + 2*gtXX*gtxxdy*gtXYdx + 2*gtxxdy*gtXY*gtXYdy + 4*GammatXyz*GammatYxx*gtXZ + 4*GammatYxy*GammatYyz*gtXZ + 4*GammatXzz*GammatZxx*gtXZ + 4*GammatYzz*GammatZxy*gtXZ + 4*GammatXxz*GammatZxz*gtXZ + 4*GammatYxz*GammatZyz*gtXZ + 4*GammatZxz*GammatZzz*gtXZ - gtXXdx*gtxxdz*gtXZ + gtxxdx*gtXXdz*gtXZ - 2*gtXYdx*gtxydz*gtXZ + 2*gtxxdy*gtXYdz*gtXZ + 4*GammatXxx*(GammatXxy*gtXY + GammatXxz*gtXZ) + 2*gtXXdx*gtXZ*gtxzdx + 2*gtXX*gtxxdz*gtXZdx + 2*gtXY*gtxydz*gtXZdx + 2*gtXYdx*gtXZ*gtxzdy - 2*gtXY*gtXZdx*gtxzdy + 2*gtxxdz*gtXY*gtXZdy + 2*gtxxdz*gtXZ*gtXZdz + 2*(GammatXxy*GammatXxy)*gtYY + 4*GammatXyy*GammatYxy*gtYY + 2*(GammatYyy*GammatYyy)*gtYY + 4*GammatXyz*GammatZxy*gtYY + 4*GammatYyz*GammatZyy*gtYY + 2*(GammatZyz*GammatZyz)*gtYY - gtxxdy*gtXXdy*gtYY + 2*gtXXdy*gtxydx*gtYY + 2*gtxydz*gtXZdy*gtYY - 2*gtxzdy*gtXZdy*gtYY + 2*gtXY*gtXYdx*gtyydx + 2*gtXYdy*gtYY*gtyydx + 2*gtXX*gtxydy*gtYYdx - gtXX*gtyydx*gtYYdx + gtXY*gtYYdx*gtyydy + 2*gtXY*gtxydy*gtYYdy - gtXY*gtyydx*gtYYdy + gtYY*gtyydy*gtYYdy - gtXZ*gtYYdx*gtyydz + 2*gtxydy*gtXZ*gtYYdz - gtXZ*gtyydx*gtYYdz + 4*GammatXyz*GammatYxy*gtYZ + 4*GammatXyy*GammatYxz*gtYZ + 4*GammatYyy*GammatYyz*gtYZ + 4*GammatXzz*GammatZxy*gtYZ + 4*GammatXyz*GammatZxz*gtYZ + 4*GammatYzz*GammatZyy*gtYZ + 4*GammatYyz*GammatZyz*gtYZ + 4*GammatZyz*GammatZzz*gtYZ - gtXXdy*gtxxdz*gtYZ - gtxxdy*gtXXdz*gtYZ + 2*gtXXdz*gtxydx*gtYZ - 2*gtXYdy*gtxydz*gtYZ + 2*gtXXdy*gtxzdx*gtYZ + 2*gtXYdy*gtxzdy*gtYZ + 2*gtxydz*gtXZdz*gtYZ - 2*gtxzdy*gtXZdz*gtYZ + 2*gtXYdz*gtyydx*gtYZ - gtYYdy*gtyydz*gtYZ + gtyydy*gtYYdz*gtYZ + 4*GammatXxy*(GammatYxx*gtXX + GammatYxy*gtXY + GammatYxz*gtXZ + GammatXxz*gtYZ) + 2*gtXYdx*gtXZ*gtyzdx + 2*gtXY*gtXZdx*gtyzdx + 2*gtXZdy*gtYY*gtyzdx + 2*gtXYdy*gtYZ*gtyzdx + 2*gtXZdz*gtYZ*gtyzdx + 2*gtXX*gtxydz*gtYZdx + 2*gtXX*gtxzdy*gtYZdx + 2*gtXY*gtyydz*gtYZdx - 2*gtXX*gtyzdx*gtYZdx + 2*gtXZ*gtYYdx*gtyzdy + 2*gtYYdy*gtYZ*gtyzdy + 2*gtXY*gtxydz*gtYZdy + 2*gtXY*gtxzdy*gtYZdy + 2*gtYY*gtyydz*gtYZdy - 2*gtXY*gtyzdx*gtYZdy + 2*gtxydz*gtXZ*gtYZdz + 2*gtXZ*gtxzdy*gtYZdz + 2*gtyydz*gtYZ*gtYZdz - 2*gtXZ*gtyzdx*gtYZdz + 2*(GammatXxz*GammatXxz)*gtZZ + 4*GammatXyz*GammatYxz*gtZZ + 2*(GammatYyz*GammatYyz)*gtZZ + 4*GammatXzz*GammatZxz*gtZZ + 4*GammatYzz*GammatZyz*gtZZ + 2*(GammatZzz*GammatZzz)*gtZZ - gtxxdz*gtXXdz*gtZZ - 2*gtxydz*gtXYdz*gtZZ + 2*gtXXdz*gtxzdx*gtZZ + 2*gtXYdz*gtxzdy*gtZZ - gtyydz*gtYYdz*gtZZ + 2*gtXYdz*gtyzdx*gtZZ + 2*gtYYdz*gtyzdy*gtZZ + 2*gtXZ*gtXZdx*gtzzdx + 2*gtXZdy*gtYZ*gtzzdx + 2*gtXZdz*gtZZ*gtzzdx + 2*gtXZ*gtYZdx*gtzzdy + 2*gtYZ*gtYZdy*gtzzdy + 2*gtYZdz*gtZZ*gtzzdy + 2*gtXY*gtxzdz*gtZZdy + 2*gtYY*gtyzdz*gtZZdy - gtXY*gtzzdx*gtZZdy - gtYY*gtzzdy*gtZZdy + gtYZ*gtZZdy*gtzzdz + gtZZdx*(2*gtXX*gtxzdz + 2*gtXY*gtyzdz - gtXX*gtzzdx - gtXY*gtzzdy + gtXZ*gtzzdz) + 2*gtXZ*gtxzdz*gtZZdz + 2*gtYZ*gtyzdz*gtZZdz - gtXZ*gtzzdx*gtZZdz - gtYZ*gtzzdy*gtZZdz + gtZZ*gtzzdz*gtZZdz))/6. + 4*(phidx*phidx) - 2*phidxdx + (2*gtxx*(gtXX*(-2*(phidx*phidx) + phidxdx) + 2*gtXZ*phidxdz - 2*gtYY*(phidy*phidy) + 2*gtXY*(phidxdy - 2*phidx*phidy) + gtYY*phidydy + 2*gtYZ*phidydz - 4*gtXZ*phidx*phidz - 4*gtYZ*phidy*phidz - 2*gtZZ*(phidz*phidz) + gtZZ*phidzdz))/3.;
    double const RddTFxy = (-3*(2*GammatXxx*GammatXxy + 2*GammatXyy*GammatYxx + 2*GammatXxy*GammatYxy + 2*GammatYxy*GammatYyy + 2*GammatXyz*GammatZxx + 2*GammatXxz*GammatZxy + 2*GammatYyz*GammatZxy + 2*GammatYxz*GammatZyy + 2*GammatZxz*GammatZyz - GammatXdy*gtxx + gtxxdx*gtXXdy - GammatXdx*gtxy - GammatYdy*gtxy + gtXXdx*gtxydx + gtXX*gtxydxdx + 2*gtXY*gtxydxdy + gtXYdx*gtxydy + gtxxdy*gtXYdy + gtxydx*gtXYdy - GammatZdy*gtxz + 2*gtxydxdz*gtXZ + gtxydz*gtXZdx + gtxxdz*gtXZdy + gtxzdx*gtXZdy - GammatYdx*gtyy + gtxydydy*gtYY + gtXYdx*gtyydx + gtYYdx*gtyydy + gtxydy*gtYYdy - GammatZdx*gtyz + 2*gtxydydz*gtYZ + gtXZdx*gtyzdx + gtyydz*gtYZdx + gtYZdx*gtyzdy + gtxydz*gtYZdy + gtxzdy*gtYZdy + gtxydzdz*gtZZ + gtyzdz*gtZZdx + gtxzdz*gtZZdy) + gtxy*(-2*(GammatXdx + GammatYdy + GammatZdz) + 2*(GammatXxx*GammatXxx)*gtXX + 2*(GammatYxy*GammatYxy)*gtXX + 4*GammatXxz*GammatZxx*gtXX + 4*GammatYxz*GammatZxy*gtXX + 2*(GammatZxz*GammatZxz)*gtXX + gtXX*gtxxdx*gtXXdx + 4*GammatXyy*GammatYxx*gtXY + 4*GammatYxy*GammatYyy*gtXY + 4*GammatXyz*GammatZxx*gtXY + 4*GammatXxz*GammatZxy*gtXY + 4*GammatYyz*GammatZxy*gtXY + 4*GammatYxz*GammatZyy*gtXY + 4*GammatZxz*GammatZyz*gtXY - gtXXdx*gtxxdy*gtXY + gtxxdx*gtXXdy*gtXY + 2*gtXXdx*gtXY*gtxydx + 2*gtXX*gtxxdy*gtXYdx + 2*gtxxdy*gtXY*gtXYdy + 4*GammatXyz*GammatYxx*gtXZ + 4*GammatYxy*GammatYyz*gtXZ + 4*GammatXzz*GammatZxx*gtXZ + 4*GammatYzz*GammatZxy*gtXZ + 4*GammatXxz*GammatZxz*gtXZ + 4*GammatYxz*GammatZyz*gtXZ + 4*GammatZxz*GammatZzz*gtXZ - gtXXdx*gtxxdz*gtXZ + gtxxdx*gtXXdz*gtXZ - 2*gtXYdx*gtxydz*gtXZ + 2*gtxxdy*gtXYdz*gtXZ + 4*GammatXxx*(GammatXxy*gtXY + GammatXxz*gtXZ) + 2*gtXXdx*gtXZ*gtxzdx + 2*gtXX*gtxxdz*gtXZdx + 2*gtXY*gtxydz*gtXZdx + 2*gtXYdx*gtXZ*gtxzdy - 2*gtXY*gtXZdx*gtxzdy + 2*gtxxdz*gtXY*gtXZdy + 2*gtxxdz*gtXZ*gtXZdz + 2*(GammatXxy*GammatXxy)*gtYY + 4*GammatXyy*GammatYxy*gtYY + 2*(GammatYyy*GammatYyy)*gtYY + 4*GammatXyz*GammatZxy*gtYY + 4*GammatYyz*GammatZyy*gtYY + 2*(GammatZyz*GammatZyz)*gtYY - gtxxdy*gtXXdy*gtYY + 2*gtXXdy*gtxydx*gtYY + 2*gtxydz*gtXZdy*gtYY - 2*gtxzdy*gtXZdy*gtYY + 2*gtXY*gtXYdx*gtyydx + 2*gtXYdy*gtYY*gtyydx + 2*gtXX*gtxydy*gtYYdx - gtXX*gtyydx*gtYYdx + gtXY*gtYYdx*gtyydy + 2*gtXY*gtxydy*gtYYdy - gtXY*gtyydx*gtYYdy + gtYY*gtyydy*gtYYdy - gtXZ*gtYYdx*gtyydz + 2*gtxydy*gtXZ*gtYYdz - gtXZ*gtyydx*gtYYdz + 4*GammatXyz*GammatYxy*gtYZ + 4*GammatXyy*GammatYxz*gtYZ + 4*GammatYyy*GammatYyz*gtYZ + 4*GammatXzz*GammatZxy*gtYZ + 4*GammatXyz*GammatZxz*gtYZ + 4*GammatYzz*GammatZyy*gtYZ + 4*GammatYyz*GammatZyz*gtYZ + 4*GammatZyz*GammatZzz*gtYZ - gtXXdy*gtxxdz*gtYZ - gtxxdy*gtXXdz*gtYZ + 2*gtXXdz*gtxydx*gtYZ - 2*gtXYdy*gtxydz*gtYZ + 2*gtXXdy*gtxzdx*gtYZ + 2*gtXYdy*gtxzdy*gtYZ + 2*gtxydz*gtXZdz*gtYZ - 2*gtxzdy*gtXZdz*gtYZ + 2*gtXYdz*gtyydx*gtYZ - gtYYdy*gtyydz*gtYZ + gtyydy*gtYYdz*gtYZ + 4*GammatXxy*(GammatYxx*gtXX + GammatYxy*gtXY + GammatYxz*gtXZ + GammatXxz*gtYZ) + 2*gtXYdx*gtXZ*gtyzdx + 2*gtXY*gtXZdx*gtyzdx + 2*gtXZdy*gtYY*gtyzdx + 2*gtXYdy*gtYZ*gtyzdx + 2*gtXZdz*gtYZ*gtyzdx + 2*gtXX*gtxydz*gtYZdx + 2*gtXX*gtxzdy*gtYZdx + 2*gtXY*gtyydz*gtYZdx - 2*gtXX*gtyzdx*gtYZdx + 2*gtXZ*gtYYdx*gtyzdy + 2*gtYYdy*gtYZ*gtyzdy + 2*gtXY*gtxydz*gtYZdy + 2*gtXY*gtxzdy*gtYZdy + 2*gtYY*gtyydz*gtYZdy - 2*gtXY*gtyzdx*gtYZdy + 2*gtxydz*gtXZ*gtYZdz + 2*gtXZ*gtxzdy*gtYZdz + 2*gtyydz*gtYZ*gtYZdz - 2*gtXZ*gtyzdx*gtYZdz + 2*(GammatXxz*GammatXxz)*gtZZ + 4*GammatXyz*GammatYxz*gtZZ + 2*(GammatYyz*GammatYyz)*gtZZ + 4*GammatXzz*GammatZxz*gtZZ + 4*GammatYzz*GammatZyz*gtZZ + 2*(GammatZzz*GammatZzz)*gtZZ - gtxxdz*gtXXdz*gtZZ - 2*gtxydz*gtXYdz*gtZZ + 2*gtXXdz*gtxzdx*gtZZ + 2*gtXYdz*gtxzdy*gtZZ - gtyydz*gtYYdz*gtZZ + 2*gtXYdz*gtyzdx*gtZZ + 2*gtYYdz*gtyzdy*gtZZ + 2*gtXZ*gtXZdx*gtzzdx + 2*gtXZdy*gtYZ*gtzzdx + 2*gtXZdz*gtZZ*gtzzdx + 2*gtXZ*gtYZdx*gtzzdy + 2*gtYZ*gtYZdy*gtzzdy + 2*gtYZdz*gtZZ*gtzzdy + 2*gtXY*gtxzdz*gtZZdy + 2*gtYY*gtyzdz*gtZZdy - gtXY*gtzzdx*gtZZdy - gtYY*gtzzdy*gtZZdy + gtYZ*gtZZdy*gtzzdz + gtZZdx*(2*gtXX*gtxzdz + 2*gtXY*gtyzdz - gtXX*gtzzdx - gtXY*gtzzdy + gtXZ*gtzzdz) + 2*gtXZ*gtxzdz*gtZZdz + 2*gtYZ*gtyzdz*gtZZdz - gtXZ*gtzzdx*gtZZdz - gtYZ*gtzzdy*gtZZdz + gtZZ*gtzzdz*gtZZdz) - 12*phidxdy + 24*phidx*phidy + 4*gtxy*(gtXX*(-2*(phidx*phidx) + phidxdx) + 2*gtXZ*phidxdz - 2*gtYY*(phidy*phidy) + 2*gtXY*(phidxdy - 2*phidx*phidy) + gtYY*phidydy + 2*gtYZ*phidydz - 4*gtXZ*phidx*phidz - 4*gtYZ*phidy*phidz - 2*gtZZ*(phidz*phidz) + gtZZ*phidzdz))/6.;
    double const RddTFxz = (-3*(2*GammatXxx*GammatXxz + 2*GammatXyz*GammatYxx + 2*GammatXxy*GammatYxz + 2*GammatYxy*GammatYyz + 2*GammatXzz*GammatZxx + 2*GammatYzz*GammatZxy + 2*GammatXxz*GammatZxz + 2*GammatYxz*GammatZyz + 2*GammatZxz*GammatZzz - GammatXdz*gtxx + gtxxdx*gtXXdz - GammatYdz*gtxy + gtxxdy*gtXYdz + gtxydx*gtXYdz - GammatXdx*gtxz - GammatZdz*gtxz + gtXXdx*gtxzdx + gtXX*gtxzdxdx + 2*gtXY*gtxzdxdy + 2*gtXZ*gtxzdxdz + gtXYdx*gtxzdy + gtXZdx*gtxzdz + gtxxdz*gtXZdz + gtxzdx*gtXZdz + gtxzdydy*gtYY + gtxydy*gtYYdz - GammatYdx*gtyz + 2*gtxzdydz*gtYZ + gtXYdx*gtyzdx + gtYYdx*gtyzdy + gtYZdx*gtyzdz + gtxydz*gtYZdz + gtxzdy*gtYZdz - GammatZdx*gtzz + gtxzdzdz*gtZZ + gtXZdx*gtzzdx + gtYZdx*gtzzdy + gtZZdx*gtzzdz + gtxzdz*gtZZdz) + gtxz*(-2*(GammatXdx + GammatYdy + GammatZdz) + 2*(GammatXxx*GammatXxx)*gtXX + 2*(GammatYxy*GammatYxy)*gtXX + 4*GammatXxz*GammatZxx*gtXX + 4*GammatYxz*GammatZxy*gtXX + 2*(GammatZxz*GammatZxz)*gtXX + gtXX*gtxxdx*gtXXdx + 4*GammatXyy*GammatYxx*gtXY + 4*GammatYxy*GammatYyy*gtXY + 4*GammatXyz*GammatZxx*gtXY + 4*GammatXxz*GammatZxy*gtXY + 4*GammatYyz*GammatZxy*gtXY + 4*GammatYxz*GammatZyy*gtXY + 4*GammatZxz*GammatZyz*gtXY - gtXXdx*gtxxdy*gtXY + gtxxdx*gtXXdy*gtXY + 2*gtXXdx*gtXY*gtxydx + 2*gtXX*gtxxdy*gtXYdx + 2*gtxxdy*gtXY*gtXYdy + 4*GammatXyz*GammatYxx*gtXZ + 4*GammatYxy*GammatYyz*gtXZ + 4*GammatXzz*GammatZxx*gtXZ + 4*GammatYzz*GammatZxy*gtXZ + 4*GammatXxz*GammatZxz*gtXZ + 4*GammatYxz*GammatZyz*gtXZ + 4*GammatZxz*GammatZzz*gtXZ - gtXXdx*gtxxdz*gtXZ + gtxxdx*gtXXdz*gtXZ - 2*gtXYdx*gtxydz*gtXZ + 2*gtxxdy*gtXYdz*gtXZ + 4*GammatXxx*(GammatXxy*gtXY + GammatXxz*gtXZ) + 2*gtXXdx*gtXZ*gtxzdx + 2*gtXX*gtxxdz*gtXZdx + 2*gtXY*gtxydz*gtXZdx + 2*gtXYdx*gtXZ*gtxzdy - 2*gtXY*gtXZdx*gtxzdy + 2*gtxxdz*gtXY*gtXZdy + 2*gtxxdz*gtXZ*gtXZdz + 2*(GammatXxy*GammatXxy)*gtYY + 4*GammatXyy*GammatYxy*gtYY + 2*(GammatYyy*GammatYyy)*gtYY + 4*GammatXyz*GammatZxy*gtYY + 4*GammatYyz*GammatZyy*gtYY + 2*(GammatZyz*GammatZyz)*gtYY - gtxxdy*gtXXdy*gtYY + 2*gtXXdy*gtxydx*gtYY + 2*gtxydz*gtXZdy*gtYY - 2*gtxzdy*gtXZdy*gtYY + 2*gtXY*gtXYdx*gtyydx + 2*gtXYdy*gtYY*gtyydx + 2*gtXX*gtxydy*gtYYdx - gtXX*gtyydx*gtYYdx + gtXY*gtYYdx*gtyydy + 2*gtXY*gtxydy*gtYYdy - gtXY*gtyydx*gtYYdy + gtYY*gtyydy*gtYYdy - gtXZ*gtYYdx*gtyydz + 2*gtxydy*gtXZ*gtYYdz - gtXZ*gtyydx*gtYYdz + 4*GammatXyz*GammatYxy*gtYZ + 4*GammatXyy*GammatYxz*gtYZ + 4*GammatYyy*GammatYyz*gtYZ + 4*GammatXzz*GammatZxy*gtYZ + 4*GammatXyz*GammatZxz*gtYZ + 4*GammatYzz*GammatZyy*gtYZ + 4*GammatYyz*GammatZyz*gtYZ + 4*GammatZyz*GammatZzz*gtYZ - gtXXdy*gtxxdz*gtYZ - gtxxdy*gtXXdz*gtYZ + 2*gtXXdz*gtxydx*gtYZ - 2*gtXYdy*gtxydz*gtYZ + 2*gtXXdy*gtxzdx*gtYZ + 2*gtXYdy*gtxzdy*gtYZ + 2*gtxydz*gtXZdz*gtYZ - 2*gtxzdy*gtXZdz*gtYZ + 2*gtXYdz*gtyydx*gtYZ - gtYYdy*gtyydz*gtYZ + gtyydy*gtYYdz*gtYZ + 4*GammatXxy*(GammatYxx*gtXX + GammatYxy*gtXY + GammatYxz*gtXZ + GammatXxz*gtYZ) + 2*gtXYdx*gtXZ*gtyzdx + 2*gtXY*gtXZdx*gtyzdx + 2*gtXZdy*gtYY*gtyzdx + 2*gtXYdy*gtYZ*gtyzdx + 2*gtXZdz*gtYZ*gtyzdx + 2*gtXX*gtxydz*gtYZdx + 2*gtXX*gtxzdy*gtYZdx + 2*gtXY*gtyydz*gtYZdx - 2*gtXX*gtyzdx*gtYZdx + 2*gtXZ*gtYYdx*gtyzdy + 2*gtYYdy*gtYZ*gtyzdy + 2*gtXY*gtxydz*gtYZdy + 2*gtXY*gtxzdy*gtYZdy + 2*gtYY*gtyydz*gtYZdy - 2*gtXY*gtyzdx*gtYZdy + 2*gtxydz*gtXZ*gtYZdz + 2*gtXZ*gtxzdy*gtYZdz + 2*gtyydz*gtYZ*gtYZdz - 2*gtXZ*gtyzdx*gtYZdz + 2*(GammatXxz*GammatXxz)*gtZZ + 4*GammatXyz*GammatYxz*gtZZ + 2*(GammatYyz*GammatYyz)*gtZZ + 4*GammatXzz*GammatZxz*gtZZ + 4*GammatYzz*GammatZyz*gtZZ + 2*(GammatZzz*GammatZzz)*gtZZ - gtxxdz*gtXXdz*gtZZ - 2*gtxydz*gtXYdz*gtZZ + 2*gtXXdz*gtxzdx*gtZZ + 2*gtXYdz*gtxzdy*gtZZ - gtyydz*gtYYdz*gtZZ + 2*gtXYdz*gtyzdx*gtZZ + 2*gtYYdz*gtyzdy*gtZZ + 2*gtXZ*gtXZdx*gtzzdx + 2*gtXZdy*gtYZ*gtzzdx + 2*gtXZdz*gtZZ*gtzzdx + 2*gtXZ*gtYZdx*gtzzdy + 2*gtYZ*gtYZdy*gtzzdy + 2*gtYZdz*gtZZ*gtzzdy + 2*gtXY*gtxzdz*gtZZdy + 2*gtYY*gtyzdz*gtZZdy - gtXY*gtzzdx*gtZZdy - gtYY*gtzzdy*gtZZdy + gtYZ*gtZZdy*gtzzdz + gtZZdx*(2*gtXX*gtxzdz + 2*gtXY*gtyzdz - gtXX*gtzzdx - gtXY*gtzzdy + gtXZ*gtzzdz) + 2*gtXZ*gtxzdz*gtZZdz + 2*gtYZ*gtyzdz*gtZZdz - gtXZ*gtzzdx*gtZZdz - gtYZ*gtzzdy*gtZZdz + gtZZ*gtzzdz*gtZZdz) - 12*phidxdz + 24*phidx*phidz + 4*gtxz*(gtXX*(-2*(phidx*phidx) + phidxdx) + 2*gtXZ*phidxdz - 2*gtYY*(phidy*phidy) + 2*gtXY*(phidxdy - 2*phidx*phidy) + gtYY*phidydy + 2*gtYZ*phidydz - 4*gtXZ*phidx*phidz - 4*gtYZ*phidy*phidz - 2*gtZZ*(phidz*phidz) + gtZZ*phidzdz))/6.;
    double const RddTFyy = -(GammatXxy*GammatXxy) - 2*GammatXyy*GammatYxy - GammatYyy*GammatYyy - 2*GammatXyz*GammatZxy - 2*GammatYyz*GammatZyy - GammatZyz*GammatZyz + GammatXdy*gtxy - gtXXdy*gtxydx + GammatYdy*gtyy - gtXYdy*(gtxydy + gtyydx) - (gtXX*gtyydxdx)/2. - gtXY*gtyydxdy - gtXZ*gtyydxdz - gtyydy*gtYYdy - (gtYY*gtyydydy)/2. + GammatZdy*gtyz - gtyydydz*gtYZ - gtXZdy*(gtxydz + gtyzdx) - (gtyydz + gtyzdy)*gtYZdy - (gtyydzdz*gtZZ)/2. - gtyzdz*gtZZdy + (gtyy*(-2*(GammatXdx + GammatYdy + GammatZdz) + 2*(GammatXxx*GammatXxx)*gtXX + 2*(GammatYxy*GammatYxy)*gtXX + 4*GammatXxz*GammatZxx*gtXX + 4*GammatYxz*GammatZxy*gtXX + 2*(GammatZxz*GammatZxz)*gtXX + gtXX*gtxxdx*gtXXdx + 4*GammatXyy*GammatYxx*gtXY + 4*GammatYxy*GammatYyy*gtXY + 4*GammatXyz*GammatZxx*gtXY + 4*GammatXxz*GammatZxy*gtXY + 4*GammatYyz*GammatZxy*gtXY + 4*GammatYxz*GammatZyy*gtXY + 4*GammatZxz*GammatZyz*gtXY - gtXXdx*gtxxdy*gtXY + gtxxdx*gtXXdy*gtXY + 2*gtXXdx*gtXY*gtxydx + 2*gtXX*gtxxdy*gtXYdx + 2*gtxxdy*gtXY*gtXYdy + 4*GammatXyz*GammatYxx*gtXZ + 4*GammatYxy*GammatYyz*gtXZ + 4*GammatXzz*GammatZxx*gtXZ + 4*GammatYzz*GammatZxy*gtXZ + 4*GammatXxz*GammatZxz*gtXZ + 4*GammatYxz*GammatZyz*gtXZ + 4*GammatZxz*GammatZzz*gtXZ - gtXXdx*gtxxdz*gtXZ + gtxxdx*gtXXdz*gtXZ - 2*gtXYdx*gtxydz*gtXZ + 2*gtxxdy*gtXYdz*gtXZ + 4*GammatXxx*(GammatXxy*gtXY + GammatXxz*gtXZ) + 2*gtXXdx*gtXZ*gtxzdx + 2*gtXX*gtxxdz*gtXZdx + 2*gtXY*gtxydz*gtXZdx + 2*gtXYdx*gtXZ*gtxzdy - 2*gtXY*gtXZdx*gtxzdy + 2*gtxxdz*gtXY*gtXZdy + 2*gtxxdz*gtXZ*gtXZdz + 2*(GammatXxy*GammatXxy)*gtYY + 4*GammatXyy*GammatYxy*gtYY + 2*(GammatYyy*GammatYyy)*gtYY + 4*GammatXyz*GammatZxy*gtYY + 4*GammatYyz*GammatZyy*gtYY + 2*(GammatZyz*GammatZyz)*gtYY - gtxxdy*gtXXdy*gtYY + 2*gtXXdy*gtxydx*gtYY + 2*gtxydz*gtXZdy*gtYY - 2*gtxzdy*gtXZdy*gtYY + 2*gtXY*gtXYdx*gtyydx + 2*gtXYdy*gtYY*gtyydx + 2*gtXX*gtxydy*gtYYdx - gtXX*gtyydx*gtYYdx + gtXY*gtYYdx*gtyydy + 2*gtXY*gtxydy*gtYYdy - gtXY*gtyydx*gtYYdy + gtYY*gtyydy*gtYYdy - gtXZ*gtYYdx*gtyydz + 2*gtxydy*gtXZ*gtYYdz - gtXZ*gtyydx*gtYYdz + 4*GammatXyz*GammatYxy*gtYZ + 4*GammatXyy*GammatYxz*gtYZ + 4*GammatYyy*GammatYyz*gtYZ + 4*GammatXzz*GammatZxy*gtYZ + 4*GammatXyz*GammatZxz*gtYZ + 4*GammatYzz*GammatZyy*gtYZ + 4*GammatYyz*GammatZyz*gtYZ + 4*GammatZyz*GammatZzz*gtYZ - gtXXdy*gtxxdz*gtYZ - gtxxdy*gtXXdz*gtYZ + 2*gtXXdz*gtxydx*gtYZ - 2*gtXYdy*gtxydz*gtYZ + 2*gtXXdy*gtxzdx*gtYZ + 2*gtXYdy*gtxzdy*gtYZ + 2*gtxydz*gtXZdz*gtYZ - 2*gtxzdy*gtXZdz*gtYZ + 2*gtXYdz*gtyydx*gtYZ - gtYYdy*gtyydz*gtYZ + gtyydy*gtYYdz*gtYZ + 4*GammatXxy*(GammatYxx*gtXX + GammatYxy*gtXY + GammatYxz*gtXZ + GammatXxz*gtYZ) + 2*gtXYdx*gtXZ*gtyzdx + 2*gtXY*gtXZdx*gtyzdx + 2*gtXZdy*gtYY*gtyzdx + 2*gtXYdy*gtYZ*gtyzdx + 2*gtXZdz*gtYZ*gtyzdx + 2*gtXX*gtxydz*gtYZdx + 2*gtXX*gtxzdy*gtYZdx + 2*gtXY*gtyydz*gtYZdx - 2*gtXX*gtyzdx*gtYZdx + 2*gtXZ*gtYYdx*gtyzdy + 2*gtYYdy*gtYZ*gtyzdy + 2*gtXY*gtxydz*gtYZdy + 2*gtXY*gtxzdy*gtYZdy + 2*gtYY*gtyydz*gtYZdy - 2*gtXY*gtyzdx*gtYZdy + 2*gtxydz*gtXZ*gtYZdz + 2*gtXZ*gtxzdy*gtYZdz + 2*gtyydz*gtYZ*gtYZdz - 2*gtXZ*gtyzdx*gtYZdz + 2*(GammatXxz*GammatXxz)*gtZZ + 4*GammatXyz*GammatYxz*gtZZ + 2*(GammatYyz*GammatYyz)*gtZZ + 4*GammatXzz*GammatZxz*gtZZ + 4*GammatYzz*GammatZyz*gtZZ + 2*(GammatZzz*GammatZzz)*gtZZ - gtxxdz*gtXXdz*gtZZ - 2*gtxydz*gtXYdz*gtZZ + 2*gtXXdz*gtxzdx*gtZZ + 2*gtXYdz*gtxzdy*gtZZ - gtyydz*gtYYdz*gtZZ + 2*gtXYdz*gtyzdx*gtZZ + 2*gtYYdz*gtyzdy*gtZZ + 2*gtXZ*gtXZdx*gtzzdx + 2*gtXZdy*gtYZ*gtzzdx + 2*gtXZdz*gtZZ*gtzzdx + 2*gtXZ*gtYZdx*gtzzdy + 2*gtYZ*gtYZdy*gtzzdy + 2*gtYZdz*gtZZ*gtzzdy + 2*gtXY*gtxzdz*gtZZdy + 2*gtYY*gtyzdz*gtZZdy - gtXY*gtzzdx*gtZZdy - gtYY*gtzzdy*gtZZdy + gtYZ*gtZZdy*gtzzdz + gtZZdx*(2*gtXX*gtxzdz + 2*gtXY*gtyzdz - gtXX*gtzzdx - gtXY*gtzzdy + gtXZ*gtzzdz) + 2*gtXZ*gtxzdz*gtZZdz + 2*gtYZ*gtyzdz*gtZZdz - gtXZ*gtzzdx*gtZZdz - gtYZ*gtzzdy*gtZZdz + gtZZ*gtzzdz*gtZZdz))/6. + 4*(phidy*phidy) - 2*phidydy + (2*gtyy*(gtXX*(-2*(phidx*phidx) + phidxdx) + 2*gtXZ*phidxdz - 2*gtYY*(phidy*phidy) + 2*gtXY*(phidxdy - 2*phidx*phidy) + gtYY*phidydy + 2*gtYZ*phidydz - 4*gtXZ*phidx*phidz - 4*gtYZ*phidy*phidz - 2*gtZZ*(phidz*phidz) + gtZZ*phidzdz))/3.;
    double const RddTFyz = (-3*(2*GammatXxy*GammatXxz + 2*GammatXyy*GammatYxz + 2*GammatYyy*GammatYyz + 2*GammatXzz*GammatZxy + 2*GammatXyz*(GammatYxy + GammatZxz) + 2*GammatYzz*GammatZyy + 2*GammatYyz*GammatZyz + 2*GammatZyz*GammatZzz - GammatXdz*gtxy + gtXXdz*gtxydx + gtxydy*gtXYdz - GammatXdy*gtxz + gtXXdy*gtxzdx + gtXYdy*gtxzdy + gtXZdy*gtxzdz + gtxydz*gtXZdz - GammatYdz*gtyy + gtXYdz*gtyydx + gtyydy*gtYYdz - GammatYdy*gtyz - GammatZdz*gtyz + gtXYdy*gtyzdx + gtXZdz*gtyzdx + gtXX*gtyzdxdx + 2*gtXY*gtyzdxdy + 2*gtXZ*gtyzdxdz + gtYYdy*gtyzdy + gtYY*gtyzdydy + 2*gtYZ*gtyzdydz + gtYZdy*gtyzdz + gtyydz*gtYZdz + gtyzdy*gtYZdz - GammatZdy*gtzz + gtyzdzdz*gtZZ + gtXZdy*gtzzdx + gtYZdy*gtzzdy + gtZZdy*gtzzdz + gtyzdz*gtZZdz) + gtyz*(-2*(GammatXdx + GammatYdy + GammatZdz) + 2*(GammatXxx*GammatXxx)*gtXX + 2*(GammatYxy*GammatYxy)*gtXX + 4*GammatXxz*GammatZxx*gtXX + 4*GammatYxz*GammatZxy*gtXX + 2*(GammatZxz*GammatZxz)*gtXX + gtXX*gtxxdx*gtXXdx + 4*GammatXyy*GammatYxx*gtXY + 4*GammatYxy*GammatYyy*gtXY + 4*GammatXyz*GammatZxx*gtXY + 4*GammatXxz*GammatZxy*gtXY + 4*GammatYyz*GammatZxy*gtXY + 4*GammatYxz*GammatZyy*gtXY + 4*GammatZxz*GammatZyz*gtXY - gtXXdx*gtxxdy*gtXY + gtxxdx*gtXXdy*gtXY + 2*gtXXdx*gtXY*gtxydx + 2*gtXX*gtxxdy*gtXYdx + 2*gtxxdy*gtXY*gtXYdy + 4*GammatXyz*GammatYxx*gtXZ + 4*GammatYxy*GammatYyz*gtXZ + 4*GammatXzz*GammatZxx*gtXZ + 4*GammatYzz*GammatZxy*gtXZ + 4*GammatXxz*GammatZxz*gtXZ + 4*GammatYxz*GammatZyz*gtXZ + 4*GammatZxz*GammatZzz*gtXZ - gtXXdx*gtxxdz*gtXZ + gtxxdx*gtXXdz*gtXZ - 2*gtXYdx*gtxydz*gtXZ + 2*gtxxdy*gtXYdz*gtXZ + 4*GammatXxx*(GammatXxy*gtXY + GammatXxz*gtXZ) + 2*gtXXdx*gtXZ*gtxzdx + 2*gtXX*gtxxdz*gtXZdx + 2*gtXY*gtxydz*gtXZdx + 2*gtXYdx*gtXZ*gtxzdy - 2*gtXY*gtXZdx*gtxzdy + 2*gtxxdz*gtXY*gtXZdy + 2*gtxxdz*gtXZ*gtXZdz + 2*(GammatXxy*GammatXxy)*gtYY + 4*GammatXyy*GammatYxy*gtYY + 2*(GammatYyy*GammatYyy)*gtYY + 4*GammatXyz*GammatZxy*gtYY + 4*GammatYyz*GammatZyy*gtYY + 2*(GammatZyz*GammatZyz)*gtYY - gtxxdy*gtXXdy*gtYY + 2*gtXXdy*gtxydx*gtYY + 2*gtxydz*gtXZdy*gtYY - 2*gtxzdy*gtXZdy*gtYY + 2*gtXY*gtXYdx*gtyydx + 2*gtXYdy*gtYY*gtyydx + 2*gtXX*gtxydy*gtYYdx - gtXX*gtyydx*gtYYdx + gtXY*gtYYdx*gtyydy + 2*gtXY*gtxydy*gtYYdy - gtXY*gtyydx*gtYYdy + gtYY*gtyydy*gtYYdy - gtXZ*gtYYdx*gtyydz + 2*gtxydy*gtXZ*gtYYdz - gtXZ*gtyydx*gtYYdz + 4*GammatXyz*GammatYxy*gtYZ + 4*GammatXyy*GammatYxz*gtYZ + 4*GammatYyy*GammatYyz*gtYZ + 4*GammatXzz*GammatZxy*gtYZ + 4*GammatXyz*GammatZxz*gtYZ + 4*GammatYzz*GammatZyy*gtYZ + 4*GammatYyz*GammatZyz*gtYZ + 4*GammatZyz*GammatZzz*gtYZ - gtXXdy*gtxxdz*gtYZ - gtxxdy*gtXXdz*gtYZ + 2*gtXXdz*gtxydx*gtYZ - 2*gtXYdy*gtxydz*gtYZ + 2*gtXXdy*gtxzdx*gtYZ + 2*gtXYdy*gtxzdy*gtYZ + 2*gtxydz*gtXZdz*gtYZ - 2*gtxzdy*gtXZdz*gtYZ + 2*gtXYdz*gtyydx*gtYZ - gtYYdy*gtyydz*gtYZ + gtyydy*gtYYdz*gtYZ + 4*GammatXxy*(GammatYxx*gtXX + GammatYxy*gtXY + GammatYxz*gtXZ + GammatXxz*gtYZ) + 2*gtXYdx*gtXZ*gtyzdx + 2*gtXY*gtXZdx*gtyzdx + 2*gtXZdy*gtYY*gtyzdx + 2*gtXYdy*gtYZ*gtyzdx + 2*gtXZdz*gtYZ*gtyzdx + 2*gtXX*gtxydz*gtYZdx + 2*gtXX*gtxzdy*gtYZdx + 2*gtXY*gtyydz*gtYZdx - 2*gtXX*gtyzdx*gtYZdx + 2*gtXZ*gtYYdx*gtyzdy + 2*gtYYdy*gtYZ*gtyzdy + 2*gtXY*gtxydz*gtYZdy + 2*gtXY*gtxzdy*gtYZdy + 2*gtYY*gtyydz*gtYZdy - 2*gtXY*gtyzdx*gtYZdy + 2*gtxydz*gtXZ*gtYZdz + 2*gtXZ*gtxzdy*gtYZdz + 2*gtyydz*gtYZ*gtYZdz - 2*gtXZ*gtyzdx*gtYZdz + 2*(GammatXxz*GammatXxz)*gtZZ + 4*GammatXyz*GammatYxz*gtZZ + 2*(GammatYyz*GammatYyz)*gtZZ + 4*GammatXzz*GammatZxz*gtZZ + 4*GammatYzz*GammatZyz*gtZZ + 2*(GammatZzz*GammatZzz)*gtZZ - gtxxdz*gtXXdz*gtZZ - 2*gtxydz*gtXYdz*gtZZ + 2*gtXXdz*gtxzdx*gtZZ + 2*gtXYdz*gtxzdy*gtZZ - gtyydz*gtYYdz*gtZZ + 2*gtXYdz*gtyzdx*gtZZ + 2*gtYYdz*gtyzdy*gtZZ + 2*gtXZ*gtXZdx*gtzzdx + 2*gtXZdy*gtYZ*gtzzdx + 2*gtXZdz*gtZZ*gtzzdx + 2*gtXZ*gtYZdx*gtzzdy + 2*gtYZ*gtYZdy*gtzzdy + 2*gtYZdz*gtZZ*gtzzdy + 2*gtXY*gtxzdz*gtZZdy + 2*gtYY*gtyzdz*gtZZdy - gtXY*gtzzdx*gtZZdy - gtYY*gtzzdy*gtZZdy + gtYZ*gtZZdy*gtzzdz + gtZZdx*(2*gtXX*gtxzdz + 2*gtXY*gtyzdz - gtXX*gtzzdx - gtXY*gtzzdy + gtXZ*gtzzdz) + 2*gtXZ*gtxzdz*gtZZdz + 2*gtYZ*gtyzdz*gtZZdz - gtXZ*gtzzdx*gtZZdz - gtYZ*gtzzdy*gtZZdz + gtZZ*gtzzdz*gtZZdz) - 12*phidydz + 24*phidy*phidz + 4*gtyz*(gtXX*(-2*(phidx*phidx) + phidxdx) + 2*gtXZ*phidxdz - 2*gtYY*(phidy*phidy) + 2*gtXY*(phidxdy - 2*phidx*phidy) + gtYY*phidydy + 2*gtYZ*phidydz - 4*gtXZ*phidx*phidz - 4*gtYZ*phidy*phidz - 2*gtZZ*(phidz*phidz) + gtZZ*phidzdz))/6.;
    double const RddTFzz = (-6*(GammatXxz*GammatXxz) - 12*GammatXyz*GammatYxz - 6*(GammatYyz*GammatYyz) - 12*GammatXzz*GammatZxz - 12*GammatYzz*GammatZyz - 6*(GammatZzz*GammatZzz) + 6*GammatXdz*gtxz - 6*gtXXdz*gtxzdx + 6*GammatYdz*gtyz - 6*gtXYdz*(gtxzdy + gtyzdx) - 6*gtYYdz*gtyzdy + 6*GammatZdz*gtzz - 6*gtXZdz*(gtxzdz + gtzzdx) - 3*gtXX*gtzzdxdx - 6*gtXY*gtzzdxdy - 6*gtXZ*gtzzdxdz - 6*gtYZdz*(gtyzdz + gtzzdy) - 3*gtYY*gtzzdydy - 6*gtYZ*gtzzdydz - 6*gtzzdz*gtZZdz + gtzz*(-2*(GammatXdx + GammatYdy + GammatZdz) + 2*(GammatXxx*GammatXxx)*gtXX + 2*(GammatYxy*GammatYxy)*gtXX + 4*GammatXxz*GammatZxx*gtXX + 4*GammatYxz*GammatZxy*gtXX + 2*(GammatZxz*GammatZxz)*gtXX + gtXX*gtxxdx*gtXXdx + 4*GammatXyy*GammatYxx*gtXY + 4*GammatYxy*GammatYyy*gtXY + 4*GammatXyz*GammatZxx*gtXY + 4*GammatXxz*GammatZxy*gtXY + 4*GammatYyz*GammatZxy*gtXY + 4*GammatYxz*GammatZyy*gtXY + 4*GammatZxz*GammatZyz*gtXY - gtXXdx*gtxxdy*gtXY + gtxxdx*gtXXdy*gtXY + 2*gtXXdx*gtXY*gtxydx + 2*gtXX*gtxxdy*gtXYdx + 2*gtxxdy*gtXY*gtXYdy + 4*GammatXyz*GammatYxx*gtXZ + 4*GammatYxy*GammatYyz*gtXZ + 4*GammatXzz*GammatZxx*gtXZ + 4*GammatYzz*GammatZxy*gtXZ + 4*GammatXxz*GammatZxz*gtXZ + 4*GammatYxz*GammatZyz*gtXZ + 4*GammatZxz*GammatZzz*gtXZ - gtXXdx*gtxxdz*gtXZ + gtxxdx*gtXXdz*gtXZ - 2*gtXYdx*gtxydz*gtXZ + 2*gtxxdy*gtXYdz*gtXZ + 4*GammatXxx*(GammatXxy*gtXY + GammatXxz*gtXZ) + 2*gtXXdx*gtXZ*gtxzdx + 2*gtXX*gtxxdz*gtXZdx + 2*gtXY*gtxydz*gtXZdx + 2*gtXYdx*gtXZ*gtxzdy - 2*gtXY*gtXZdx*gtxzdy + 2*gtxxdz*gtXY*gtXZdy + 2*gtxxdz*gtXZ*gtXZdz + 2*(GammatXxy*GammatXxy)*gtYY + 4*GammatXyy*GammatYxy*gtYY + 2*(GammatYyy*GammatYyy)*gtYY + 4*GammatXyz*GammatZxy*gtYY + 4*GammatYyz*GammatZyy*gtYY + 2*(GammatZyz*GammatZyz)*gtYY - gtxxdy*gtXXdy*gtYY + 2*gtXXdy*gtxydx*gtYY + 2*gtxydz*gtXZdy*gtYY - 2*gtxzdy*gtXZdy*gtYY + 2*gtXY*gtXYdx*gtyydx + 2*gtXYdy*gtYY*gtyydx + 2*gtXX*gtxydy*gtYYdx - gtXX*gtyydx*gtYYdx + gtXY*gtYYdx*gtyydy + 2*gtXY*gtxydy*gtYYdy - gtXY*gtyydx*gtYYdy + gtYY*gtyydy*gtYYdy - gtXZ*gtYYdx*gtyydz + 2*gtxydy*gtXZ*gtYYdz - gtXZ*gtyydx*gtYYdz + 4*GammatXyz*GammatYxy*gtYZ + 4*GammatXyy*GammatYxz*gtYZ + 4*GammatYyy*GammatYyz*gtYZ + 4*GammatXzz*GammatZxy*gtYZ + 4*GammatXyz*GammatZxz*gtYZ + 4*GammatYzz*GammatZyy*gtYZ + 4*GammatYyz*GammatZyz*gtYZ + 4*GammatZyz*GammatZzz*gtYZ - gtXXdy*gtxxdz*gtYZ - gtxxdy*gtXXdz*gtYZ + 2*gtXXdz*gtxydx*gtYZ - 2*gtXYdy*gtxydz*gtYZ + 2*gtXXdy*gtxzdx*gtYZ + 2*gtXYdy*gtxzdy*gtYZ + 2*gtxydz*gtXZdz*gtYZ - 2*gtxzdy*gtXZdz*gtYZ + 2*gtXYdz*gtyydx*gtYZ - gtYYdy*gtyydz*gtYZ + gtyydy*gtYYdz*gtYZ + 4*GammatXxy*(GammatYxx*gtXX + GammatYxy*gtXY + GammatYxz*gtXZ + GammatXxz*gtYZ) + 2*gtXYdx*gtXZ*gtyzdx + 2*gtXY*gtXZdx*gtyzdx + 2*gtXZdy*gtYY*gtyzdx + 2*gtXYdy*gtYZ*gtyzdx + 2*gtXZdz*gtYZ*gtyzdx + 2*gtXX*gtxydz*gtYZdx + 2*gtXX*gtxzdy*gtYZdx + 2*gtXY*gtyydz*gtYZdx - 2*gtXX*gtyzdx*gtYZdx + 2*gtXZ*gtYYdx*gtyzdy + 2*gtYYdy*gtYZ*gtyzdy + 2*gtXY*gtxydz*gtYZdy + 2*gtXY*gtxzdy*gtYZdy + 2*gtYY*gtyydz*gtYZdy - 2*gtXY*gtyzdx*gtYZdy + 2*gtxydz*gtXZ*gtYZdz + 2*gtXZ*gtxzdy*gtYZdz + 2*gtyydz*gtYZ*gtYZdz - 2*gtXZ*gtyzdx*gtYZdz + 2*(GammatXxz*GammatXxz)*gtZZ + 4*GammatXyz*GammatYxz*gtZZ + 2*(GammatYyz*GammatYyz)*gtZZ + 4*GammatXzz*GammatZxz*gtZZ + 4*GammatYzz*GammatZyz*gtZZ + 2*(GammatZzz*GammatZzz)*gtZZ - gtxxdz*gtXXdz*gtZZ - 2*gtxydz*gtXYdz*gtZZ + 2*gtXXdz*gtxzdx*gtZZ + 2*gtXYdz*gtxzdy*gtZZ - gtyydz*gtYYdz*gtZZ + 2*gtXYdz*gtyzdx*gtZZ + 2*gtYYdz*gtyzdy*gtZZ + 2*gtXZ*gtXZdx*gtzzdx + 2*gtXZdy*gtYZ*gtzzdx + 2*gtXZdz*gtZZ*gtzzdx + 2*gtXZ*gtYZdx*gtzzdy + 2*gtYZ*gtYZdy*gtzzdy + 2*gtYZdz*gtZZ*gtzzdy + 2*gtXY*gtxzdz*gtZZdy + 2*gtYY*gtyzdz*gtZZdy - gtXY*gtzzdx*gtZZdy - gtYY*gtzzdy*gtZZdy + gtYZ*gtZZdy*gtzzdz + gtZZdx*(2*gtXX*gtxzdz + 2*gtXY*gtyzdz - gtXX*gtzzdx - gtXY*gtzzdy + gtXZ*gtzzdz) + 2*gtXZ*gtxzdz*gtZZdz + 2*gtYZ*gtyzdz*gtZZdz - gtXZ*gtzzdx*gtZZdz - gtYZ*gtzzdy*gtZZdz + gtZZ*gtzzdz*gtZZdz) - 3*gtZZ*gtzzdzdz + 4*(gtzz*(gtXX*(-2*(phidx*phidx) + phidxdx) + 2*gtXZ*phidxdz - 2*gtYY*(phidy*phidy) + 2*gtXY*(phidxdy - 2*phidx*phidy) + gtYY*phidydy + 2*gtYZ*phidydz) - 4*gtzz*(gtXZ*phidx + gtYZ*phidy)*phidz + (6 - 2*gtzz*gtZZ)*(phidz*phidz) + (-3 + gtzz*gtZZ)*phidzdz))/6.;
    double const DiDjalpTFxx = alpdxdx - alpdx*GammatXxx - alpdy*GammatYxx - alpdz*GammatZxx - 4*alpdx*phidx - (gtxx*(-((-alpdxdx + alpdx*GammatXxx + alpdy*GammatYxx + alpdz*GammatZxx)*gtXX) - 2*(-alpdxdy + alpdx*GammatXxy + alpdy*GammatYxy + alpdz*GammatZxy)*gtXY - 2*(-alpdxdz + alpdx*GammatXxz + alpdy*GammatYxz + alpdz*GammatZxz)*gtXZ - (-alpdydy + alpdx*GammatXyy + alpdy*GammatYyy + alpdz*GammatZyy)*gtYY - 2*(-alpdydz + alpdx*GammatXyz + alpdy*GammatYyz + alpdz*GammatZyz)*gtYZ - (-alpdzdz + alpdx*GammatXzz + alpdy*GammatYzz + alpdz*GammatZzz)*gtZZ - 4*(alpdx*gtXX + alpdy*gtXY + alpdz*gtXZ)*phidx - 4*(alpdy*gtYY + alpdz*gtYZ)*phidy - 4*(alpdy*gtYZ + alpdz*gtZZ)*phidz - 4*alpdx*(gtXY*phidy + gtXZ*phidz)))/3.;
    double const DiDjalpTFxy = alpdxdy - alpdx*GammatXxy - alpdy*GammatYxy - alpdz*GammatZxy - 2*(alpdy*phidx + alpdx*phidy) - (gtxy*(-((-alpdxdx + alpdx*GammatXxx + alpdy*GammatYxx + alpdz*GammatZxx)*gtXX) - 2*(-alpdxdy + alpdx*GammatXxy + alpdy*GammatYxy + alpdz*GammatZxy)*gtXY - 2*(-alpdxdz + alpdx*GammatXxz + alpdy*GammatYxz + alpdz*GammatZxz)*gtXZ - (-alpdydy + alpdx*GammatXyy + alpdy*GammatYyy + alpdz*GammatZyy)*gtYY - 2*(-alpdydz + alpdx*GammatXyz + alpdy*GammatYyz + alpdz*GammatZyz)*gtYZ - (-alpdzdz + alpdx*GammatXzz + alpdy*GammatYzz + alpdz*GammatZzz)*gtZZ - 4*(alpdx*gtXX + alpdy*gtXY + alpdz*gtXZ)*phidx - 4*(alpdy*gtYY + alpdz*gtYZ)*phidy - 4*(alpdy*gtYZ + alpdz*gtZZ)*phidz - 4*alpdx*(gtXY*phidy + gtXZ*phidz)))/3.;
    double const DiDjalpTFxz = alpdxdz - alpdx*GammatXxz - alpdy*GammatYxz - alpdz*GammatZxz - 2*(alpdz*phidx + alpdx*phidz) - (gtxz*(-((-alpdxdx + alpdx*GammatXxx + alpdy*GammatYxx + alpdz*GammatZxx)*gtXX) - 2*(-alpdxdy + alpdx*GammatXxy + alpdy*GammatYxy + alpdz*GammatZxy)*gtXY - 2*(-alpdxdz + alpdx*GammatXxz + alpdy*GammatYxz + alpdz*GammatZxz)*gtXZ - (-alpdydy + alpdx*GammatXyy + alpdy*GammatYyy + alpdz*GammatZyy)*gtYY - 2*(-alpdydz + alpdx*GammatXyz + alpdy*GammatYyz + alpdz*GammatZyz)*gtYZ - (-alpdzdz + alpdx*GammatXzz + alpdy*GammatYzz + alpdz*GammatZzz)*gtZZ - 4*(alpdx*gtXX + alpdy*gtXY + alpdz*gtXZ)*phidx - 4*(alpdy*gtYY + alpdz*gtYZ)*phidy - 4*(alpdy*gtYZ + alpdz*gtZZ)*phidz - 4*alpdx*(gtXY*phidy + gtXZ*phidz)))/3.;
    double const DiDjalpTFyy = alpdydy - alpdx*GammatXyy - alpdy*GammatYyy - alpdz*GammatZyy - 4*alpdy*phidy - (gtyy*(-((-alpdxdx + alpdx*GammatXxx + alpdy*GammatYxx + alpdz*GammatZxx)*gtXX) - 2*(-alpdxdy + alpdx*GammatXxy + alpdy*GammatYxy + alpdz*GammatZxy)*gtXY - 2*(-alpdxdz + alpdx*GammatXxz + alpdy*GammatYxz + alpdz*GammatZxz)*gtXZ - (-alpdydy + alpdx*GammatXyy + alpdy*GammatYyy + alpdz*GammatZyy)*gtYY - 2*(-alpdydz + alpdx*GammatXyz + alpdy*GammatYyz + alpdz*GammatZyz)*gtYZ - (-alpdzdz + alpdx*GammatXzz + alpdy*GammatYzz + alpdz*GammatZzz)*gtZZ - 4*(alpdx*gtXX + alpdy*gtXY + alpdz*gtXZ)*phidx - 4*(alpdy*gtYY + alpdz*gtYZ)*phidy - 4*(alpdy*gtYZ + alpdz*gtZZ)*phidz - 4*alpdx*(gtXY*phidy + gtXZ*phidz)))/3.;
    double const DiDjalpTFyz = alpdydz - alpdx*GammatXyz - alpdy*GammatYyz - alpdz*GammatZyz - 2*(alpdz*phidy + alpdy*phidz) - (gtyz*(-((-alpdxdx + alpdx*GammatXxx + alpdy*GammatYxx + alpdz*GammatZxx)*gtXX) - 2*(-alpdxdy + alpdx*GammatXxy + alpdy*GammatYxy + alpdz*GammatZxy)*gtXY - 2*(-alpdxdz + alpdx*GammatXxz + alpdy*GammatYxz + alpdz*GammatZxz)*gtXZ - (-alpdydy + alpdx*GammatXyy + alpdy*GammatYyy + alpdz*GammatZyy)*gtYY - 2*(-alpdydz + alpdx*GammatXyz + alpdy*GammatYyz + alpdz*GammatZyz)*gtYZ - (-alpdzdz + alpdx*GammatXzz + alpdy*GammatYzz + alpdz*GammatZzz)*gtZZ - 4*(alpdx*gtXX + alpdy*gtXY + alpdz*gtXZ)*phidx - 4*(alpdy*gtYY + alpdz*gtYZ)*phidy - 4*(alpdy*gtYZ + alpdz*gtZZ)*phidz - 4*alpdx*(gtXY*phidy + gtXZ*phidz)))/3.;
    double const DiDjalpTFzz = alpdzdz - alpdx*GammatXzz - alpdy*GammatYzz - alpdz*GammatZzz - 4*alpdz*phidz - (gtzz*(-((-alpdxdx + alpdx*GammatXxx + alpdy*GammatYxx + alpdz*GammatZxx)*gtXX) - 2*(-alpdxdy + alpdx*GammatXxy + alpdy*GammatYxy + alpdz*GammatZxy)*gtXY - 2*(-alpdxdz + alpdx*GammatXxz + alpdy*GammatYxz + alpdz*GammatZxz)*gtXZ - (-alpdydy + alpdx*GammatXyy + alpdy*GammatYyy + alpdz*GammatZyy)*gtYY - 2*(-alpdydz + alpdx*GammatXyz + alpdy*GammatYyz + alpdz*GammatZyz)*gtYZ - (-alpdzdz + alpdx*GammatXzz + alpdy*GammatYzz + alpdz*GammatZzz)*gtZZ - 4*(alpdx*gtXX + alpdy*gtXY + alpdz*gtXZ)*phidx - 4*(alpdy*gtYY + alpdz*gtYZ)*phidy - 4*(alpdy*gtYZ + alpdz*gtZZ)*phidz - 4*alpdx*(gtXY*phidy + gtXZ*phidz)))/3.;


    /* Equation for traceless conformal extrinsic curvature */
    res[ATXXL+0] = Atxxdx*betaX + Atxxdy*betaY + 2*Atxy*betaYdx + Atxxdz*betaZ + 2*Atxz*betaZdx - (2*Atxx*(-3*betaXdx + Dtibetai))/3. - 2*alp*(Atxx*Atxx)*gtXX - 2*alp*(Atxy*Atxy*gtYY + 2*Atxy*Atxz*gtYZ + Atxz*Atxz*gtZZ) + alp*Atxx*(-4*Atxy*gtXY - 4*Atxz*gtXZ + K) + (8*alp*gtxx*pi*S)/3. - DiDjalpTFxx*exp(-4.*phi) + alp*(RddTFxx - 8*pi*Sxx)*exp(-4.*phi);
    res[ATXXL+1] = Atxydx*betaX + Atxx*betaXdy + Atxydy*betaY + Atyy*betaYdx + Atxydz*betaZ + Atyz*betaZdx + Atxz*betaZdy + Atxy*(betaXdx + betaYdy - (2*Dtibetai)/3.) - 2*alp*(Atxy*Atxy*gtXY + Atxy*Atxz*gtXZ + Atxx*(Atxy*gtXX + Atyy*gtXY + Atyz*gtXZ) + Atxy*Atyy*gtYY + Atxz*Atyy*gtYZ + Atxy*Atyz*gtYZ + Atxz*Atyz*gtZZ) + alp*Atxy*K + (8*alp*gtxy*pi*S)/3. - DiDjalpTFxy*exp(-4.*phi) + alp*(RddTFxy - 8*pi*Sxy)*exp(-4.*phi);
    res[ATXXL+2] = Atxzdx*betaX + Atxx*betaXdz + Atxzdy*betaY + Atyz*betaYdx + Atxy*betaYdz + Atxzdz*betaZ + Atzz*betaZdx + Atxz*(betaXdx + betaZdz - (2*Dtibetai)/3.) - 2*alp*Atxx*(Atxz*gtXX + Atyz*gtXY + Atzz*gtXZ) - 2*alp*Atxy*(Atxz*gtXY + Atyz*gtYY + Atzz*gtYZ) + alp*Atxz*(-2*(Atxz*gtXZ + Atyz*gtYZ + Atzz*gtZZ) + K) + (8*alp*gtxz*pi*S)/3. - DiDjalpTFxz*exp(-4.*phi) + alp*(RddTFxz - 8*pi*Sxz)*exp(-4.*phi);
    res[ATXXL+3] = Atyydx*betaX + 2*Atxy*betaXdy + Atyydy*betaY + Atyydz*betaZ + 2*Atyz*betaZdy - (2*Atyy*(-3*betaYdy + Dtibetai))/3. - 2*alp*(Atxy*Atxy*gtXX + 2*Atxy*(Atyy*gtXY + Atyz*gtXZ) + Atyy*Atyy*gtYY + 2*Atyy*Atyz*gtYZ + Atyz*Atyz*gtZZ) + alp*Atyy*K + (8*alp*gtyy*pi*S)/3. - DiDjalpTFyy*exp(-4.*phi) + alp*(RddTFyy - 8*pi*Syy)*exp(-4.*phi);
    res[ATXXL+4] = Atyzdx*betaX + Atxz*betaXdy + Atxy*betaXdz + Atyzdy*betaY + Atyy*betaYdz + Atyzdz*betaZ + Atzz*betaZdy + Atyz*(betaYdy + betaZdz - (2*Dtibetai)/3.) - 2*alp*Atxy*(Atxz*gtXX + Atyz*gtXY + Atzz*gtXZ) - 2*alp*(Atxz*Atyy*gtXY + Atxz*Atyz*gtXZ + Atyy*Atyz*gtYY + Atyz*Atyz*gtYZ + Atyy*Atzz*gtYZ + Atyz*Atzz*gtZZ) + alp*Atyz*K + (8*alp*gtyz*pi*S)/3. - DiDjalpTFyz*exp(-4.*phi) + alp*(RddTFyz - 8*pi*Syz)*exp(-4.*phi);
    res[ATXXL+5] = Atzzdx*betaX + 2*Atxz*betaXdz + Atzzdy*betaY + 2*Atyz*betaYdz + Atzzdz*betaZ - (2*Atzz*(-3*betaZdz + Dtibetai))/3. - 2*alp*(Atxz*Atxz*gtXX + 2*Atxz*(Atyz*gtXY + Atzz*gtXZ) + Atyz*Atyz*gtYY + 2*Atyz*Atzz*gtYZ + Atzz*Atzz*gtZZ) + alp*Atzz*K + (8*alp*gtzz*pi*S)/3. - DiDjalpTFzz*exp(-4.*phi) + alp*(RddTFzz - 8*pi*Szz)*exp(-4.*phi);


    /* Equation for Gamama tilde */
    double const GammatXdt = (-6*alpdx*AtXX - 6*alpdy*AtXY - 6*alpdz*AtXZ + (-betaXdx + 2*(betaYdy + betaZdz))*GammatX - 3*betaXdz*GammatZ + (4*betaXdxdx + betaYdxdy + betaZdxdz)*gtXX + (7*betaXdxdy + betaYdydy + betaZdydz)*gtXY + (7*betaXdxdz + betaYdydz + betaZdzdz)*gtXZ + 3*(betaX*GammatXdx + betaY*GammatXdy + betaZ*GammatXdz - betaXdy*GammatY + betaXdydy*gtYY + 2*betaXdydz*gtYZ + betaXdzdz*gtZZ) + 2*alp*(3*AtYY*GammatXyy + 6*AtYZ*GammatXyz + 3*AtZZ*GammatXzz + 3*AtXX*(GammatXxx + 6*phidx) + 6*AtXY*(GammatXxy + 3*phidy) + 6*AtXZ*(GammatXxz + 3*phidz) - 2*(gtXX*(Kdx + 12*pi*Sx) + gtXY*(Kdy + 12*pi*Sy) + gtXZ*(Kdz + 12*pi*Sz))))/3.; res[GAMMAXL+1] = GammatXdt ;
    double const GammatYdt = (-6*alpdx*AtXY - 6*alpdy*AtYY - 6*alpdz*AtYZ + (2*betaXdx - betaYdy + 2*betaZdz)*GammatY - 3*betaYdz*GammatZ + (betaXdxdx + 7*betaYdxdy + betaZdxdz)*gtXY + (betaXdxdy + 4*betaYdydy + betaZdydz)*gtYY + (betaXdxdz + 7*betaYdydz + betaZdzdz)*gtYZ + 3*(-(betaYdx*GammatX) + betaX*GammatYdx + betaY*GammatYdy + betaZ*GammatYdz + betaYdxdx*gtXX + 2*betaYdxdz*gtXZ + betaYdzdz*gtZZ) + 2*alp*(3*AtXX*GammatYxx + 6*AtXZ*GammatYxz + 3*AtZZ*GammatYzz + 6*AtXY*(GammatYxy + 3*phidx) + 3*AtYY*(GammatYyy + 6*phidy) + 6*AtYZ*(GammatYyz + 3*phidz) - 2*(gtXY*(Kdx + 12*pi*Sx) + gtYY*(Kdy + 12*pi*Sy) + gtYZ*(Kdz + 12*pi*Sz))))/3.; res[GAMMAXL+2] = GammatYdt ;
    double const GammatZdt = (-6*alpdx*AtXZ - 6*alpdy*AtYZ - 6*alpdz*AtZZ + (2*(betaXdx + betaYdy) - betaZdz)*GammatZ + (betaXdxdx + betaYdxdy + 7*betaZdxdz)*gtXZ + 3*(-(betaZdx*GammatX) - betaZdy*GammatY + betaX*GammatZdx + betaY*GammatZdy + betaZ*GammatZdz + betaZdxdx*gtXX + 2*betaZdxdy*gtXY + betaZdydy*gtYY) + (betaXdxdy + betaYdydy + 7*betaZdydz)*gtYZ + (betaXdxdz + betaYdydz + 4*betaZdzdz)*gtZZ + 2*alp*(3*AtXX*GammatZxx + 6*AtXY*GammatZxy + 3*AtYY*GammatZyy + 6*AtXZ*(GammatZxz + 3*phidx) + 6*AtYZ*(GammatZyz + 3*phidy) + 3*AtZZ*(GammatZzz + 6*phidz) - 2*(gtXZ*(Kdx + 12*pi*Sx) + gtYZ*(Kdy + 12*pi*Sy) + gtZZ*(Kdz + 12*pi*Sz))))/3.; res[GAMMAXL+3] = GammatZdt ;


    /* Equation for conformal factor */
    res[PHIL] = (Dtibetai - alp*K)/6. + betaX*phidx + betaY*phidy + betaZ*phidz;


    /* Equation for conformal metric */
    res[GTXXL+0] = -2*alp*Atxx + 2*betaXdx*gtxx - (2*Dtibetai*gtxx)/3. + betaX*gtxxdx + betaY*gtxxdy + betaZ*gtxxdz + 2*betaYdx*gtxy + 2*betaZdx*gtxz;
    res[GTXXL+1] = -2*alp*Atxy + betaXdy*gtxx + (betaXdx + betaYdy - (2*Dtibetai)/3.)*gtxy + betaX*gtxydx + betaY*gtxydy + betaZ*gtxydz + betaZdy*gtxz + betaYdx*gtyy + betaZdx*gtyz;
    res[GTXXL+2] = -2*alp*Atxz + betaXdz*gtxx + betaYdz*gtxy + (betaXdx + betaZdz - (2*Dtibetai)/3.)*gtxz + betaX*gtxzdx + betaY*gtxzdy + betaZ*gtxzdz + betaYdx*gtyz + betaZdx*gtzz;
    res[GTXXL+3] = -2*alp*Atyy + 2*betaXdy*gtxy + 2*betaYdy*gtyy - (2*Dtibetai*gtyy)/3. + betaX*gtyydx + betaY*gtyydy + betaZ*gtyydz + 2*betaZdy*gtyz;
    res[GTXXL+4] = -2*alp*Atyz + betaXdz*gtxy + betaXdy*gtxz + betaYdz*gtyy + (betaYdy + betaZdz - (2*Dtibetai)/3.)*gtyz + betaX*gtyzdx + betaY*gtyzdy + betaZ*gtyzdz + betaZdy*gtzz;
    res[GTXXL+5] = -2*alp*Atzz + 2*betaXdz*gtxz + 2*betaYdz*gtyz + 2*betaZdz*gtzz - (2*Dtibetai*gtzz)/3. + betaX*gtzzdx + betaY*gtzzdy + betaZ*gtzzdz;


    /* Equation for trace of extrinsic curvature */
    res[KL] = betaX*Kdx + betaY*Kdy + betaZ*Kdz + alp*(Atxx*AtXX + 2*Atxy*AtXY + 2*Atxz*AtXZ + Atyy*AtYY + 2*Atyz*AtYZ + Atzz*AtZZ + (K*K)/3. + 4*pi*(Edens + S)) + (-(alpdxdx*gtXX) + alpdy*GammatYxx*gtXX + alpdz*GammatZxx*gtXX - 2*alpdxdy*gtXY + 2*alpdy*GammatYxy*gtXY + 2*alpdz*GammatZxy*gtXY - 2*alpdxdz*gtXZ + 2*alpdy*GammatYxz*gtXZ + 2*alpdz*GammatZxz*gtXZ - alpdydy*gtYY + alpdy*GammatYyy*gtYY + alpdz*GammatZyy*gtYY - 2*alpdydz*gtYZ + 2*alpdy*GammatYyz*gtYZ + 2*alpdz*GammatZyz*gtYZ - alpdzdz*gtZZ + alpdy*GammatYzz*gtZZ + alpdz*GammatZzz*gtZZ - 2*alpdy*gtXY*phidx - 2*alpdz*gtXZ*phidx - 2*alpdy*gtYY*phidy - 2*alpdz*gtYZ*phidy - 2*alpdy*gtYZ*phidz - 2*alpdz*gtZZ*phidz + alpdx*(GammatXxx*gtXX + 2*GammatXxy*gtXY + 2*GammatXxz*gtXZ + GammatXyy*gtYY + 2*GammatXyz*gtYZ + GammatXzz*gtZZ - 2*gtXX*phidx - 2*gtXY*phidy - 2*gtXZ*phidz))*exp(-4.*phi);


    /* 1 + log slicing condition */
    res[ALPL] = alpdx*betaX + alpdy*betaY + alpdz*betaZ - 2*alp*K;

    /* Gamma driver */
    res[BETAXL+0] = BX*k1;
    res[BETAXL+1] = BY*k1;
    res[BETAXL+2] = BZ*k1;
    res[BXL+0] = -(BX*eta) + GammatXdt;
    res[BXL+1] = -(BY*eta) + GammatYdt;
    res[BXL+2] = -(BZ*eta) + GammatZdt;

    /* All done! */
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
    double const Edens = (Ttt - 2*betaZ*Ttz + betaX*betaX*Txx + 2*betaX*(-Ttx + betaY*Txy + betaZ*Txz) + betaY*(-2*Tty + betaY*Tyy + 2*betaZ*Tyz) + betaZ*betaZ*Tzz)/(alp*alp);
    double const Sx = Ttx/alp - (betaX*Txx)/alp - (betaY*Txy)/alp - (betaZ*Txz)/alp;
    double const Sy = Tty/alp - (betaX*Txy)/alp - (betaY*Tyy)/alp - (betaZ*Tyz)/alp;
    double const Sz = Ttz/alp - (betaX*Txz)/alp - (betaY*Tyz)/alp - (betaZ*Tzz)/alp;



    std::array<double,4> res;
    int ww=0;

    /* Hamiltonian constraint */
    res[ww] = -(Atxx*AtXX) - 2*Atxy*AtXY - 2*Atxz*AtXZ - Atyy*AtYY - 2*Atyz*AtYZ - Atzz*AtZZ + (2*(K*K))/3. - 2*Edens*M_PI + ((GammatXdx + GammatYdy + GammatZdz + (-2*(GammatXxx*GammatXxx)*gtXX - 2*(GammatYxy*GammatYxy)*gtXX - 4*GammatXxz*GammatZxx*gtXX - 4*GammatYxz*GammatZxy*gtXX - 2*(GammatZxz*GammatZxz)*gtXX - gtXX*gtxxdx*gtXXdx - 4*GammatXyy*GammatYxx*gtXY - 4*GammatYxy*GammatYyy*gtXY - 4*GammatXyz*GammatZxx*gtXY - 4*GammatXxz*GammatZxy*gtXY - 4*GammatYyz*GammatZxy*gtXY - 4*GammatYxz*GammatZyy*gtXY - 4*GammatZxz*GammatZyz*gtXY + gtXXdx*gtxxdy*gtXY - gtxxdx*gtXXdy*gtXY - 2*gtXXdx*gtXY*gtxydx - 2*gtXX*gtxxdy*gtXYdx - 2*gtxxdy*gtXY*gtXYdy - 4*GammatXyz*GammatYxx*gtXZ - 4*GammatYxy*GammatYyz*gtXZ - 4*GammatXzz*GammatZxx*gtXZ - 4*GammatYzz*GammatZxy*gtXZ - 4*GammatXxz*GammatZxz*gtXZ - 4*GammatYxz*GammatZyz*gtXZ - 4*GammatZxz*GammatZzz*gtXZ + gtXXdx*gtxxdz*gtXZ - gtxxdx*gtXXdz*gtXZ + 2*gtXYdx*gtxydz*gtXZ - 2*gtxxdy*gtXYdz*gtXZ - 4*GammatXxx*(GammatXxy*gtXY + GammatXxz*gtXZ) - 2*gtXXdx*gtXZ*gtxzdx - 2*gtXX*gtxxdz*gtXZdx - 2*gtXY*gtxydz*gtXZdx - 2*gtXYdx*gtXZ*gtxzdy + 2*gtXY*gtXZdx*gtxzdy - 2*gtxxdz*gtXY*gtXZdy - 2*gtxxdz*gtXZ*gtXZdz - 2*(GammatXxy*GammatXxy)*gtYY - 4*GammatXyy*GammatYxy*gtYY - 2*(GammatYyy*GammatYyy)*gtYY - 4*GammatXyz*GammatZxy*gtYY - 4*GammatYyz*GammatZyy*gtYY - 2*(GammatZyz*GammatZyz)*gtYY + gtxxdy*gtXXdy*gtYY - 2*gtXXdy*gtxydx*gtYY - 2*gtxydz*gtXZdy*gtYY + 2*gtxzdy*gtXZdy*gtYY - 2*gtXY*gtXYdx*gtyydx - 2*gtXYdy*gtYY*gtyydx - 2*gtXX*gtxydy*gtYYdx + gtXX*gtyydx*gtYYdx - gtXY*gtYYdx*gtyydy - 2*gtXY*gtxydy*gtYYdy + gtXY*gtyydx*gtYYdy - gtYY*gtyydy*gtYYdy + gtXZ*gtYYdx*gtyydz - 2*gtxydy*gtXZ*gtYYdz + gtXZ*gtyydx*gtYYdz - 4*GammatXyz*GammatYxy*gtYZ - 4*GammatXyy*GammatYxz*gtYZ - 4*GammatYyy*GammatYyz*gtYZ - 4*GammatXzz*GammatZxy*gtYZ - 4*GammatXyz*GammatZxz*gtYZ - 4*GammatYzz*GammatZyy*gtYZ - 4*GammatYyz*GammatZyz*gtYZ - 4*GammatZyz*GammatZzz*gtYZ + gtXXdy*gtxxdz*gtYZ + gtxxdy*gtXXdz*gtYZ - 2*gtXXdz*gtxydx*gtYZ + 2*gtXYdy*gtxydz*gtYZ - 2*gtXXdy*gtxzdx*gtYZ - 2*gtXYdy*gtxzdy*gtYZ - 2*gtxydz*gtXZdz*gtYZ + 2*gtxzdy*gtXZdz*gtYZ - 2*gtXYdz*gtyydx*gtYZ + gtYYdy*gtyydz*gtYZ - gtyydy*gtYYdz*gtYZ - 4*GammatXxy*(GammatYxx*gtXX + GammatYxy*gtXY + GammatYxz*gtXZ + GammatXxz*gtYZ) - 2*gtXYdx*gtXZ*gtyzdx - 2*gtXY*gtXZdx*gtyzdx - 2*gtXZdy*gtYY*gtyzdx - 2*gtXYdy*gtYZ*gtyzdx - 2*gtXZdz*gtYZ*gtyzdx - 2*gtXX*gtxydz*gtYZdx - 2*gtXX*gtxzdy*gtYZdx - 2*gtXY*gtyydz*gtYZdx + 2*gtXX*gtyzdx*gtYZdx - 2*gtXZ*gtYYdx*gtyzdy - 2*gtYYdy*gtYZ*gtyzdy - 2*gtXY*gtxydz*gtYZdy - 2*gtXY*gtxzdy*gtYZdy - 2*gtYY*gtyydz*gtYZdy + 2*gtXY*gtyzdx*gtYZdy - 2*gtxydz*gtXZ*gtYZdz - 2*gtXZ*gtxzdy*gtYZdz - 2*gtyydz*gtYZ*gtYZdz + 2*gtXZ*gtyzdx*gtYZdz - 2*(GammatXxz*GammatXxz)*gtZZ - 4*GammatXyz*GammatYxz*gtZZ - 2*(GammatYyz*GammatYyz)*gtZZ - 4*GammatXzz*GammatZxz*gtZZ - 4*GammatYzz*GammatZyz*gtZZ - 2*(GammatZzz*GammatZzz)*gtZZ + gtxxdz*gtXXdz*gtZZ + 2*gtxydz*gtXYdz*gtZZ - 2*gtXXdz*gtxzdx*gtZZ - 2*gtXYdz*gtxzdy*gtZZ + gtyydz*gtYYdz*gtZZ - 2*gtXYdz*gtyzdx*gtZZ - 2*gtYYdz*gtyzdy*gtZZ - 2*gtXZ*gtXZdx*gtzzdx - 2*gtXZdy*gtYZ*gtzzdx - 2*gtXZdz*gtZZ*gtzzdx - 2*gtXZ*gtYZdx*gtzzdy - 2*gtYZ*gtYZdy*gtzzdy - 2*gtYZdz*gtZZ*gtzzdy - 2*gtXY*gtxzdz*gtZZdy - 2*gtYY*gtyzdz*gtZZdy + gtXY*gtzzdx*gtZZdy + gtYY*gtzzdy*gtZZdy - gtYZ*gtZZdy*gtzzdz + gtZZdx*(gtXX*(-2*gtxzdz + gtzzdx) + gtXY*(-2*gtyzdz + gtzzdy) - gtXZ*gtzzdz) - 2*gtXZ*gtxzdz*gtZZdz - 2*gtYZ*gtyzdz*gtZZdz + gtXZ*gtzzdx*gtZZdz + gtYZ*gtzzdy*gtZZdz - gtZZ*gtzzdz*gtZZdz)/2.)/8. - 8*(gtXX*(phidx*phidx) + 2*gtXY*phidx*phidy + gtYY*(phidy*phidy) + 2*gtXZ*phidx*phidz + 2*gtYZ*phidy*phidz + gtZZ*(phidz*phidz)) + 8*(gtXX*(GammatXxx*phidx - phidxdx + GammatYxx*phidy + GammatZxx*phidz) + 2*gtXY*(GammatXxy*phidx - phidxdy + GammatYxy*phidy + GammatZxy*phidz) + 2*gtXZ*(GammatXxz*phidx - phidxdz + GammatYxz*phidy + GammatZxz*phidz) + gtYY*(GammatXyy*phidx + GammatYyy*phidy - phidydy + GammatZyy*phidz) + 2*gtYZ*(GammatXyz*phidx + GammatYyz*phidy - phidydz + GammatZyz*phidz) + gtZZ*(GammatXzz*phidx + GammatYzz*phidy + GammatZzz*phidz - phidzdz)))*exp(-4.*phi); ww++;

    /* Momentum constraints */
    res[ww] = (Atxxdx - 2*(Atxx*GammatXxx + Atxy*GammatYxx + Atxz*GammatZxx))*gtXX - 2*(-Atxydx + Atxx*GammatXxy + Atyy*GammatYxx + Atxy*(GammatXxx + GammatYxy) + Atyz*GammatZxx + Atxz*GammatZxy)*gtXY - 2*(-Atxzdx + Atxx*GammatXxz + Atyz*GammatYxx + Atxy*GammatYxz + Atzz*GammatZxx + Atxz*(GammatXxx + GammatZxz))*gtXZ + (Atyydx - 2*(Atxy*GammatXxy + Atyy*GammatYxy + Atyz*GammatZxy))*gtYY - 2*(-Atyzdx + Atxz*GammatXxy + Atxy*GammatXxz + Atyy*GammatYxz + Atzz*GammatZxy + Atyz*(GammatYxy + GammatZxz))*gtYZ + (Atzzdx - 2*(Atxz*GammatXxz + Atyz*GammatYxz + Atzz*GammatZxz))*gtZZ - (2*Kdx)/3. + 6*(Atxx*(gtXX*phidx + gtXY*phidy + gtXZ*phidz) + Atxy*(gtXY*phidx + gtYY*phidy + gtYZ*phidz) + Atxz*(gtXZ*phidx + gtYZ*phidy + gtZZ*phidz)); ww++;
    res[ww] = (Atxxdy - 2*(Atxx*GammatXxy + Atxy*GammatYxy + Atxz*GammatZxy))*gtXX - 2*(-Atxydy + Atxx*GammatXyy + Atyy*GammatYxy + Atxy*(GammatXxy + GammatYyy) + Atyz*GammatZxy + Atxz*GammatZyy)*gtXY - 2*(-Atxzdy + Atxx*GammatXyz + Atyz*GammatYxy + Atxy*GammatYyz + Atzz*GammatZxy + Atxz*(GammatXxy + GammatZyz))*gtXZ + (Atyydy - 2*(Atxy*GammatXyy + Atyy*GammatYyy + Atyz*GammatZyy))*gtYY - 2*(-Atyzdy + Atxz*GammatXyy + Atxy*GammatXyz + Atyy*GammatYyz + Atzz*GammatZyy + Atyz*(GammatYyy + GammatZyz))*gtYZ + (Atzzdy - 2*(Atxz*GammatXyz + Atyz*GammatYyz + Atzz*GammatZyz))*gtZZ - (2*Kdy)/3. + 6*(Atxy*(gtXX*phidx + gtXY*phidy + gtXZ*phidz) + Atyy*(gtXY*phidx + gtYY*phidy + gtYZ*phidz) + Atyz*(gtXZ*phidx + gtYZ*phidy + gtZZ*phidz)); ww++;
    res[ww] = (Atxxdz - 2*(Atxx*GammatXxz + Atxy*GammatYxz + Atxz*GammatZxz))*gtXX - 2*(-Atxydz + Atxx*GammatXyz + Atyy*GammatYxz + Atxy*(GammatXxz + GammatYyz) + Atyz*GammatZxz + Atxz*GammatZyz)*gtXY - 2*(-Atxzdz + Atxx*GammatXzz + Atyz*GammatYxz + Atxy*GammatYzz + Atzz*GammatZxz + Atxz*(GammatXxz + GammatZzz))*gtXZ + (Atyydz - 2*(Atxy*GammatXyz + Atyy*GammatYyz + Atyz*GammatZyz))*gtYY - 2*(-Atyzdz + Atxz*GammatXyz + Atxy*GammatXzz + Atyy*GammatYzz + Atzz*GammatZyz + Atyz*(GammatYyz + GammatZzz))*gtYZ + (Atzzdz - 2*(Atxz*GammatXzz + Atyz*GammatYzz + Atzz*GammatZzz))*gtZZ - (2*Kdz)/3. + 6*(Atxz*(gtXX*phidx + gtXY*phidy + gtXZ*phidz) + Atyz*(gtXY*phidx + gtYY*phidy + gtYZ*phidz) + Atzz*(gtXZ*phidx + gtYZ*phidy + gtZZ*phidz)); ww++;



    /* All done! */
    return std::move(res);
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
                , std::array<double,GRACE_NSPACEDIM> const& )

INSTANTIATE_TEMPLATE(2) ; 
INSTANTIATE_TEMPLATE(4) ; 
#undef INSTANTIATE_TEMPLATE
}