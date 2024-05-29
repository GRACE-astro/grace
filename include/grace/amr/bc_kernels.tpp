/**
 * @file bc_kernels.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-23
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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
#ifndef GRACE_AMR_BC_KERNELS_TPP 
#define GRACE_AMR_BC_KERNELS_TPP

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/data_structures/macros.hh>

namespace grace { namespace amr {

struct outgoing_bc_t 
{
    template< typename ViewT>
    static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    void apply(ViewT& u, int ngz, int n0, VECD(int j, int k), int face, int64_t iq)
    {
      EXPR(
      int const I0 = EXPRD((face==0) * ngz 
                         + (face==1) * (n0+ngz-1)
                         + (face/2==1) * (j+ngz), 
                         + (face/2==2) * (j+ngz))  ;,
      int const J0 = EXPRD((face==2) * ngz 
                         + (face==3) * (n0+ngz-1)
                         + (face/2==0) * (j+ngz), 
                         + (face/2==2) * (k+ngz))  ;,
      int const K0 = EXPRD((face==4) * ngz 
                         + (face==5) * (n0+ngz-1)
                         + (face/2==0) * (k+ngz), 
                         + (face/2==1) * (k+ngz))  ;)
      for(int ig=0; ig<ngz; ++ig)
      {
            EXPR(
            int I = EXPRD((face==0) * ig 
                  + (face==1) * (n0+ngz+ig)
                  + (face/2==1) * (j+ngz), 
                  + (face/2==2) * (j+ngz))  ;,
            int J = EXPRD((face==2) * ig 
                  + (face==3) * (n0+ngz+ig)
                  + (face/2==0) * (j+ngz), 
                  + (face/2==2) * (k+ngz))  ;,
            int K = EXPRD((face==4) * ig 
                  + (face==5) * (n0+ngz+ig)
                  + (face/2==0) * (k+ngz), 
                  + (face/2==1) * (k+ngz))  ;)
            u(VEC(I,J,K),iq) = u(VEC(I0,J0,K0),iq); 
      }       
    }
} ;


}}

#endif /* GRACE_AMR_BC_KERNELS_TPP */