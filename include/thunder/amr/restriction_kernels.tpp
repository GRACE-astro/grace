/**
 * @file restriction_kernels.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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

#ifndef THUNDER_AMR_RESTRICTION_KERNELS_TPP
#define THUNDER_AMR_RESTRICTION_KERNELS_TPP

#include <thunder_config.h>

namespace thunder { namespace amr { 

template< typename StateViewT 
        , typename VolViewT >  
struct restrictor_t {
    size_t VEC( nx, ny, nz ) ; //!< Quadrant extents (unchanged)
    int ngz                  ; //!< Number of ghost cells
    StateViewT state         ; //!< Old state 
    VolViewT   vol_child     ; //!< Child volume
    VolViewT   vol_parent    ; //!< Parent volume

    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    operator() ( VEC( int const& i
                    , int const& j 
                    , int const& k )
                , int* iq
                , int iq_parent 
                , int const& ivar ) const 
    {
        /*****************************************/
        /* We are assuming cell centered data,   */
        /* which implies we need to average over */
        /* P4EST_CHILDREN cells of the children  */
        /* quadrants.                            */
        /*****************************************/

        /* Indices in child quadrant            */ 
        EXPR(
            const int i0 = (2*i) % nx + ngz;,
            const int j0 = (2*j) % ny + ngz;,
            const int k0 = (2*k) % nz + ngz;
        ) 
        /* Index of child containing point */ 
        int const iq_child =
            iq[ EXPR(
                  (2*i)/nx
              , + (2*j)/ny * 2 
              , + (2*k)/nz * 2 * 2 )
              ];
        
        return (EXPR(
              state(VEC(i0,j0,k0),ivar,iq_child)*vol_child(VEC(i0,j0,k0),iq_child)
            + state(VEC(i0+1,j0,k0),ivar,iq_child)*vol_child(VEC(i0+1,j0,k0),iq_child),
            + state(VEC(i0,j0+1,k0),ivar,iq_child)*vol_child(VEC(i0,j0+1,k0),iq_child)
            + state(VEC(i0+1,j0+1,k0),ivar,iq_child)*vol_child(VEC(i0+1,j0+1,k0),iq_child),
            + state(VEC(i0,j0,k0+1),ivar,iq_child)*vol_child(VEC(i0,j0,k0+1),iq_child)
            + state(VEC(i0,j0+1,k0+1),ivar,iq_child)*vol_child(VEC(i0,j0+1,k0+1),iq_child)
            + state(VEC(i0+1,j0,k0+1),ivar,iq_child)*vol_child(VEC(i0+1,j0,k0+1),iq_child)
            + state(VEC(i0+1,j0+1,k0+1),ivar,iq_child)*vol_child(VEC(i0+1,j0+1,k0+1),iq_child)
        )) / vol_parent(VEC(i+ngz,j+ngz,k+ngz),iq_parent) ; 

    }
} ;

} } /* namespace thunder::amr */ 

#endif /* THUNDER_AMR_RESTRICTION_KERNELS_TPP */ 