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

template< typename AvgT 
        , typename StateViewT
        , typename CoordViewT >  
struct restrictor_t {
    size_t VEC( nx, ny, nz ) ; //!< Quadrant extents (unchanged)
    int ngz                  ; //!< Number of ghost cells
    StateViewT state         ; //!< Old state 
    CoordViewT idx_child     ; //!< Old idx 

    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    operator() ( VEC( int const& i
                    , int const& j 
                    , int const& k )
                , int* iq
                , int const& ivar ) const 
    {
        /* We are assuming cell centered data,   */
        /* which implies we need to average over */
        /* P4EST_CHILDREN cells of the children  */
        /* quadrants.                            */

        /* Indices in parent quadrant            */ 
        EXPR(
            const int i0 = (2*i) % nx ;,
            const int j0 = (2*j) % ny ;,
            const int k0 = (2*k) % nz ;
        ) 
        /* Index of child containing point */ 
        int const iq_child =
            iq[ EXPR(
                  (2*i)/nx
              , + (2*j)/ny * 2 
              , + (2*k)/nz * 2 * 2 )
              ];
        /* Coordinates in parent quadrant        */ 
        EXPR(
            const double x0 = (i + 0.5) / idx_child(iq_child,0) * 2. ;,
            const double y0 = (j + 0.5) / idx_child(iq_child,1) * 2. ;,
            const double z0 = (k + 0.5) / idx_child(iq_child,2) * 2. ;
        ) 

        size_t constexpr stencil = AvgT::stencil_size ;
        size_t constexpr npoints = EXPR(stencil,*stencil,*stencil) ; 
        int x_param[THUNDER_NSPACEDIM*npoints] ; 
        AvgT::get_parametric_coordinates(x_param) ; 
        double x[THUNDER_NSPACEDIM*npoints] ;
        double y[npoints]; 
        int s_min = Kokkos::floor(stencil/2) - 1;
        for(size_t is=0; is<npoints;++is)
        {
            EXPR(
            x[THUNDER_NSPACEDIM*is + 0UL] 
                = (i0+0.5 - s_min + x_param[THUNDER_NSPACEDIM*is + 0UL]) / idx_child(iq_child,0) ;,
            x[THUNDER_NSPACEDIM*is + 1UL] 
                = (j0+0.5 - s_min + x_param[THUNDER_NSPACEDIM*is + 1UL]) / idx_child(iq_child,1) ;, 
            x[THUNDER_NSPACEDIM*is + 2UL] 
                = (k0+0.5 - s_min + x_param[THUNDER_NSPACEDIM*is + 2UL]) / idx_child(iq_child,2) ; 
            )
            y[is] = state( VEC(
                ngz + i0 - s_min + x_param[THUNDER_NSPACEDIM*is + 0UL],
                ngz + j0 - s_min + x_param[THUNDER_NSPACEDIM*is + 1UL],
                ngz + k0 - s_min + x_param[THUNDER_NSPACEDIM*is + 2UL]
                ),ivar,iq_child) ;

        }
        AvgT averager(x,y) ; 
        return averager.interpolate(VEC(x0,y0,z0)) ; 
    }
} ;

} } /* namespace thunder::amr */ 

#endif /* THUNDER_AMR_RESTRICTION_KERNELS_TPP */ 