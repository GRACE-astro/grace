/**
 * @file prolongation_kernels.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
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

#ifndef THUNDER_AMR_PROLONGATION_KERNELS_TPP
#define THUNDER_AMR_PROLONGATION_KERNELS_TPP

#include <thunder_config.h>

namespace thunder { namespace amr { 

template< typename StateViewT
        , typename CoordViewT >  
struct prolongator_t {
    size_t VEC( nx, ny, nz ) ; 
    StateViewT state ; 
    CoordViewT idx   ; 

    template< typename InterpT >
    double THUNDER_HOST_DEVICE 
    operator() ( VEC( int const& i
                    , int const& j 
                    , int const& k )
                , int const& iq 
                , int const& ivar 
                , int const& ichild )
    {
        /* 
        * First we compute the coordinates
        * of the requested point in the parent's 
        * coordinates 
        */ 
        #ifdef THUNDER_3D 
        double const x0 = 
              ( ichild % 2 ) / ( idx(iq,0) * 2.0 )
            + ( i + 0.5 ) / idx(iq,0) ; 

        double const y0 = 
              (( ichild / 2 ) % 2) / ( idx(iq,1) * 2.0 )
            + ( j + 0.5 ) / idx(iq,1) ; 

        double const z0 = 
              (( ichild / 2 ) / 2) / ( idx(iq,2) * 2.0 )
            + ( k + 0.5 ) / idx(iq,2) ; 
        #else 
        double const x0 = 
              ( ichild % 2 ) / ( idx(iq,0) * 2.0 )
            + ( i + 0.5 ) / idx(iq,0) ; 

        double const y0 = 
              ( ichild / 2 ) / ( idx(iq,1) * 2.0 )
            + ( j + 0.5 ) / idx(iq,1) ;
        #endif 
        /* 
        * Then we need to find the index 
        * in the parent quadrant closest 
        * to the requested index in the child
        * quadrant. 
        */ 
        #ifdef THUNDER_3D 
        size_t const i0 = 
              ( ( ichild % 2 ) * nx + i ) / 2 ;

        double const j0 = 
              ((( ichild / 2 ) % 2)* ny + j ) / 2 ;

        double const k0 = 
              ((( ichild / 2 ) / 2)*nz + k  ) / 2 ;
        #else 
        size_t const i0 = 
              ( ( ichild % 2 ) * nx + i ) / 2 ;

        double const j0 = 
              (( ichild / 2 )* ny + j ) / 2 ;
        #endif


        /* 
        *  Then we construct a stencil of 
        *  the appropriate size for the 
        *  interpolator.
        */ 
        size_t const stencil = InterpT::stencil_size ; 
        double x_interp[ THUNDER_NSPACEDIM * EXPR( stencil, *stencil, *stencil)] ; 
        double y_interp[ EXPR( stencil, *stencil, *stencil) ] ;
        int x_param[ THUNDER_NSPACEDIM * EXPR( stencil, *stencil, *stencil) ]
        InterpT::get_parametric_coordinates(x_param) ; 
        int s_min = stencil/2 - 1; 
        for( size_t istencil=0; istencil<EXPR( stencil, *stencil, *stencil); ++istencil)
        {   
            EXPR(
            x_interp[ THUNDER_NSPACEDIM*istencil + 0UL ] = 
                (i0 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 0UL]) / idx(iq,0);,
            x_interp[ THUNDER_NSPACEDIM*istencil + 1UL ] = 
                (j0 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 1UL]) / idx(iq,1);,
            x_interp[ THUNDER_NSPACEDIM*istencil + 2UL ] = 
                (k0 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 2UL]) / idx(iq,2); 
            )
            y_interp[ istencil ] = state(VEC(
                i0 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 0UL],
                j0 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 1UL]
                k0 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 2UL]
            ),iq,ivar) ; 
        }
        InterpT interpolator(x_interp,y_interp) ; 
        /* 
        * Finally we call the interpolator 
        * to obtain the desired prolongated 
        * value. 
        */ 
        return interpolator.interpolate(VEC(x0,y0,z0))
    }
}  ; 


}} /* namespace thunder::amr */ 

#endif 