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

/**
 * @brief Prolongation functor
 * 
 * @tparam InterpT    Type of interpolator
 * @tparam StateViewT Type of state vector 
 * @tparam CoordViewT Type of coordinate spacing vector
 */
template< typename InterpT      // Type of interpolator
        , typename StateViewT   // Type of state vector 
        , typename CoordViewT > // Type of coordinate spacing vector
struct prolongator_t {
    long VEC( nx, ny, nz )   ; //!< Quadrant extents (unchanged)
    int ngz                  ; //!< Number of ghost cells 
    StateViewT state         ; //!< Old state
    CoordViewT idx_parent    ; //!< Old idx 

    /**
     * @brief Prolongate requested variable at the requested
     *        point in the child quadrant
     * 
     * @param i       x-index of point in child quadrant
     * @param j       y-index of point in child quadrant 
     * @param z       z-index of point in child quadrant (3D only)
     * @param iq      (Parent) quadrant index
     * @param ivar    Variable index 
     * @param ichild  Child quadrant index in z-ordering
     * @return double Prolongated variable at requested point 
     *                in the child quadrant.
     */
    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    operator() ( VEC( int const& i
                    , int const& j 
                    , int const& k )
                , int const& iq 
                , int const& ivar 
                , int const& ichild ) const 
    {
        /* 
        * First we need to find the index 
        * in the parent quadrant closest 
        * to the requested index in the child
        * quadrant. 
        */ 
        EXPR( 
        int const iquad_x = ichild % 2 ;, 
        int const iquad_y = static_cast<int>(Kokkos::floor(ichild/2))%2;,
        int const iquad_z = Kokkos::floor(Kokkos::floor(ichild/2)/2);
        )
        EXPR(
        int const i0 = 
              Kokkos::floor((iquad_x * nx + i ) / 2) ;,

        int const j0 = 
              Kokkos::floor((iquad_y * ny + j ) / 2) ;,

        int const k0 = 
              Kokkos::floor((iquad_z * nz + k ) / 2) ; 
        )
        /* 
        * Then we compute the coordinates
        * of the requested point in the child's 
        * coordinates 
        */ 
        EXPR(
            const double x0 = (iquad_x*nx + i + 0.5) / idx_parent(iq,0) / 2.;,
            const double y0 = (iquad_y*ny + j + 0.5) / idx_parent(iq,1) / 2.;,
            const double z0 = (iquad_z*nz + k + 0.5) / idx_parent(iq,2) / 2.;
        ) 
        /* 
        *  Then we construct a stencil of 
        *  the appropriate size for the 
        *  interpolator.
        */ 
        size_t constexpr stencil = InterpT::stencil_size              ;
        size_t constexpr npoints = EXPR( stencil, *stencil, *stencil) ; 
        double x_interp[ THUNDER_NSPACEDIM * npoints ] ;
        int x_param[ THUNDER_NSPACEDIM * npoints ]     ; 
        double y_interp[ npoints ]                     ;
        InterpT::get_parametric_coordinates(x_param) ; 
        int s_min = 0; 
        for( size_t istencil=0; istencil<npoints; ++istencil)
        {   
            EXPR(
            x_interp[ THUNDER_NSPACEDIM*istencil + 0UL ] = 
                (i0+0.5 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 0UL]) / idx_parent(iq,0);,
            x_interp[ THUNDER_NSPACEDIM*istencil + 1UL ] = 
                (j0+0.5 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 1UL]) / idx_parent(iq,1);,
            x_interp[ THUNDER_NSPACEDIM*istencil + 2UL ] = 
                (k0+0.5 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 2UL]) / idx_parent(iq,2); 
            )
            y_interp[ istencil ] = state(VEC(
                ngz + i0 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 0UL],
                ngz + j0 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 1UL],
                ngz + k0 - s_min + x_param[THUNDER_NSPACEDIM*istencil + 2UL]
            ),iq,ivar) ; 
        }
        InterpT interpolator(x_interp,y_interp) ; 
        /* 
        * Finally we call the interpolator 
        * to obtain the desired prolongated 
        * value. 
        */ 
        return interpolator.interpolate(VEC(x0,y0,z0)) ; 
    }
}  ; 


}} /* namespace thunder::amr */ 

#endif 