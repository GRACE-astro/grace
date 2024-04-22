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
 * \ingroup amr
 * @tparam InterpT    Type of interpolator
 * @tparam StateViewT Type of state vector 
 * @tparam CoordViewT Type of coordinate spacing vector
 */
template< typename InterpT      // Type of interpolator
        , typename StateViewT   // Type of state vector 
        , typename CoordViewT   // Type of coordinate spacing vector
        , typename VolViewT >   // Type of cell volume
struct prolongator_t {
    long VEC( nx, ny, nz )   ; //!< Quadrant extents (unchanged)
    int ngz                  ; //!< Number of ghost cells 
    StateViewT state         ; //!< Old state
    CoordViewT dx_parent     ; //!< Old dx
    CoordViewT dx_child      ; //!< Old dx 
    VolViewT   vol_parent    ; //!< Old cell volumes
    VolViewT   vol_child     ; //!< New cell volumes
    CoordViewT x_parent      ; //!< Old quadrant coordinates
    CoordViewT x_child       ; //!< New quadrant coordinates


    /**
     * @brief Prolongate requested variable at the requested
     *        point in the child quadrant
     * \ingroup amr
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
                , int const& iq_parent
                , int const& iq_child
                , int const& ivar 
                , int const& ichild) const 
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
        return InterpT::interpolate(
              VEC(i+ngz,j+ngz,k+ngz)
            , VEC(i0+ngz,j0+ngz,k0+ngz)
            , iq_child, iq_parent, ngz, ivar
            , x_child
            , x_parent
            , dx_child
            , dx_parent 
            , state 
            , vol_child
            , vol_parent
        ) ; 
    }
}  ; 


}} /* namespace thunder::amr */ 

#endif 