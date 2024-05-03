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
#include <thunder/utils/inline.h>
#include <thunder/utils/device.h> 
#include <thunder/utils/math.hh>
#include <thunder/data_structures/macros.hh> 

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
        int const iquad_y = static_cast<int>(math::floor_int(ichild/2))%2;,
        int const iquad_z = math::floor_int(math::floor_int(ichild/2)/2);
        )
        EXPR(
        int const i0 = 
              math::floor_int((iquad_x * nx + i ) / 2) ;,

        int const j0 = 
              math::floor_int((iquad_y * ny + j ) / 2) ;,

        int const k0 = 
              math::floor_int((iquad_z * nz + k ) / 2) ; 
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
struct ghostzone_prolongator_t {
    long VEC( nx, ny, nz )   ; //!< Quadrant extents (unchanged)
    int ngz                  ; //!< Number of ghost cells 
    StateViewT state         ; //!< State array 
    StateViewT halo          ; //!< Halo state array
    VolViewT   vol           ; //!< Old cell volumes
    VolViewT   vol_halo      ; //!< New cell volumes
    CoordViewT x             ; //!< Old quadrant coordinates
    CoordViewT x_halo        ; //!< New quadrant coordinates


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
    operator() ( VEC( int const& ig
                    , int const& i1 
                    , int const& i2 )
                , VEC( int const& i_f
                     , int const& j_f  
                     , int const& k_f )
                , VEC( double const& dx_fine
                     , double const& dy_fine
                     , double const& dz_fine )
                , int64_t const& iq_coarse
                , int64_t const& iq_fine
                , int const& ivar 
                , int const& ichild
                , int polarity 
                , int which_face_coarse
                , int which_face_fine 
                , bool is_halo_coarse 
                , bool is_halo_fine ) const 
    {
      EXPRD(
      int64_t n1 = (which_face_coarse/2==0) * ny + ((which_face_coarse/2==1) * nx) + ((which_face_coarse/2==2) * nx) ;,
      int64_t n2 = (which_face_coarse/2==0) * nz + ((which_face_coarse/2==1) * nz) + ((which_face_coarse/2==2) * ny) ;
      )

      auto& u = is_halo_coarse ? halo : state ; 
      
      auto x_coarse  = is_halo_coarse ? x_halo : x ; 
      auto vol_coarse = is_halo_coarse ? vol_halo : vol ;  
      
      auto x_fine  = is_halo_fine ? x_halo : x ; 
      auto vol_fine = is_halo_fine ? vol_halo : vol ;  

      /* 
      * First we need to find the index 
      * in the parent quadrant closest 
      * to the requested index in the child
      * quadrant. 
      */ 
      EXPRD( 
      int const iquad_1 = ichild % 2 ;, 
      int const iquad_2 = static_cast<int>(math::floor_int(ichild/2))%2;
      )
      int const VECD( I1{ math::floor_int((iquad_1 * n1 + i1 ) / 2) + ngz }
                    , I2{ math::floor_int((iquad_2 * n2 + i2 ) / 2) + ngz } ) ; 
      EXPR(
      int const i_c = 
            (which_face_coarse == 0) * (
            (!polarity) * (ngz + math::floor_int( ig / 2 ) )
      +     (polarity)  * (ngz + math::floor_int( (ngz-1-ig) / 2 ) )
            )  
      + (which_face_coarse == 1) * (
            (!polarity) * (nx + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) )
      +     (polarity)  * (nx + ngz - 1 - math::floor_int( ig / 2 ) )
            )
      + (which_face_coarse/2!=0) * I1 ;,

      int const j_c = EXPR(
            (which_face_coarse == 2) * (
            (!polarity) * (ngz + math::floor_int( ig / 2 ) )
      +     (polarity)  * (ngz + math::floor_int( (ngz-1-ig) / 2 ) )
            )  
      + (which_face_coarse == 3) * (
            (!polarity) * (ny + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) )
      +     (polarity)  * (ny + ngz - 1 - math::floor_int( ig / 2 ) )
            ),
      + (which_face_coarse/2==0) * I1, 
      + (which_face_coarse/2==2) * I2 );,

      int const k_c =  
            (which_face_coarse == 4) * (
            (!polarity) * (ngz + math::floor_int( ig / 2 ) )
      +     (polarity)  * (ngz + math::floor_int( (ngz-1-ig) / 2 ) )
            )  
      + (which_face_coarse == 5) * (
            (!polarity) * (nz + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) )
      +     (polarity)  * (nz + ngz - 1 - math::floor_int( ig / 2 ) )
            )
      + (which_face_coarse/2!=2) * I2 ;
      )
      
      return InterpT::interpolate(
            VEC(i_f,j_f,k_f)
          , VEC(i_c,j_c,k_c)
          , iq_fine, iq_coarse, ngz, ivar
          , x_fine
          , x_coarse
          , VEC(dx_fine, dy_fine, dz_fine)
          , VEC(2.*dx_fine, 2.*dy_fine, 2.*dz_fine)
          , u 
          , vol_fine
          , vol_coarse
      ) ; 
  }
}  ; 


}} /* namespace thunder::amr */ 

#endif 