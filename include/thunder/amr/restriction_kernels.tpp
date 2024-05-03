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
#include <thunder/utils/inline.h>
#include <thunder/utils/device.h> 
#include <thunder/utils/math.hh>
#include <thunder/data_structures/macros.hh> 

namespace thunder { namespace amr { 
/**
 * @brief Helper class for restriction
 * \ingroup amr 
 * @tparam StateViewT Type of variable view
 * @tparam VolViewT   Type of cell volume view
 * 
 * The <code>operator()</code> of this class performs 
 * a volume averaged restriction of fine data onto a coarse
 * cell. The refinement factor is assumed to be two. 
 */
template< typename StateViewT 
        , typename VolViewT >  
struct restrictor_t {
    size_t VEC( nx, ny, nz ) ; //!< Quadrant extents (unchanged)
    int ngz                  ; //!< Number of ghost cells
    StateViewT state         ; //!< Old state 
    VolViewT   vol_child     ; //!< Child volume
    VolViewT   vol_parent    ; //!< Parent volume
    /**
     * @brief Restrict variable from fine to coarse grid.
     * 
     * @param i x-index in parent quadrant (zero-offset).
     * @param j y-index in parent quadrant (zero-offset).
     * @param k z-index in parent quadrant (zero-offset).
     * @param iq Quadrant indices of children.
     * @param iq_parent Quadrant index of parent. 
     * @param ivar Variable index. 
     * @return double The restricted variable at the coarse point.
     * 
     * The restriction operator acts on the fine data as follows
     * 
     * \f[
     * \frac{\sum_{\{i,j,k\} \in I^{(i_0,j_0,k_0)}_{\rm c} } U^{l+1}_{i,j,k}\,V^{l+1}_{i,j,k}}
     *  {\sum_{\{i,j,k\} \in I^{(i_0,j_0,k_0)}_{\rm c} } V^{l+1}_{i,j,k} }
     * \f]
     * Where \f$ I^{(i_0,j_0,k_0)}_{\rm c} \f$ contains the indices of children cells of 
     * the coarse cell \f$(i_0,j_0,k_0)\f$.
     */
    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    operator() ( VEC( int const& i
                    , int const& j 
                    , int const& k )
                , int iq[P4EST_CHILDREN]
                , int iq_parent 
                , int const& ivar ) const 
    {
        /*****************************************/
        /* We are assuming cell centered data,   */
        /* which implies we need to average over */
        /* P4EST_CHILDREN cells of the children  */
        /* quadrants.                            */
        /*****************************************/

        /* Indices in child quadrant                */ 
        EXPR(
            const int i0 = (2*i) % nx + ngz;,
            const int j0 = (2*j) % ny + ngz;,
            const int k0 = (2*k) % nz + ngz;
        ) 
        /* Index of child quadrant containing point */ 
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


/**
 * @brief Helper class for restriction
 * \ingroup amr 
 * @tparam StateViewT Type of variable view
 * @tparam VolViewT   Type of cell volume view
 * 
 * The <code>operator()</code> of this class performs 
 * a volume averaged restriction of fine data onto a coarse
 * cell. The refinement factor is assumed to be two. 
 */
template< typename StateViewT 
        , typename VolViewT >  
struct ghostzone_restrictor_t {
    int ngz                  ; //!< Number of ghost cells
    StateViewT state         ; //!< Fine state 
    StateViewT halo          ; //!< Fine halo state
    VolViewT   vol           ; //!< Cell volumes
    VolViewT   halo_vol      ; //!< Halo cell volumes
    /**
     * @brief Restrict variable from fine to coarse grid.
     * 
     * @param i x-index in parent quadrant (zero-offset).
     * @param j y-index in parent quadrant (zero-offset).
     * @param k z-index in parent quadrant (zero-offset).
     * @param iq Quadrant indices of children.
     * @param iq_parent Quadrant index of parent. 
     * @param ivar Variable index. 
     * @return double The restricted variable at the coarse point.
     * 
     * The restriction operator acts on the fine data as follows
     * 
     * \f[
     * \frac{\sum_{\{i,j,k\} \in I^{(i_0,j_0,k_0)}_{\rm c} } U^{l+1}_{i,j,k}\,V^{l+1}_{i,j,k}}
     *  {\sum_{\{i,j,k\} \in I^{(i_0,j_0,k_0)}_{\rm c} } V^{l+1}_{i,j,k} }
     * \f]
     * Where \f$ I^{(i_0,j_0,k_0)}_{\rm c} \f$ contains the indices of children cells of 
     * the coarse cell \f$(i_0,j_0,k_0)\f$.
     */
    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    operator() ( VEC( int const& ig
                    , int const& i1 
                    , int const& i2 )
                , VEC( int const& ng 
                     , int const& n1 
                     , int const& n2 )
                , int64_t const iqs_b[THUNDER_FACE_CHILDREN]
                , int const& ivar 
                , int8_t which_face_b 
                , int polarity
                , int const is_halos_b[THUNDER_FACE_CHILDREN] ) const 
    {
        /*****************************************/
        /* We are assuming cell centered data,   */
        /* which implies we need to average over */
        /* P4EST_CHILDREN cells of the children  */
        /* quadrants.                            */
        /*****************************************/

        /* Determine which fine quadrant contains    */
        /* the cell in question.                     */
        const int ichild = EXPRD(
                (2*i1)/n1 
            , + (2*i2)/n2 * 2
            ) ; 

        int const  iq_b = iqs_b[ ichild ] ; 
        bool const is_halo_b = is_halos_b[ ichild ] ; 
        /* Select source state and volumes arrays    */
        auto& u = is_halo_b ? halo : state ; 
        auto& cvol = is_halo_b ? halo_vol : vol ; 
        /* Find indices in fine quadrant             */ 
        int const VEC( Ig{ (2*ig)%ng }, I1{ (2*i1)%n1 + ngz }, I2{ (2*i2)%n2 + ngz } ) ; 
        EXPR(
            const int i0 = 
                (which_face_b == 0) * (
                    (!polarity) * ( ngz + Ig       )
                +   (polarity)  * ( 3*ngz - Ig - 2 ) 
                ) 
            +   (which_face_b == 1) * (
                    (!polarity) * ( ng - 3*ngz + Ig )
                +   (polarity)  * ( ng - ngz - 2 - Ig ) 
                )
            +   (which_face_b/2 != 0) * I1 ;, 

            const int j0 = EXPR(
                (which_face_b == 2) * (
                    (!polarity) * ( ngz + Ig       )
                +   (polarity)  * ( 3*ngz - Ig - 2 ) 
                ) 
            +   (which_face_b == 3) * (
                    (!polarity) * ( ng - 3*ngz + Ig )
                +   (polarity)  * ( ng - ngz - 2 - Ig ) 
                ),
            +   (which_face_b/2 == 0) * I1, 
            +   (which_face_b/2 == 2) * I2) ;, 
            const int k0 = 
                (which_face_b == 4) * (
                    (!polarity) * ( ngz + Ig       )
                +   (polarity)  * ( 3*ngz - Ig - 2 ) 
                ) 
            +   (which_face_b == 5) * (
                    (!polarity) * ( ng - 3*ngz + Ig )
                +   (polarity)  * ( ng - ngz - 2 - Ig ) 
                )
            +   (which_face_b/2 != 2) * I2 ;
        ) 
        
        
        return (EXPR(
              u(VEC(i0,j0,k0),ivar,iq_b)*cvol(VEC(i0,j0,k0),iq_b)
            + u(VEC(i0+1,j0,k0),ivar,iq_b)*cvol(VEC(i0+1,j0,k0),iq_b),
            + u(VEC(i0,j0+1,k0),ivar,iq_b)*cvol(VEC(i0,j0+1,k0),iq_b)
            + u(VEC(i0+1,j0+1,k0),ivar,iq_b)*cvol(VEC(i0+1,j0+1,k0),iq_b),
            + u(VEC(i0,j0,k0+1),ivar,iq_b)*cvol(VEC(i0,j0,k0+1),iq_b)
            + u(VEC(i0,j0+1,k0+1),ivar,iq_b)*cvol(VEC(i0,j0+1,k0+1),iq_b)
            + u(VEC(i0+1,j0,k0+1),ivar,iq_b)*cvol(VEC(i0+1,j0,k0+1),iq_b)
            + u(VEC(i0+1,j0+1,k0+1),ivar,iq_b)*cvol(VEC(i0+1,j0+1,k0+1),iq_b)
        )) / (EXPR(
              cvol(VEC(i0,j0,k0),iq_b)   + cvol(VEC(i0+1,j0,k0),iq_b  ),
            + cvol(VEC(i0,j0+1,k0),iq_b) + cvol(VEC(i0+1,j0+1,k0),iq_b),
            + cvol(VEC(i0,j0,k0+1),iq_b) + cvol(VEC(i0,j0+1,k0+1),iq_b)
            + cvol(VEC(i0+1,j0,k0+1),iq_b) + cvol(VEC(i0+1,j0+1,k0+1),iq_b)
        )) ; 

    }
} ;

} } /* namespace thunder::amr */ 

#endif /* THUNDER_AMR_RESTRICTION_KERNELS_TPP */ 