/**
 * @file prolongation.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-04-08
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

#ifndef GRACE_UTILS_PROLONGATION_HH
#define GRACE_UTILS_PROLONGATION_HH

#include <grace_config.h>
#include <grace/utils/device/device.h>
#include <grace/utils/inline.h> 
#include <grace/utils/numerics/math.hh>
#include <grace/utils/numerics/matrix_helpers.tpp>
#include <grace/utils/numerics/lagrange_interpolators.hh>

#include <Kokkos_Core.hpp> 

namespace utils {


/**
 * @brief Helper struct to perform slope limited 2nd order
 *        prolongation of data from coarse to fine grid.
 * \ingroup amr
 * @tparam LimT Limiter type, see <code>limiters.hh</code> for 
 *              supported options.
 */
template< typename LimT > 
struct linear_prolongator_t
{
    /**
     * @brief Return slope limited interpolated value of coarse 
     *        state at fine grid point
     *
     * @tparam VarViewT Type of variable view
     * @tparam CoordViewT Type of quadrant coordinate view
     * @tparam VolViewT Type of cell volume view
     * @param i_f x-index of fine cell (ngz-offset)
     * @param j_f y-index of fine cell (ngz-offset)
     * @param k_f z-index of fine cell (ngz-offset)
     * @param i_c x-index of coarse cell (ngz-offset)
     * @param j_c y-index of coarse cell (ngz-offset)
     * @param k_c z-index of coarse cell (ngz-offset)
     * @param q_f Fine quadrant idx 
     * @param q_c Coarse quadrant idx
     * @param ngz Number of ghost cells
     * @param ivar Var index
     * @param fine_coords Fine quadrant coordinates view
     * @param coarse_coords Coarse quadrant coordinates view
     * @param fine_dx Fine cell spacing view
     * @param coarse_dx Coarse cell spacing view
     * @param coarse_view Coarse state view
     * @param fine_vol Fine cell volume view
     * @param coarse_vol Coarse cell volume view
     * @return double Interpolated value at fine cell.
     * 
     * The prolongation operator acts on the coarse data as follows
     * 
     * \f[
     *  U^l_{i_c,j_c,k_c} + \sum_{i_d=1}^{N_d} \Delta\bar{U} 
     *      \frac{x^{i_d, l+1}_{i_f,j_f,k_f}-x^{i_d, l}_{i_c,j_c,k_c} }{\Delta x^(i_d,l)} 
     *      \left( 1 - \frac{ V^{l+1}_{i_f,j_f,k_f} }{ \sum V^{l+1}_{i^\prime_f,j^\prime_f,k^\prime_f} } \right)
     * \f]
     * Where \f$(i_f,j_f,k_f)\f$ are the indices of the children of cell \f$(i_c,j_c,k_c)\f$
     * and the partial volume sum extends over the two fine cells along direction \f$i_d\f$.
     * The limited slope \f$\Delta\bar{U}\f$ is computed with the limiter given by <code>LimT</code>.
     */
    template< typename VarViewT
            , typename CoordViewT
            , typename VolViewT >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate(  VEC(int i_f, int j_f, int k_f)
                , VEC(int i_c, int j_c, int k_c)
                , int64_t q_f, int64_t q_c, int ngz, int ivar
                , CoordViewT& fine_coords
                , CoordViewT& coarse_coords 
                , CoordViewT& fine_dx  
                , CoordViewT& coarse_dx  
                , VarViewT& coarse_view 
                , VolViewT& fine_vol 
                , VolViewT& coarse_vol )
    {
        LimT limiter{} ; 
        /* Get coordinates of cell centres for fine and coarse cell */
        EXPR(
        double x_f = fine_coords(0,q_f) + fine_dx(0,q_f) * (i_f - ngz + 0.5) ; 
        double x_c = coarse_coords(0,q_c) + coarse_dx(0,q_c) * (i_c - ngz + 0.5) ;,
        double y_f = fine_coords(1,q_f) + fine_dx(1,q_f) * (j_f - ngz + 0.5) ; 
        double y_c = coarse_coords(1,q_c) + coarse_dx(1,q_c) * (j_c - ngz + 0.5) ;,
        double z_f = fine_coords(2,q_f) + fine_dx(2,q_f) * (k_f - ngz + 0.5) ; 
        double z_c = coarse_coords(2,q_c) + coarse_dx(2,q_c) * (k_c - ngz + 0.5) ;
        )
        double eta ; 
        double slopeR ; 
        double slopeL ; 
        double u_fine{0.};
        EXPR(
        int const sign_x = math::sgn(x_f-x_c);,
        int const sign_y = math::sgn(y_f-y_c);,
        int const sign_z = math::sgn(z_f-z_c);
        )
        EXPR(
        eta = sign_x * 0.5  
        * (1.-fine_vol(VEC(i_f,j_f,k_f),q_f) /( fine_vol(VEC(i_f,j_f,k_f),q_f)
                                              + fine_vol(VEC(i_f-sign_x,j_f,k_f),q_f))) ; 
        slopeL = coarse_view(VEC(i_c,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c-1,j_c,k_c),ivar,q_c);
        slopeR = coarse_view(VEC(i_c+1,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c),ivar,q_c);
        u_fine += eta*limiter(slopeL,slopeR);,
        
        eta = sign_y * 0.5    
        * (1.-fine_vol(VEC(i_f,j_f,k_f),q_f) /( fine_vol(VEC(i_f,j_f,k_f),q_f)
                                              + fine_vol(VEC(i_f,j_f-sign_y,k_f),q_f))) ; 
        slopeL = coarse_view(VEC(i_c,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c-1,k_c),ivar,q_c);
        slopeR = coarse_view(VEC(i_c,j_c+1,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c),ivar,q_c);
        u_fine += eta*limiter(slopeL,slopeR);,

        eta = sign_z * 0.5 
        * (1.-fine_vol(VEC(i_f,j_f,k_f),q_f) /( fine_vol(VEC(i_f,j_f,k_f),q_f)
                                              + fine_vol(VEC(i_f,j_f,k_f-sign_z),q_f))) ; 
        slopeL = coarse_view(VEC(i_c,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c-1),ivar,q_c);
        slopeR = coarse_view(VEC(i_c,j_c,k_c+1),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c),ivar,q_c);
        u_fine += eta*limiter(slopeL,slopeR);

        )
        return u_fine + coarse_view(VEC(i_c,j_c,k_c),ivar,q_c); 
    }
    #ifdef GRACE_CARTESIAN_COORDINATES
    /**
     * @brief Return slope limited interpolated value of coarse 
     *        state at fine grid point. Overload valid for 
     *        Cartesian coordinates that does not require 
     *        cell volumes as input.
     *
     * @tparam VarViewT Type of variable view
     * @tparam CoordViewT Type of quadrant coordinate view
     * @tparam VolViewT Type of cell volume view
     * @param i_f x-index of fine cell (ngz-offset)
     * @param j_f y-index of fine cell (ngz-offset)
     * @param k_f z-index of fine cell (ngz-offset)
     * @param i_c x-index of coarse cell (ngz-offset)
     * @param j_c y-index of coarse cell (ngz-offset)
     * @param k_c z-index of coarse cell (ngz-offset)
     * @param q_f Fine quadrant idx 
     * @param q_c Coarse quadrant idx
     * @param ngz Number of ghost cells
     * @param ivar Var index
     * @param fine_coords Fine quadrant coordinates view
     * @param coarse_coords Coarse quadrant coordinates view
     * @param sign_x Whether the fine point lies ahead or behind the 
     *               center of the coarse cell along the x-axis.
     * @param sign_y Whether the fine point lies ahead or behind the 
     *               center of the coarse cell along the y-axis.
     * @param sign_z Whether the fine point lies ahead or behind the 
     *               center of the coarse cell along the z-axis.
     * @param coarse_view Coarse state view
     * @return double Interpolated value at fine cell.
     * 
     * The prolongation operator acts on the coarse data as follows
     * 
     * \f[
     *  U^l_{i_c,j_c,k_c} + \sum_{i_d=1}^{N_d} \Delta\bar{U} 
     *      \frac{x^{i_d, l+1}_{i_f,j_f,k_f}-x^{i_d, l}_{i_c,j_c,k_c} }{\Delta x^(i_d,l)} 
     *      \left( 1 - \frac{ V^{l+1}_{i_f,j_f,k_f} }{ \sum V^{l+1}_{i^\prime_f,j^\prime_f,k^\prime_f} } \right)
     * \f]
     * Where \f$(i_f,j_f,k_f)\f$ are the indices of the children of cell \f$(i_c,j_c,k_c)\f$
     * and the partial volume sum extends over the two fine cells along direction \f$i_d\f$.
     * The limited slope \f$\Delta\bar{U}\f$ is computed with the limiter given by <code>LimT</code>.
     */
    template< typename VarViewT >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate(  VEC(int i_f, int j_f, int k_f)
                , VEC(int i_c, int j_c, int k_c)
                , int64_t q_f, int64_t q_c, int ngz, int ivar
                , VEC(int const sign_x, int const sign_y, int const sign_z)
                , VarViewT& coarse_view )
    {
        LimT limiter{} ; 
        double eta ; 
        double slopeR ; 
        double slopeL ; 
        double u_fine{0.};
        EXPR(
        eta = sign_x * 0.25 ; 
        slopeL = coarse_view(VEC(i_c,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c-1,j_c,k_c),ivar,q_c);
        slopeR = coarse_view(VEC(i_c+1,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c),ivar,q_c);
        u_fine += eta*limiter(slopeL,slopeR);,
        
        eta = sign_y * 0.25 ; 
        slopeL = coarse_view(VEC(i_c,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c-1,k_c),ivar,q_c);
        slopeR = coarse_view(VEC(i_c,j_c+1,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c),ivar,q_c);
        u_fine += eta*limiter(slopeL,slopeR);,

        eta = sign_z * 0.25 ; 
        slopeL = coarse_view(VEC(i_c,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c-1),ivar,q_c);
        slopeR = coarse_view(VEC(i_c,j_c,k_c+1),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c),ivar,q_c);
        u_fine += eta*limiter(slopeL,slopeR);
        )
        return u_fine + coarse_view(VEC(i_c,j_c,k_c),ivar,q_c); 
    }

    #endif 
    /**
     * @brief Return slope limited interpolated value of coarse 
     *        state at fine grid point
     *
     * @tparam VarViewT Type of variable view
     * @tparam CoordViewT Type of quadrant coordinate view
     * @tparam VolViewT Type of cell volume view
     * @param i_f x-index of fine cell (ngz-offset)
     * @param j_f y-index of fine cell (ngz-offset)
     * @param k_f z-index of fine cell (ngz-offset)
     * @param i_c x-index of coarse cell (ngz-offset)
     * @param j_c y-index of coarse cell (ngz-offset)
     * @param k_c z-index of coarse cell (ngz-offset)
     * @param q_f Fine quadrant idx 
     * @param q_c Coarse quadrant idx
     * @param ngz Number of ghost cells
     * @param ivar Var index
     * @param fine_coords Fine quadrant coordinates view
     * @param coarse_coords Coarse quadrant coordinates view
     * @param sign_x Whether the fine point lies ahead or behind the 
     *               center of the coarse cell along the first axis.
     * @param sign_y Whether the fine point lies ahead or behind the 
     *               center of the coarse cell along the second axis.
     * @param sign_z Whether the fine point lies ahead or behind the 
     *               center of the coarse cell along the third axis.
     * @param coarse_view Coarse state view
     * @param fine_vol Fine cell volume view
     * @param coarse_vol Coarse cell volume view
     * @return double Interpolated value at fine cell.
     * 
     * The prolongation operator acts on the coarse data as follows
     * 
     * \f[
     *  U^l_{i_c,j_c,k_c} + \sum_{i_d=1}^{N_d} \Delta\bar{U} 
     *      \frac{x^{i_d, l+1}_{i_f,j_f,k_f}-x^{i_d, l}_{i_c,j_c,k_c} }{\Delta x^(i_d,l)} 
     *      \left( 1 - \frac{ V^{l+1}_{i_f,j_f,k_f} }{ \sum V^{l+1}_{i^\prime_f,j^\prime_f,k^\prime_f} } \right)
     * \f]
     * Where \f$(i_f,j_f,k_f)\f$ are the indices of the children of cell \f$(i_c,j_c,k_c)\f$
     * and the partial volume sum extends over the two fine cells along direction \f$i_d\f$.
     * The limited slope \f$\Delta\bar{U}\f$ is computed with the limiter given by <code>LimT</code>.
     */
    template< typename VarViewT
            , typename VolViewT >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate(  VEC(int i_f, int j_f, int k_f)
                , VEC(int i_c, int j_c, int k_c)
                , int64_t q_f, int64_t q_c, int ngz, int ivar
                , VEC(int const sign_x, int const sign_y, int const sign_z)
                , VarViewT& coarse_view 
                , VolViewT& fine_vol )
    {
        LimT limiter{} ; 
        double eta ; 
        double slopeR ; 
        double slopeL ; 
        double u_fine{0.};
        EXPR(
        eta = sign_x * 0.5  
        * (1.-fine_vol(VEC(i_f,j_f,k_f),q_f) /( fine_vol(VEC(i_f,j_f,k_f),q_f)
                                              + fine_vol(VEC(i_f-sign_x,j_f,k_f),q_f))) ; 
        slopeL = coarse_view(VEC(i_c,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c-1,j_c,k_c),ivar,q_c);
        slopeR = coarse_view(VEC(i_c+1,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c),ivar,q_c);
        u_fine += eta*limiter(slopeL,slopeR);,
        
        eta = sign_y * 0.5    
        * (1.-fine_vol(VEC(i_f,j_f,k_f),q_f) /( fine_vol(VEC(i_f,j_f,k_f),q_f)
                                              + fine_vol(VEC(i_f,j_f-sign_y,k_f),q_f))) ; 
        slopeL = coarse_view(VEC(i_c,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c-1,k_c),ivar,q_c);
        slopeR = coarse_view(VEC(i_c,j_c+1,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c),ivar,q_c);
        u_fine += eta*limiter(slopeL,slopeR);,

        eta = sign_z * 0.5 
        * (1.-fine_vol(VEC(i_f,j_f,k_f),q_f) /( fine_vol(VEC(i_f,j_f,k_f),q_f)
                                              + fine_vol(VEC(i_f,j_f,k_f-sign_z),q_f))) ; 
        slopeL = coarse_view(VEC(i_c,j_c,k_c),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c-1),ivar,q_c);
        slopeR = coarse_view(VEC(i_c,j_c,k_c+1),ivar,q_c) - coarse_view(VEC(i_c,j_c,k_c),ivar,q_c);
        u_fine += eta*limiter(slopeL,slopeR);
        )
        return u_fine + coarse_view(VEC(i_c,j_c,k_c),ivar,q_c); 
    }
} ; 

// Note: this struct should probably be named with the "_corner_" subscript
// to indicate what it does 

template< size_t order >
struct lagrange_prolongator_t
{
    template< typename CoarseViewT
            , typename FineViewT >
    static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate(  VEC(int i_f, int j_f, int k_f)
                , VEC(int i_c, int j_c, int k_c)
                , CoarseViewT& coarse_view 
                , FineViewT& fine_view ) 
    {
        using interp_t = corner_staggered_lagrange_interp_t<order> ; 
        /* first copy the common corners */
        fine_view(VEC(i_f,j_f,k_f)) = coarse_view(VEC(i_c,j_c,k_c)) ; 
        fine_view(VEC(i_f+2,j_f,k_f)) = coarse_view(VEC(i_c+1,j_c,k_c)) ; 
        fine_view(VEC(i_f,j_f+2,k_f)) = coarse_view(VEC(i_c,j_c+1,k_c)) ;
        fine_view(VEC(i_f+2,j_f+2,k_f)) = coarse_view(VEC(i_c+1,j_c+1,k_c)) ; 
        #ifdef GRACE_3D
        fine_view(VEC(i_f+2,j_f,k_f+2)) = coarse_view(VEC(i_c+1,j_c,k_c+1)) ; 
        fine_view(VEC(i_f,j_f+2,k_f+2)) = coarse_view(VEC(i_c,j_c+1,k_c+1)) ; 
        fine_view(VEC(i_f+2,j_f+2,k_f+2)) = coarse_view(VEC(i_c+1,j_c+1,k_c+1)) ; 
        fine_view(VEC(i_f,j_f,k_f+2)) = coarse_view(VEC(i_c,j_c,k_c+1)) ; 
        #endif 
        /* Then interpolate along edges */
        fine_view(VEC(i_f+1,j_f,k_f)) = 
            interp_t::template oned_interp<0,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        fine_view(VEC(i_f,j_f+1,k_f)) =
            interp_t::template oned_interp<1,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        #ifdef GRACE_3D 
        fine_view(VEC(i_f,j_f,k_f+1)) =
            interp_t::template oned_interp<2,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        #endif
        fine_view(VEC(i_f+1,j_f+2,k_f)) =
            interp_t::template oned_interp<0,CoarseViewT>(coarse_view,VEC(i_c,j_c+1,k_c)) ; 
        fine_view(VEC(i_f+2,j_f+1,k_f)) =
            interp_t::template oned_interp<1,CoarseViewT>(coarse_view,VEC(i_c+1,j_c,k_c)) ;
        #ifdef GRACE_3D
        fine_view(VEC(i_f+1,j_f,k_f+2)) =
            interp_t::template oned_interp<0,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c+1)) ;
        fine_view(VEC(i_f,j_f+1,k_f+2)) =
            interp_t::template oned_interp<1,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c+1)) ;
        fine_view(VEC(i_f+2,j_f,k_f+1)) =
            interp_t::template oned_interp<2,CoarseViewT>(coarse_view,VEC(i_c+1,j_c,k_c)) ;
        fine_view(VEC(i_f+2,j_f+2,k_f+1)) =
            interp_t::template oned_interp<2,CoarseViewT>(coarse_view,VEC(i_c+1,j_c+1,k_c)) ;
        fine_view(VEC(i_f,j_f+2,k_f+1)) =
            interp_t::template oned_interp<2,CoarseViewT>(coarse_view,VEC(i_c,j_c+1,k_c)) ;
        fine_view(VEC(i_f+1,j_f+2,k_f+2)) =
            interp_t::template oned_interp<0,CoarseViewT>(coarse_view,VEC(i_c,j_c+1,k_c+1)) ;
        fine_view(VEC(i_f+2,j_f+1,k_f+2)) =
            interp_t::template oned_interp<1,CoarseViewT>(coarse_view,VEC(i_c+1,j_c,k_c+1)) ;
        #endif 
        /* Now do a 2D lagrange for corners within coarse face*/
        fine_view(VEC(i_f+1,j_f+1,k_f)) =
            interp_t::template twod_interp<0,1,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ;
        #ifdef GRACE_3D
        fine_view(VEC(i_f+1,j_f,k_f+1)) =
            interp_t::template twod_interp<0,2,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        fine_view(VEC(i_f,j_f+1,k_f+1)) =
            interp_t::template twod_interp<1,2,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ;
        
        fine_view(VEC(i_f+2,j_f+1,k_f+1)) =
            interp_t::template twod_interp<1,2,CoarseViewT>(coarse_view,VEC(i_c+1,j_c,k_c)) ;
        fine_view(VEC(i_f+1,j_f+2,k_f+1)) =
            interp_t::template twod_interp<0,2,CoarseViewT>(coarse_view,VEC(i_c,j_c+1,k_c)) ;
        fine_view(VEC(i_f+1,j_f+1,k_f+2)) =
            interp_t::template twod_interp<0,1,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c+1)) ;

        /* Finally the one at the center */
        fine_view(VEC(i_f+1,j_f+1,k_f+1)) =
            interp_t::template threed_interp<CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        #endif 
    }
} ;  


/**
 * @brief Helper struct to perform 
 *        prolongation of edge-staggered data from coarse to fine grid.
 * \ingroup amr
 * @tparam order Order of the interpolation
 * @tparam stagger_direction Orthogonal direction to the ones in which the variable is staggered
 */
template<size_t order, size_t stagger_direction>
struct lagrange_edge_prolongator_t
{
    /**
     * @brief Return the interpolated value of coarse 
     *        edge-centred variable state at a fine edge
     *
     * @tparam CoarseViewT Type of variable view
     * @tparam FineViewT Type of variable view 
     * @param i_f x-index of fine edge (ngz-offset)
     * @param j_f y-index of fine edge (ngz-offset)
     * @param k_f z-index of fine edge (ngz-offset)
     * @param i_c x-index of coarse edge (ngz-offset)
     * @param j_c y-index of coarse edge (ngz-offset)
     * @param k_c z-index of coarse edge (ngz-offset)
     * @param coarse_view Coarse state view
     * @param fine_view Fine state view
     */

    template< typename CoarseViewT
            , typename FineViewT >
    static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate(  VEC(int i_f, int j_f, int k_f)
                , VEC(int i_c, int j_c, int k_c)
                , CoarseViewT& coarse_view 
                , FineViewT& fine_view );  
};

// template specialization for order 2 (highest needed for vector-potential based MHD at the moment)

template <size_t stagger_direction>
struct lagrange_edge_prolongator_t<2,stagger_direction>
{   

    using num_order = std::integral_constant<size_t, 2>;
    static constexpr size_t sD = stagger_direction;  // Define sD as a constant for easy reference
    
    template< typename CoarseViewT
            , typename FineViewT >
    static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate(  VEC(int i_f, int j_f, int k_f)
                , VEC(int i_c, int j_c, int k_c)
                , CoarseViewT& coarse_view 
                , FineViewT& fine_view ){

        // (somewhat) helpful picture in 2D, for A^z (xc-dx/2,yc-dy/2,zc)
        // here, edir is up (^), facedir is (>)
        //       ____________________________
        //      /                           /|
        //     /                           / |
        //    /                           /  |
        //   *==========================*/   |
        //   |             |            |    |          
        //   |             |            |    |
        //   +      c      x     c      |   /|
        //   |             |            |  / |
        //   |             |            | /  |
        //   o=============|============o/   |
        //   |             |            |    |
        //   |             |            |    / 
        //   +      c      x     c      |   / 
        //   |             |            |  / 
        //   |             |            | /
        //   *===========================*
        //
        // the points to interpolate at in 1d are denoted with (+), in 2d with (x)
        // the coarse edges with (o). (c) are fine cell-centres
        // the leftmost (o) is the parent coarse edge
        // if we assume that we fill out A^z components  from A^z_coarse(i,j,k),
        // we will be filling out fine edges 
        // in 1d : A^z(i,j,k),  A^z(i,j,k+1)
        // in 2d:  A^z(i+1,j,k), A^z(i,j+1,k), A^z(i+1,j,k+1),A^z(i,j+1,k+1)
        // in 3d:  A^z(i+1,j+1,k), A^z(i+1,j+1,k+1)
        // things should make sense once we substitute: fdir1=0, fdir2=1, sD=2

        using interp_t = edge_staggered_lagrange_interp_t<num_order::value,stagger_direction> ; 
    
        // these are the (+) point above and below the marker denoted by (o)
        constexpr size_t child_down = 0; //down (below) is always 0, as we begin from 'more negative' values
        constexpr size_t child_up = 1;
        using utils::delta;
        constexpr size_t fdir1 = std::get<0>(get_complementary_dirs<sD>());
        constexpr size_t fdir2 = std::get<1>(get_complementary_dirs<sD>());

        /*  interpolate along edges for two fine children edges */
        fine_view(VEC(i_f,j_f,k_f)) 
        = interp_t::template oned_interp<child_down,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        fine_view(VEC(i_f+delta(0,sD),j_f+delta(1,sD),k_f+delta(2,sD))) 
        = interp_t::template oned_interp<child_up,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 

        /* Now do a 2D lagrange for fine edges within coarse face, these are denoted by (x)*/
        // lower children
        fine_view(VEC(i_f+delta(0,fdir1),j_f+delta(1,fdir1),k_f+delta(2,fdir1))) 
        = interp_t::template twod_interp<child_down,fdir1,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        fine_view(VEC(i_f+delta(0,fdir2),j_f+delta(1,fdir2),k_f+delta(2,fdir2))) 
        = interp_t::template twod_interp<child_down,fdir2,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        // upper children
        fine_view(VEC(i_f+delta(0,fdir1)+delta(0,sD),j_f+delta(1,fdir1)+delta(1,sD),k_f+delta(2,fdir1)+delta(2,sD))) 
        = interp_t::template twod_interp<child_up,fdir1,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        fine_view(VEC(i_f+delta(0,fdir2)+delta(0,sD),j_f+delta(1,fdir2)+delta(1,sD),k_f+delta(2,fdir2)+delta(2,sD))) 
        = interp_t::template twod_interp<child_up,fdir2,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 

        /* Finally the other two children, within the general volume of the coarse cell */
        //lower child
         fine_view(VEC(i_f+delta(0,fdir1)+delta(0,fdir2),  j_f+delta(1,fdir1)+delta(1,fdir2),  k_f+delta(2,fdir1)+delta(2,fdir2))) 
        = interp_t::template threed_interp<child_down,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 

        //upper child 
        fine_view(VEC(i_f+1,j_f+1,k_f+1)) 
        = interp_t::template threed_interp<child_up,CoarseViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 

        // remember: adding 2 * delta in the fine view is equivalent to adding +1 in the coarse view 
        // because the grid is staggered, we need one more index in the directions orthogonal to the staggering:
        fine_view(VEC(i_f+2*delta(0,fdir1),j_f+2*delta(1,fdir1),k_f+2*delta(2,fdir1))) 
        = interp_t::template oned_interp<child_down,CoarseViewT>(coarse_view,VEC(i_c+delta(0,fdir1),j_c+delta(1,fdir1),k_c+delta(2,fdir1))) ; 
        fine_view(VEC(i_f+2*delta(0,fdir2),j_f+2*delta(1,fdir2),k_f+2*delta(2,fdir2))) 
        = interp_t::template oned_interp<child_down,CoarseViewT>(coarse_view,VEC(i_c+delta(0,fdir2),j_c+delta(1,fdir2),k_c+delta(2,fdir2))) ; 
        
        fine_view(VEC(i_f+2*delta(0,fdir1)+delta(0,sD),j_f+2*delta(1,fdir1)+delta(1,sD),k_f+2*delta(2,fdir1)+delta(2,sD))) 
        = interp_t::template oned_interp<child_up,CoarseViewT>(coarse_view,VEC(i_c+delta(0,fdir1),j_c+delta(1,fdir1),k_c+delta(2,fdir1))) ; 
        fine_view(VEC(i_f+2*delta(0,fdir2)+delta(0,sD),j_f+2*delta(1,fdir2)+delta(1,sD),k_f+2*delta(2,fdir2)+delta(2,sD))) 
        = interp_t::template oned_interp<child_up,CoarseViewT>(coarse_view,VEC(i_c+delta(0,fdir2),j_c+delta(1,fdir2),k_c+delta(2,fdir2))) ; 
        
        // two-d interpolation, in one direction the fine increment is multiplied by 2 
        fine_view(VEC(i_f+2*delta(0,fdir1)+delta(0,fdir2),j_f+2*delta(1,fdir1)+delta(1,fdir2),k_f+2*delta(2,fdir1)+delta(2,fdir2))) 
        = interp_t::template twod_interp<child_down,fdir2,CoarseViewT>(coarse_view,VEC(i_c+delta(0,fdir1),j_c+delta(1,fdir1),k_c+delta(2,fdir1))) ; 
        // two-d interpolation, in one direction the fine increment is multiplied by 2 
        fine_view(VEC(i_f+2*delta(0,fdir1)+delta(0,fdir2)+delta(0,sD),j_f+2*delta(1,fdir1)+delta(1,fdir2)+delta(1,sD),k_f+2*delta(2,fdir1)+delta(2,fdir2)+delta(2,sD))) 
        = interp_t::template twod_interp<child_up,fdir2,CoarseViewT>(coarse_view,VEC(i_c+delta(0,fdir1),j_c+delta(1,fdir1),k_c+delta(2,fdir1))) ; 

        // exchange the role of fdir1 and f2
        // two-d interpolation, in one direction the fine increment is multiplied by 2 
        fine_view(VEC(i_f+2*delta(0,fdir2)+delta(0,fdir1),j_f+2*delta(1,fdir2)+delta(1,fdir1),k_f+2*delta(2,fdir2)+delta(2,fdir1))) 
        = interp_t::template twod_interp<child_down,fdir1,CoarseViewT>(coarse_view,VEC(i_c+delta(0,fdir2),j_c+delta(1,fdir2),k_c+delta(2,fdir2))) ; 
        // two d interpolation, in one direction the fine increment is multiplied by 2 
        fine_view(VEC(i_f+2*delta(0,fdir2)+delta(0,fdir1)+delta(0,sD),j_f+2*delta(1,fdir2)+delta(1,fdir1)+delta(1,sD),k_f+2*delta(2,fdir2)+delta(2,fdir1)+delta(2,sD))) 
        = interp_t::template twod_interp<child_up,fdir1,CoarseViewT>(coarse_view,VEC(i_c+delta(0,fdir2),j_c+delta(1,fdir2),k_c+delta(2,fdir2))) ; 

        // these fine edges are already aligned with the next coarse edge:
        fine_view(VEC(i_f+2*delta(0,fdir1)+2*delta(0,fdir2),j_f+2*delta(1,fdir1)+2*delta(1,fdir2),k_f+2*delta(2,fdir1)+2*delta(2,fdir2))) 
        = interp_t::template oned_interp<child_down,CoarseViewT>(coarse_view,VEC(i_c+1,j_c+1,k_c)) ; 
        // this (child-up) edge is also aligned with the next coarse edge
        fine_view(VEC(i_f+2*delta(0,fdir1)+2*delta(0,fdir2)+delta(0,sD),j_f+2*delta(1,fdir1)+2*delta(1,fdir2)+delta(1,sD),k_f+2*delta(2,fdir1)+2*delta(2,fdir2)+delta(2,sD))) 
        = interp_t::template oned_interp<child_up,CoarseViewT>(coarse_view,VEC(i_c+1,j_c+1,k_c)) ; 


    }
};

}

#endif /* GRACE_UTILS_PROLONGATION_HH */