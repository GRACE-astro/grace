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
#include <grace/utils/device.h>
#include <grace/utils/inline.h> 
#include <grace/utils/math.hh>
#include <grace/utils/matrix_helpers.tpp>
#include <grace/data_structures/macros.hh>

#include <Kokkos_Core.hpp> 

namespace utils {
namespace detail {

template< size_t idir
        , size_t order > 
struct oned_lagrange_interp_t {
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate( view_t& view, VEC(int ic, int jc, int kc)) ; 
} ; 

template< size_t idir > 
struct oned_lagrange_interp_t<idir,2> {
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate( view_t& view, VEC(int ic, int jc, int kc)) {
        return 0.5*( view(VEC(ic, jc, kc)) 
                   + view(VEC(ic+utils::delta(idir,0), jc+utils::delta(idir,1), kc+utils::delta(idir,2))) ) ; 
    }
} ; 

template< size_t idir > 
struct oned_lagrange_interp_t<idir,4> {
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate( view_t& view, VEC(int ic, int jc, int kc)) {
        using utils::delta ;
        return  (-view(VEC(ic - delta(0,idir),jc - delta(1,idir),kc - delta(2,idir))) + 9*(view(VEC(ic,jc,kc)) + view(VEC(ic + delta(0,idir),jc + delta(1,idir),kc + delta(2,idir)))) - 
     view(VEC(ic + 2*delta(0,idir),jc + 2*delta(1,idir),kc + 2*delta(2,idir))))/16. ; 
    }
} ;

template< size_t idir
        , size_t jdir
        , size_t order > 
struct twod_lagrange_interp_t {
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate( view_t& view, VEC(int i, int j, int k)) ; 
} ; 

template< size_t idir
        , size_t jdir > 
struct twod_lagrange_interp_t<idir,jdir,2> {
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate( view_t& view, VEC(int ic, int jc, int kc)) {
        using utils::delta; 
        static_assert( not (idir == jdir), "jdir and idir should never coincide in 2D lagrange.") ; 
        return (view(VEC(ic,jc,kc)) + view(VEC(ic + delta(0,idir),jc + delta(1,idir),kc + delta(2,idir))) + view(VEC(ic + delta(0,jdir),jc + delta(1,jdir),kc + delta(2,jdir))) + 
     view(VEC(ic + delta(0,idir) + delta(0,jdir),jc + delta(1,idir) + delta(1,jdir),kc + delta(2,idir) + delta(2,jdir))))/4.;
    }
} ; 

template< size_t idir
        , size_t jdir > 
struct twod_lagrange_interp_t<idir,jdir,4> {
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate( view_t& view, VEC(int ic, int jc, int kc)) {
        using utils::delta; 
        static_assert( not (idir == jdir), "jdir and idir should never coincide in 2D lagrange.") ; 
        static constexpr double A0 = 1./256. ; 
        static constexpr double A1 = 9  ;
        static constexpr double A2 = 81  ; 
        return (81*view(VEC(ic,jc,kc)) - 9*view(VEC(ic - delta(0,idir),jc - delta(1,idir),kc - delta(2,idir))) + 81*view(VEC(ic + delta(0,idir),jc + delta(1,idir),kc + delta(2,idir))) - 9*view(VEC(ic - delta(0,jdir),jc - delta(1,jdir),kc - delta(2,jdir))) + 
     view(VEC(ic - delta(0,idir) - delta(0,jdir),jc - delta(1,idir) - delta(1,jdir),kc - delta(2,idir) - delta(2,jdir))) - 9*view(VEC(ic + delta(0,idir) - delta(0,jdir),jc + delta(1,idir) - delta(1,jdir),kc + delta(2,idir) - delta(2,jdir))) + 
     view(VEC(ic + 2*delta(0,idir) - delta(0,jdir),jc + 2*delta(1,idir) - delta(1,jdir),kc + 2*delta(2,idir) - delta(2,jdir))) + 81*view(VEC(ic + delta(0,jdir),jc + delta(1,jdir),kc + delta(2,jdir))) - 
     9*view(VEC(ic - delta(0,idir) + delta(0,jdir),jc - delta(1,idir) + delta(1,jdir),kc - delta(2,idir) + delta(2,jdir))) + 81*view(VEC(ic + delta(0,idir) + delta(0,jdir),jc + delta(1,idir) + delta(1,jdir),kc + delta(2,idir) + delta(2,jdir))) - 
     9*(view(VEC(ic + 2*delta(0,idir),jc + 2*delta(1,idir),kc + 2*delta(2,idir))) + view(VEC(ic + 2*delta(0,idir) + delta(0,jdir),jc + 2*delta(1,idir) + delta(1,jdir),kc + 2*delta(2,idir) + delta(2,jdir)))) - 
     9*view(VEC(ic + 2*delta(0,jdir),jc + 2*delta(1,jdir),kc + 2*delta(2,jdir))) + view(VEC(ic - delta(0,idir) + 2*delta(0,jdir),jc - delta(1,idir) + 2*delta(1,jdir),kc - delta(2,idir) + 2*delta(2,jdir))) - 
     9*view(VEC(ic + delta(0,idir) + 2*delta(0,jdir),jc + delta(1,idir) + 2*delta(1,jdir),kc + delta(2,idir) + 2*delta(2,jdir))) + 
     view(VEC(ic + 2*delta(0,idir) + 2*delta(0,jdir),jc + 2*delta(1,idir) + 2*delta(1,jdir),kc + 2*delta(2,idir) + 2*delta(2,jdir))))/256.;

    }
} ;

template< size_t order >
struct threed_lagrange_interp_t {
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate( view_t& view, VEC(int ic, int jc, int kc)) ; 
} ; 

template<> 
struct threed_lagrange_interp_t<2> {
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate( view_t& view, VEC(int ic, int jc, int kc)) {
        return (view(VEC(ic,jc,kc)) + view(VEC(ic,jc,1 + kc)) + view(VEC(ic,1 + jc,kc)) + view(VEC(ic,1 + jc,1 + kc)) + view(VEC(1 + ic,jc,kc)) + view(VEC(1 + ic,jc,1 + kc)) + view(VEC(1 + ic,1 + jc,kc)) + 
     view(VEC(1 + ic,1 + jc,1 + kc)))/8.; 
    }
} ;

template<> 
struct threed_lagrange_interp_t<4> {
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate( view_t& view, VEC(int ic, int jc, int kc)) {
        return (-view(VEC(-1 + ic,-1 + jc,-1 + kc)) + 9*view(VEC(-1 + ic,-1 + jc,kc)) + 9*view(VEC(-1 + ic,-1 + jc,1 + kc)) - view(VEC(-1 + ic,-1 + jc,2 + kc)) + 9*view(VEC(-1 + ic,jc,-1 + kc)) - 81*view(VEC(-1 + ic,jc,kc)) - 
     81*view(VEC(-1 + ic,jc,1 + kc)) + 9*view(VEC(-1 + ic,jc,2 + kc)) + 9*view(VEC(-1 + ic,1 + jc,-1 + kc)) - 81*view(VEC(-1 + ic,1 + jc,kc)) - 81*view(VEC(-1 + ic,1 + jc,1 + kc)) + 9*view(VEC(-1 + ic,1 + jc,2 + kc)) - 
     view(VEC(-1 + ic,2 + jc,-1 + kc)) + 9*view(VEC(-1 + ic,2 + jc,kc)) + 9*view(VEC(-1 + ic,2 + jc,1 + kc)) - view(VEC(-1 + ic,2 + jc,2 + kc)) + 9*view(VEC(ic,-1 + jc,-1 + kc)) - 81*view(VEC(ic,-1 + jc,kc)) - 
     81*view(VEC(ic,-1 + jc,1 + kc)) + 9*view(VEC(ic,-1 + jc,2 + kc)) - 81*view(VEC(ic,jc,-1 + kc)) + 729*view(VEC(ic,jc,kc)) + 729*view(VEC(ic,jc,1 + kc)) - 81*view(VEC(ic,jc,2 + kc)) - 81*view(VEC(ic,1 + jc,-1 + kc)) + 
     729*view(VEC(ic,1 + jc,kc)) + 729*view(VEC(ic,1 + jc,1 + kc)) - 81*view(VEC(ic,1 + jc,2 + kc)) + 9*view(VEC(ic,2 + jc,-1 + kc)) - 81*view(VEC(ic,2 + jc,kc)) - 81*view(VEC(ic,2 + jc,1 + kc)) + 9*view(VEC(ic,2 + jc,2 + kc)) + 
     9*view(VEC(1 + ic,-1 + jc,-1 + kc)) - 81*view(VEC(1 + ic,-1 + jc,kc)) - 81*view(VEC(1 + ic,-1 + jc,1 + kc)) + 9*view(VEC(1 + ic,-1 + jc,2 + kc)) - 81*view(VEC(1 + ic,jc,-1 + kc)) + 729*view(VEC(1 + ic,jc,kc)) + 
     729*view(VEC(1 + ic,jc,1 + kc)) - 81*view(VEC(1 + ic,jc,2 + kc)) - 81*view(VEC(1 + ic,1 + jc,-1 + kc)) + 729*view(VEC(1 + ic,1 + jc,kc)) + 729*view(VEC(1 + ic,1 + jc,1 + kc)) - 81*view(VEC(1 + ic,1 + jc,2 + kc)) + 
     9*view(VEC(1 + ic,2 + jc,-1 + kc)) - 81*view(VEC(1 + ic,2 + jc,kc)) - 81*view(VEC(1 + ic,2 + jc,1 + kc)) + 9*view(VEC(1 + ic,2 + jc,2 + kc)) - view(VEC(2 + ic,-1 + jc,-1 + kc)) + 9*view(VEC(2 + ic,-1 + jc,kc)) + 
     9*view(VEC(2 + ic,-1 + jc,1 + kc)) - view(VEC(2 + ic,-1 + jc,2 + kc)) + 9*view(VEC(2 + ic,jc,-1 + kc)) - 81*view(VEC(2 + ic,jc,kc)) - 81*view(VEC(2 + ic,jc,1 + kc)) + 9*view(VEC(2 + ic,jc,2 + kc)) + 
     9*view(VEC(2 + ic,1 + jc,-1 + kc)) - 81*view(VEC(2 + ic,1 + jc,kc)) - 81*view(VEC(2 + ic,1 + jc,1 + kc)) + 9*view(VEC(2 + ic,1 + jc,2 + kc)) - view(VEC(2 + ic,2 + jc,-1 + kc)) + 
     9*(view(VEC(2 + ic,2 + jc,kc)) + view(VEC(2 + ic,2 + jc,1 + kc))) - view(VEC(2 + ic,2 + jc,2 + kc)))/4096. ; 

    }
} ; 

}
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

template< size_t order >
struct lagrange_prolongator_t
{
    template< typename VarViewT >
    static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate(  VEC(int i_f, int j_f, int k_f)
                , VEC(int i_c, int j_c, int k_c)
                , VarViewT& coarse_view 
                , VarViewT& fine_view ) 
    {
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
            detail::oned_lagrange_interp_t<0,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        fine_view(VEC(i_f,j_f+1,k_f)) =
            detail::oned_lagrange_interp_t<1,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        #ifdef GRACE_3D 
        fine_view(VEC(i_f,j_f,k_f+1)) =
            detail::oned_lagrange_interp_t<2,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        #endif
        fine_view(VEC(i_f+1,j_f+2,k_f)) =
            detail::oned_lagrange_interp_t<0,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c+1,k_c)) ; 
        fine_view(VEC(i_f+2,j_f+1,k_f)) =
            detail::oned_lagrange_interp_t<1,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c+1,j_c,k_c)) ;
        #ifdef GRACE_3D
        fine_view(VEC(i_f+1,j_f,k_f+2)) =
            detail::oned_lagrange_interp_t<0,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c,k_c+1)) ;
        fine_view(VEC(i_f,j_f+1,k_f+2)) =
            detail::oned_lagrange_interp_t<1,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c,k_c+1)) ;
        fine_view(VEC(i_f+2,j_f,k_f+1)) =
            detail::oned_lagrange_interp_t<2,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c+1,j_c,k_c)) ;
        fine_view(VEC(i_f+2,j_f+2,k_f+1)) =
            detail::oned_lagrange_interp_t<2,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c+1,j_c+1,k_c)) ;
        fine_view(VEC(i_f,j_f+2,k_f+1)) =
            detail::oned_lagrange_interp_t<2,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c+1,k_c)) ;
        fine_view(VEC(i_f+1,j_f+2,k_f+2)) =
            detail::oned_lagrange_interp_t<0,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c+1,k_c+1)) ;
        fine_view(VEC(i_f+2,j_f+1,k_f+2)) =
            detail::oned_lagrange_interp_t<1,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c+1,j_c,k_c+1)) ;
        #endif 
        /* Now do a 2D lagrange for corners within coarse face*/
        fine_view(VEC(i_f+1,j_f+1,k_f)) =
            detail::twod_lagrange_interp_t<0,1,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c,k_c)) ;
        #ifdef GRACE_3D
        fine_view(VEC(i_f+1,j_f,k_f+1)) =
            detail::twod_lagrange_interp_t<0,2,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        fine_view(VEC(i_f,j_f+1,k_f+1)) =
            detail::twod_lagrange_interp_t<1,2,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c,k_c)) ;
        
        fine_view(VEC(i_f+2,j_f+1,k_f+1)) =
            detail::twod_lagrange_interp_t<1,2,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c+1,j_c,k_c)) ;
        fine_view(VEC(i_f+1,j_f+2,k_f+1)) =
            detail::twod_lagrange_interp_t<0,2,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c+1,k_c)) ;
        fine_view(VEC(i_f+1,j_f+1,k_f+2)) =
            detail::twod_lagrange_interp_t<0,1,order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c,k_c+1)) ;

        /* Finally the one at the center */
        fine_view(VEC(i_f+1,j_f+1,k_f+1)) =
            detail::threed_lagrange_interp_t<order>::template interpolate<VarViewT>(coarse_view,VEC(i_c,j_c,k_c)) ; 
        #endif 
    }
} ;  
}
#endif /* GRACE_UTILS_PROLONGATION_HH */