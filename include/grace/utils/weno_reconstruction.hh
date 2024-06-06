/**
 * @file weno_reconstruction.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-06
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#include <grace/utils/math.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device.h>
#include <grace/utils/limiters.hh>
#include <grace/utils/matrix_helpers.tpp>

#ifndef GRACE_UTILS_WENO_RECONSTRUCTION_HH
#define GRACE_UTILS_WENO_RECONSTRUCTION_HH 

/**
 * \defgroup numerics Numeric helpers.
 * 
 */

#define WENO_EPS 1.e-15 

#define U0 u(VEC(i,j,k))
#define UM(d) u(VEC(i-d*utils::delta(0,idir),j-d*utils::delta(1,idir),k-d*utils::delta(2,idir)))
#define UP(d) u(VEC(i+d*utils::delta(0,idir),j+d*utils::delta(1,idir),k+d*utils::delta(2,idir)))

namespace grace {

template< size_t order >
struct weno_reconstructor_t ; 
/**
 * @brief Piecewise linear WENO reconstruction
 * \ingroup numerics 
 */
template<>
struct weno_reconstructor_t<3> 
{
 private:
    static constexpr double d0 = 2./3.; 
    static constexpr double d1 = 1./3.;
    
 public: 

    template< typename ViewT >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() (
          ViewT& u 
        , VEC( int const i
             , int const j 
             , int const k)
        , double& uL
        , double& uR 
        , int8_t idir )
    {

        std::array<double,3> const beta {
            math::int_pow<2>(UM(1)-UM(2)),
            math::int_pow<2>(U0 - UM(1)),
            math::int_pow<2>(UP(1) - U0)
        } ;

        std::array<double,2> const alphaL { 
            d0 / math::int_pow<2>( WENO_EPS + beta[0] ),
            d1 / math::int_pow<2>( WENO_EPS + beta[1] )
        } ; 
        std::array<double,2> const alphaR { 
            d1 / math::int_pow<2>( WENO_EPS + beta[1] ),
            d0 / math::int_pow<2>( WENO_EPS + beta[2] )
        } ; 
        double const wL = 1./( alphaL[0] + alphaL[1] ) ; 
        double const wR = 1./( alphaR[0] + alphaR[1] ) ; 
        
        uR = 0.5 * wR * ( alphaR[0] * (  U0    +    UP(1) ) 
                        + alphaR[1] * ( -UM(1) + 3.*U0    )) ;
        
        uL = 0.5 * wL * ( alphaL[0] * ( -U0    + 3.*UM(1) ) 
                        + alphaL[1] * (  UM(2) +    UM(1) )) ;
    }
} ;

/**
 * @brief Piecewise parabolic WENO reconstruction
 * \ingroup numerics 
 */
template<>
struct weno_reconstructor_t<5> 
{
 private:
    static constexpr double d0 = 0.3; 
    static constexpr double d1 = 0.6; 
    static constexpr double d2 = 0.1; 
    
    static constexpr double WENO5_12_BY_13 = 12.0/13.0 ; 
    static constexpr double WENO5_1_BY_6   = 1.0/6.0   ; 
    
 public: 

    template< typename ViewT >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() (
          ViewT& u 
        , VEC( int const i
             , int const j 
             , int const k)
        , double& uL
        , double& uR 
        , int8_t idir )
    {
        std::array<double,4> const gamma {
             WENO5_13_BY_12 * math::int_pow<2>(U0-2.*UP(1)+UP(2)),
             WENO5_13_BY_12 * math::int_pow<2>(UM(1)-2.*U0+UP(1)),
             WENO5_13_BY_12 * math::int_pow<2>(UM(2)-2.*UM(1)+U0),
             WENO5_13_BY_12 * math::int_pow<2>(UM(3)-2.*UM(2)+UM(1))
        } ; 

        std::array<double,3> const betaL {
            gamma[0] + 0.25 * math::int_pow<2>(3.*U0-4.*UP(1)+UP(2)) ,
            gamma[1] + 0.25 * math::int_pow<2>(UM(1)-UP(1)) ,
            gamma[2] + 0.25 * math::int_pow<2>(UM(2)-4.*UM(1)+3.*U) 
        } ;

        std::array<double,3> const betaR {
            gamma[1] + 0.25 * math::int_pow<2>(3.*UM(1)-4.*U+UP(1)) ,
            gamma[2] + 0.25 * math::int_pow<2>(UM(2)-U0) ,
            gamma[3] + 0.25 * math::int_pow<2>(UM(3)-4.*UM(2)+3.*UM(1)) 
        } ;

        std::array<double,3> const alphaL { 
            d2 / math::int_pow<2>( WENO_EPS + betaL[0] ),
            d1 / math::int_pow<2>( WENO_EPS + betaL[1] ),
            d0 / math::int_pow<2>( WENO_EPS + betaL[2] )
        } ; 
        std::array<double,3> const alphaR { 
            d0 / math::int_pow<2>( WENO_EPS + betaR[0] ),
            d1 / math::int_pow<2>( WENO_EPS + betaR[1] ),
            d2 / math::int_pow<2>( WENO_EPS + betaR[2] )
        } ; 
        double const wL = 1./(alphaL[0] + alphaL[1] + alphaL[2]) ; 
        double const wR = 1./(alphaR[0] + alphaR[1] + alphaR[2]) ; 
        
        uR = WENO5_1_BY_6 * wR * ( alphaR[0] * ( -UP(2) + 5.*UP(1) + 2*U0) 
                                 + alphaR[1] * ( -UM(1) + 5.*U0 + 2*UP(1)) 
                                 + alphaR[2] * ( 2.*UM(2) - 7.*UM(1) + 11*U0) ) ;
        uL = WENO5_1_BY_6 * wL * ( alphaL[0] * ( 2.*UP(1) - 7.*U0 + 11*UM(1)) 
                                 + alphaL[1] * ( -U0 + 5.*UM(1) + 2*UM(2)) 
                                 + alphaL[2] * ( -UM(3) + 5.*UM(2) + 2.*UM(1)) ) ;
    }
} ; 

}


#undef U0
#undef UP
#undef UM 
#undef WENO_EPS

#endif /* GRACE_UTILS_WENO_RECONSTRUCTION_HH */