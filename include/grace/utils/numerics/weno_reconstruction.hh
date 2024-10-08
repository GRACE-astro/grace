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

#include <grace/utils/numerics/math.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>
#include <grace/utils/numerics/limiters.hh>
#include <grace/utils/numerics/matrix_helpers.tpp>

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
/**
 * @brief WENO reconstruction class. Specialized
 *        to 3rd and 5th order below.
 * 
 * @tparam order Formal order of accuracy of the method
 *         on smooth data and away from extrema.
 */
template< size_t order >
struct weno_reconstructor_t ; 

/**
 * @brief Piecewise linear WENO reconstruction
 * \ingroup numerics 
 * NB: This class is constructed to act on cell-centered 
 * data which is interpreted to represent volume averaged
 * quantities. This class does \b not produce correct 
 * results for Finite Differencing. 
 */
template<>
struct weno_reconstructor_t<3> 
{
 private:
    static constexpr double d0 = 1./3.; 
    static constexpr double d1 = 2./3.;
    
 public: 
    /**
     * @brief Perform 3rd order accurate WENO reconstruction
     *        of a cell-centered (volume averaged) variable.
     * 
     * @tparam ViewT Type of variable view.
     * @param u Variable view.
     * @param uL Reconstructed value at \f$i-1/2-\epsilon\f$.
     * @param uR Reconstructed value at \f$i-1/2+\epsilon\f$.
     * @param idir Direction of reconstruction.
     * 
     * The weights for WENO reconstruction are taken from Jiang, Liu 1996
     * (https://apps.dtic.mil/sti/tr/pdf/ADA301993.pdf), see Tab. 1 and 2.
     * NB: Following GRACE convention on reconstruction, we want out of this 
     * routine the values  \f$U_{i-1/2 \pm \epsilon}\f$. This is different 
     * from the notation of most papers / codes where the reconstruction is 
     * performed at \f$U_{i-1/2 + \epsilon}\f$ and \f$U_{i+1/2 - \epsilon}\f$.
     * For this reason some of the coefficients are in a different order from
     * what appears in the tables.
     */
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
            d0 / math::int_pow<2>( WENO_EPS + beta[2] ),
            d1 / math::int_pow<2>( WENO_EPS + beta[1] )
        } ; 
        double const wL = 1./( alphaL[0] + alphaL[1] ) ; 
        double const wR = 1./( alphaR[0] + alphaR[1] ) ; 

        uR = 0.5 * wR * ( alphaR[0] * ( 3.*U0    -    UP(1) ) 
                        + alphaR[1] * (    UM(1) +    U0    )) ;
        
        uL = 0.5 * wL * ( alphaL[1] * (  U0    +    UM(1) ) 
                        + alphaL[0] * ( -UM(2) + 3.*UM(1) )) ;
    }
} ;

/**
 * @brief Piecewise parabolic WENO reconstruction
 * \ingroup numerics
 * NB: This class is constructed to act on cell-centered 
 * data which is interpreted to represent volume averaged
 * quantities. This class does \b not produce correct 
 * results for Finite Differencing. 
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
    /**
     * @brief Perform 5rd order accurate WENO reconstruction
     *        of a cell-centered (volume averaged) variable.
     * 
     * @tparam ViewT Type of variable view.
     * @param u Variable view.
     * @param uL Reconstructed value at \f$i-1/2-\epsilon\f$.
     * @param uR Reconstructed value at \f$i-1/2+\epsilon\f$.
     * @param idir Direction of reconstruction.
     * 
     * The weights for WENO reconstruction are taken from Jiang, Liu 1996
     * https://apps.dtic.mil/sti/tr/pdf/ADA301993.pdf, (see Tab. 1 and 2)
     * as well as Titarev, Toro https://www.newton.ac.uk/files/preprints/ni03057.pdf
     * (see Eqs. (25-29)).
     * NB: Following GRACE convention on reconstruction, we want out of this 
     * routine the values  \f$U_{i-1/2 \pm \epsilon}\f$. This is different 
     * from the notation of most papers / codes where the reconstruction is 
     * performed at \f$U_{i-1/2 + \epsilon}\f$ and \f$U_{i+1/2 - \epsilon}\f$.
     * For this reason some of the coefficients are in a different order from
     * what appears in the tables.
     */
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
             WENO5_12_BY_13 * math::int_pow<2>(U0-2.*UP(1)+UP(2)),
             WENO5_12_BY_13 * math::int_pow<2>(UM(1)-2.*U0+UP(1)),
             WENO5_12_BY_13 * math::int_pow<2>(UM(2)-2.*UM(1)+U0),
             WENO5_12_BY_13 * math::int_pow<2>(UM(3)-2.*UM(2)+UM(1))
        } ; 

        std::array<double,3> const betaL {
            gamma[0] + 0.25 * math::int_pow<2>(3.*U0-4.*UP(1)+UP(2)) ,
            gamma[1] + 0.25 * math::int_pow<2>(UM(1)-UP(1)) ,
            gamma[2] + 0.25 * math::int_pow<2>(UM(2)-4.*UM(1)+3.*U0) 
        } ;

        std::array<double,3> const betaR {
            gamma[1] + 0.25 * math::int_pow<2>(3.*UM(1)-4.*U0+UP(1)) ,
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