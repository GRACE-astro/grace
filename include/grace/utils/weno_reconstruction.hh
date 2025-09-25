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
    static constexpr double d0 = 0.1; 
    static constexpr double d1 = 0.6; 
    static constexpr double d2 = 0.3; 
    
    static constexpr double WENO5_13_BY_12 = 13.0/12.0 ;  // fix me ? 13/12!
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
        

    // shorthand accessor
    auto U = [&](int ii,int jj,int kk){ return u(VEC(ii,jj,kk)); };

    // relative indices
    int im3=i-3*utils::delta(0,idir), im2=i-2*utils::delta(0,idir), im1=i-1*utils::delta(0,idir);
    int ip1=i+1*utils::delta(0,idir), ip2=i+2*utils::delta(0,idir);
    int jm3=j-3*utils::delta(1,idir), jm2=j-2*utils::delta(1,idir), jm1=j-1*utils::delta(1,idir);
    int jp1=j+1*utils::delta(1,idir), jp2=j+2*utils::delta(1,idir);
    #ifdef GRACE_3D
    int km3=k-3*utils::delta(2,idir), km2=k-2*utils::delta(2,idir), km1=k-1*utils::delta(2,idir);
    int kp1=k+1*utils::delta(2,idir), kp2=k+2*utils::delta(2,idir);
    #else
    int km3=0, km2=0, km1=0, kp1=0, kp2=0;
    #endif

    double um3 = U(im3,jm3,km3);
    double um2 = U(im2,jm2,km2);
    double um1 = U(im1,jm1,km1);
    double u0  = U(i,j,k);
    double up1 = U(ip1,jp1,kp1);
    double up2 = U(ip2,jp2,kp2);

    // Shared constants
    // Smooth WENO weights: (Jiang & Shu 1996)

    const double beta_coeff[2]{13.0/12.0, 0.25};
    const double eps = 1.0e-42;

    //========================
    // Compute uR at i-1/2
    //========================
    {
        double beta[3];
        beta[0] = beta_coeff[0]*math::int_pow<2>(um2+u0-2.0*um1)
                + beta_coeff[1]*math::int_pow<2>(um2+3.0*u0-4.0*um1);
        beta[1] = beta_coeff[0]*math::int_pow<2>(um1+up1-2.0*u0)
                + beta_coeff[1]*math::int_pow<2>(um1-up1);
        beta[2] = beta_coeff[0]*math::int_pow<2>(up2+u0-2.0*up1)
                + beta_coeff[1]*math::int_pow<2>(up2+3.0*u0-4.0*up1);

        // WENO-Z: Borges et al. 2008, Castro et al. 2011
        double tau5 = fabs(beta[0]-beta[2]);

        double ind[3];
        ind[0] = math::int_pow<2>(tau5/(beta[0]+eps));
        ind[1] = math::int_pow<2>(tau5/(beta[1]+eps));
        ind[2] = math::int_pow<2>(tau5/(beta[2]+eps));

        double f[3];
        f[0] = ( 2.0*up2 - 7.0*up1 + 11.0*u0 );
        f[1] = (-1.0*up1 + 5.0*u0 + 2.0*um1);
        f[2] = ( 2.0*u0   + 5.0*um1 - um2 );

        double alpha[3];
        alpha[0] = 0.1*(1.0+ind[2]);
        alpha[1] = 0.6*(1.0+ind[1]);
        alpha[2] = 0.3*(1.0+ind[0]);
        double asum = 6.0*(alpha[0]+alpha[1]+alpha[2]);

        uR = (f[0]*alpha[0]+f[1]*alpha[1]+f[2]*alpha[2])/asum;
    }

    //========================
    // Compute uL at i-1/2
    //========================
    {
        double beta[3];
        beta[0] = beta_coeff[0]*math::int_pow<2>(um3+um1-2.0*um2)
                + beta_coeff[1]*math::int_pow<2>(um3+3.0*um1-4.0*um2);
        beta[1] = beta_coeff[0]*math::int_pow<2>(um2+u0-2.0*um1)
                + beta_coeff[1]*math::int_pow<2>(um2-u0);
        beta[2] = beta_coeff[0]*math::int_pow<2>(up1+um1-2.0*u0)
                + beta_coeff[1]*math::int_pow<2>(up1+3.0*um1-4.0*u0);

        double tau5 = fabs(beta[0]-beta[2]);

        double ind[3];
        ind[0] = math::int_pow<2>(tau5/(beta[0]+eps));
        ind[1] = math::int_pow<2>(tau5/(beta[1]+eps));
        ind[2] = math::int_pow<2>(tau5/(beta[2]+eps));

        double f[3];
        f[0] = ( 2.0*um3 - 7.0*um2 + 11.0*um1 );
        f[1] = (-1.0*um2 + 5.0*um1 + 2.0*u0 );
        f[2] = ( 2.0*um1 + 5.0*u0   - up1   );

        double alpha[3];
        alpha[0] = 0.1*(1.0+ind[0]);
        alpha[1] = 0.6*(1.0+ind[1]);
        alpha[2] = 0.3*(1.0+ind[2]);
        double asum = 6.0*(alpha[0]+alpha[1]+alpha[2]);

        uL = (f[0]*alpha[0]+f[1]*alpha[1]+f[2]*alpha[2])/asum;
    }


    return ; 


    }

 } ; 

}


#undef U0
#undef UP
#undef UM 
#undef WENO_EPS

#endif /* GRACE_UTILS_WENO_RECONSTRUCTION_HH */