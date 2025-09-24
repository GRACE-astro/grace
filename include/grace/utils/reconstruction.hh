/**
 * @file reconstruction.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-13
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#ifndef GRACE_UTILS_RECONSTRUCTION_HH 
#define GRACE_UTILS_RECONSTRUCTION_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/limiters.hh>
#include <grace/utils/matrix_helpers.tpp>

#include <grace/data_structures/variable_properties.hh>

namespace grace {
/**
 * @brief Class for slope-limited, second order accurate
 *        reconstruction.
 * \ingroup numerics
 * @tparam limiter_t Limiter type.
 */
template< typename limiter_t >
struct slope_limited_reconstructor_t  
{
    /**
     * @brief Compute reconstruction of state 
     *        at the left and right of interface.
     * 
     * @tparam ViewT Variable view type.
     * @param u Variable view.
     * @param uL Left state.
     * @param uR Right state.
     * @param idir Direction of reconstruction.
     * The reconstruction is performed as 
     * \f{eqnarray*}{
     *  u^L_i &:=& u_{i-1/2-\epsilon} = u_{i-1} + 0.5 \Delta u_{i-1}~, \\ 
     *  u^R_i &:=& u_{i-1/2+\epsilon} = u_{i} - 0.5 \Delta u_{i}~.     \\
     * \f}
     * Where \f$\Delta u_i\f$ is the limited slope computed as:
     * \f[
     * \Delta u_i = \text{limiter}(u_i-u_{i-1}, u_{i+1}-u_i).
     * \f]
     * NB: The limiter can be minmod or monotonized-central. See 
     * the relative APIs in the documentation of limiters.hh
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
        limiter_t limiter{} ; 

        int const ip  = i + utils::delta(0,idir)   ; 
        int const im  = i - utils::delta(0,idir)   ; 
        int const imm = i - 2*utils::delta(0,idir) ; 

        int const jp  = j + utils::delta(1,idir)   ; 
        int const jm  = j - utils::delta(1,idir)   ; 
        int const jmm = j - 2*utils::delta(1,idir) ;
        
        #ifdef GRACE_3D 
        int const kp  = k + utils::delta(2,idir)   ; 
        int const km  = k - utils::delta(2,idir)   ; 
        int const kmm = k - 2*utils::delta(2,idir) ;
        #endif 

        double slopeL = u(VEC(i,j,k)) - u(VEC(im,jm,km)) ; 
        double slopeR = u(VEC(ip,jp,kp)) - u(VEC(i,j,k)) ; 

        uR = u(VEC(i,j,k)) - 0.5 * limiter(slopeL,slopeR) ; 

        slopeL = u(VEC(im,jm,km)) - u(VEC(imm,jmm,kmm)) ; 
        slopeR = u(VEC(i,j,k)) - u(VEC(im,jm,km))       ; 

        uL = u(VEC(im,jm,km)) + 0.5 * limiter(slopeL,slopeR) ; 
    }
} ;


/**
 * @brief Specialization of the above class for slope-limited, third order accurate
 *        reconstruction with biased left and right slopes 
 * \ingroup numerics
 * @tparam limiter_t Limiter type.
 */
template< >
struct slope_limited_reconstructor_t<Koren>
{
     /**
     * @brief Compute reconstruction of state 
     *        at the left and right of interface.
     * 
     * @tparam ViewT Variable view type.
     * @param u Variable view.
     * @param uL Left state.
     * @param uR Right state.
     * @param idir Direction of reconstruction.
     * The reconstruction is performed as 
     * \f{eqnarray*}{
     *  u^L_i &:=& u_{i-1/2-\epsilon} = u_{i-1} + 0.5 \Delta u_{i-1}~, \\ 
     *  u^R_i &:=& u_{i-1/2+\epsilon} = u_{i} - 0.5 \Delta u_{i}~.     \\
     * \f}
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
        Koren koren_limiter{} ; 

        int const ip  = i + utils::delta(0,idir)   ; 
        int const im  = i - utils::delta(0,idir)   ; 
        int const imm = i - 2*utils::delta(0,idir) ; 

        int const jp  = j + utils::delta(1,idir)   ; 
        int const jm  = j - utils::delta(1,idir)   ; 
        int const jmm = j - 2*utils::delta(1,idir) ;
        
        #ifdef GRACE_3D 
        int const kp  = k + utils::delta(2,idir)   ; 
        int const km  = k - utils::delta(2,idir)   ; 
        int const kmm = k - 2*utils::delta(2,idir) ;
        #endif 

        double slopeL = u(VEC(i,j,k)) - u(VEC(im,jm,km)) ; 
        double slopeR = u(VEC(ip,jp,kp)) - u(VEC(i,j,k)) ; 

        uR = u(VEC(i,j,k)) - 0.5 * koren_limiter.get_right_slope(slopeL,slopeR) ; 

        slopeL = u(VEC(im,jm,km)) - u(VEC(imm,jmm,kmm)) ; 
        slopeR = u(VEC(i,j,k)) - u(VEC(im,jm,km))       ; 

        uL = u(VEC(im,jm,km)) + 0.5 * koren_limiter.get_left_slope(slopeL,slopeR) ; 
    }
} ;
 


/**
 * @brief General class for polynomial reconstruction (non-WENO)
 * \ingroup numerics
 */
template<typename limiter_t>
struct polynomial_reconstruction{

    template<typename ViewT>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    void operator()(ViewT& u,
                    VEC(int const i, int const j, int const k),
                    double& uL,
                    double& uR,
                    int8_t idir) {
        }
};


/**
 * @brief Fifth-order MP5 limiter (Suresh & Huynh 1997, Mignone et al. 2010)
 * \ingroup numerics
 */
struct MP5 {
    double alpha = 4.0; // same as BHAC
    // double eps   = 1e-14; // 0 in BHAC
    double eps   = 0; // 0 in BHAC

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double sgn(double x) {
        return (x > 0.0) - (x < 0.0); // +1, -1, or 0
    }

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double minmod(double a, double b) {
        return 0.5 * (sgn(a) + sgn(b))
            * Kokkos::min(Kokkos::fabs(a), Kokkos::fabs(b));
    }


    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double median(double a, double b, double c) {
        double tmp1 = b - a;
        double tmp2 = c - a;
        return a + 0.5 * (sgn(tmp1) + sgn(tmp2)) * Kokkos::min(Kokkos::fabs(tmp1), Kokkos::fabs(tmp2));
    }

};

/**
 * @brief Specialization for MP5 reconstruction - NOTE: DOESN'T WORK !
 */
template<>
struct polynomial_reconstruction<MP5> {
        /**
     * @brief Compute polynomial reconstruction of state 
     *        at the left and right of interface.
     * 
     * @tparam ViewT Variable view type.
     * @param u Variable view.
     * @param uL Left state.
     * @param uR Right state.
     * @param idir Direction of reconstruction.
     * The reconstruction is performed as 
     * \f{eqnarray*}{
     *  u^L_i &:=& u_{i-1/2-\epsilon} = u_{i-1} + 0.5 \Delta u_{i-1}~, \\ 
     *  u^R_i &:=& u_{i-1/2+\epsilon} = u_{i} - 0.5 \Delta u_{i}~.     \\
     * \f}
     */
    template<typename ViewT>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    void operator()(ViewT& u,
                    VEC(int const i, int const j, int const k),
                    double& uL,
                    double& uR,
                    int8_t idir) {
        MP5 mp5{};

          // shorthand accessor
        auto U = [&](int ii,int jj,int kk){ return u(VEC(ii,jj,kk)); };

        
        // NOTE: we actually don't end up needing or using im3,jm3,km3
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

        // ---------------- first thing that BHAC reconstructs as well ----------------
        // ---------------- note, however, that BHAC reconstructs u_R understood as      u_{i-1/2 + \epsilon}
        // ---------------- and later constructs                  u_L BUT understood as  u_{i+1/2 - \epsilon} 
        // In other words, for BHAC, uL and uR are within the same cell "i"
        // for us, however, we shift the indices related to the reconstruction of uL by minus one to get the interface at the same face (i-1/2), but from the left side
        
        // ---------------- uR (from cell i, right interface at i-1/2) ----------------
        // for now, the index convention is the same as in the relevant function in BHAC   
        // this is because BHAC's routine, in it's own words, "takes the convention that the central index represents right-side interface value"
        // illustration:
        //       i-2              i-1                  i             i+1             i+2               i+3
        // |                |                |                |                |                |                 |
        // |________________|________________|________________|________________|________________|_________________|
        //                                  i-1/2
        //                                 L | R             L (AMRVAC)
        //                                                 (this is what AMRVAC docs call "right side")
        //                                                        but what for us is the uL at i+1/2
        //                                                  we need precisely this, but for    i-1/2
        //                                 L <--(i to i-1)---L
        //                                 L <---------------L
        // we are now getting R
        // index-wise, it makes sense, since it's right-biased now (takes more info from cells to the right relative to the position of the interface)

        {
            double f = (2.0 * U(im2,jm2,km2) - 13.0 * U(im1,jm1,km1) + 47 * U(i,j,k) + 27.0 * U(ip1,jp1,kp1) - 3.0 * U(ip2,jp2,kp2)) / 60.0;
        
            double a = U(ip1,jp1,kp1) - U(i,j,k);
            double b = mp5.alpha * (U(i,j,k) - U(im1,jm1,km1));
            
            double tmp = mp5.minmod(a, b);
            double fmp = U(i,j,k) + tmp;
            double ful = U(i,j,k) + b;

            // compute di and dim 
            double dim = U(i,j,k) - 2.0 * U(im1,jm1,km1) + U(im2,jm2,km2);
            double di  = U(ip1,jp1,kp1) - 2.0 * U(i,j,k) + U(im1,jm1,km1);

            double aim = 4.0 * dim - di;
            double bim = 4.0 * di - dim;
            double tmp1 = mp5.minmod(aim, bim);
                   aim = dim ; 
                   bim = di  ; 
            double tmp2 = mp5.minmod(aim, bim);
            double tmp3 = mp5.minmod(tmp1, tmp2);
            // dm4(i-1)
            double dm4m = tmp3; 
            // finally flc 
            double flc = 0.5 * (3.0 * U(i,j,k) - U(im1,jm1,km1)) + 4./3. * dm4m; 

            // now, we want dm4(i), so we repeat the above paragraph but for i instead
            // we need a missing dip value:
            double dip = U(ip2,jp2,kp2) - 2.0 * U(ip1,jp1,kp1) + U(i,j,k);
            double ai = 4.0 * di - dip;
            double bi = 4.0 * dip - di; 
                   tmp1= mp5.minmod(ai,bi);
                   ai = di;
                   bi = dip;
                   tmp2 = mp5.minmod(ai,bi);
                   tmp3 = mp5.minmod(tmp1,tmp2);
            // dm4(i)
            double dm4 = tmp3; 
            //finally fmd
            double fmd = 0.5 * (U(i,j,k) + U(ip1,jp1,kp1)) - 0.5 * dm4; 

            // form min and max:
            // we have to combine two min calls to get a ternary operator
            double fmin = Kokkos::max(  Kokkos::min(Kokkos::min(U(i,j,k), U(ip1,jp1,kp1)), fmd),
                                        Kokkos::min(Kokkos::min(U(i,j,k), ful           ), flc)        
                                     );

            double fmax = Kokkos::min(  Kokkos::max(Kokkos::max(U(i,j,k), U(ip1,jp1,kp1)), fmd),
                                        Kokkos::max(Kokkos::max(U(i,j,k), ful           ), flc)

                                    );

            tmp = mp5.median(fmin, f, fmax); 

            double flim = tmp; 
            
            uR = ((f-U(i,j,k))*(f-fmp) <= mp5.eps) ? f : flim;  // note - should we leave eps=1e-14 or eps=0?
        }

        // now we proceed to reconstruct the interface value from the left side 
        // ---------------- uL (inside cell i-1, corresponds to left interface at i-1/2) ----------------
        // in AMRVAC, the indices are at +1 wrt to what we do 

        {
            // f(i-1)
            double fim = (2.0 * U(ip2,jp2,kp2) - 13.0 * U(ip1,jp1,kp1) + 47.0 * U(i,j,k) + 27.0 * U(im1,jm1,km1) - 3.0 * U(im2,jm2,km2)  ) / 60.0;

            double aim = U(im1,jm1,km1) - U(i,j,k); 
            double bim = mp5.alpha * (U(i,j,k) - U(ip1,jp1,kp1)) ; 
            double tmp = mp5.minmod(aim, bim);

            double fmp = U(i,j,k) + tmp;
            double ful = U(i,j,k) + bim; 

            // BASE THESE ON THE LEFT NEIGHBOUR (im1), not U(i)
            // double fmp = U(im1,jm1,km1) + tmp;
            // double ful = U(im1,jm1,km1) + bim;


            // we start by computing di, dim and dimm  
            double di  = U(i,j,k) - 2.0 * U(ip1,jp1,kp1) + U(ip2,jp2,kp2) ; 
            double dim = U(im1,jm1,km1) - 2.0 * U(i,j,k) + U(ip1,jp1,kp1) ; 
            double dimm= U(im2,jm2,km2) - 2.0 * U(im1,jm1,km1) + U(i,j,k) ; 
            
            aim = 4.0 * dim - dimm;
            bim = 4.0 * dimm - dim;
            double tmp1 = mp5.minmod(aim, bim);
            aim = dim;
            bim = dimm;
            double tmp2 = mp5.minmod(aim, bim);
            double tmp3 = mp5.minmod(tmp1, tmp2);
            // finally, get dm4m (i.e. dm4(i-1))
            double dm4m = tmp3;

            // now we proceed to get dm4 (i.e. dm4(i))
            // di is already computed 
            double a = 4*di - dim; 
            double b = 4*dim - di;
            tmp1 = mp5.minmod(a,b);
            a = di;
            b = dim;
            tmp2 = mp5.minmod(a,b);
            tmp3 = mp5.minmod(tmp1,tmp2);
            // finally, get dm4:
            double dm4 = tmp3; 

            // construct fmd , flc, fmin, fmax:
            double fmd = 0.5 * (U(im1,jm1,km1) + U(i,j,k)) - 0.5 * dm4m;
            double flc = 0.5 * (3.0 * U(i,j,k) - U(ip1,jp1,kp1) )   + 4.0/3.0 * dm4; 

            double fmin = Kokkos::max( Kokkos::min(Kokkos::min(U(i,j,k), U(im1,jm1,km1)), fmd),
                                       Kokkos::min(Kokkos::min(U(i,j,k), ful           ), flc)
                                    );

            double fmax = Kokkos::min( Kokkos::max(Kokkos::max(U(i,j,k), U(im1,jm1,km1)), fmd),
                                       Kokkos::max(Kokkos::max(U(i,j,k), ful           ), flc)
                                    );   

            tmp = mp5.median(fmin,fim,fmax);
            double flim = tmp; 

            uL = ( (fim - U(i,j,k))*(fim - fmp) <   mp5.eps) ? fim : flim; 
            // uL = ( (fim - U(im1,jm1,km1))*(fim - fmp) <   mp5.eps) ? fim : flim; 


           
        }
       
    }
};


}
#endif /* GRACE_UTILS_RECONSTRUCTION_HH */