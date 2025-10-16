
/**
 * @file grhd_c2p.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-10
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

#ifndef GRACE_PHYSICS_EOS_C2P_GRHD_HH
#define GRACE_PHYSICS_EOS_C2P_GRHD_HH

#include <grace_config.h>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/utils/rootfinding.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace {

/**
 * @brief Implementation of conservative 
 *        to primitive conversion routine 
 *        for General Relativistic Hydro 
 *        following Appendix C of 
 *        https://arxiv.org/pdf/1306.4953
 * \ingroup eos
 */
template< typename eos_t >
struct grhd_c2p_t {
    /**
     * @brief Constructor.
     * 
     * @param _eos Equation of State.
     * @param _metric Metric array.
     * @param conservs Conservative variables.
     * NB: The conservatives are expected to be 
     *     undensitized when passed to the c2p.
     */
    GRACE_HOST_DEVICE
    grhd_c2p_t( eos_t const& _eos
              , metric_array_t const& _metric 
              , grmhd_cons_array_t& conservs )
    : eos(_eos), metric(_metric)
    {   

        StildeU = metric.raise({conservs[STXL],conservs[STYL],conservs[STZL]}) ; 
        auto StildeNorm = 
            Kokkos::sqrt(conservs[STXL]*StildeU[0] + conservs[STYL]*StildeU[1] + conservs[STZL]*StildeU[2] ) ; 
        conservs[TAUL] = math::max(0, conservs[TAUL]) ;
        D  = conservs[DENSL] ; 
        /* Acausal momentum */
        if ( StildeNorm > D+conservs[TAUL] ) {
            double const fact = 0.9999*(D+conservs[TAUL]) ; 
            conservs[STXL] *= fact/StildeNorm ; 
            conservs[STYL] *= fact/StildeNorm  ;
            conservs[STZL] *= fact/StildeNorm  ;
            StildeU = metric.raise({conservs[STXL],conservs[STYL],conservs[STZL]}) ; 
            StildeNorm = fact ; 
            //    Kokkos::sqrt(conservs[STXL]*StildeU[0] + conservs[STYL]*StildeU[1] + conservs[STZL]*StildeU[2] ) ; 
        } 
        ye = conservs[YESL] / D ;
        q  = conservs[TAUL] / D ; 
        r = StildeNorm / D ; 
        k = r / ( 1 + q ) ;  
    }

    /**
     * @brief Invert the primitive to conservative transformation
     *        and return primitive variables.
     * @param error c2p inversion residual.
     * @return grmhd_prims_array_t Primitives.
     * NB: When this function returns, the velocity portion 
     * of the prims array actually contains the z-vector, 
     * the pressure contains the lorentz factor and temperature
     * and entropy are left empty. This is later fixed by the 
     * calling function which will compute \f$v^i\f$, pressure, 
     * entropy and temperature by calling the EOS and adding 
     * the relevant metric components to the velocity.
     */
    grmhd_prims_array_t GRACE_HOST_DEVICE
    invert(double& error) {

        auto const func = [&] (double const& zeta) {
            return zeta - r / htilde(zeta) ; 
        } ; 
        double const zm{ 0.5*k/Kokkos::sqrt(1-math::int_pow<2>(0.5*k))} 
                   , zp{ 1e-06 + k/Kokkos::sqrt(1-math::int_pow<2>(k))} ; 
        double const zeta = utils::brent(func,zm,zp,1e-15) ; 
        double const W = Wtilde(zeta) ; 
        grmhd_prims_array_t prims ; 
        prims[RHOL] = D/W ;
        prims[YEL]  = ye ;
        /* Enforce range on eps tilde */
        double epsmin, epsmax; 
        unsigned int err ;
        eos.eps_range__rho_ye(epsmin,epsmax,prims[RHOL],prims[YEL],err) ; 
        prims[EPSL]   = math::min( epsmax
                                 , math::max( epsmin
                                            , epstilde(W,zeta) ) ) ; 
        prims[PRESSL] = W ; 
        double const h = htilde(zeta) ; 
        prims[VXL] = StildeU[0] / D / h / W; 
        prims[VYL] = StildeU[1] / D / h / W; 
        prims[VZL] = StildeU[2] / D / h / W; 
        error = func(zeta) ;
        return std::move(prims) ; 
    }

    /**
     * @brief Invert the primitive to conservative transformation
     *        and return primitive variables.
     * @param error c2p inversion residual.
     * @return grmhd_prims_array_t Primitives.
     * NB: When this function returns, the velocity portion 
     * of the prims array actually contains the z-vector, 
     * the pressure contains the lorentz factor and temperature
     * and entropy are left empty. This is later fixed by the 
     * calling function which will compute \f$v^i\f$, pressure, 
     * entropy and temperature by calling the EOS and adding 
     * the relevant metric components to the velocity.
     */
    grmhd_prims_array_t GRACE_HOST_DEVICE
    invert_with_zeta(double zeta, double& error) {

        auto const func = [&] (double const& zeta) {
            return zeta - r / htilde(zeta) ; 
        } ; 
        double const W = Wtilde(zeta) ; 
        grmhd_prims_array_t prims ; 
        prims[RHOL] = D/W ;
        prims[YEL]  = ye ;
        /* Enforce range on eps tilde */
        double epsmin, epsmax; 
        unsigned int err ;
        eos.eps_range__rho_ye(epsmin,epsmax,prims[RHOL],prims[YEL],err) ; 
        prims[EPSL]   = math::min( epsmax
                                 , math::max( epsmin
                                            , epstilde(W,zeta) ) ) ; 
        prims[PRESSL] = W ; 
        double const h = htilde(zeta) ; 
        prims[VXL] = StildeU[0] / D / h / W; 
        prims[VYL] = StildeU[1] / D / h / W; 
        prims[VZL] = StildeU[2] / D / h / W; 
        error = func(zeta) ;
        return std::move(prims) ; 
    }
    
 private:
    //! Equation of state
    eos_t const& eos ; 
    //! Metric
    metric_array_t const& metric;
    //! Conserved density
    double D  ;
    //! Electron fraction
    double ye ; 
    //! Rescaled energy
    double q  ; 
    //! Rescaled momentum
    double r  ; 
    //! Momentum / Energy ratio
    double k  ; 
    //! Momentum with upper indices 
    std::array<double,3> StildeU ; 

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    Wtilde(double const& z) const {
        return Kokkos::sqrt(1 + math::int_pow<2>(z)) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    rhotilde(double const& W) const {
        return D/W ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    epstilde(double const& W, double const& z) const {
        return W*q - z*r + math::int_pow<2>(z)/(1+W) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    atilde(double& rho, double& eps) const {
        unsigned int err ;
        double yel{ye} ; 
        auto const press = eos.press__eps_rho_ye(eps,rho,yel,err) ; 
        return press / (rho * ( 1 + eps )) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    htilde(double const& z) const {
        auto const W   = Wtilde(z) ; 
        auto rho = rhotilde(W) ; 
        double epsmin, epsmax; 
        double yel{ye} ; 
        unsigned int err; 
        eos.eps_range__rho_ye(epsmin,epsmax,rho,yel,err) ; 
        auto eps = math::max(epsmin,math::min(epsmax,epstilde(W,z))) ; 
        return (1+eps) * (1+atilde(rho,eps)) ; 
    }
} ; 

}

#endif /* GRACE_PHYSICS_EOS_C2P_GRHD_HH */