#ifndef GRACE_C2P_ENTROPY_MHD_HH
#define GRACE_C2P_ENTROPY_MHD_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/metric_utils.hh>
#include <grace/utils/rootfinding.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/eos/c2p.hh>


namespace grace {

template < typename eos_t > 
struct entropy_fix_c2p_t {

    GRACE_HOST_DEVICE
    entropy_fix_c2p_t(
		  eos_t const& _eos,
		  metric_array_t const& _metric,
		  grmhd_cons_array_t& conservs
		  ) : eos(_eos), metric(_metric)
    {
        D  = conservs[DENSL] ;
        Btilde = {conservs[BSXL]/sqrt(D),conservs[BSYL]/sqrt(D), conservs[BSZL]/sqrt(D)} ;
        r = {conservs[STXL]/D, conservs[STYL]/D, conservs[STZL]/D} ;
        r2 = metric.square_covec(r) ; 
        Btilde2 = metric.square_vec(Btilde) ; 
        rdotBtilde = r[0] * Btilde[0] + r[1] * Btilde[1] + r[2] * Btilde[2] ;
        s = conservs[ENTSL] / D ; 
        ye = conservs[YESL] / D ;  
    }

    double  GRACE_HOST_DEVICE
    invert(grmhd_prims_array_t& prims, c2p_sig_t& err) {

        err = C2P_SUCCESS ;
         
        prims[YEL] = ye ; 
        prims[ENTL] = s ; 

        static constexpr double tolerance = 1e-15 ; 

        auto const f = [this] (double x) { return this->f__x(x) ; };
        // find W 
        double x = utils::brent(f,1.,50.,tolerance) ; 
        // get rho
        prims[RHOL] = D / x ;  
        // get P, T, eps from s, rho, Ye
        double h,csnd2 ; 
        unsigned int err ;
        prims[PRESSL] = eos.press_h_csnd2_temp_eps__entropy_rho_ye(
            h,csnd2,prims[TEMPL],prims[EPSL],prims[ENTL],prims[RHOL],prims[YEL], err 
        ) ;
        // get v
 
        /*
        (*PRIMS)[ZVECX] =
        lorentz * (Stilde_up[WVX] + (*CONS)[BX_CENTER] * (SdotBtilde / hW)) /
        (hW + B2tilde);
        (*PRIMS)[ZVECY] =
            lorentz * (Stilde_up[WVY] + (*CONS)[BY_CENTER] * (SdotBtilde / hW)) /
            (hW + B2tilde);
        (*PRIMS)[ZVECZ] =
            lorentz * (Stilde_up[WVZ] + (*CONS)[BZ_CENTER] * (SdotBtilde / hW)) /
            (hW + B2tilde);
        */
        auto rU = metric.raise(r) ; 
        double const hW = h * x ; 
        prims[VXL] = (rU[0] +  Btilde[0] * (rdotBtilde/hW)) / (hW + Btilde2) ; 
        prims[VYL] = (rU[1] +  Btilde[1] * (rdotBtilde/hW)) / (hW + Btilde2) ; 
        prims[VZL] = (rU[2] +  Btilde[2] * (rdotBtilde/hW)) / (hW + Btilde2) ; 

        double const hh = (1 + prims[EPSL] + prims[PRESSL]/prims[RHOL]) ; 

        return fabs(f__x(x)) / SQR(Btilde2 + hh * x) ; 
    }

    private:

    eos_t eos ;
    metric_array_t metric ;
    double D, rdotBtilde, s, ye, Btilde2, r2 ;
    std::array<double,3> r, Btilde ;

    double KOKKOS_INLINE_FUNCTION f__x(double const& x) const 
    {
        /*
        lorentz = X;
        // 0. Update rho
        (*PRIMS)[RHOB] = (*CONS)[RHOSTAR] / lorentz;
        // 1. Compute h_cold
        // 4. Compute a by making a pressure call
        typename eos::error_type error;
        // auto entropyrange = eos::entropy_range__rho_ye((*PRIMS)[RHOB],
        // (*PRIMS)[YE],error);
        // 3. Compute eps and h
        // double entL =
        // min(entropyrange[1],max(entropyrange[0],(*PRIMS)[ENTROPY]));
        double entL = (*PRIMS)[ENTROPY];
        (*PRIMS)[PRESSURE] = eos::press_h_csnd2_temp_eps__entropy_rho_ye(
            aux_vars[0], (*PRIMS)[CS2], (*PRIMS)[TEMP], (*PRIMS)[EPS], entL,
            (*PRIMS)[RHOB], (*PRIMS)[YE], error);  // in
        auto const hW = aux_vars[0]*lorentz;
        auto const lorentz_sq = lorentz*lorentz;
        auto const residual = -lorentz_sq*stilde_sq
        +(lorentz_sq-1.)*SQ(B2tilde + hW) - lorentz_sq*(B2tilde + 2.*hW)*SQ(SdotBtilde/hW);
        */
        // update rho 
        double rho = D / x ; 
        double eps,h,csnd2,temp ; 
        unsigned int err ;
        double entL = s ; 
        double yeL = ye; 
        double press = eos.press_h_csnd2_temp_eps__entropy_rho_ye(
            h,csnd2,temp,eps,entL,rho,yeL, err 
        ) ; 
        double const hW = h * x ;
        double const W2 = SQR(x) ;  
        return -W2 * r2 + (W2-1)*SQR(Btilde2+hW) - W2 * (Btilde2 + 2*hW) * SQR(rdotBtilde/hW) ; 
        
    }
} ; 

}
#endif 