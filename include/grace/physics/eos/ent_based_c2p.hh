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

#include <grace/physics/eos/kastaun_c2p.hh>

namespace grace {

  
  template< typename eos_t > 
  struct ent_froot_t {

    KOKKOS_FUNCTION ent_froot_t(
      eos_t _eos, double _d, double _rsqr, double _rbsqr, double _bsqr, double _s, double _ye, double h0
    ) : eos(_eos), d(_d), s(_s), ye(_ye), rsqr(_rsqr), bsqr(_bsqr), rbsqr(_rbsqr), brosqr(_rsqr*_bsqr-_rbsqr)
    {
      double zsqrmax = rsqr/(h0*h0) ; 
      double wsqrmax = 1 + zsqrmax ; 
      wmax = sqrt(wsqrmax) ; 
      vsqrmax = zsqrmax/wsqrmax ; 
    }

    double KOKKOS_INLINE_FUNCTION 
    x__mu(double mu) const {
      return 1./(1. + mu*bsqr) ; 
    }

    double KOKKOS_INLINE_FUNCTION 
    rfsqr__mu_x(double mu, double x) const {
      return x * (rsqr * x + mu * (x + 1.0) * rbsqr);  
    }

    double KOKKOS_INLINE_FUNCTION
    operator() (double mu) 
    {
      err = C2P_SUCCESS ; 
      lmu                = mu ; 
      x                  = x__mu(mu) ;
      const double rfsqr = rfsqr__mu_x(mu,x) ; 
      vsqr               = rfsqr * mu * mu ; 

      if ( vsqr > vsqrmax ) {
        vsqr = vsqrmax ; 
        w    = wmax    ; 
        err = C2P_VEL_TOO_HIGH ; 
      } else {
        w = 1/sqrt(1-vsqr) ; 
      }

      double const rhomax = eos.density_maximum();
      double const rhomin = eos.density_minimum();
      rho = d/w ; 
      if ( rho >= rhomax ) {
        err = C2P_RHO_TOO_HIGH ; 
        rho = rhomax ; 
      } else if ( rho <= rhomin ) {
        err = C2P_RHO_TOO_LOW ; 
        rho = rhomin ; 
      } 

      double hh,csnd2 ; 
      unsigned int err ; 
      press = eos.press_h_csnd2_temp_eps__entropy_rho_ye(
        hh,csnd2,temp,eps,s,rho,ye,err
      ) ; 

      double const a = press/(rho*(1+eps)) ; 
      double const h = (1+eps) * (1+a) ; 

      double const hbw     = h/w      ; 
      double const newmu   = 1. / (hbw + rfsqr * mu) ;

      return mu - newmu;
    }


    eos_t eos ; 

    double d, s, ye ; 

    double rsqr, bsqr, rbsqr, brosqr, h0sqr; 

    double lmu, x, rho, w, eps, press, temp, vsqr; 

    double vsqrmax, wmax ; 

    c2p_sig_t err ; 
  } ; 

  template< typename eos_t >
  struct entropy_fix_c2p_t {

    GRACE_HOST_DEVICE
    entropy_fix_c2p_t(
		  eos_t const& _eos,
		  metric_array_t const& _metric,
		  grmhd_cons_array_t& conservs
		  ) : eos(_eos), metric(_metric), h0(_eos.enthalpy_minimum())
    {
      

      double const B2 = metric.square_vec({conservs[BSXL],conservs[BSYL], conservs[BSZL]}) ; 
      // limit tau
      conservs[DENSL] = fmax(0,conservs[DENSL]) ; 
      D  = conservs[DENSL] ;


      //conservs[TAUL]  = fmax(0.5*B2/D, conservs[TAUL]) ;
      s = conservs[ENTSL]/D ; 
      r = {conservs[STXL]/D, conservs[STYL]/D, conservs[STZL]/D} ;
      
      Btilde = {conservs[BSXL]/sqrt(D),conservs[BSYL]/sqrt(D), conservs[BSZL]/sqrt(D)} ;
      B = {conservs[BSXL],conservs[BSYL], conservs[BSZL]}; 
      r2 = metric.square_covec(r) ;
      Btilde2 = metric.square_vec(Btilde);
      r_dot_Btilde = (r[0]*Btilde[0] + r[1]*Btilde[1] + r[2]*Btilde[2]) ;
      r_dot_Btilde2 = r_dot_Btilde*r_dot_Btilde;

      r = metric.raise(r) ; 
      
      ye = conservs[YESL] / D ;

      v02 = r2 / (h0*h0 + r2 ) ;
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
    double  GRACE_HOST_DEVICE
    invert(grmhd_prims_array_t& prims, c2p_sig_t& err) {

      prims[YEL]  = ye ;
      prims[ENTL] = s  ;
      
      static constexpr double tolerance = 1e-15 ; 

      // initial bracket 
      #if 1
      double mu0 = 1/h0 ; 
      if ( r2 >= h0 ) {
        fbrack_t g(r2,Btilde2,r_dot_Btilde2,h0) ; 
        int err ; 
        mu0 = utils::rootfind_newton_raphson(0,1./h0,g,30,1e-10,err) ; 
        if ( err == 0 ) {
          mu0 *= 1+1e-10 ; 
        } else {
          mu0 = utils::brent(g,0,1./h0,tolerance)*(1+1e-10) ;
        }
      }
      #endif 

      ent_froot_t fmu(eos,D,r2,r_dot_Btilde2,Btilde2,s,ye,h0) ; 
      double mu = utils::brent(fmu, 0, mu0, tolerance) ; 
      double residual = fmu(mu) ; 

      double const W = fmu.w    ; 
      prims[EPSL]   = fmu.eps   ; 
      prims[RHOL]   = fmu.rho   ; 
      prims[PRESSL] = fmu.press ;
      prims[YEL]    = fmu.ye    ; 
      prims[TEMPL]  = fmu.temp  ; 
      prims[BXL]    = B[0] ; 
      prims[BYL]    = B[1] ; 
      prims[BZL]    = B[2] ; 

      err = fmu.err ; 

      double x = fmu.x; 

      for( int ii=0; ii<3; ++ii) 
        prims[ZXL+ii] = W * mu * x * ( r[ii] + mu * r_dot_Btilde * Btilde[ii] ) ;  
      
      return SQR(W) * fabs(residual) / (1e-50 + mu) ; 
    }
    
  private:
    eos_t eos ;
    metric_array_t metric ;
    double r2, s, Btilde2, D, ye, r_dot_Btilde, r_dot_Btilde2, h0, v02 ;
    std::array<double,3> r, Btilde, B ;
  };
    

} /* namespace grace */
#endif /*GRACE_C2P_KASTAUN_MHD_HH*/
