#ifndef GRACE_C2P_KASTAUN_MHD_HH
#define GRACE_C2P_KASTAUN_MHD_HH

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

  struct fbrack_t {
    KOKKOS_FUNCTION fbrack_t(double _rsqr, double _bsqr, double _rbsqr,  double _h0)
      : rsqr(_rsqr), bsqr(_bsqr), rbsqr(_rbsqr), h0sqr(_h0*_h0) 
    {}

    double KOKKOS_INLINE_FUNCTION 
    x__mu(double mu) const {
      return 1./(1. + mu*bsqr) ; 
    }

    double KOKKOS_INLINE_FUNCTION 
    rbsq__mu_x(double mu, double x) const {
      return x * (rsqr * x + mu * (x + 1.0) * rbsqr);  
    }

    double KOKKOS_INLINE_FUNCTION 
    h0w__mu_x(double mu, double x) const {
      return sqrt(h0sqr + rbsq__mu_x(mu,x));  
    }
    #if 1
    void KOKKOS_INLINE_FUNCTION
    operator() (double mu, double& f, double& df) const {
      double x = x__mu(mu) ; 
      double xsqr = x*x ; 
      double hw = h0w__mu_x(mu,x) ; 
      double b = x * (xsqr * rsqr + mu * (1 + x + xsqr) * rbsqr);
      f     = mu * hw - 1.;
      df    = (h0sqr + b) / hw;
    }   
    
    double KOKKOS_INLINE_FUNCTION
    operator() (double mu) const {
      double x = x__mu(mu) ; 
      double rfsqr = rbsq__mu_x(mu,x) ; 
      return mu * Kokkos::sqrt(h0sqr + rfsqr) - 1. ; 
    } 
    #endif 

    void KOKKOS_INLINE_FUNCTION 
    bracket(double& mu_min, double& mu_max) const {
      mu_min = 1./(h0sqr + rsqr) ; 
      double mu0 = 1./sqrt(h0sqr) ; 
      double rfsqr_min = rbsq__mu_x(mu0, x__mu(mu0));
      mu_max    = 1.0 / sqrt(h0sqr + rfsqr_min);
      mu_min *= (1-1e-10) ; 
      mu_max *= (1+1e-10) ; 
      if (mu_max <= mu_min) 
      { 
        mu_min = 0;
        mu_max = mu0 * (1.0 + 1e-10);
      }
    }
    double rsqr, bsqr, rbsqr, h0sqr; 
  } ; 
  
  template< typename eos_t > 
  struct froot_t {

    KOKKOS_FUNCTION froot_t(
      eos_t _eos, double _d, double _q, double _rsqr, double _rbsqr, double _bsqr, double _ye, double h0
    ) : eos(_eos), d(_d), qtot(_q), ye(_ye), rsqr(_rsqr), bsqr(_bsqr), rbsqr(_rbsqr), brosqr(_rsqr*_bsqr-_rbsqr)
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
    qf__mu_x(double mu, double x) const {
      double mux = mu*x ; 
      return qtot - 0.5 * (bsqr + mux*mux*brosqr) ;
    }

    double KOKKOS_INLINE_FUNCTION
    eps_raw__mu_qf_rfsqr_w(
      double mu, double qf, double rfsqr, double w
    ) const 
    {
      return w * (qf - mu * rfsqr*(1.0 - mu * w / (1 + w)));
    }

    void KOKKOS_INLINE_FUNCTION
    get_eps_range(double& epsmin, double& epsmax, double rho) const {
      double yel{ye} ;
      unsigned int eos_err ;
      double rhol{rho} ; 
      eos.eps_range__rho_ye(epsmin,epsmax,rhol,yel,eos_err);
    }

    double KOKKOS_INLINE_FUNCTION
    operator() (double mu) 
    {
      err = C2P_SUCCESS ; 
      lmu                = mu ; 
      x                  = x__mu(mu) ;
      const double rfsqr = rfsqr__mu_x(mu,x) ; 
      const double qf    = qf__mu_x(mu,x) ; 
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

      eps = eps_raw__mu_qf_rfsqr_w(mu,qf,rfsqr,w) ;
      double epsmin, epsmax ;  
      get_eps_range(epsmin,epsmax,rho) ;
      if ( eps >= epsmax ) {
        err = C2P_EPS_TOO_HIGH ; 
        eps = epsmax * 0.999; 
      } else if ( eps <= epsmin ) {
        err = C2P_EPS_TOO_LOW ; 
        eps = epsmin; 
      }

      double hh,csnd2 ; 
      unsigned int eos_err ; 
      press = eos.press_h_csnd2_temp_entropy__eps_rho_ye(
        hh,csnd2,temp,ent,eps,rho,ye,eos_err
      ) ; 

      double const a = press/(rho*(1+eps)) ; 
      double const h = (1+eps) * (1+a) ; 

      double const hbw_raw = (1+a) * (1+qf-mu*rfsqr) ; 
      double const hbw     = fmax(hbw_raw, h/w)      ; 
      double const newmu   = 1. / (hbw + rfsqr * mu) ;

      return mu - newmu;
    }


    eos_t eos ; 

    double d, qtot, ye ; 

    double rsqr, bsqr, rbsqr, brosqr, h0sqr; 

    double lmu, x, rho, w, eps, press, temp, ent, vsqr; 

    double vsqrmax, wmax ; 

    c2p_sig_t err ; 
  } ; 

  template< typename eos_t >
  struct kastaun_c2p_t {

    GRACE_HOST_DEVICE
    kastaun_c2p_t(
		  eos_t const& _eos,
		  metric_array_t const& _metric,
		  grmhd_cons_array_t& conservs
		  ) : eos(_eos), metric(_metric), h0(1.0+1e-10)
    {
      

      double const B2 = metric.square_vec({conservs[BSXL],conservs[BSYL], conservs[BSZL]}) ; 
      // limit tau
      conservs[DENSL] = fmax(0,conservs[DENSL]) ; 
      D  = conservs[DENSL] ;


      //conservs[TAUL]  = fmax(0.5*B2/D, conservs[TAUL]) ;
      q = conservs[TAUL]/D ;
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
    invert(grmhd_prims_array_t& prims, c2p_sig_t& c2p_errors) {

      prims[YEL] = ye ;
      
      static constexpr double tolerance = 1e-15 ; 

      // initial bracket 
      double mu0 = 1/h0 ; 
      #if 1
      if ( r2 >= h0*h0 ) {
        fbrack_t g(r2,Btilde2,r_dot_Btilde2,h0) ; 
        double mu0_min, mu0_max ; 
        g.bracket(mu0_min,mu0_max) ; 
        int rootfind_err ; 
        mu0 = utils::rootfind_newton_raphson(mu0_min,mu0_max,g,30,1e-10,rootfind_err) ; 
        if ( rootfind_err == 0 ) {
          mu0 *= 1+1e-10 ; 
        } else {
          mu0 = 1.0/h0 ;
        }
      }
      #endif 

      froot_t fmu(eos,D,q,r2,r_dot_Btilde2,Btilde2,ye,h0) ; 
      double mu = utils::brent(fmu, 0, mu0, tolerance) ; 
      double residual = fmu(mu) ; 

      double const W = fmu.w    ; 
      prims[EPSL]   = fmu.eps   ; 
      prims[RHOL]   = fmu.rho   ; 
      prims[PRESSL] = fmu.press ;
      prims[YEL]    = fmu.ye    ; 
      prims[TEMPL]  = fmu.temp  ; 
      prims[ENTL]   = fmu.ent   ;
      prims[BXL]    = B[0] ; 
      prims[BYL]    = B[1] ; 
      prims[BZL]    = B[2] ; 

      c2p_errors = fmu.err ; 

      double x = fmu.x; 

      for( int ii=0; ii<3; ++ii) 
        prims[ZXL+ii] = W * mu * x * ( r[ii] + mu * r_dot_Btilde * Btilde[ii] ) ;  
      
      return fabs(residual) ; 
    }
    
  private:
    eos_t eos ;
    metric_array_t metric ;
    double r2, q, Btilde2, D, ye, r_dot_Btilde, r_dot_Btilde2, h0, v02 ;
    std::array<double,3> r, Btilde, B ;
  };
    

} /* namespace grace */
#endif /*GRACE_C2P_KASTAUN_MHD_HH*/
