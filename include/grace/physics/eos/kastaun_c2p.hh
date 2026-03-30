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

  /**
   * @brief Auxiliary rootfind for Kastaun c2p
   */
  struct fbrack_t {
    /**
     * @brief Constructor
     */
    KOKKOS_FUNCTION fbrack_t(double _rsqr, double _bsqr, double _rbsqr,  double _h0)
      : rsqr(_rsqr), bsqr(_bsqr), rbsqr(_rbsqr), h0sqr(_h0*_h0) 
    {}
    /**
     * @brief x of mu
     */
    double KOKKOS_INLINE_FUNCTION 
    x__mu(double mu) const {
      return 1./(1. + mu*bsqr) ; 
    }
    /**
     * @brief (r B)^2 of mu, x 
     */
    double KOKKOS_INLINE_FUNCTION 
    rbsq__mu_x(double mu, double x) const {
      return x * (rsqr * x + mu * (x + 1.0) * rbsqr);  
    }

    /**
     * @brief h0 W of mu, x
     */
    double KOKKOS_INLINE_FUNCTION 
    h0w__mu_x(double mu, double x) const {
      return sqrt(h0sqr + rbsq__mu_x(mu,x));  
    }
    
    /**
     * @brief Residual of auxiliary function, returning derivative
     */
    void KOKKOS_INLINE_FUNCTION
    operator() (double mu, double& f, double& df) const {
      double x = x__mu(mu) ; 
      double xsqr = x*x ; 
      double hw = h0w__mu_x(mu,x) ; 
      double b = x * (xsqr * rsqr + mu * (1 + x + xsqr) * rbsqr);
      f     = mu * hw - 1.;
      df    = (h0sqr + b) / hw;
    }   
    
    /**
     * @brief Residual of auxiliary function, no derivative
     */
    double KOKKOS_INLINE_FUNCTION
    operator() (double mu) const {
      double x = x__mu(mu) ; 
      double rfsqr = rbsq__mu_x(mu,x) ; 
      return mu * Kokkos::sqrt(h0sqr + rfsqr) - 1. ; 
    } 

    /**
     * @brief Bracket where the root lies
     */
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
  
  /**
   * @brief Main rootfinding interface of Kastaun c2p
   */
  template< typename eos_t > 
  struct froot_t {

    /**
     * @brief Constructor
     */
    KOKKOS_FUNCTION froot_t(
      eos_t _eos, double _d, double _q, double _rsqr, double _rbsqr, double _bsqr, double _ye, double h0
    ) : eos(_eos), d(_d), qtot(_q), ye(_ye), rsqr(_rsqr), bsqr(_bsqr), rbsqr(_rbsqr), brosqr(_rsqr*_bsqr-_rbsqr)
    {
      double zsqrmax = rsqr/(h0*h0) ; 
      double wsqrmax = 1 + zsqrmax ; 
      wmax = sqrt(wsqrmax) ; 
      vsqrmax = zsqrmax/wsqrmax ; 

    }

    /**
     * @brief x of mu
     */
    double KOKKOS_INLINE_FUNCTION 
    x__mu(double mu) const {
      return 1./(1. + mu*bsqr) ; 
    }

    /**
     * @brief Non-magnetic momentum density square of mu, x
     */
    double KOKKOS_INLINE_FUNCTION 
    rfsqr__mu_x(double mu, double x) const {
      return x * (rsqr * x + mu * (x + 1.0) * rbsqr);  
    }
    
    /**
     * @brief Non-magnetic energy density of mu, x
     */
    double KOKKOS_INLINE_FUNCTION
    qf__mu_x(double mu, double x) const {
      double mux = mu*x ; 
      return qtot - 0.5 * (bsqr + mux*mux*brosqr) ;
    }

    /**
     * @brief Get eps
     */
    double KOKKOS_INLINE_FUNCTION
    eps_raw__mu_qf_rfsqr_w(
      double mu, double qf, double rfsqr, double w
    ) const 
    {
      return w * (qf - mu * rfsqr*(1.0 - mu * w / (1 + w)));
    }

    /**
     * @brief Get allowed eps range at rho, ye
     */
    void KOKKOS_INLINE_FUNCTION
    get_eps_range(double& epsmin, double& epsmax, double rho) const {
      double yel{ye} ;
      eos_err_t eos_err ;
      double rhol{rho} ; 
      eos.eps_range__rho_ye(epsmin,epsmax,rhol,yel,eos_err);
    }

    /**
     * @brief Obtain primitives
     */
    double KOKKOS_INLINE_FUNCTION
    compute_primitives(
      double mu, c2p_sig_t& err,
      double& x, double& w, double& rho, double& press, 
      double& eps, double& temp, double& entropy
    )
    {
      double lmu                = mu ; 
      x                  = x__mu(mu) ;
      const double rfsqr = rfsqr__mu_x(mu,x) ; 
      const double qf    = qf__mu_x(mu,x) ; 
      double vsqr        = rfsqr * mu * mu ; 
      if ( vsqr > vsqrmax ) {
        vsqr = vsqrmax ; 
        w    = wmax    ; 
        err.set(c2p_sig_enum_t::C2P_VEL_TOO_HIGH) ; 
      } else {
        w = 1/sqrt(1-vsqr) ; 
      }

      double const rhomax = eos.density_maximum();
      double const rhomin = eos.density_minimum();
      rho = d/w ; 
      if ( rho >= rhomax ) {
        err.set(c2p_sig_enum_t::C2P_RHO_TOO_HIGH) ; 
        rho = rhomax ; 
      } else if ( rho <= rhomin ) {
        err.set(c2p_sig_enum_t::C2P_RHO_TOO_LOW) ; 
        rho = rhomin ; 
      } 

      eps = eps_raw__mu_qf_rfsqr_w(mu,qf,rfsqr,w) ;
      double epsmin, epsmax ;  
      get_eps_range(epsmin,epsmax,rho) ;
      epsmax = Kokkos::fmin(epsmax, eos.get_c2p_eps_max()) ; 
      if ( eps >= epsmax ) {
        err.set(c2p_sig_enum_t::C2P_EPS_TOO_HIGH); 
        eps = epsmax * 0.999; 
      } else if ( eps < epsmin ) {
        err.set(c2p_sig_enum_t::C2P_EPS_TOO_LOW) ; 
        eps = epsmin; 
      }

      double hh,csnd2 ; 
      // rho and eps are always in bound here 
      eos_err_t eos_err ; 
      press = eos.press_h_csnd2_temp_entropy__eps_rho_ye(
        hh,csnd2,temp,entropy,eps,rho,ye,eos_err
      ) ; 

      // return the residual 
      double const a = press/(rho*(1+eps)) ; 
      double const h = (1+eps) * (1+a) ; 

      double const hbw_raw = (1+a) * (1+qf-mu*rfsqr) ; 
      double const hbw     = fmax(hbw_raw, h/w)      ; 
      double const newmu   = 1. / (hbw + rfsqr * mu) ;

      return mu - newmu;
      
    }
    
    /**
     * @brief Call to c2p residual
     */
    double KOKKOS_INLINE_FUNCTION
    operator() (double mu) 
    {
      const double lmu   = mu ; 
      const double x     = x__mu(mu) ;
      const double rfsqr = rfsqr__mu_x(mu,x) ; 
      const double qf    = qf__mu_x(mu,x) ; 
      double vsqr        = rfsqr * mu * mu ; 
      double w           = 1/sqrt(1-vsqr) ; 
      if ( vsqr > vsqrmax ) {
        vsqr = vsqrmax ; 
        w    = wmax    ; 
      } 

      double const rhomax = eos.density_maximum();
      double const rhomin = eos.density_minimum();
      double rho = Kokkos::fmin(rhomax, Kokkos::fmax(rhomin,d/w)) ; 
      

      double eps = eps_raw__mu_qf_rfsqr_w(mu,qf,rfsqr,w) ;
      double epsmin, epsmax ;  
      get_eps_range(epsmin,epsmax,rho) ;
      eps = Kokkos::fmin(epsmax, Kokkos::fmax(epsmin, eps)) ; 
      
      // rho and eps are always in bound here 
      eos_err_t eos_err ; 
      double const press = eos.press__eps_rho_ye(
        eps,rho,ye,eos_err
      ) ; 

      double const a = press/(rho*(1+eps)) ; 
      double const h = (1+eps) * (1+a) ; 

      double const hbw_raw = (1+a) * (1+qf-mu*rfsqr) ; 
      double const hbw     = fmax(hbw_raw, h/w)      ; 
      double const newmu   = 1. / (hbw + rfsqr * mu) ;

      return mu - newmu;
    }

    /************************************************/
    eos_t eos ; //!< Equation of state handle 
    /************************************************/
    double d, qtot, ye ; //!< Density, energy density, ye
    /************************************************/
    double rsqr, bsqr, rbsqr, brosqr, h0sqr; //!< Helpers
    /************************************************/
    double vsqrmax, wmax ; //!< Maximum v^2 and W
    /************************************************/
  } ; 
  /************************************************/
  template< typename eos_t >
  struct kastaun_c2p_t {

    GRACE_HOST_DEVICE
    kastaun_c2p_t(
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
      
      static constexpr double tolerance = 1e-15 ; 

      // initial bracket 
      double mu0 = 1/h0 ; 
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

      froot_t fmu(eos,D,q,r2,r_dot_Btilde2,Btilde2,ye,h0) ; 
      double mu = utils::brent(fmu, 0, mu0, tolerance) ; 

      // now compute all primitives, this call handles 
      // errors! 
      double x, w, eps, rho, press, temp, entropy ; 
      double residual = fmu.compute_primitives(
        mu, c2p_errors, x, w, rho, press, eps, temp, entropy
      ) ; 

      prims[EPSL]   = eps       ; 
      prims[RHOL]   = rho       ; 
      prims[PRESSL] = press     ;
      prims[TEMPL]  = temp      ; 
      prims[ENTL]   = entropy   ;
      prims[BXL]    = B[0]      ; 
      prims[BYL]    = B[1]      ; 
      prims[BZL]    = B[2]      ; 


      for( int ii=0; ii<3; ++ii) 
        prims[ZXL+ii] = w * mu * x * ( r[ii] + mu * r_dot_Btilde * Btilde[ii] ) ;  
      
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
