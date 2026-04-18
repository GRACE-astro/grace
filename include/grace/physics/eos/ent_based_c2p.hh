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
    ) : eos(_eos), d(_d), s(_s), ye(_ye), rsqr(_rsqr), bsqr(_bsqr), rbsqr(_rbsqr)
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
      const double x     = x__mu(mu) ;
      const double rfsqr = rfsqr__mu_x(mu,x) ; 
      double vsqr        = rfsqr * mu * mu ; 
      double w           = 1/sqrt(1-vsqr) ; 
      if ( vsqr > vsqrmax ) {
        vsqr = vsqrmax ; 
        w    = wmax    ; 
      } 

      double const rhomax = eos.density_maximum();
      double const rhomin = eos.density_minimum();
      double rho          = Kokkos::fmin(rhomax,Kokkos::fmax(rhomin,d/w)) ; 

      double hh,csnd2,temp,eps ; 
      eos_err_t eos_err ; 
      double press = eos.press_h_csnd2_temp_eps__entropy_rho_ye(
        hh,csnd2,temp,eps,s,rho,ye,eos_err
      ) ; 

      double const a = press/(rho*(1+eps)) ; 
      double const h = (1+eps) * (1+a) ; 

      double const hbw     = h/w      ; 
      double const newmu   = 1. / (hbw + rfsqr * mu) ;

      return mu - newmu;
    }

    double KOKKOS_INLINE_FUNCTION
    compute_primitives (
      double mu, c2p_sig_t& err,
      double& x, double& w, double& rho, double& press, 
      double& eps, double& temp, double& entropy
    )
    {

      entropy = s ;

      x                  = x__mu(mu) ;
      const double rfsqr = rfsqr__mu_x(mu,x) ; 
      double vsqr        = rfsqr * mu * mu ; 
      w = 1/sqrt(1-vsqr) ; 
      if ( vsqr > vsqrmax ) {
        vsqr = vsqrmax ; 
        w    = wmax    ; 
        err.set(c2p_sig_enum_t::C2P_VEL_TOO_HIGH) ; 
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

      double hh,csnd2 ; 
      eos_err_t eos_err ; 
      press = eos.press_h_csnd2_temp_eps__entropy_rho_ye(
        hh,csnd2,temp,eps,entropy,rho,ye,eos_err
      ) ; 
      // handle errors, rho always in bound, need to 
      // check entropy and ye 
      if (eos_err.test(EOS_ERROR_T::EOS_ENTROPY_TOO_LOW)) {
        err.set(C2P_ENT_TOO_LOW) ;
      }   
      if (eos_err.test(EOS_ERROR_T::EOS_ENTROPY_TOO_HIGH)) {
        err.set(C2P_ENT_TOO_HIGH) ; 
      }

      double const a = press/(rho*(1+eps)) ; 
      double const h = (1+eps) * (1+a) ; 

      double const hbw     = h/w      ; 
      double const newmu   = 1. / (hbw + rfsqr * mu) ;

      return mu - newmu;
    }


    eos_t eos ; 

    double d, s, ye ; 

    double rsqr, bsqr, rbsqr ; 

    double vsqrmax, wmax ; 

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
      
      static constexpr double tolerance = 1e-15 ; 

      // initial bracket 
      double mu0 = 1/h0 ; 
      if ( r2 >= h0*h0 ) {
        fbrack_t g(r2,Btilde2,r_dot_Btilde2,h0) ; 
        int rerr ; 
        mu0 = utils::rootfind_newton_raphson(0,1./h0,g,30,1e-10,rerr) ; 
        if ( rerr == 0 ) {
          mu0 *= 1+1e-10 ; 
        } else {
          mu0 = utils::brent(g,0,1./h0,tolerance)*(1+1e-10) ;
        }
      }

      ent_froot_t fmu(eos,D,r2,r_dot_Btilde2,Btilde2,s,ye,h0) ; 
      double mu = utils::brent(fmu, 0, mu0, tolerance) ; 

      double x, w, eps, rho, press, temp, entropy ; 
      double residual = fmu.compute_primitives(
        mu, err, x, w, rho, press, eps, temp, entropy
      ) ;

      prims[EPSL]   = eps   ; 
      prims[RHOL]   = rho   ; 
      prims[PRESSL] = press ;
      prims[TEMPL]  = temp  ; 
      prims[ENTL]   = entropy ; // we set it here in case it was clamped 
      prims[BXL]    = B[0] ; 
      prims[BYL]    = B[1] ; 
      prims[BZL]    = B[2] ; 

      for( int ii=0; ii<3; ++ii) 
        prims[ZXL+ii] = w * mu * x * ( r[ii] + mu * r_dot_Btilde * Btilde[ii] ) ;  
      
      return fabs(residual) ; 
    }
    
  private:
    eos_t eos ;
    metric_array_t metric ;
    double r2, s, Btilde2, D, ye, r_dot_Btilde, r_dot_Btilde2, h0, v02 ;
    std::array<double,3> r, Btilde, B ;
  };
    

} /* namespace grace */
#endif /*GRACE_C2P_KASTAUN_MHD_HH*/
