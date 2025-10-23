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

#define SQR(a) (a)*(a)

namespace grace {

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

      r2 = metric.square_covec(r) ;
      Btilde2 = metric.square_vec(Btilde);
      r_dot_Btilde2 = SQR(r[0]*Btilde[0] + r[1]*Btilde[1] + r[2]*Btilde[2]) ; 
      Brorth2 = Btilde2 * r2 - r_dot_Btilde2 ;

      r = metric.raise(r) ; 
      
      ye = conservs[YESL] / D ;

      v02 = fmin(1.0-1e-5, r2 / (h0*h0 + r2 )) ;
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
    invert(grmhd_prims_array_t& prims) {

      prims[YEL] = ye ;
      prims[BXL] = Btilde[0] * sqrt(D) ;
      prims[BYL] = Btilde[1] * sqrt(D) ;
      prims[BZL] = Btilde[2] * sqrt(D) ;

      static constexpr double tolerance = 1e-15 ; 
      auto const fa = [this] (double mu) { return this->fa__mu(mu) ; };
      double mu0 = sqrt(r2) < h0 ? 1./h0 : 1e-12 + utils::brent(fa, 0, 1./h0, tolerance) ;

      auto const fmu = [this] (double mu) {return this->f__mu(mu);};
      double mu = utils::brent(fmu, 0, mu0, tolerance) ; 

      double What = What__mu(mu) ; 
      double eps = epshat__mu(mu) ; 
      double rhohat = D/What ; 
      double yel{ye} ;
      double epsmin,epsmax;
      unsigned int err ;
      eos.eps_range__rho_ye(epsmin,epsmax,rhohat,yel,err);
      eps = fmax(epsmin,fmin(epsmax,eps)) ;
      
      prims[RHOL] = rhohat ;
      prims[EPSL] = eps ;

      auto chi = chi__mu(mu) ;
      for( int ii=0; ii<3; ++ii) prims[VXL+ii] = mu * chi * ( r[ii] + mu * sqrt(r_dot_Btilde2)*Btilde[ii] ) ;  
            
      return f__mu(mu) ; 
    }
    
  private:
    eos_t eos ;
    metric_array_t metric ;
    double r2, q, Btilde2, Brorth2, D, ye, r_dot_Btilde2, h0, v02 ;
    std::array<double,3> r, Btilde ;

    double KOKKOS_INLINE_FUNCTION
    chi__mu(double mu) const {
      return 1./(1.+mu*Btilde2) ; 
    }
    double KOKKOS_INLINE_FUNCTION
    rbar2__mu(double mu) const {
      double chi = chi__mu(mu) ;
      return r2 * SQR(chi) + mu * chi * ( 1.0 + chi) * r_dot_Btilde2 ; 
    }
    
    double KOKKOS_INLINE_FUNCTION
    fa__mu(double mu) const {
      return mu * sqrt(h0*h0 + rbar2__mu(mu)) - 1. ; 
    }

    double KOKKOS_INLINE_FUNCTION
    qbar__mu(double mu) const {
      return q - 0.5 * Btilde2 - 0.5 * SQR(mu) * SQR(chi__mu(mu)) * Brorth2 ; 
    }

    double KOKKOS_INLINE_FUNCTION
    vhat2__mu(double mu) const {
      return fmin(v02, mu * mu * rbar2__mu(mu)) ; 
    }

    double KOKKOS_INLINE_FUNCTION
    What__mu(double mu) const {
      return 1. / sqrt(1. - vhat2__mu(mu)) ; 
    }

    double KOKKOS_INLINE_FUNCTION
    epshat__mu(double mu) const {
      double const qbar = qbar__mu(mu) ;
      double const rbar2 = rbar2__mu(mu) ;
      double const vhat2 = vhat2__mu(mu) ;
      double const What = What__mu(mu) ;
       
      return What * ( qbar - mu * rbar2 ) + vhat2 * SQR(What)/(1+What) ; 
    }

    double KOKKOS_INLINE_FUNCTION
    rhohat__mu(double mu) const {
      return D / What__mu(mu) ; 
    }
    
    double KOKKOS_INLINE_FUNCTION
    phat__mu(double rhohat, double epshat) const {
      double yehat{ye} ; 
      unsigned int err ;      
      auto press = eos.press__eps_rho_ye(epshat,rhohat,yehat,err) ;
      return press ; 
    }
    
    double KOKKOS_INLINE_FUNCTION
    ahat__mu(double mu) const {
      double rhohat = rhohat__mu(mu) ;
      double epshat = epshat__mu(mu) ;
      double yel{ye} ;
      double epsmin,epsmax; 
      unsigned int err ; 
      eos.eps_range__rho_ye(epsmin,epsmax,rhohat,yel,err); 
      epshat = fmax(epsmin,fmin(epsmax,epshat)) ; 
      double phat = phat__mu(rhohat,epshat) ;
      return phat / (rhohat * (1+epshat) ) ; 
    }

    double KOKKOS_INLINE_FUNCTION
    nuA__mu(double mu) const {
      double const What = What__mu(mu) ;
      double const epshat = epshat__mu(mu) ;
      double const ahat = ahat__mu(mu) ;
      return (1.+ahat) * ( 1 + epshat ) / What ; 
    }

    double KOKKOS_INLINE_FUNCTION
    nuB__mu(double mu) const {
      double const qbar = qbar__mu(mu) ; 
      double const ahat = ahat__mu(mu) ;
      double const rbar2 = rbar2__mu(mu) ; 
      return (1.+ahat) * ( 1 + qbar - mu * rbar2 ); 
    }

    double KOKKOS_INLINE_FUNCTION
    nuhat__mu(double mu) const {
      return fmax(nuA__mu(mu),nuB__mu(mu)); 
    }

    double KOKKOS_INLINE_FUNCTION
    f__mu(double mu) const {
      double nuhat = nuhat__mu(mu) ;
      double const rbar2 = rbar2__mu(mu) ; 
      return mu - 1./(nuhat + mu * rbar2) ; 
    }
    
  };
    

} /* namespace grace */
#undef SQR
#endif /*GRACE_C2P_KASTAUN_MHD_HH*/
