#ifndef GRACE_ID_PUNCTURE_HH
#define GRACE_ID_PUNCTURE_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

namespace grace {


template < typename eos_t >
struct puncture_id_t {
    using state_t = grace::var_array_t ; 
    
    puncture_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double m )
        : _eos(eos)
        , _pcoords(pcoords), _m(m)
    {} 

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE GRACE_DEVICE_EXTERNAL_LINKAGE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 
        id.rho     = 0.0 ; 
        id.press   = 0.0   ;
        id.eps = 0.0 ; 
        id.temp = 0.0 ;
        id.entropy = 0.0 ; 
        id.bx = 0.0 ;
        id.by = 0.0 ;
        id.bz = 0.0 ;
        id.vx = 0; id.vy = 0.; id.vz = 0.;
        id.ye = 0;
        double const r = sqrt(fmax(
            SQR(_pcoords(i,j,k,0,q)) + SQR(_pcoords(i,j,k,1,q)) +  SQR(_pcoords(i,j,k,2,q)),
            1e-12
        )) ;

        id.betax = 0; id.betay=0; id.betaz = 0; 

        double const psi = (1. + 0.5 * _m/r) ; 
        double const psim2 = 1/(psi*psi) ; 
        double const psi4 = SQR(psi)*SQR(psi) ;

        id.alp = psim2 ; 

        id.gxx = psi4; id.gyy = psi4; id.gzz = psi4;
        id.gxy = 0; id.gxz = 0; id.gyz = 0 ;
        id.kxx = 0; id.kyy = 0; id.kzz = 0 ;
        id.kxy = 0; id.kxz =0 ; id.kyz = 0 ; 

        return std::move(id) ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _m;          //!< Left and right states  
} ; 


}

#endif 