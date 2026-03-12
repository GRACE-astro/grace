#ifndef GRACE_ID_LINEAR_GW_HH
#define GRACE_ID_LINEAR_GW_HH

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
struct linear_gw_id_t {
    using state_t = grace::var_array_t ; 
    
    linear_gw_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords )
        : _eos(eos)
        , _pcoords(pcoords)
    {
    } 

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
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

        double x = _pcoords(i,j,k,0,q) ; 
        double const A = 1e-8 ; 
        double const Hs = A * sin(2*M_PI*x) ; 
        double const Hc = A * M_PI * cos(2*M_PI*x) ; 

        id.alp = 1.0 ; 
        id.betax = 0; id.betay=0; id.betaz = 0; 

        id.gxx = 1 - Hs; 
        id.gyy = 1 + Hs; 
        id.gzz = 1;
        id.gxy = 0; id.gxz = 0; id.gyz = 0 ;

        id.kxx = 1 + Hc; 
        id.kyy = 1 + Hc; 
        id.kzz = 0 ;

        id.kxy = 0 ;  
        id.kxz = 0 ; 
        id.kyz = 0 ; 


        return std::move(id) ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
} ; 


}

#endif /* GRACE_ID_LINEAR_GW_HH */