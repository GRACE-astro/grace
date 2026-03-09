#ifndef GRACE_ID_ROBUST_STABILITY_HH
#define GRACE_ID_ROBUST_STABILITY_HH

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
struct robust_stability_id_t {
    using state_t = grace::var_array_t ; 
    
    robust_stability_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords )
        : _eos(eos)
        , _pcoords(pcoords)
    {
        _rho = get_param<int>("grmhd", "robust_stability_rho") ; 
        random_pool = Kokkos::Random_XorShift64_Pool<>(/*seed=*/12345);
    } 


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

        auto generator = random_pool.get_state();

        auto const eps = [&]() -> double {
            return generator.drand(-1e-10/_rho/_rho, 1e-10/_rho/_rho) ; 
        } ; 


        id.alp = 1 + eps() ;  
        id.betax = eps(); id.betay=eps(); id.betaz = eps(); 

        id.gxx = 1 + eps(); 
        id.gyy = 1 + eps(); 
        id.gzz = 1 + eps();
        id.gxy = eps(); 
        id.gxz = eps(); 
        id.gyz = eps();

        id.kxx = eps(); 
        id.kyy = eps(); 
        id.kzz = eps(); 

        id.kxy = eps(); 
        id.kxz = eps(); 
        id.kyz = eps(); 

        random_pool.free_state(generator);

        return std::move(id) ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    int _rho ; 
    Kokkos::Random_XorShift64_Pool<> random_pool ; 
} ; 


}

#endif /* GRACE_ID_LINEAR_GW_HH */