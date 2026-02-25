/**
 * @file fuka_id.hh
 * @author Konrad Topolski (konrad.topolski@uni-hamburg.de)
 * @brief 
 * @date 2025-02-17
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
#ifndef GRACE_PHYSICS_ID_FUKA_HH
#define GRACE_PHYSICS_ID_FUKA_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/utils/rootfinding.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/variable_indices.hh>

/* KADATH includes - match library's API */
#include <grace/physics/id/import_kadath.hh> 

namespace grace {


template < typename eos_t >
struct fuka_id_t {
    using state_t = grace::var_array_t ; 
    using sview_t = typename Kokkos::View<double ****, grace::default_space> ; 
    using vview_t = typename Kokkos::View<double *****, grace::default_space> ; 

    fuka_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , std::string const& id_type
        , std::string const& id_dir
        , std::string const& fname 
    ) : _pcoords(pcoords), _eos(eos)
    {
        DECLARE_GRID_EXTENTS;
        using namespace grace ;
        using namespace Kokkos ; 
        
        GRACE_VERBOSE("Setting FUKA initial data.") ; 

        GRACE_VERBOSE("Initial data type is: {}.", id_type ) ;
        GRACE_VERBOSE("Directory: {}.",id_dir) ;
        GRACE_VERBOSE("Filename: {}.",fname) ;
        
        auto& aux   = variable_list::get().getaux() ; 
        auto& state = variable_list::get().getstate() ;
        auto& idx   = grace::variable_list::get().getinvspacings()   ; 

        auto& coord_system = grace::coordinate_system::get() ; 
  
        _rho   = sview_t("rho_fuka", nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _eps   = sview_t("eps_fuka", nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _press = sview_t("press_fuka", nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _vel   = vview_t("vel_fuka", 3,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 

        _alp   = sview_t("lapse_fuka", nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _beta  = vview_t("shift_fuka", 3,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _g     = vview_t("metric_fuka", 6,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _k     = vview_t("ext_curv_fuka", 6,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ;

        auto _hrho = Kokkos::create_mirror_view(_rho) ; 
        auto _heps = Kokkos::create_mirror_view(_eps) ; 
        auto _hpress = Kokkos::create_mirror_view(_press) ; 
        auto _hvel = Kokkos::create_mirror_view(_vel) ; 

        auto _halp = Kokkos::create_mirror_view(_alp) ; 
        auto _hbeta = Kokkos::create_mirror_view(_beta) ; 
        auto _hg = Kokkos::create_mirror_view(_g) ; 
        auto _hk = Kokkos::create_mirror_view(_k) ; 
        
        int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
        std::vector<std::array<double, GRACE_NSPACEDIM>> cells_pcoords;
        const bool has_matter = (id_type=="NS" || id_type=="BNS" || id_type=="BHNS");

        int64_t const nfields= has_matter? 4+6+6+6 : 4+6+6 ;

        std::vector<std::reference_wrapper<double>> local_vars; // holds references to (nfields x npoints) values
        local_vars.reserve(nfields*ncells); 

        for( int64_t icell=0; icell<ncells; ++icell) {
            size_t const i = icell%(nx+2*ngz); 
            size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
            #ifdef GRACE_3D 
            size_t const k = 
                (icell/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ; 
            size_t const q = 
                (icell/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ;
            #else 
            size_t const q = (icell/(nx+2*ngz)/(ny+2*ngz)) ; 
            #endif 
            /* Physical coordinates of cell center */
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                true
            ) ; 
        
            cells_pcoords.push_back(pcoords); 

            // The ordering here follows Kadath enums convention:
            // ALP = 0 
            // BETAX = 1

            // access:
            // local_vars[icell*nfields+K_MAT::FIELD_NUM] 

            local_vars.push_back(std::ref(_halp(i,j,k,q)));
            local_vars.push_back(std::ref(_hbeta(0,i,j,k,q)));
            local_vars.push_back(std::ref(_hbeta(1,i,j,k,q)));
            local_vars.push_back(std::ref(_hbeta(2,i,j,k,q)));

            local_vars.push_back(std::ref(_hg(0,i,j,k,q)));
            local_vars.push_back(std::ref(_hg(1,i,j,k,q)));
            local_vars.push_back(std::ref(_hg(2,i,j,k,q)));
            local_vars.push_back(std::ref(_hg(3,i,j,k,q)));
            local_vars.push_back(std::ref(_hg(4,i,j,k,q)));
            local_vars.push_back(std::ref(_hg(5,i,j,k,q)));

            local_vars.push_back(std::ref(_hk(0,i,j,k,q)));
            local_vars.push_back(std::ref(_hk(1,i,j,k,q)));
            local_vars.push_back(std::ref(_hk(2,i,j,k,q)));
            local_vars.push_back(std::ref(_hk(3,i,j,k,q)));
            local_vars.push_back(std::ref(_hk(4,i,j,k,q)));
            local_vars.push_back(std::ref(_hk(5,i,j,k,q)));

            // MATTER:
            if(has_matter){
                local_vars.push_back(std::ref(_hrho(i,j,k,q)));
                local_vars.push_back(std::ref(_heps(i,j,k,q)));
                local_vars.push_back(std::ref(_hpress(i,j,k,q)));
                local_vars.push_back(std::ref(_hvel(0,i,j,k,q)));
                local_vars.push_back(std::ref(_hvel(1,i,j,k,q)));
                local_vars.push_back(std::ref(_hvel(2,i,j,k,q)));
            }

        }
    
        // the import happens on host; send off the packed references to the kadath exporters
        KadathImporter(id_type, id_dir+"/"+fname, local_vars, 
                                    cells_pcoords, nfields, ncells);

        // reuse device views  
        Kokkos::deep_copy(_rho  , _hrho  );
        Kokkos::deep_copy(_eps  , _heps  );
        Kokkos::deep_copy(_press, _hpress  );
        Kokkos::deep_copy(_vel  , _hvel  );
        Kokkos::deep_copy(_alp  , _halp  );
        Kokkos::deep_copy(_beta , _hbeta  );
        Kokkos::deep_copy(_g    , _hg  );
        Kokkos::deep_copy(_k    , _hk  );

    }

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 
        
        
        double const rho_atm{1e-14} ; 
        unsigned int eos_err ;

        double rho = _rho(VEC(i,j,k),q);

        if ( rho < (1.+1e-3) * rho_atm || !Kokkos::isfinite(rho)) {
            id.rho = rho_atm ; 
            // get ye at beta eq
            id.ye = _eos.ye_beta_eq__rho_cold(id.rho, eos_err) ;
            // get pressure from EOS
            id.press = _eos.press_cold__rho_ye(id.rho,id.ye,eos_err) ; 
            // set velocities 
            id.vx = id.vy = id.vz = 0.0 ;
        } else {
            id.rho = rho ; 
            // get ye at beta eq
            id.ye = _eos.ye_beta_eq__rho_cold(id.rho, eos_err) ;
            // get pressure from EOS
            id.press = _eos.press_cold__rho_ye(id.rho,id.ye,eos_err) ;
            // alternatively, we could just import:
            // id.press = _press(VEC(i,j,k),q));
            // set velocities 
            id.vx =  _vel(0,VEC(i,j,k),q) ;  
            id.vy =  _vel(1,VEC(i,j,k),q) ; 
            id.vz =  _vel(2,VEC(i,j,k),q) ; 
        }

        // is this needed?

       
        
             
        // B field is set elsewhere    
        id.bx = id.by = id.bz = 0.0 ; 

        // metric 
        id.alp =  _alp(VEC(i,j,k),q) ; 

        id.betax =  _beta(0,VEC(i,j,k),q);
        id.betay =  _beta(1,VEC(i,j,k),q);
        id.betaz =  _beta(2,VEC(i,j,k),q);

        id.gxx = _g(0,VEC(i,j,k),q) ; 
        id.gxy = _g(1,VEC(i,j,k),q) ; 
        id.gxz = _g(2,VEC(i,j,k),q) ; 
        id.gyy = _g(3,VEC(i,j,k),q) ; 
        id.gyz = _g(4,VEC(i,j,k),q) ; 
        id.gzz = _g(5,VEC(i,j,k),q) ;
        
        id.kxx = _k(0,VEC(i,j,k),q) ; 
        id.kxy = _k(1,VEC(i,j,k),q)  ; 
        id.kxz = _k(2,VEC(i,j,k),q)  ; 
        id.kyy = _k(3,VEC(i,j,k),q)  ; 
        id.kyz = _k(4,VEC(i,j,k),q)  ; 
        id.kzz = _k(5,VEC(i,j,k),q)  ;
        
        return id ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers

    
    sview_t _rho, _eps, _press; 
    sview_t _alp ; 
    vview_t _vel, _g, _k, _beta ; 

} ; 

}

#endif /* GRACE_PHYSICS_ID_FUKA_HH */