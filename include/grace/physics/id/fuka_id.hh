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
        
        atmo_params = get_atmo_params() ;
        zero_shift = get_param<bool>("grmhd","fuka","set_shift_to_zero") ; 

        GRACE_VERBOSE("Setting FUKA initial data.") ; 

        GRACE_VERBOSE("Initial data type is: {}.", id_type ) ;
        GRACE_VERBOSE("Directory: {}.",id_dir) ;
        GRACE_VERBOSE("Filename: {}.",fname) ;
        
        auto& aux   = variable_list::get().getaux() ; 
        auto& state = variable_list::get().getstate() ;
        auto& idx   = grace::variable_list::get().getinvspacings()   ; 

        auto& coord_system = grace::coordinate_system::get() ; 

        const bool has_matter = (id_type=="NS" || id_type=="BNS" || id_type=="BHNS");
        int64_t const nfields= has_matter? 4+6+6+4 : 4+6+6 ;

        _data  = vview_t("data_fuka", nfields, nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        
        int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
        
        

        std::vector<double> xx(ncells), yy(ncells), zz(ncells) ; 

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
        
            xx[icell] = pcoords[0]; 
            yy[icell] = pcoords[1]; 
            zz[icell] = pcoords[2]; 
        }
    
        // the import happens on host; send off the packed references to the kadath exporters
        KadathImporter(id_type, id_dir+"/"+fname, 
                        xx,yy,zz,_data, nfields, ncells,nx,ny,nz,ngz);

    }

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE GRACE_DEVICE_EXTERNAL_LINKAGE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ;     
        eos_err_t eos_err ;
        bool reset_eps{false} ; // fixme 
        double e = _data(16,i,j,k,q) ; 
        id.rho = _eos.rho__energy_cold_impl(e, eos_err) ; 
        id.eps = e/id.rho - 1. ; 

        auto rho_atm = atmo_params.rho_fl ; 
        auto ye_atm = atmo_params.ye_fl ; 
        auto temp_atm = atmo_params.temp_fl ; 

        if ( id.rho < (1.+1e-3) * rho_atm || !Kokkos::isfinite(id.rho)) {
            id.rho   = rho_atm ; 
            id.ye    = ye_atm   ; 
            id.temp  = temp_atm ; 
            // get the rest 
            double dummy ;
            id.press = _eos.press_eps_csnd2_entropy__temp_rho_ye(
                id.eps, dummy, id.entropy, id.temp, id.rho, id.ye, eos_err
            ) ; 
            // set velocities 
            id.vx = id.vy = id.vz = 0.0 ;
        } else {
            // get ye at beta eq
            id.ye = _eos.ye_cold__rho(id.rho, eos_err) ;
            // get velocities 
            id.vx =  _data(17,VEC(i,j,k),q) ;  
            id.vy =  _data(18,VEC(i,j,k),q) ; 
            id.vz =  _data(19,VEC(i,j,k),q) ; 
            if (reset_eps) {
                // assume "zero" temperature
                // for ideal gas t_atmo **must** be 
                // K rho_atmo^(Gamma-1) with K from the
                // ID for this to be self-consistent.
                double h, csnd2 ;
                id.temp = temp_atm ; 
                id.press = _eos.press_eps_csnd2_entropy__temp_rho_ye_impl(
                    id.eps, csnd2, id.entropy, id.temp, id.rho, id.ye, eos_err
                ) ; 
            } else {
                // get pressure and the rest assuming eps from
                // the ID is good enough
                double h, csnd2 ;
                id.press = _eos.press_h_csnd2_temp_entropy__eps_rho_ye(
                    h, csnd2, id.temp, id.entropy, id.eps, id.rho, id.ye, eos_err
                ) ; 
            }
        }

        // B field is set elsewhere    
        id.bx = id.by = id.bz = 0.0 ; 

        // metric 
        id.alp =  _data(0,VEC(i,j,k),q) ; 

        if ( zero_shift ) {
            id.betax =  id.betay = id.betaz = 0;
        } else {
            id.betax =  _data(1,VEC(i,j,k),q);
            id.betay =  _data(2,VEC(i,j,k),q);
            id.betaz =  _data(3,VEC(i,j,k),q);
        }

        id.gxx = _data(4,VEC(i,j,k),q) ; 
        id.gxy = _data(5,VEC(i,j,k),q) ; 
        id.gxz = _data(6,VEC(i,j,k),q) ; 
        id.gyy = _data(7,VEC(i,j,k),q) ; 
        id.gyz = _data(8,VEC(i,j,k),q) ; 
        id.gzz = _data(9,VEC(i,j,k),q) ;
        
        id.kxx = _data(10,VEC(i,j,k),q) ; 
        id.kxy = _data(11,VEC(i,j,k),q)  ; 
        id.kxz = _data(12,VEC(i,j,k),q)  ; 
        id.kyy = _data(13,VEC(i,j,k),q)  ; 
        id.kyz = _data(14,VEC(i,j,k),q)  ; 
        id.kzz = _data(15,VEC(i,j,k),q)  ;
        
        return id ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    atmo_params_t atmo_params                      ;  //!< Atmosphere properties 

    vview_t _data ; 

    bool zero_shift ; 

} ; 

}

#endif /* GRACE_PHYSICS_ID_FUKA_HH */
