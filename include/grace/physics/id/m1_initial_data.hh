/**
 * @file m1_initial_data.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-24
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

#ifndef GRACE_PHYSICS_ID_M1_HH
#define GRACE_PHYSICS_ID_M1_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/coordinates/coordinate_systems.hh>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

namespace grace {

struct m1_id_t {
    double erad1, nrad1, fradx1, frady1, fradz1 ; //! lower indices
    #ifdef M1_NU_THREESPECIES
    double erad2, nrad2, fradx2, frady2, fradz2 ;
    double erad3, nrad3, fradx3, frady3, fradz3 ;
    #endif
    #ifdef M1_NU_FIVESPECIES
    double erad4, nrad4, fradx4, frady4, fradz4 ;
    double erad5, nrad5, fradx5, frady5, fradz5 ;
    #endif
} ; 

struct zero_m1_id_t {
    zero_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords)
    {}

    // NB this assumes all optically thin! 
    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        /* we assume coords are spherical here! */
        double rtp[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 

        auto E_atmo = atmo.E_fl * Kokkos::pow(rtp[0], atmo.E_fl_scaling) ; 
        auto eps_atmo = atmo.eps_fl * Kokkos::pow(rtp[0], atmo.eps_fl_scaling) ; 

        bool excise = excision.excise_by_radius ? rtp[0] <= excision.r_ex : false ; /*we don't have alp here*/
        
        if ( excise ) {
            id.erad1 = excision.E_ex ; 
            id.nrad1 = excision.E_ex / excision.eps_ex ; 
            #ifdef M1_NU_THREESPECIES
            id.erad2 = excision.E_ex ; 
            id.nrad2 = excision.E_ex / excision.eps_ex ; 
            id.erad3 = excision.E_ex ; 
            id.nrad3 = excision.E_ex / excision.eps_ex ; 
            #endif
            #ifdef M1_NU_FIVESPECIES
            id.erad4 = excision.E_ex ; 
            id.nrad4 = excision.E_ex / excision.eps_ex ; 
            id.erad5 = excision.E_ex ; 
            id.nrad5 = excision.E_ex / excision.eps_ex ; 
            #endif
        } else {
            id.erad1 = E_atmo ;
            id.nrad1 = E_atmo / eps_atmo ; 
            #ifdef M1_NU_THREESPECIES
            id.erad2 = E_atmo ;
            id.nrad2 = E_atmo / eps_atmo ; 
            id.erad3 = E_atmo ;
            id.nrad3 = E_atmo / eps_atmo ; 
            #endif
            #ifdef M1_NU_FIVESPECIES
            id.erad4 = E_atmo ;
            id.nrad4 = E_atmo / eps_atmo ; 
            id.erad5 = E_atmo ;
            id.nrad5 = E_atmo / eps_atmo ; 
            #endif
        }
        id.fradx1 = id.frady1 = id.fradz1 = 0. ; 
        #ifdef M1_NU_THREESPECIES
        id.fradx2 = id.frady2 = id.fradz2 = 0. ; 
        id.fradx3 = id.frady3 = id.fradz3 = 0. ; 
        #endif
        #ifdef M1_NU_FIVESPECIES
        id.fradx4 = id.frady4 = id.fradz4 = 0. ; 
        id.fradx5 = id.frady5 = id.fradz5 = 0. ; 
        #endif
        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
} ; 

struct equil_m1_id_t {
    equil_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        var_array_t _aux,
        scalar_array_t<GRACE_NSPACEDIM> const _dx,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), aux(_aux), dx(_dx), pcoords(_pcoords)
    {}

    // NB this assumes all optically thin! 
    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        /* we assume coords are spherical here! */
        double rtp[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 
        /* Get eas to check regime */
        m1_eas_array_t eas ; 
        eas[KAL] = aux(VEC(i,j,k),KAPPAA1_,q) ; 
        eas[KSL] = aux(VEC(i,j,k),KAPPAS1_,q) ; 
        eas[ETAL] = aux(VEC(i,j,k),ETA1_,q) ; 

        // this is l / dx --> small == thick
        double A = fmin(1.,1./((eas[KAL]+eas[KSL]+1e-20)*dx(0,q))) ; 

        if ( A < 2./3. ) {
            // here we set variables to their 
            // equilibrium value 
        } else if ( A >= 1. ) {

        } else {

        }

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    var_array_t aux; 
    scalar_array_t<GRACE_NSPACEDIM> dx;
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
} ;

struct straight_beam_m1_id_t {
    straight_beam_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 
        
        id.erad1 = atmo.E_fl ;
        id.fradx1 = id.frady1 = id.fradz1 = 0. ; 

        if ( xyz[0] <= -0.25 and 
            xyz[1] < 0.0625 and xyz[1] > - 0.0625 and 
            xyz[2] < 0.0625 and xyz[2] > - 0.0625) {
            id.erad1 = id.fradx1 = 1.0 ; 
        }

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
} ;

struct scattering_diffusion_m1_id_t {
    scattering_diffusion_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords,
        double _ks, double _t0
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords), ks(_ks), t0(_t0)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 
        double r2 = SQR(xyz[0])+SQR(xyz[1])+SQR(xyz[2]);
        double r = sqrt(r2) ; 
        id.erad1 = Kokkos::pow(ks/t0,3./2.) * Kokkos::exp(-3*ks*r2/(4.*t0)) ; 

        double Hr = r/(2.*t0) * id.erad1 ; 

        id.fradx1 = xyz[0]/r * Hr ; 
        id.frady1 = xyz[1]/r * Hr ; 
        id.fradz1 = xyz[2]/r * Hr ; 

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    double ks, t0;
} ;

struct moving_scattering_diffusion_m1_id_t {
    moving_scattering_diffusion_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords,
        double _v0
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords), v0(_v0)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 

        id.erad1 = Kokkos::exp(-9.0*SQR(xyz[0])) ; 

        double const W2 = 1./(1-SQR(v0)) ; 
        double J = 3.*id.erad1  / (4.*W2-1.); 

        id.fradx1 = 4./3. * J * W2 * v0 ; 
        id.frady1 = id.fradz1 = 0. ;  

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    double v0;
} ;


struct emitting_sphere_m1_id_t {
    emitting_sphere_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 

        double r2 = SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2]) ; 
        double r = sqrt(r2) ; 

        if ( r < 1. ) {
            id.erad1 = 1. ; 
            id.fradx1=id.frady1=id.fradz1 = 0 ; 
        } else {
            id.erad1 = 1/r2 ;
            id.fradx1 = 0.5/r2 * xyz[0]/r ; 
            id.frady1 = 0.5/r2 * xyz[1]/r ; 
            id.fradz1 = 0.5/r2 * xyz[2]/r ; 
        }

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
} ;

struct curved_beam_m1_id_t {
    curved_beam_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords,
        var_array_t _state
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords), state(_state)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 
        
        id.erad1 = atmo.E_fl ; id.nrad1 = atmo.E_fl / atmo.eps_fl ; 
        id.fradx1 = id.frady1 = id.fradz1 = 0. ;  

        if ( xyz[0] <= 0.015625 and 
            xyz[1] < 0.25 and xyz[1] > - 0.25 and 
            xyz[2] <= 3.5 and xyz[2] >= 3.0 ) {
            id.erad1 = 1.0 ; 
            // F_i F^i = E * E  
            metric_array_t metric ; 
            FILL_METRIC_ARRAY(metric,this->state,q,i,j,k) ; 
            double FY = metric.beta(1) * id.erad1 / metric.alp() ; 
            double FZ = metric.beta(2) * id.erad1 / metric.alp() ; 

            double const a = metric.gamma(0) ; 
            double const b = 2 * FY * metric.gamma(1) + 2 * FZ * metric.gamma(2) ; 
            double const c = - 0.9999 * SQR(id.erad1) + SQR(FY) * metric.gamma(3) + SQR(FZ) * metric.gamma(5) + 2. * FY * FZ * metric.gamma(4) ; 

            double const FX = (-b + sqrt(SQR(b)-4.*a*c))/(2.*a) ; 
            auto Fd = metric.lower({FX,FY,FZ}) ; 
            id.fradx1 = Fd[0] ; id.frady1 = Fd[1] ; id.fradz1 = Fd[2] ; 
        }

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    var_array_t state ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
} ;

} /* namespace grace */
#endif /*GRACE_PHYSICS_ID_M1_HH*/
