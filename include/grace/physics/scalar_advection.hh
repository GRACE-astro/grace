/**
 * @file scalar_advection.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-13
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#ifndef GRACE_PHYSICS_SCALAR_ADVECTION_EQUATION_HH
#define GRACE_PHYSICS_SCALAR_ADVECTION_EQUATION_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>

#include <grace/evolution/hrsc_evolution_system.hh> 

#include <grace/data_structures/variable_indices.hh>
#include <grace/evolution/evolution_kernel_tags.hh>

#include <Kokkos_Core.hpp>

namespace grace {


struct scalar_advection_system_t 
    : public hrsc_evolution_system_t<scalar_advection_system_t>
{
    scalar_advection_system_t( grace::var_array_t<GRACE_NSPACEDIM> state_
                             , grace::var_array_t<GRACE_NSPACEDIM> aux_ 
                             , VEC( double ax_ 
                                  , double ay_ 
                                  , double az_))
     : hrsc_evolution_system_t<scalar_advection_system_t>(state_,aux_)
     , VEC(ax(ax_),ay(ay_),az(az_))
    { } ;

    /**
     * @brief Compute Burgers' fluxes in direction \f$x^1\f$
     * 
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename riemann_t 
            , typename recon_t 
            , typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_x_flux_impl( thread_team_t& team 
                       , VEC( const int i 
                       ,      const int j 
                       ,      const int k)
                       , int ngz
                       , grace::flux_array_t const  fluxes
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt 
                       , double const dtfact ) const 
    {
        getflux<0,riemann_t,recon_t>(VEC(i,j,k),team.league_rank(),ngz,fluxes,dx,dt,dtfact);
    }
    /**
     * @brief Compute Burgers' fluxes in direction \f$x^2\f$
     * 
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename riemann_t 
            , typename recon_t 
            , typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_y_flux_impl( thread_team_t& team 
                       , VEC( const int i 
                       ,      const int j 
                       ,      const int k)
                       , int ngz
                       , grace::flux_array_t const  fluxes
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt 
                       , double const dtfact ) const 
    {
        getflux<1,riemann_t,recon_t>(VEC(i,j,k),team.league_rank(),ngz,fluxes,dx,dt,dtfact);
    }
    /**
     * @brief Compute Burgers' fluxes in direction \f$x^3\f$
     * 
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename riemann_t 
            , typename recon_t 
            , typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_z_flux_impl( thread_team_t& team 
                       , VEC( const int i 
                       ,      const int j 
                       ,      const int k)
                       , int ngz
                       , grace::flux_array_t const  fluxes
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt 
                       , double const dtfact ) const 
    {
        getflux<2,riemann_t,recon_t>(VEC(i,j,k),team.league_rank(),ngz,fluxes,dx,dt,dtfact);
    }

    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_source_terms( thread_team_t& team 
                         , VEC( const int i 
                         ,      const int j 
                         ,      const int k)
                         , grace::scalar_array_t<GRACE_NSPACEDIM> const idx
                         , grace::var_array_t<GRACE_NSPACEDIM> const state_new 
                         , double const dt 
                         , double const dtfact ) const 
    { }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_auxiliaries(  VEC( const int i 
                        ,      const int j 
                        ,      const int k) 
                        , int64_t q ) const 
    { }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_max_eigenspeed( VEC( const int i 
                          ,      const int j 
                          ,      const int k) 
                          , int64_t q ) const 
    {
        return Kokkos::max(
              Kokkos::fabs(ax)
            #ifdef GRACE_3D 
            , Kokkos::max(Kokkos::fabs(ay),Kokkos::fabs(az))
            #else
            , Kokkos::fabs(ay)
            #endif 
        ) ; 
    }

 private:
    template< int idir 
            , typename riemann_t 
            , typename recon_t   >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    getflux(  VEC( const int i 
            ,      const int j 
            ,      const int k)
            , const int64_t q 
            , int ngz 
            , grace::flux_array_t const fluxes
            , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
            , double const dt 
            , double const dtfact ) const 
    {
        recon_t reconstructor{} ;
        auto u = Kokkos::subview(  this->_state
                                 , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()) 
                                 , U_ 
                                 , q ) ; 
        double a ; 
        if( idir == 0 ) {
            a = ax ;
        } else if ( idir == 1 ) {
            a = ay ; 
        }
        #ifdef GRACE_3D 
        else {
            a = az ; 
        }
        #endif 
        double uR,uL ; 
        reconstructor(u,VEC(i+ngz,j+ngz,k+ngz),uL,uR,idir) ; 
        //uR = u(VEC(i+this->_ngz,j+this->_ngz,k+this->_ngz)) ; 
        //uL = u(VEC(i+this->_ngz-utils::delta(0,idir),j+this->_ngz-utils::delta(1,idir),k+this->_ngz-utils::delta(2,idir))) ; 
        fluxes(VEC(i,j,k),U_,idir,q) 
            = a > 0 ? a * uL
                    : a * uR; 
    }

 private:
    double VEC(ax,ay,az) ; 

} ; 

void set_scalar_advection_initial_data() ; 

}

#endif /* GRACE_PHYSICS_SCALAR_ADVECTION_EQUATION_HH */