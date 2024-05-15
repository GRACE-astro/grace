/**
 * @file burgers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-13
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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

#ifndef THUNDER_PHYSICS_BURGERS_EQUATION_HH
#define THUNDER_PHYSICS_BURGERS_EQUATION_HH

#include <thunder_config.h>

#include <thunder/utils/thunder_utils.hh>

#include <thunder/evolution/hrsc_evolution_system.hh> 

#include <thunder/data_structures/variable_indices.hh>
#include <thunder/evolution/evolution_kernel_tags.hh>

#include <Kokkos_Core.hpp>

namespace thunder {

template< typename recon_t 
        , typename riemann_t > 
struct burgers_equation_system_t 
    : public hrsc_evolution_system_t<burgers_equation_system_t<recon_t,riemann_t>>
{
    burgers_equation_system_t( thunder::var_array_t<THUNDER_NSPACEDIM> state_
                             , thunder::var_array_t<THUNDER_NSPACEDIM> aux_   ) 
     : hrsc_evolution_system_t<burgers_equation_system_t<recon_t,riemann_t>>(state_,aux_)
    { } ;

    template< typename thread_team_t >
    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    compute_x_flux( thread_team_t& team 
                  , VEC( const int i 
                  ,      const int j 
                  ,      const int k)
                  , int ngz
                  , thunder::flux_array_t& fluxes) const 
    {
        getflux<0>(VEC(i,j,k),team.league_rank(),ngz,fluxes); 
    }
    template< typename thread_team_t >
    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    compute_y_flux( thread_team_t& team 
                  , VEC( const int i 
                  ,      const int j 
                  ,      const int k)
                  , int ngz
                  , thunder::flux_array_t& fluxes) const 
    {
        getflux<1>(VEC(i,j,k),team.league_rank(),ngz,fluxes); 
    }
    template< typename thread_team_t >
    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    compute_z_flux(  thread_team_t& team 
                  , VEC( const int i 
                  ,      const int j 
                  ,      const int k)
                  , int ngz
                  , thunder::flux_array_t& fluxes) const 
    {
        getflux<2>(VEC(i,j,k),team.league_rank(),ngz,fluxes); 
    }

    template< typename thread_team_t >
    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    compute_source_terms( thread_team_t& team 
                         , VEC( const int i 
                         ,      const int j 
                         ,      const int k)
                         , thunder::var_array_t<THUNDER_NSPACEDIM>& state_new
                         , double const dt 
                         , double const dtfact ) const 
    { }

    template< typename thread_team_t >
    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    compute_auxiliaries( thread_team_t& team 
                        , VEC( const int i 
                        ,      const int j 
                        ,      const int k) ) const 
    { }

 private:
    template< int idir >
    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    getflux(  VEC( const int i 
            ,      const int j 
            ,      const int k)
            , const int64_t q 
            , int ngz
            , thunder::flux_array_t& fluxes) const 
    {
        recon_t reconstructor{} ; 
        riemann_t solver{}      ;
        auto u = Kokkos::subview(  this->_state
                                 , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()) 
                                 , U_ 
                                 , q ) ; 
        double uR,uL ; 
        reconstructor(u,VEC(i+ngz,j+ngz,k+ngz),uL,uR,idir) ; 
        double const fL = 0.5 * math::int_pow<2>(uL) ; 
        double const fR = 0.5 * math::int_pow<2>(uR) ; 
        
        double const cmin = - math::min(0., math::min(uL,uR)) ; 
        double const cmax = math::max(0., math::max(uL,uR))   ; 

        fluxes(VEC(i,j,k),U_,idir,q) 
            = solver(fL,fR,uL,uR,cmin,cmax) ; 
    }

     

} ; 

}

#endif /* THUNDER_PHYSICS_BURGERS_EQUATION_HH */