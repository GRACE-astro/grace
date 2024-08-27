/**
 * @file burgers.hh
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

#ifndef GRACE_PHYSICS_BURGERS_EQUATION_HH
#define GRACE_PHYSICS_BURGERS_EQUATION_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>

#include <grace/evolution/hrsc_evolution_system.hh> 

#include <grace/data_structures/variable_indices.hh>
#include <grace/evolution/evolution_kernel_tags.hh>

#include <Kokkos_Core.hpp>

namespace grace {
/**
 * @brief Burgers' equation evolution module.
 * \ingroup physics
 * 
 * This class implements the necessary methods to evolve 
 * Burgers' equation with HRSC methods in GRACE. The Burgers'
 * equation is a minimal example of non-linear hyperbolic PDE 
 * which can be written in conservative form as 
 * \f[
 * \partial_t U + \frac{1}{2} \partial_x U^2 = 0
 * \f]
 * Since the PDE is nonlinear, it can develop discontinuities 
 * in a finite time when starting from smooth data. Within GRACE,
 * the Burgers' equation module serves mainly the purpose of providing 
 * a simple and lightweight test for the full evolution infrastructure.
 * Standard tests for the validity of GRACE include The evolution of 
 * so-called N-wave initial data as well as a variety of shocktubes.
 * See the User's guide to learn how to run these standard tests.
 * Burgers' equation is fairly simple, and this class serves as a 
 * simple to read template on how to implement a system of equations 
 * to be solved with HRSC methods in GRACE.
 */
struct burgers_equation_system_t 
    : public hrsc_evolution_system_t<burgers_equation_system_t>
{
    /**
     * @brief Construct a new Burgers' system object.
     * 
     * @param state_ Current state array.
     * @param aux_   Current auxiliary array.
     */
    burgers_equation_system_t( grace::var_array_t<GRACE_NSPACEDIM> state_
                             , grace::var_array_t<GRACE_NSPACEDIM> aux_   ) 
     : hrsc_evolution_system_t<burgers_equation_system_t>(state_,aux_)
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
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
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
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
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
    /**
     * @brief Compute geometric source terms for Burgers' equation.
     * 
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param state_new State where sources are added.
     * @param dt Timestep.
     * @param dtfact Timestep factor.
     */
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
    /**
     * @brief Compute Burgers' auxiliary quantities.
     * 
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param q Quadrant index.
     */
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_auxiliaries(  VEC( const int i 
                        ,      const int j 
                        ,      const int k) 
                        , int64_t q ) const 
    { }
    /**
     * @brief Compute maximum absolute value eigenspeed.
     * 
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param q Quadrant index.
     * @return double Maximum eigenspeed of Burgers' equations.
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_max_eigenspeed( VEC( const int i 
                          ,      const int j 
                          ,      const int k) 
                          , int64_t q ) const 
    {
        return Kokkos::fabs(
            this->_state(VEC(i,j,k),U_,q)
        ) ; 
    }

 private:
    /**
     * @brief Get the Burgers' equation flux
     *        in a prescribed direction.
     * 
     * @tparam idir Direction.
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param q Quadrant index
     * @param ngz Number of ghost-zones.
     * @param fluxes Flux array.
     * 
     * This function performs reconstruction of the 
     * evolved variable \f$U\f$ and computes the 
     * physical flux at the interface \f$(i-1/2+\epsilon,j,k)\f$
     * (assuming <code>idir==0</code>) using the Riemann solver.
     */
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
        double const cmax =   math::max(0., math::max(uL,uR))   ; 

        fluxes(VEC(i,j,k),U_,idir,q) 
            = solver(fL,fR,uL,uR,cmin,cmax) ; 
    }

     

} ;

/**
 * @brief Set the initial data for Burgers'
 *        equation. Parameters are used to 
 *        control which test-case to initialize.
 * \ingroup physics
 */
void set_burgers_initial_data() ; 

}

#endif /* GRACE_PHYSICS_BURGERS_EQUATION_HH */