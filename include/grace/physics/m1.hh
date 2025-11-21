/**
 * @file m1.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2024-11-21
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
#ifndef GRACE_PHYSICS_M1_HH
#define GRACE_PHYSICS_M1_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/evolution/hrsc_evolution_system.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/utils/reconstruction.hh>
#include <grace/utils/weno_reconstruction.hh>
#include <grace/utils/riemann_solvers.hh>
#include <grace/utils/advanced_riemann_solvers.hh>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace grace {
//**************************************************************************************************/ 
//**************************************************************************************************
/**
 * @brief M1 equations system.
 * \ingroup physics 
 */
//**************************************************************************************************/
struct m1_equations_system_t
    : public hrsc_evolution_system_t<m1_equations_system_t>
{
    private:
    //! Base class type 
    using base_t = hrsc_evolution_system_t<grmhd_equations_system_t<eos_t>>;

    public:

    m1_equations_system_t(grace::var_array_t state_
                        , grace::staggered_variable_arrays_t stag_state_
                        , grace::var_array_t aux_ )
    : base_t(state_,stag_state_,aux_)
    {} ; 

    /**
     * @brief Compute M1 fluxes in direction \f$x^1\f$
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
            , typename recon_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_x_flux_impl( int const q 
                       , VEC( const int i 
                       ,      const int j 
                       ,      const int k)
                       , grace::flux_array_t const  fluxes
                       , grace::flux_array_t const  vbar
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt 
                       , double const dtfact ) const 
    {
        getflux<0,riemann_t,recon_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
    }
    /**
     * @brief Compute M1 fluxes in direction \f$x^2\f$
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
            , typename recon_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_y_flux_impl( int const q 
                       , VEC( const int i 
                       ,      const int j 
                       ,      const int k)
                       , grace::flux_array_t const  fluxes
                       , grace::flux_array_t const  vbar
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt 
                       , double const dtfact ) const
    {
        getflux<1,riemann_t,recon_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
    }
    /**
     * @brief Compute M1 fluxes in direction \f$x^3\f$
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
            , typename recon_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_z_flux_impl( int const q 
                       , VEC( const int i 
                       ,      const int j 
                       ,      const int k)
                       , grace::flux_array_t const  fluxes
                       , grace::flux_array_t const  vbar
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt 
                       , double const dtfact ) const
    {
        getflux<2,riemann_t,recon_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
    }
    

    /**
     * @brief Compute geometric source terms for M1 equations.
     * 
     * @tparam thread_team_t Thread team type.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param idx Inverse cell coordinate spacings.
     * @param state_new State where sources are added.
     * @param dt Timestep.
     * @param dtfact Timestep factor.
     */
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_source_terms( const int q 
                         , VEC( const int i 
                         ,      const int j 
                         ,      const int k)
                         , grace::scalar_array_t<GRACE_NSPACEDIM> const idx
                         , grace::var_array_t const state_new
                         , double const dt 
                         , double const dtfact ) const 
    {
        using namespace grace  ;
        using namespace Kokkos ;

    }


    private:
    /***********************************************************************/
    //! Number of reconstructed variables.
    static constexpr unsigned int M1_NUM_RECON_VARS = 4*N_M1_RAD_SPECIES ; 
    //! Equation of State object.
    eos_t _eos ;    
    //! Parameters for atmosphere
    atmo_params_t atmo_params;
    //! Parameters for excision
    excision_params_t excision_params; 
    /***********************************************************************/
    /***********************************************************************/
    /**
     * @brief Compute fluxes for m1 equations.
     * 
     * @tparam idir Direction the fluxes are computed in.
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @param i zero-offset x cell index.
     * @param j zero-offset y cell index.
     * @param k zero-offset z cell index.
     * @param q quadrant index.
     * @param ngz Number of ghost-zones.
     * @param fluxes Flux array.
     */
    template< int idir 
            , typename riemann_t
            , typename recon_t   >
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    getflux(  VEC( const int i 
            ,      const int j 
            ,      const int k)
            , const int64_t q 
            , grace::flux_array_t const fluxes
            , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
            , double const dt 
            , double const dtfact ) const 
    {
        /***********************************************************************/
        /* Initialize reconstructor and riemann solver                         */
        /***********************************************************************/
        recon_t reconstructor{} ; 
        /***********************************************************************/
        /* 3rd order interpolation of metric at cell interface                 */
        /***********************************************************************/
        metric_array_t metric_face ; 
        COMPUTE_FCVAL(metric_face,this->_state,i,j,k,q,idir) ; 
        /***********************************************************************/
        /*              Reconstruct primitive variables                        */
        /***********************************************************************/
        std::array<int, 4>
            recon_indices{
                  ERAD_
                , FRADX_
                , FRADY_
                , FRADZ_
            } ; 
        /* Local indices in prims array (note z^k -> v^k) */
        std::array<int, 4>
            recon_indices_loc{
                  ERADL
                , FXL
                , FYL 
                , FZL 
            } ;
        /* Reconstruction                                  */
        m1_prims_array_t primL, primR ; 
        #pragma unroll 4
        for( int ivar=0; ivar<4; ++ivar) {
            auto u = Kokkos::subview( this->_state
                                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()) 
                                    , recon_indices[ivar] 
                                    , q ) ;
            reconstructor( u, VEC(i,j,k)
                         , primL[recon_indices_loc[ivar]]
                         , primR[recon_indices_loc[ivar]]
                         , idir) ;
        }
        // now we need to reconstruct zvec from the hydro 
        std::array<int, 3>
            recon_indices{
                  ZVECX_
                , ZVECY_
                , ZVECZ_
            } ; 
        /* Local indices in prims array (note z^k -> v^k) */
        std::array<int, 3>
            recon_indices_loc{
                  ZXL 
                , ZYL 
                , ZZL
            } ;
        #pragma unroll 3
        for( int ivar=0; ivar<3; ++ivar) {
            auto u = Kokkos::subview( this->_aux
                                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()) 
                                    , recon_indices[ivar] 
                                    , q ) ;
            reconstructor( u, VEC(i,j,k)
                         , primL[recon_indices_loc[ivar]]
                         , primR[recon_indices_loc[ivar]]
                         , idir) ;
        }
        // note that at this stage F is actually F/E, we need to fix that here
        for( int ii=0; ii<3; ++ii) {
            primL[FXL+ii] *= primL[ERADL] ; 
            primR[FXL+ii] *= primR[ERADL] ;  
        }
        // closures 
        m1_closure_t cl{
            primL[ERADL],
            {primL[FXL], primL[FYL], primL[FZL]},
            {primL[ZXL], primL[ZYL], primL[ZZL]},
            metric_face
        }, 
        cr{
            primR[ERADL],
            {primR[FXL], primR[FYL], primR[FZL]},
            {primR[ZXL], primR[ZYL], primR[ZZL]},
            metric_face
        }; 

        cl.update_closure(0) ; cr.update_closure(0) ;  
        // compute P^i_j 
        int imap[3][3] = {
            {0,1,2}, {1,3,4}, {2,4,5}
        } ; 
        auto const& PUU_l = cl.PUU ; auto const& PUU_r = cr.PUU ; 
        auto const PUD_l = metric_face.lower_vec(
            {PUU_l[imap[idir][0]], PUU_l[imap[idir][1]],PUU_l[imap[idir][2]]}
        ) ; 
        auto const PUD_r = metric_face.lower_vec(
            {PUU_r[imap[idir][0]], PUU_r[imap[idir][1]],PUU_r[imap[idir][2]]}
        ) ; 
        // compute the A factor for asymptotic flux correction 
        double const kappa = this->_aux(VEC(i,j,k),KAPPAA_,q) + this->_aux(VEC(i,j,k),KAPPAS_,q) ; 
        double const A = min(1, idx(idir,q) / kappa ) ; 
        // compute one component of the upper-index flux for the E flux 
        
        double FUd_l = metric_face.invgamma(imap[idir][0]) * primL[FXL] 
                     + metric_face.invgamma(imap[idir][1]) * primL[FYL]
                     + metric_face.invgamma(imap[idir][2]) * primL[FZL] ; 
        double FUd_r = metric_face.invgamma(imap[idir][0]) * primR[FXL] 
                     + metric_face.invgamma(imap[idir][1]) * primR[FYL]
                     + metric_face.invgamma(imap[idir][2]) * primR[FZL] ; 

        // compute wave speeds 
        double cmin, cmax ; 
        double cpr, cmr, cpl, cml;
        compute_cp_cm<idir>(cpl,cml, cl, metric_face) ;
        compute_cp_cm<idir>(cpr,cmr, cr, metric_face) ; 
        cmin = -Kokkos::min(0., Kokkos::min(cml,cmr)) ; 
        cmax =  Kokkos::max(0., Kokkos::max(cpl,cpr)) ; 
        /* Add some diffusion in weakly hyperbolic limit */
        if( cmin < 1e-12 and cmax < 1e-12 ) { cmin=1; cmax=1; }

        // compute the fluxes 
        // E 
        double E_l = primL[ERADL] * metric_face.sqrtg() ;
        double E_r = primR[ERADL] * metric_face.sqrtg() ; 
        double f_E_l = metric_face.sqrtg() * (metric_face.alp() * FUd_l - metric_face.beta(idir) * primL[ERADL]) ; 
        double f_E_r = metric_face.sqrtg() * (metric_face.alp() * FUd_r - metric_face.beta(idir) * primR[ERADL]) ; 
        fluxes(VEC(i,j,k),ERAD_,q) = (cmax*f_E_l + cmin*f_E_r - A * cmax * cmin * (E_r-E_l))/(cmax+cmin) ; 
        // Fx 
        double Fx_l = primL[FXL] * metric_face.sqrtg() ;
        double Fx_r = primR[FXL] * metric_face.sqrtg() ;
        double f_Fx_l = metric_face.sqrtg() * (metric_face.alp() * PUD_l[0] - metric_face.beta(idir) * primL[FXL]) ; 
        double f_Fx_r = metric_face.sqrtg() * (metric_face.alp() * PUD_r[0] - metric_face.beta(idir) * primR[FXL]) ; 
        fluxes(VEC(i,j,k),FRADX_,q) = (SQR(A)*(cmax*f_Fx_l + cmin*f_Fx_r) - A * cmax * cmin * (Fx_r-Fx_l))/(cmax+cmin) 
                                    + (1-SQR(A)) * 0.5 * (f_Fx_l+f_Fx_r); 
        // Fy 
        double Fy_l = primL[FYL] * metric_face.sqrtg() ;
        double Fy_r = primR[FYL] * metric_face.sqrtg() ;
        double f_Fy_l = metric_face.sqrtg() * (metric_face.alp() * PUD_l[1] - metric_face.beta(idir) * primL[FYL]) ; 
        double f_Fy_r = metric_face.sqrtg() * (metric_face.alp() * PUD_r[1] - metric_face.beta(idir) * primR[FYL]) ;
        fluxes(VEC(i,j,k),FRADY_,q) = (SQR(A)*(cmax*f_Fy_l + cmin*f_Fy_r) - A * cmax * cmin * (Fy_r-Fy_l))/(cmax+cmin) 
                                    + (1-SQR(A)) * 0.5 * (f_Fy_l+f_Fy_r); 
        // Fz 
        double Fz_l = primL[FZL] * metric_face.sqrtg() ;
        double Fz_r = primR[FZL] * metric_face.sqrtg() ;
        double f_Fz_l = metric_face.sqrtg() * (metric_face.alp() * PUD_l[2] - metric_face.beta(idir) * primL[FZL]) ; 
        double f_Fz_r = metric_face.sqrtg() * (metric_face.alp() * PUD_r[2] - metric_face.beta(idir) * primR[FZL]) ;
        fluxes(VEC(i,j,k),FRADZ_,q) = (SQR(A)*(cmax*f_Fz_l + cmin*f_Fz_r) - A * cmax * cmin * (Fz_r-Fz_l))/(cmax+cmin) 
                                    + (1-SQR(A)) * 0.5 * (f_Fz_l+f_Fz_r); 
    }

    template< size_t idir >
    void compute_cp_cm(
        double& cp, double &cm, m1_closure_t const& cl, metric_array_t const& metric
    )
    {
        double const cpthin = -metric.beta(idir) + metric.alp() * fabs(cl.FU[idir])/cl.F ; 
        double const cmthin = -metric.beta(idir) - metric.alp() * fabs(cl.FU[idir])/cl.F ; 

        double const p = metric.alp() * cl.vU[idir]/cl.W ; 
        int const icomp = (idir==0) * 0 + (idir==1)*2 + (idir==2)*5 ;
        double const r = sqrt(SQR(metric.alp()) * metric.invgamma(icomp) * (2.*cl.W2 + 1.) - 2.*cl.W2 * SQR(p));

        double const cpthick = min(
            -metric.beta(idir) + (2.*p*cl.W2+r)/(2.*cl.W2+1.), -metric.beta(idir) + p 
        ) ; 
        double const cmthick = min(
            -metric.beta(idir) + (2.*p*cl.W2-r)/(2.*cl.W2+1.), -metric.beta(idir) + p 
        ) ; 

        double dthin = cl.chi * 1.5 - 0.5 ; 
        double dthick = 1.5 - cl.chi * 1.5 ;
        cp = dthin * cpthin + dthick * cpthick ;
        cm = dthin * cmthin + dthick * cmthick ; 

    }

} ; 

} /* namespace grace */

#endif /*GRACE_PHYSICS_M1_HH*/