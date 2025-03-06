/**
 * @file bssn.hh
 * @author  Carlo Musolino
 * @brief 
 * @date 2024-09-03
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

#ifndef GRACE_PHYSICS_BSSN
#define GRACE_PHYSICS_BSSN

#include <grace_config.h> 

#include <grace/utils/grace_utils.hh>
#include <grace/utils/numerics/global_interpolators.hh>
#include <grace/utils/numerics/kreiss_olinger.hh>

#include <grace/data_structures/variable_properties.hh>
#include <grace/data_structures/variables.hh>

#include <grace/evolution/fd_evolution_system.hh>

#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/grmhd_metric_utils.hh>
#include <grace/physics/bssn_helpers.hh>

#include <Kokkos_Core.hpp>

#include <array>

//**************************************************************************************************
namespace grace {
//**************************************************************************************************
/**
 * @brief BSSN right-hand side computation.
 * 
 * @tparam der_order Truncation order of the scheme
 * @param i x cell index 
 * @param j y cell index 
 * @param k z cell index
 * @param q quadrant index
 * @param state Staggered state array
 * @param Tmunu Energy momentum tensor (covariant indices)
 * @param idx Inverse cell spacing 
 * @return grace::bssn_state_t The right-hand side of the BSSN equations.
 */
template< size_t der_order >
grace::bssn_state_t GRACE_HOST_DEVICE 
compute_bssn_rhs( VEC(int i, int j, int k), int q
                , grace::var_array_t<GRACE_NSPACEDIM> const state
                , std::array<std::array<double,4>,4> const& Tmunu
                , std::array<double,GRACE_NSPACEDIM> const& idx
                , double const k1, double const eta );
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Compute constraints violations in the BSSN formulation.
 * 
 * @tparam der_order Truncation order used in the computation
 * @param i x cell index
 * @param j y cell index
 * @param k z cell index
 * @param q quadrant index
 * @param state Staggered state array
 * @param Tdd Energy momentum tensor (covariant indices)
 * @param idx Inverse cell spacing
 * @return std::array<double,4> The constraint violations at the requested point
 */
template< size_t der_order >
std::array<double,4> GRACE_HOST_DEVICE 
compute_bssn_constraint_violations(
      VEC(int i, int j, int k), int q
    , grace::var_array_t<GRACE_NSPACEDIM> const state
    , std::array<std::array<double,4>,4> const& Tdd
    , std::array<double,GRACE_NSPACEDIM> const& idx
) ; 
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Forward declaration of a routine for computing the Newman-Penrose scalar 
 * 
 * @tparam der_order Truncation order used in the computation
 * @param i x cell index
 * @param j y cell index
 * @param k z cell index
 * @param q quadrant index
 * @param state Staggered state array
 * @param idx Inverse cell spacing
 * @return std::array<double,2> The real and imaginary parts of Psi_4
 */
template< size_t der_order >
std::array<double,2> GRACE_HOST_DEVICE 
compute_psi4(
      VEC(int i, int j, int k), int q
    , grace::coord_array_t<GRACE_NSPACEDIM> const pcoords
    , grace::var_array_t<GRACE_NSPACEDIM> const state
    , std::array<double,GRACE_NSPACEDIM> const& idx
) ; 

//**************************************************************************************************
/**
 * @brief Baumgarte-Shapiro-Shibata-Nakamura equations
 * \ingroup physics
 */
//**************************************************************************************************
struct bssn_system_t 
    : public fd_evolution_system_t<bssn_system_t> 
{
 private:
    //**************************************************************************************************   
    using base_t = fd_evolution_system_t<bssn_system_t>  ;
    //**************************************************************************************************
 public:
    //**************************************************************************************************
    /**
     * @brief Construct a new_bssn_system_t object
     * 
     * @param state_  State array
     * @param aux_    Auxiliary array
     * @param sstate_ Staggered state array
     */
    bssn_system_t( grace::var_array_t<GRACE_NSPACEDIM> state_ 
                 , grace::var_array_t<GRACE_NSPACEDIM> aux_ 
                 , grace::staggered_variable_arrays_t  sstate_ )
        : base_t(state_,aux_,sstate_) 
    {} 
    //**************************************************************************************************
    //**************************************************************************************************
    /**
     * @brief Compute pointwise update for BSSN equations
     * 
     * @tparam der_order Truncation order of the scheme
     * @param i x cell index 
     * @param j y cell index 
     * @param k z cell index
     * @param q Quadrant index
     * @param _idx Inverse cell spacing
     * @param state_new New state array
     * @param sstate_new New staggered state array
     * @param dt Timestep
     * @param dtfact Timestep factor
     */
    template< size_t der_order >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_update_custom( int const q 
                       , VEC( int const i 
                            , int const j 
                            , int const k)
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const _idx 
                       , grace::var_array_t<GRACE_NSPACEDIM> const state_new 
                       , grace::staggered_variable_arrays_t const sstate_new 
                       , double const dt 
                       , double const dtfact
                       , double const t  
                       , double const x ) const
    {
        auto& cstate = this->_sstate.corner_staggered_fields    ;
        auto& cstate_new = sstate_new.corner_staggered_fields    ;

        std::array<double, GRACE_NSPACEDIM> idx{ VEC(_idx(0,q), _idx(1,q), _idx(2,q))} ;  

        //auto Tmunu = get_Tmunu_lower(VEC(i,j,k),q) ; 
        // FIXME: Tmunu is set to zero because we are testing vacuum 
        std::array<std::array<double,4>,4> Tmunu {{{0},{0},{0},{0}}} ; 
        double const k1 = 0.1; double const eta = 0.25; // FIXME 
        bssn_state_t update = compute_bssn_rhs<der_order>(VEC(i,j,k),q,cstate,Tmunu,idx,k1,eta)  ;   
        #if 0
        // FIXME: Apply Kreiss-Olinger dissipation
        int ii = 0 ; 
        std::array<double, 3> dx { 1/idx[0], 1/idx[1], 1/idx[2] } ;  
        for( int ivar=PHI_; ivar<= BZ_; ++ivar ) {
            update[ii] += apply_kreiss_olinger_dissipation<der_order,5>(
                VEC(i,j,k),ivar,q,cstate,dx,0.1
            ) ; 
            ii++ ; 
        }
        #endif
        double const A  = 1e-08 ; 
        double const A2 = 1e-16 ; 
        double const d  = 1     ; 
        double const sin2L = Kokkos::sin( 4 * M_PI * (x-t) / d ) ; 
        double const sinL  = Kokkos::sin( 2 * M_PI * (x-t) / d ) ;
        double const cosL  = Kokkos::cos( 2 * M_PI * (x-t) / d ) ;
        double const cos2L = Kokkos::cos( 4 * M_PI * (x-t) / d ) ;
        double const denom = 3 * d * Kokkos::pow( 2 - A2 + A2 * cos2L, 4./3. ) ;
        double const cbrt2 = Kokkos::cbrt(2) ; 

        double const Axx = 2 * A2 * M_PI * cbrt2 * sin2L / denom ;
        double const Ayy = cbrt2 * A * M_PI * cosL * ( 6 - A2 + A2*cos2L + 4 * A * sinL ) / denom ; 
        double const Azz = cbrt2 * A * M_PI * cosL * ( - 6 + A2 - A2*cos2L + 4 * A * sinL ) / denom ;

        double const KTR = 2 * M_PI * A2 / d * sin2L / math::int_pow<2>( 1 - A2 * sinL ) ; 

        // Apply update
        cstate_new(VEC(i,j,k),PHI_,q) += dt * dtfact * update[PHIL] ;
        cstate_new(VEC(i,j,k),K_,q)   += dt * dtfact * update[KL]   ;
        int ww = 0 ; 
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ww++;
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ww++;
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ww++;
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ww++;
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ww++;
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ww++;
        ww = 0 ; 
        // XX 
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ww++;
        // XY 
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ww++;
        // XZ  
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ww++;
        // YY 
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ww++;
        // YZ
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ww++;
        // ZZ
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ww++;
        ww = 0 ; 
        cstate_new(VEC(i,j,k),GAMMAX_+ww,q) += dt * dtfact * update[GAMMAXL+ww] ; ww++;
        cstate_new(VEC(i,j,k),GAMMAX_+ww,q) += dt * dtfact * update[GAMMAXL+ww] ; ww++;
        cstate_new(VEC(i,j,k),GAMMAX_+ww,q) += dt * dtfact * update[GAMMAXL+ww] ; ww++;

        cstate_new(VEC(i,j,k),ALP_,q) = 1. ; // += dt * dtfact * update[ALPL] ; 

        cstate_new(VEC(i,j,k),BETAX_,q) = 0 ; //+= dt * dtfact * update[BETAXL] ;
        cstate_new(VEC(i,j,k),BETAY_,q) = 0 ; //+= dt * dtfact * update[BETAYL] ;
        cstate_new(VEC(i,j,k),BETAZ_,q) = 0 ; //+= dt * dtfact * update[BETAZL] ;

        cstate_new(VEC(i,j,k),BX_,q) = 0 ; //+= dt * dtfact * update[BXL] ;
        cstate_new(VEC(i,j,k),BY_,q) = 0 ; //+= dt * dtfact * update[BYL] ;
        cstate_new(VEC(i,j,k),BZ_,q) = 0 ; //+= dt * dtfact * update[BZL] ;

        // Apply algebraic constraints 
        impose_algebraic_constraints(cstate_new,VEC(i,j,k),q) ; 

    }
    //**************************************************************************************************
    //**************************************************************************************************
    template< size_t der_order >
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
    compute_constraint_violations( 
        grace::var_array_t<GRACE_NSPACEDIM> saux, std::array<double,3> const& idx, 
        VEC(int i, int j, int k), int64_t q ) const 
    {
        using namespace grace  ; 
        using namespace Kokkos ;

        std::array<std::array<double,4>,4> Tdd {{{0},{0},{0},{0}}} ; 

        auto constr_loc = 
                    compute_bssn_constraint_violations<der_order>(VEC(i,j,k),q,this->_sstate.corner_staggered_fields,Tdd,idx) ;

        saux(VEC(i,j,k),HAM_ ,q) = constr_loc[0] ;
        saux(VEC(i,j,k),MOMX_,q) = constr_loc[1] ;
        saux(VEC(i,j,k),MOMY_,q) = constr_loc[2] ;
        saux(VEC(i,j,k),MOMZ_,q) = constr_loc[3] ;
        
    } ;
    // *********************************************************************************************
    // compute psi4:
    template< size_t der_order >
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
    compute_newman_penrose( 
        grace::var_array_t<GRACE_NSPACEDIM> saux,
        coord_array_t<GRACE_NSPACEDIM> const pcoords,
         std::array<double,3> const& idx, 
        VEC(int i, int j, int k), int64_t q ) const 
    {
        using namespace grace  ; 
        using namespace Kokkos ;

        auto psi4Complex = compute_psi4<der_order>( VEC(i,j,k)
                                            ,q
                                            ,pcoords   //coordinates
                                            ,this->_sstate.corner_staggered_fields
                                            ,idx) ;

        saux(VEC(i,j,k),PSI4RE_ ,q)  = psi4Complex[0] ;
        saux(VEC(i,j,k),PSI4IM_ ,q)  = psi4Complex[1] ;
        
    } ;
    /**
     * @brief Compute constraint violations on the whole grid.
     * 
     * @tparam der_order Truncation order of the scheme.
     */
    template< size_t der_order >
    void compute_auxiliaries() const 
    {
        DECLARE_GRID_EXTENTS ; 

        using namespace grace  ; 
        using namespace Kokkos ;
        // TODO define a enum for the constraints "local" indices 
        var_array_t<GRACE_NSPACEDIM> bssn_constraints(
            "bssn_constraints_tmp", VEC(nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz), 4, nq 
        ) ; 
        auto _idx = grace::variable_list::get().getinvspacings() ; 
        /* Compute the constraint violations on cell corners */
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
            policy{
                {VEC(ngz,ngz,ngz),0},{VEC(nx+1+ngz,ny+1+ngz,nz+1+ngz),nq}
            } ; 
        parallel_for(
            GRACE_EXECUTION_TAG("BSSN","compute_constraint_violations"),
            policy,
            KOKKOS_LAMBDA(VEC(int i, int j, int k), int q)
            {   
                std::array<double, GRACE_NSPACEDIM> idx{ VEC(_idx(0,q), _idx(1,q), _idx(2,q))} ; 
                //auto Tdd = get_Tmunu_lower(VEC(i,j,k),q) ; // FIXME should not capture (*this)
                std::array<std::array<double,4>,4> Tdd {{{0},{0},{0},{0}}} ; 
                auto constr_loc = 
                    compute_bssn_constraint_violations<der_order>(VEC(i,j,k),q,this->_sstate.corner_staggered_fields,Tdd,idx) ;
                #pragma unroll
                for( int ic=0; ic<4; ++ic) 
                    bssn_constraints(VEC(i,j,k),ic,q) = constr_loc[ic] ; 
            }
        ) ; 

        /* Transfer to cell centers */
        View<int*> out_idx{"out_interp_indices", 4} ; 
        auto h_out_idx = create_mirror_view(out_idx) ; 
        h_out_idx(0) = HAM  ;  
        h_out_idx(1) = MOMX ;
        h_out_idx(2) = MOMY ; 
        h_out_idx(3) = MOMZ ; 
        deep_copy(out_idx,h_out_idx) ; 
        auto sview_aux = subview(this->_aux, VEC(ALL(),ALL(),ALL()), Kokkos::pair{HAM,MOMZ+1}, ALL()) ; 
        interp_corner_to_center_scatter_out<2>(
            bssn_constraints, sview_aux, out_idx
        ) ; 

    }
    //**************************************************************************************************
    //**************************************************************************************************
    /**
     * @brief Return maximum eigenspeed of BSSN equations
     * @return double 
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_max_eigenspeed( VEC( const int i
                               , const int j
                               , const int k)
                          , const int64_t q ) const
    {
        return 1. ; 
    } 
    //**************************************************************************************************
 private:
    //**************************************************************************************************
    /**
     * @brief Compute covariant energy momentum tensor
     * 
     * @param i x cell index
     * @param j y cell index
     * @param k z cell index
     * @param q quadrant index
     * @return std::array<std::array<double,4>,4> Covariant index energy momentum tensor
     */
    std::array<std::array<double,4>,4> GRACE_HOST_DEVICE 
    get_Tmunu_lower(VEC(int i, int j, int k), int q) const
    {
        auto& state  = this->_state                             ;
        auto& cstate = this->_sstate.corner_staggered_fields    ;
        auto& aux    = this->_aux                               ;


        auto const metric = grace::metric_array_t {
            {
                cstate(VEC(i,j,k),GTXX_,q),
                cstate(VEC(i,j,k),GTXY_,q),
                cstate(VEC(i,j,k),GTXZ_,q),
                cstate(VEC(i,j,k),GTYY_,q),
                cstate(VEC(i,j,k),GTYZ_,q),
                cstate(VEC(i,j,k),GTZZ_,q)
            }, 
            cstate(VEC(i,j,k),PHI_,q), 
            {
                cstate(VEC(i,j,k),BETAX_,q),
                cstate(VEC(i,j,k),BETAY_,q),
                cstate(VEC(i,j,k),BETAZ_,q)  
            }, 
            cstate(VEC(i,j,k),ALP_,q)
        } ; 

        double W;
        grmhd_prims_array_t prims = get_primitives_cell_corner(
            aux,metric,W,VEC(i,j,k),q
        );

        // Fill T_{\mu\nu} = \rho h u_\mu u_\nu + P g_{\mu \nu} 
        std::array<std::array<double,4>,4> Tmunu ;
        double const u0 =  W/metric.alp() ; 
        std::array<double,4> uU { u0, prims[VXL]*u0, prims[VYL]*u0, prims[VZL]*u0 } ; 
        auto uD = metric.lower_4vec(uU)  ; 
        auto gdd = metric.gmunu()     ; 
        int idx4[4][4] = {
            {0,1,2,3},
            {1,4,5,6},
            {2,5,7,8},
            {3,6,8,9}
        } ; 
        for( int mu=0; mu<4; ++mu ) {
            for( int nu=0; nu<4; ++nu) {
                // TODO missing b field contribution 
                Tmunu[mu][nu] = (prims[RHOL]*(1+prims[EPSL]) + prims[PRESSL]) * uD[mu] * uD[nu] 
                              + prims[PRESSL] * gdd[idx4[mu][nu]] ;
            }
        }

        return Tmunu ; 
    }
    //**************************************************************************************************
    //**************************************************************************************************
    void GRACE_HOST_DEVICE 
    impose_algebraic_constraints(grace::var_array_t<GRACE_NSPACEDIM> state, VEC(int i, int j, int k), int q) const 
    {
        /* First impose the det(gtilde) = 1 constraint */
        double const gtxx = state(VEC(i,j,k),GTXX_+0,q);
        double const gtxy = state(VEC(i,j,k),GTXX_+1,q);
        double const gtxz = state(VEC(i,j,k),GTXX_+2,q);
        double const gtyy = state(VEC(i,j,k),GTXX_+3,q);
        double const gtyz = state(VEC(i,j,k),GTXX_+4,q);
        double const gtzz = state(VEC(i,j,k),GTXX_+5,q);

        double const detgt     = -(gtxz*gtxz*gtyy) + 2*gtxy*gtxz*gtyz - gtxx*(gtyz*gtyz) - gtxy*gtxy*gtzz + gtxx*gtyy*gtzz;
        double const cbrtdetgt = Kokkos::cbrt(detgt);

        state(VEC(i,j,k),GTXX_+0,q) /= cbrtdetgt ; 
        state(VEC(i,j,k),GTXX_+1,q) /= cbrtdetgt ; 
        state(VEC(i,j,k),GTXX_+2,q) /= cbrtdetgt ; 
        state(VEC(i,j,k),GTXX_+3,q) /= cbrtdetgt ; 
        state(VEC(i,j,k),GTXX_+4,q) /= cbrtdetgt ; 
        state(VEC(i,j,k),GTXX_+5,q) /= cbrtdetgt ; 

        /* And the trace-free Aij constraint next */

        double const gtXX=(-(gtyz*gtyz) + gtyy*gtzz)/detgt ;
        double const gtXY=(gtxz*gtyz - gtxy*gtzz)/detgt    ;
        double const gtXZ=(-(gtxz*gtyy) + gtxy*gtyz)/detgt ;
        double const gtYY=(-(gtxz*gtxz) + gtxx*gtzz)/detgt ;
        double const gtYZ=(gtxy*gtxz - gtxx*gtyz)/detgt    ;
        double const gtZZ=(-(gtxy*gtxy) + gtxx*gtyy)/detgt ; 

        double const Atxx = state(VEC(i,j,k),ATXX_+0,q);
        double const Atxy = state(VEC(i,j,k),ATXX_+1,q);
        double const Atxz = state(VEC(i,j,k),ATXX_+2,q);
        double const Atyy = state(VEC(i,j,k),ATXX_+3,q);
        double const Atyz = state(VEC(i,j,k),ATXX_+4,q);
        double const Atzz = state(VEC(i,j,k),ATXX_+5,q);

        double const ATR = Atxx*gtXX + 2*Atxy*gtXY + 2*Atxz*gtXZ + Atyy*gtYY + 2*Atyz*gtYZ + Atzz*gtZZ ; 
        
        state(VEC(i,j,k),ATXX_+0,q) -= 1./3. * state(VEC(i,j,k),GTXX_+0,q) * ATR ; 
        state(VEC(i,j,k),ATXX_+1,q) -= 1./3. * state(VEC(i,j,k),GTXX_+1,q) * ATR ; 
        state(VEC(i,j,k),ATXX_+2,q) -= 1./3. * state(VEC(i,j,k),GTXX_+2,q) * ATR ; 
        state(VEC(i,j,k),ATXX_+3,q) -= 1./3. * state(VEC(i,j,k),GTXX_+3,q) * ATR ; 
        state(VEC(i,j,k),ATXX_+4,q) -= 1./3. * state(VEC(i,j,k),GTXX_+4,q) * ATR ; 
        state(VEC(i,j,k),ATXX_+5,q) -= 1./3. * state(VEC(i,j,k),GTXX_+5,q) * ATR ; 

    }
} ; 
//**************************************************************************************************
//**************************************************************************************************
} // namespace grace 
//**************************************************************************************************
#endif