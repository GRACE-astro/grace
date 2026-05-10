/**
 * @file outflow_diagnostics.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2026-01-15
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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
#ifndef GRACE_IO_OUTFLOW_DIAGNOSTICS_HH
#define GRACE_IO_OUTFLOW_DIAGNOSTICS_HH


#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/metric_utils.hh>

#include <grace/IO/output_diagnostics.hh>
#include <grace/IO/diagnostics/diagnostic_base.hh>
#include <grace/IO/spherical_surfaces.hh>

#include <grace/data_structures/variable_indices.hh>

#include <array>
#include <vector>
#include <string> 


namespace grace {

struct outflows: 
    public diagnostic_base_t<outflows> 
{
    using base_t = diagnostic_base_t<outflows>; 

    #ifndef GRACE_ENABLE_Z4C_METRIC
    enum loc_var_idx_t : int {
        GXXL=0, GXYL, GXZL, GYYL, GYZL, GZZL,
        BETAXL, BETAYL, BETAZL, ALPL, NUM_VARS 
    } ; 
    #else
    enum loc_var_idx_t : int {
        GTXXL=0, GTXYL, GTXZL, GTYYL, GTYZL, GTZZL, CHIL,
        BETAXL, BETAYL, BETAZL, ALPL, NUM_VARS
    } ; 
    #endif 
    enum loc_aux_idx_t : int { 
        RHOL=0, EPSL, PRESSL, ZXL, ZYL, ZZL, NUM_AUX
    };
    enum diag_var_idx_t : int {
        GEO_UNBOUND=0, BERN_UNBOUND, TOT, N_DIAG_VARS
    } ; 

    static constexpr size_t n_fluxes = static_cast<size_t>(3); /* geodesic bernoulli total mass flux */

    static std::vector<std::string> flux_names ; 


    outflows()
        : base_t("outflows")
    {
        #ifndef GRACE_ENABLE_Z4C_METRIC
        this->var_interp_idx = std::vector<int>({GXX_, GXY_, GXZ_, GYY_, GYZ_, GZZ_, BETAX_, BETAY_, BETAZ_, ALP_});
        #else 
        this->var_interp_idx = std::vector<int>({GTXX_, GTXY_, GTXZ_, GTYY_, GTYZ_, GTZZ_, CHI_, BETAX_, BETAY_, BETAZ_, ALP_});
        #endif
        this->aux_interp_idx = std::vector<int>({RHO_,EPS_,PRESS_,ZVECX_,ZVECY_,ZVECZ_});
    }

    std::array<double,n_fluxes> 
    compute_local_fluxes(
        Kokkos::View<double**> ivals_d, 
        Kokkos::View<double**> ivals_aux_d, 
        spherical_surface_iface const& detector 
    )  ;
}; 


#ifdef GRACE_ENABLE_M1
// ---------------------------------------------------------------------------
/**
 * @brief Radiation energy and number luminosity through spherical detectors.
 *
 * For each registered detector sphere the diagnostic integrates
 *   L_E^(s) = ∫ (FRADX_s * x/r + FRADY_s * y/r + FRADZ_s * z/r) r² dΩ
 * per neutrino species s.  The conserved flux variables FRADX/Y/Z already
 * carry the factor sqrt(g), so the integral is in code-unit energy flux.
 *
 * Uses the same "outflows" parameter block (detector_names) as the mass
 * outflow diagnostic.
 */
// ---------------------------------------------------------------------------
struct m1_outflows :
    public diagnostic_base_t<m1_outflows>
{
    using base_t = diagnostic_base_t<m1_outflows>;

    // Local index into ivals (state variables interpolated to sphere)
    enum loc_var_idx_t : int {
        // Metric
        BETAXL=0, BETAYL, BETAZL, ALPL,
        E1L, FX1L, FY1L, FZ1L,
#if defined(M1_NU_THREESPECIES)
        E2L, FX2L, FY2L, FZ2L,
        E3L, FX3L, FY3L, FZ3L,
#elif defined(M1_NU_FIVESPECIES)
        E2L, FX2L, FY2L, FZ2L,
        E3L, FX3L, FY3L, FZ3L,
        E4L, FX4L, FY4L, FZ4L,
        E5L, FX5L, FY5L, FZ5L,
#endif
        NUM_VARS
    };

    // No aux variables needed
    enum loc_aux_idx_t : int { NUM_AUX = 0 };

    // One luminosity entry per species
#if defined(M1_NU_THREESPECIES)
    static constexpr size_t n_fluxes = 3;
#elif defined(M1_NU_FIVESPECIES)
    static constexpr size_t n_fluxes = 5;
#else
    static constexpr size_t n_fluxes = 1;
#endif

    static std::vector<std::string> flux_names;

    m1_outflows()
        : base_t("outflows")   // reuse outflows detector list
    {
        // Radiation flux components per species (state array indices)
        this->var_interp_idx = {
            BETAX_, BETAY_, BETAZ_, ALP_,
            ERAD1_, FRADX1_, FRADY1_, FRADZ1_
#if defined(M1_NU_THREESPECIES) || defined(M1_NU_FIVESPECIES)
            , ERAD2_, FRADX2_, FRADY2_, FRADZ2_
            , ERAD3_, FRADX3_, FRADY3_, FRADZ3_
#endif
#ifdef M1_NU_FIVESPECIES
            , ERAD4_, FRADX4_, FRADY4_, FRADZ4_
            , ERAD5_, FRADX5_, FRADY5_, FRADZ5_
#endif
        };
        this->aux_interp_idx = {};
    }

    std::array<double, n_fluxes>
    compute_local_fluxes(
        Kokkos::View<double**> ivals_d,
        Kokkos::View<double**> ivals_aux_d,
        spherical_surface_iface const& detector
    );
};
#endif /* GRACE_ENABLE_M1 */


}

#endif /*GRACE_IO_OUTFLOW_DIAGNOSTICS_HH*/
