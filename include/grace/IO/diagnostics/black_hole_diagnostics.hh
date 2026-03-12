/**
 * @file black_hole_diagnostics.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-17
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
#ifndef GRACE_IO_BH_DIAGNOSTICS_HH
#define GRACE_IO_BH_DIAGNOSTICS_HH
#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>

#include <grace/utils/metric_utils.hh>

#include <grace/utils/device_vector.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/config/config_parser.hh>

#include <grace/IO/diagnostics/diagnostic_base.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>


namespace grace {

struct bh_diagnostics: 
    public diagnostic_base_t<bh_diagnostics> 
{
    using base_t = diagnostic_base_t<bh_diagnostics>; 
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
        RHOL=0, EPSL, PRESSL, VELXL, VELYL, VELZL,
        BXL, BYL, BZL, NUM_AUX
    };
    enum diag_var_idx_t : int {
        MDOT=0, EDOT, LDOT, PHI, N_DIAG_VARS
    } ; 

    //! Number of fluxes computed by this class, needed 
    //! for the base class.
    static constexpr size_t n_fluxes = static_cast<size_t>(N_DIAG_VARS);

    static std::vector<std::string> flux_names ; 

    bh_diagnostics() 
        : base_t("bh_diagnostics")
    {
        #ifndef GRACE_ENABLE_Z4C_METRIC
        this->var_interp_idx = std::vector<int>({GXX_, GXY_, GXZ_, GYY_, GYZ_, GZZ_, BETAX_, BETAY_, BETAZ_, ALP_});
        #else 
        this->var_interp_idx = std::vector<int>({GTXX_, GTXY_, GTXZ_, GTYY_, GTYZ_, GTZZ_, CHI_, BETAX_, BETAY_, BETAZ_, ALP_});
        #endif 
        this->aux_interp_idx = std::vector<int>({RHO_, EPS_, PRESS_, ZVECX_, ZVECY_, ZVECZ_, BX_, BY_, BZ_});

    }

    std::array<double,n_fluxes> 
    compute_local_fluxes(
        Kokkos::View<double**> ivals_d, 
        Kokkos::View<double**> ivals_aux_d, 
        spherical_surface_iface const& detector 
    )  ;

} ; 

}

#endif /*GRACE_IO_BH_DIAGNOSTICS_HH*/