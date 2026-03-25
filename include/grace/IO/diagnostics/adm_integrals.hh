/**
 * @file adm_integrals.hh
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

#ifndef GRACE_IO_ADM_INTEGRALS_HH
#define GRACE_IO_ADM_INTEGRALS_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/metric_utils.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/IO/diagnostics/black_hole_diagnostics.hh>

#include <array>
#include <vector>
#include <string> 


namespace grace {

struct adm_integrals: 
    public diagnostic_base_t<adm_integrals> 
{
    using base_t = diagnostic_base_t<adm_integrals>; 

    static constexpr size_t n_fluxes = static_cast<size_t>(1 + 3 + 3 /*E P J*/);

    static std::vector<std::string> flux_names ; 


    adm_integrals()
        : base_t("gw_integrals")
    {
        #ifdef GRACE_ENABLE_Z4C_METRIC
        this->var_interp_idx = std::vector<int>({
            GTXX,GTXY,GTXZ,GTYY,GTYZ,GTZZ,
            ATXX,ATXY,ATXZ,ATYY,ATYZ,ATZZ,
            KHAT,THETA,CHI
        });
        #else 
        this->var_interp_idx = std::vector<int>({
            GXX,GXY,GXZ,GYY,GYZ,GZZ,
            KXX,KXY,KXZ,KYY,KYZ,KZZ
        });
        #endif 
    }

    std::array<double,n_fluxes> 
    compute_local_fluxes(
        Kokkos::View<double**> ivals_d, 
        Kokkos::View<double**> ivals_aux_d, 
        spherical_surface_iface const& detector 
    )  ;
}; 

}

#endif 