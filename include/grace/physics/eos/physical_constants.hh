/**
 * @file physical_constants.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
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

namespace grace { namespace physical_constants {

static constexpr const double cm_to_CU       = 6.77269222552442e-06;
static constexpr const double s_to_CU        = 2.03040204956746e05 ;
static constexpr const double densCGS_to_CU  = 1.61887093132742e-18;  
static constexpr const double pressCGS_to_CU = 1.80123683248503e-39; 
static constexpr const double epsCGS_to_CU   = 1.11265005605362e-21;
static constexpr const double CU_to_densCGS  = 6.17714470405638e17 ;
static constexpr const double CU_to_epsCGS   = 8.98755178736818e20 ;
static constexpr const double CU_to_pressCGS = 5.55174079257738e38 ;

static constexpr const double c2_CGS   = 8.9875517873681764e+20;
static constexpr const double mnuc_MeV = 931.494061; 
static constexpr const double mnuc_CGS = 1.660539040e-24; 
static constexpr const double mnuc_Msun = mnuc_CGS * pressCGS_to_CU*(cm_to_CU*cm_to_CU*cm_to_CU)*c2_CGS;
static constexpr const double MeV_to_erg = 1.60217733e-6;
static constexpr const double cm3_to_fm3 = 1.0e39;
static constexpr const double avogadro  = 6.0221367e23;
static constexpr const double m_neutron_MeV = 939.565379;

static constexpr const double hbarc_MeV_fm = 197.326978812;

} } /* namespace grace::physical_constants */