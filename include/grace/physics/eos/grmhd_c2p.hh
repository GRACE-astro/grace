
/**
 * @file grmhd_c2p.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-11-16
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

#ifndef GRACE_PHYSICS_EOS_C2P_GRMHD_HH
#define GRACE_PHYSICS_EOS_C2P_GRMHD_HH

#include <grace_config.h>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/utils/rootfinding.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/eos/grmhd_c2p_kastaun.hh>
#include <grace/physics/eos/grmhd_c2p_pure_hydro.hh>

#include <Kokkos_Core.hpp>

namespace grace {

/**
 * @brief Generic interface for providing the c2p inversion routine  
 * @tparam eos_t Type of EOS.
 * @tparam c2p_formulation_t c2p scheme (Newman, Palenzuela, Kastaun, etc)
 * @details The `c2p_formulation_t` parameter must be a template class that takes a single 
 * template argument (`eos_t`). Examples include inversion schemes such as:
 * i)   Newman
 * ii)  Palenzuela
 * iii) Kastaun
 * Parameters passed on to the constructor of the c2p scheme are:
 * @param _eos EOS object passed to the inversion scheme.
 * @param _metric Metric utilities required for the inversion.
 * @param conservs Conservative variables at a single grid cell. (by reference in case of a subsequent p2c call, should such design choice be made )
 */


template< typename eos_t , template <typename> typename c2p_formulation_t >
struct grmhd_c2p_t {
 
    /** @brief 
     * Constructor - takes the same parameters as any underlying c2p scheme is supposed to
     * @param _eos EOS object passed to the inversion scheme.
     * @param _metric Metric utilities required for the inversion.
     * @param conservs Conservative variables at a single grid cell. (by reference in case of a subsequent p2c call, should such design choice be made )
     **/
    GRACE_HOST_DEVICE
    grmhd_c2p_t(eos_t const& _eos,
                metric_array_t const& _metric, 
                grmhd_cons_array_t& conservs ) : 
                c2p(_eos,_metric, conservs)
                {};

 
 
    grmhd_prims_array_t GRACE_HOST_DEVICE
    invert(double& error) {
        return c2p.invert(error);
    }

    private:
    
    c2p_formulation_t<eos_t> c2p; 
}


// Explicit template instantiation for hybrid eos: 
#define INSTANTIATE_TEMPLATE(EOS, C2P_SCHEME) \
extern template<> struct grmhd_c2p_t<EOS,C2P_SCHEME>; 

// hybrid eos
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>, \
                grmhd_c2p_kastaun<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>) ;

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>, \
                grhd_c2p<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>) ; // pure hydro c2p

#undef INSTANTIATE_TEMPLATE


}

#endif /* GRACE_PHYSICS_EOS_C2P_GRMHD_HH */