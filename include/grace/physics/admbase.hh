/**
 * @file admbase.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-11
 * 
 * @copyright This file is part of MagMA.
 * MagMA is an evolution framework that uses Discontinuous Galerkin
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

#ifndef GRACE_PHYSICS_ADMBASE_HH 
#define GRACE_PHYSICS_ADMBASE_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/grace_utils.hh>
#include <grace/errors/error.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/config/config_parser.hh>

#include <string>

namespace grace 
{

enum static_metric_t {
    MINKOWSKI=0,
    NUM_STATIC_METRICS
} ; 

class adm_equations_system_t {

 public:
    adm_equations_system_t()  = default ; 

    adm_equations_system_t(grace::var_array_t<GRACE_NSPACEDIM> const aux)
        : _aux(aux)
    {
        std::string metric_type = 
            grace::get_param<std::string>("admbase","metric_kind") ;

        if ( metric_type == "Minkowski" ) {
            _static_metric_t = MINKOWSKI ; 
        } else {
            ERROR("Metric type " << metric_type << " not supported.") ;
        }
    }; 

    ~adm_equations_system_t() = default ;

    /**
     * @brief Compute GRMHD auxiliary quantities.
     *        This is essentially a call to c2p.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param q Quadrant index.
     */
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( auxiliaries_computation_kernel_t _tag
               , VEC( const int i 
               ,      const int j 
               ,      const int k)
               , int64_t q ) const
    {
        if ( _static_metric_t == MINKOWSKI ) { 
            _aux(VEC(i,j,k),GXX_,q) = 1. ;
            _aux(VEC(i,j,k),GYY_,q) = 1. ;
            _aux(VEC(i,j,k),GZZ_,q) = 1. ;
            _aux(VEC(i,j,k),GXY_,q) = 0. ;
            _aux(VEC(i,j,k),GXZ_,q) = 0. ;
            _aux(VEC(i,j,k),GYZ_,q) = 0. ;

            _aux(VEC(i,j,k),KXX_,q) = 0. ;
            _aux(VEC(i,j,k),KYY_,q) = 0. ;
            _aux(VEC(i,j,k),KZZ_,q) = 0. ;
            _aux(VEC(i,j,k),KXY_,q) = 0. ;
            _aux(VEC(i,j,k),KXZ_,q) = 0. ;
            _aux(VEC(i,j,k),KYZ_,q) = 0. ;

            _aux(VEC(i,j,k),BETAX_,q) = 0. ;
            _aux(VEC(i,j,k),BETAY_,q) = 0. ;
            _aux(VEC(i,j,k),BETAZ_,q) = 0. ;

            _aux(VEC(i,j,k),ALP_,q) = 1. ;
        }
    }

 private:
    grace::var_array_t<GRACE_NSPACEDIM> _aux ; 
    bool   _metric_is_static{true} ; 
    size_t _static_metric_t ; 

} ; 

void set_admbase_id() ;


} /* namespace grace */

 #endif /* GRACE_PHYSICS_ADMBASE_HH */