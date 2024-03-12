/**
 * @file quadrant.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-01
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


#ifndef THUNDER_AMR_QUADRANT_HH 
#define THUNDER_AMR_QUADRANT_HH 

#include <thunder/amr/p4est_headers.hh> 

namespace thunder { namespace amr { 

template< size_t ndim = THUNDER_NSPACEDIM > 
class quadrant_t
{
 private:
    p4est_quadrant_t * pquad_ ; 

 public: 

    quadrant_t( p4est_quadrant_t * quad_ ) : pquad_(quad_) {} ; 

    ~quadrant_t() = default; 

    std::array< p4est_qcoord_t, ndim > THUNDER_ALWAYS_INLINE 
    qcoords(bool use_current_level=true)
    {   
        std::array< p4est_qcoord_t, ndim > ret ; 
        ret[0] = static_cast<p4est_qcoord_t>( pquad_ -> x ) ; 
        ret[1] = static_cast<p4est_qcoord_t>( pquad_ -> y ) ;  
        if constexpr ( ndim == 3 ) { 
            ret[2] = static_cast<p4est_qcoord_t>( pquad_ -> z ) ;  
        }  
        if ( use_current_level ) {
            for(auto& xx: ret) xx >> ( P4EST_MAXLEVEL - (int) pquad_->level ) ; 
        }  
        return ret ; 
    }

    int THUNDER_ALWAYS_INLINE level() { return static_cast<int>( pquad_->level ) ; }

    uint64_t THUNDER_ALWAYS_INLINE 
    get_linearid(int level) {
        return p4est_quadrant_linear_id(pquad_, level) ; 
    }

    template< typename T > 
    void THUNDER_ALWAYS_INLINE 
    set_user_data(T const & data) 
    {
        memcpy(pquad_->p.user_data, (void*) &data, sizeof(T)) ; 
    }
} ; 

}} /* thunder::amr */
 
#endif /* THUNDER_AMR_QUADRANT_HH */