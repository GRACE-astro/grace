/**
 * @file tree.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-02-29
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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

 #ifndef THUNDER_AMR_TREE_HH 
 #define THUNDER_AMR_TREE_HH


#include <thunder/utils/inline.h>
#include <thunder/utils/device.h> 
#include <thunder/utils/sc_wrappers.hh>
#include <thunder_config.h>

#include <thunder/amr/p4est_headers.hh>
#include <thunder/amr/quadrant.hh> 

namespace thunder { namespace amr {

//**************************************************************************************************
/**
 * @brief Thin wrapper around p4est_tree_t 
 * \ingroup amr 
 * 
 * @tparam ndim Number of spatial dimensions. 
 * 
 */
template< std::size_t ndim = THUNDER_NSPACEDIM> // number of spatial dimensions 
class tree_t 
{
 //**************************************************************************************************
 public: 
 //**************************************************************************************************
 tree_t(p4est_tree_t* _ptree): ptree_(_ptree) {} ;

 THUNDER_ALWAYS_INLINE sc_array_view_t<p4est_quadrant_t*> get_quadrants() 
 {
    return sc_array_view_t<p4est_quadrant_t*>( _ptree->quadrants ) ;  
 }

 THUNDER_ALWAYS_INLINE quadrant_t<ndim> get_quadrant( size_t iquad )
 {
    sc_array_view_t<p4est_quadrant_t*> quads{ _ptree->quadrants } ; 
    ASSERT_DBG( iquad < quads.size()
              , "Requested out of bounds quadrant." ) ;
    return quadrant_t<ndim>(quads[i]) ; 
 }
 p4est_tree_t* get() const { return ptree_ ; }


 //**************************************************************************************************
 private:
    p4est_tree_t * ptree_ ; //!< Pointer to p4est_tree 
} ; 

} } /* thunder::amr */ 

 #endif /* THUNDER_AMR_TREE_HH */