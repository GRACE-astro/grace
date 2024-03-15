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
 * Designed to make access of members easier and safer for user code.
 */
class tree_t 
{
 //**************************************************************************************************
 public: 
   //**************************************************************************************************
   /**
    * @brief Construct a new tree_t starting from the p4est pointer.
    */
   tree_t(p4est_tree_t* ptree): _ptree(ptree) {} ;
   /**
    * @brief Return an array of p4est_quadrant_t containing the 
    *        local quadrants on this tree.
    */
   THUNDER_ALWAYS_INLINE sc_array_view_t<p4est_quadrant_t> quadrants() 
   {
      return sc_array_view_t<p4est_quadrant_t>{&(_ptree->quadrants)};  
   }
   /**
    * @brief Return the ith local quadrant on this tree.
    * 
    * @param iquad Index in the local quadrant list.
    * @return The quadrant_t object 
    */
   THUNDER_ALWAYS_INLINE quadrant_t quadrant( size_t iquad )
   {
      sc_array_view_t<p4est_quadrant_t> quads{&(_ptree->quadrants)} ; 
      ASSERT_DBG( iquad < quads.size()
               , "Requested out of bounds quadrant." ) ;
      return quadrant_t(&quads[iquad]) ; 
   }
   /**
    * @brief Return the local offset of the local quadrants on this tree. 
    * This is the cumulative sum of all quadrants from local trees on this rank.
    * All local <code>Kokkos::View</code> objects should be accessed with this 
    * offset. 
    */
   THUNDER_ALWAYS_INLINE size_t
   quadrants_offset() const { return _ptree->quadrants_offset ; } ;
   /**
    * @brief Return the number of local quadrants on this tree.
    */
   THUNDER_ALWAYS_INLINE size_t num_quadrants() 
   {
      return sc_array_view_t<p4est_quadrant_t>( &(_ptree->quadrants) ).size() ;  
   }
   /**
    * @brief Get the pointer to the underlying p4est_tree_t.
    */
   p4est_tree_t* get() const { return _ptree ; }

 //**************************************************************************************************
 private:
    p4est_tree_t * _ptree ; //!< Pointer to p4est_tree 
} ; 

} } /* thunder::amr */ 

 #endif /* THUNDER_AMR_TREE_HH */