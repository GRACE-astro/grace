/**
 * @file forest.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-02-29
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#ifndef GRACE_AMR_FOREST_HH
#define GRACE_AMR_FOREST_HH

#include <grace_config.h> 

#include <grace/utils/inline.h>
#include <grace/utils/singleton_holder.hh> 
#include <grace/utils/creation_policies.hh>
#include <grace/utils/sc_wrappers.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/amr/p4est_headers.hh>

#include <grace/amr/tree.hh>
#include <grace/amr/quadrant.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/parallel/mpi_wrappers.hh> 

namespace grace { namespace amr {
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief Wrapper around the p4est structure.
 * 
 * \ingroup amr 
 * 
 */
//*****************************************************************************************************
class forest_impl_t 
{
 private:
   //*****************************************************************************************************
    p4est_t * _p4est ; //!< Pointer to the p4est object 
   //*****************************************************************************************************
 public: 
    //*****************************************************************************************************
    /**
     * @brief Get the trees array
     * 
     * @return sc_array_view_t<p4est_tree_t> Array view containing the local trees on this rank.
     * NB: the entries in this array are only valid starting at <code>first_local_tree</code> 
     *     and up to <code>last_local_tree</code> 
     */
    sc_array_view_t<p4est_tree_t> GRACE_ALWAYS_INLINE
    trees() const { return sc_array_view_t<p4est_tree_t>(_p4est->trees) ; } ; 
    //*****************************************************************************************************
    /**
     * @brief Get a <code>tree_t</code> containing the ith local tree
     * 
     * @param which_tree Tree index (must be a valid index of local tree)
     * 
     * @return Return the requested tree wrapped into a <code>tree_t</code>
     * NB: The index must be comprised between <code>first_local_tree</code> 
     *     and <code>last_local_tree</code>. This is only enforced in 
     *     debug mode.
     */
    tree_t GRACE_ALWAYS_INLINE
    tree(size_t which_tree) const { 
      ASSERT_DBG( which_tree >= _p4est->first_local_tree 
              and which_tree <= _p4est->last_local_tree,
              "Requested tree number " << which_tree << " but"
              " first local tree is " << _p4est->first_local_tree << " and"
              " last local tree is " << _p4est->last_local_tree << '\n' ) ; 
      return tree_t( &(trees()[which_tree]) ); 
    } ;
    //*****************************************************************************************************
    /**
     * @brief Get first valid index of tree array on this rank.
     */
    size_t GRACE_ALWAYS_INLINE 
    first_local_tree() const { return static_cast<size_t>( _p4est->first_local_tree) ; }
    //*****************************************************************************************************
    /**
     * @brief Get last valid index of tree array on this rank.
     */
    size_t GRACE_ALWAYS_INLINE
    last_local_tree() const { return static_cast<size_t>( _p4est->last_local_tree) ; }
    //*****************************************************************************************************
    /**
     * @brief Get number of local quadrants on this rank.
     */
    size_t GRACE_ALWAYS_INLINE
    local_num_quadrants() const { return static_cast<size_t>( _p4est->local_num_quadrants) ; }
    //*****************************************************************************************************
    /**
     * @brief Get pointer to underlying p4est object. 
     */
    GRACE_ALWAYS_INLINE p4est_t* 
    get() const { return _p4est ; }
   //*****************************************************************************************************
   /**
   * @brief Get pointer to underlying p4est object. 
   */
    GRACE_ALWAYS_INLINE int
    global_quadrant_offset(size_t rank) const 
    { 
      return _p4est->global_first_quadrant[rank] ; 
    }
   //*****************************************************************************************************
   GRACE_ALWAYS_INLINE static_readonly_view_t<size_t,4> 
   get_grid_properties() const {return _grid_properties ; }
 private:
   //*****************************************************************************************************
    /**
     * @brief Never construct a new forest_impl_t object
     */
    forest_impl_t() ; 
    //*****************************************************************************************************
    /**
     * @brief Never construct a new forest_impl_t object
     */
    forest_impl_t(p4est_t * _forest_ptr) ; 
   //*****************************************************************************************************
    /**
     * @brief Never destroy the forest_impl_t object
     * 
     */
    ~forest_impl_t() ; 
   //*****************************************************************************************************
    friend class utils::singleton_holder<forest_impl_t, memory::default_create> ;          //!< Give access 
    friend class memory::new_delete_creator<forest_impl_t, memory::new_delete_allocator> ; //!< Give access
    static constexpr unsigned int longevity = AMR_FOREST ; //!< Longevity of p4est object. 
   //*****************************************************************************************************
   static_readonly_view_t<size_t,4> _grid_properties ; 
   //*****************************************************************************************************
} ; 
//*****************************************************************************************************
/**
 * @brief GRACE forest singleton type. This 
 *        global object can be accessed from user code 
 *        to get information about the grid structure.
 * \ingroup amr
 */
using forest = utils::singleton_holder<forest_impl_t > ; 
//*****************************************************************************************************
//*****************************************************************************************************
}} /* grace::amr */

#endif /* GRACE_AMR_FOREST_HH */
