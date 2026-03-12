/**
 * @file connectivity.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Thin wrapper around a p4est_connectivity.
 * @version 0.1
 * @date 2023-03-13
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

#ifndef GRACE_AMR_CONNECTIVITY_HH
#define GRACE_AMR_CONNECTIVITY_HH

#include <grace_config.h>

#include <grace/amr/p4est_headers.hh>
#include <grace/errors/assert.hh>

#include <grace/utils/inline.h>
#include <grace/utils/singleton_holder.hh>
#include <grace/utils/creation_policies.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <grace/amr/connectivities_impl.hh>

#include <array>

namespace grace { namespace amr { 
//**************************************************************************************************
/**
 * \defgroup amr Grid handling routines.
 * \details See the [full documentation](doc/amr.md) for details.
 */
//**************************************************************************************************
/**
 * @brief Wrapper around p4est connectivity. 
 * \ingroup amr 
 *
 * In the code this is wrapped by singleton_holder. See p4est_connectivity.h for API information.
 */
class connectivity_impl_t
{
    //**************************************************************************************************
    public:
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief check integrity of stored connectivity 
     * 
     * @return true if the connectivity is valid
     * @return false if the connectivity is invalid
     */
    bool is_valid() const {return static_cast<bool>( p4est_connectivity_is_valid(pconn_) ) ;} ; 
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief get raw pointer to p4est connectivity
     * 
     * @return p4est_connectivity_t* the connectivity 
     */
    p4est_connectivity_t* get() { return pconn_ ; }; 
    //**************************************************************************************************

     
    //**************************************************************************************************
    /**
     * @brief Get coordinates of tree vertex 
     * 
     * @param which_tree Tree index 
     * @param which_vertex Vertex index (in Morton ordering)
     * @return The coordinates of the requested vertex.
     */
    GRACE_ALWAYS_INLINE std::array<double,GRACE_NSPACEDIM> 
    vertex_coordinates( size_t which_tree, size_t which_vertex ) const { 
      size_t nv = get_tree_vertex(which_tree, which_vertex) ; 
      #ifndef GRACE_3D 
        return std::array<double, 2> {  pconn_->vertices[ 3UL*nv     ]
                                     ,  pconn_->vertices[ 3UL*nv + 1UL] } ;
      #else 
        return std::array<double, 3> {  pconn_->vertices[ 3UL*nv     ]
                                     ,  pconn_->vertices[ 3UL*nv + 1UL] 
                                     ,  pconn_->vertices[ 3UL*nv + 2UL] } ;
      #endif 
    }; 
    //**************************************************************************************************
    /**
     * @brief Get coordinate extents of given tree.
     * @param which_tree Tree index 
     * @return Array containing coordinate extents of the tree in each direction.
     */
    GRACE_ALWAYS_INLINE std::array<double, GRACE_NSPACEDIM> 
    tree_coordinate_extents(size_t which_tree) const { 
      ASSERT_DBG( which_tree < pconn_->num_trees,
                  "In tree_coordinate_extents: "
                  "requested tree " << which_tree 
                  << "does not exist (num_trees= " << pconn_->num_trees << ")" ) ; 
      auto const l_coords = vertex_coordinates(which_tree, 0UL) ; 
      auto const x_l = l_coords[0] ; 
      auto const x_r = vertex_coordinates(which_tree, 1UL)[0] ; 
      auto const y_l = l_coords[1] ; 
      auto const y_r = vertex_coordinates(which_tree, 2UL)[1] ; 
      #ifndef GRACE_3D
      return std::array<double,2> { x_r-x_l, y_r-y_l } ; 
      #else 
      auto const z_l = l_coords[2] ; 
      auto const z_r = vertex_coordinates(which_tree, 4UL)[2] ; 
      return std::array<double,3> { x_r-x_l, y_r-y_l, z_r-z_l } ;
      #endif 
    };
    //**************************************************************************************************
    /**
     * @brief Find tree connecting across a face.
     * @param which_tree Tree index
     * @param which_face Face index in z-order 
     * @return Index of tree across requested face.
     */
    GRACE_ALWAYS_INLINE int
    tree_to_tree(size_t which_tree, int which_face) const { 
      return pconn_->tree_to_tree[ which_tree * P4EST_FACES + which_face ] ;  
    };
    //**************************************************************************************************
    /**
     * @brief Find face connecting across a face.
     * @param which_tree Tree index
     * @param which_face Face index in z-order 
     * @return Index of face across requested face.
     */
    GRACE_ALWAYS_INLINE int8_t
    tree_to_face(size_t which_tree, int which_face) const { 
      return pconn_->tree_to_face[ which_tree * P4EST_FACES + which_face ] % P4EST_FACES;  
    };
    //**************************************************************************************************
    /**
     * @brief Determine whether coordinates flip across tree boundary.
     * @param which_tree Tree index
     * @param which_face Face index in z-order 
     * @return 1 if trees have opposite polarity, 0 otherwise
     */
    GRACE_ALWAYS_INLINE int
    tree_to_tree_polarity(size_t which_tree, int which_face) const { 
      return t2t_polarity_[ which_tree * P4EST_FACES + which_face ] ;  
    };
    //**************************************************************************************************
    /**
     * @brief Determine whether coordinates flip across tree boundary.
     * @param which_tree Tree index
     * @param which_face Face index in z-order 
     * @return 1 if trees have opposite polarity, 0 otherwise
     */
    GRACE_ALWAYS_INLINE std::vector<int>
    tree_to_tree_polarity_array() const { 
      return t2t_polarity_ ;  
    };
    //**Checkpointing***********************************************************************************
    /**
     * @brief save to file 
     *
     * @param fname name of file where to save the connectivity. 
     */
    void save(std::string const& fname) const { 
      ASSERT( ! p4est_connectivity_save( fname.c_str(), pconn_ ), "Connectivity could not be written to disk."  ) ; 
    };
    //**************************************************************************************************
    /**
     * @brief load from file 
     *
     * @param fname name of file from which to load the connectivity. 
     */
    void load(std::string const& fname) { 
      ASSERT( 0, "Not implemented."  ) ; 
    }; 
    //**************************************************************************************************
    
    //**************************************************************************************************
    private:
    //**************************************************************************************************
    /**
     * @brief Get tree_to_vertex entry 
     * \cond grace_detail
     * @param which_tree Tree index 
     * @param which_vertex Vertex index (in Morton ordering)
     * @return The index of the requested vertex
     */
    GRACE_ALWAYS_INLINE size_t
    get_tree_vertex(size_t which_tree, size_t which_vertex) const { 
        return pconn_->tree_to_vertex[ which_tree * P4EST_CHILDREN  + which_vertex ] ; 
    }; 
    //**************************************************************************************************
    static constexpr unsigned int longevity = AMR_CONNECTIVITY ; //!< Longevity 
    //**************************************************************************************************
    /**
     * @brief Construct a new connectivity object
     * 
     * Only used by \code singleton_holder::initialize() \endcode.
     */
    connectivity_impl_t() ; 
    /**
     * @brief Construct a new connectivity object
     * 
     * Only used by \code singleton_holder::initialize() \endcode.
     */
    connectivity_impl_t(p4est_connectivity_t * _pconn) ; 
    //**************************************************************************************************
    /**
     * @brief Destroy the connectivity object
     * 
     * Only used by \code ~singleton_holder \endcode.
     */
    ~connectivity_impl_t() = default;
    //**************************************************************************************************
    p4est_connectivity_t * pconn_  ;  //!< The p4est connectivity object 
    std::vector<int> t2t_polarity_ ;  //!< Polarity of tree boundaries 
    //**************************************************************************************************
    friend class utils::singleton_holder<connectivity_impl_t, memory::default_create> ; //!< Give access 
    friend class memory::new_delete_creator<connectivity_impl_t, memory::new_delete_allocator> ;
};
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Global connectivity object. Used as interface from user code to 
 *        obtain access to the unique connectivity instance at runtime.
 * \ingroup amr 
 */
using connectivity = utils::singleton_holder< connectivity_impl_t > ; 
//*****************************************************************************************************
//*****************************************************************************************************
namespace detail {
#ifdef GRACE_3D 
enum spherical_grid_tree_t {
  CARTESIAN_TREE=0,
  MX_TREE,
  PX_TREE,
  MY_TREE,
  PY_TREE,
  MZ_TREE,
  PZ_TREE,
  MXL_TREE,
  PXL_TREE,
  MYL_TREE,
  PYL_TREE,
  MZL_TREE,
  PZL_TREE,
  NUM_TREES 
} ; 
#else 
enum spherical_grid_tree_t {
  CARTESIAN_TREE=0,
  MX_TREE,
  PX_TREE,
  MY_TREE,
  PY_TREE,
  MXL_TREE,
  PXL_TREE,
  MYL_TREE,
  PYL_TREE,
  NUM_TREES 
} ;
#endif 
}

} } // namespace grace::amr 

#endif /* GRACE_AMR_CONNECTIVITY_HH */