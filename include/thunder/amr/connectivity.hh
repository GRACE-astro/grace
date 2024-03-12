/**
 * @file connectivity.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Thin wrapper around a p4est_connectivity.
 * @version 0.1
 * @date 2023-03-13
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

#ifndef THUNDER_AMR_CONNECTIVITY_HH
#define THUNDER_AMR_CONNECTIVITY_HH

#include <thunder_config.h>

#include <thunder/amr/p4est_headers.hh>

#include <thunder/utils/inline.h>
#include <thunder/utils/singleton_holder.hh>
#include <thunder/utils/creation_policies.hh>
#include <thunder/utils/lifetime_tracker.hh>

#include <thunder/amr/connectivities_impl.hh>

namespace thunder { namespace amr { 
//**************************************************************************************************
/**
 * \defgroup amr Grid handling routines.
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
     * @brief get raw pointer to p4est connectivity
     * 
     * @return p4est_connectivity_t* the connectivity 
     */
    THUNDER_ALWAYS_INLINE std::array<double, THUNDER_NSPACEDIM> 
    get_tree_vertex(size_t which_tree, size_t which_vertex) const { 
      #ifndef THUNDER_3D 
        return std::array<double, 2> {  pconn_->tree_to_vertex[ which_tree * P4EST_CHILDREN  + which_vertex      ]
                                     ,  pconn_->tree_to_vertex[ which_tree * P4EST_CHILDREN  + which_vertex + 1UL] } ;
      #else 
        return std::array<double, 3> {  pconn_->tree_to_vertex[ which_tree * P4EST_CHILDREN  + 3UL * which_vertex      ]
                                     ,  pconn_->tree_to_vertex[ which_tree * P4EST_CHILDREN  + 3UL * which_vertex + 1UL]
                                     ,  pconn_->tree_to_vertex[ which_tree * P4EST_CHILDREN  + 3UL * which_vertex + 2UL] } ;
      #endif 
    }; 
    //**************************************************************************************************

    THUNDER_ALWAYS_INLINE std::array<double,THUNDER_NSPACEDIM> 
    get_vertex_coordinates( size_t which_tree, size_t which_vertex ) const { 
      size_t nv = pconn_->tree_to_vertex[ P4EST_CHILDREN * which_tree + which_vertex ] ; 
      #ifndef THUNDER_3D 
        return std::array<double, 2> {  pconn_->vertices[ 3UL*nv     ]
                                     ,  pconn_->vertices[ 3UL*nv + 1UL] } ;
      #else 
        return std::array<double, 3> {  pconn_->vertices[ 3UL*nv     ]
                                     ,  pconn_->vertices[ 3UL*nv + 1UL] 
                                     ,  pconn_->vertices[ 3UL*nv + 2UL] } ;
      #endif 
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
    static constexpr unsigned int longevity = thunder::AMR_CONNECTIVITY ; //!< Longevity 
    //**************************************************************************************************
    /**
     * @brief Construct a new connectivity object
     * 
     * Only used by \code singleton_holder::initialize() \endcode.
     */
    connectivity_impl_t() ; 
    //**************************************************************************************************
    /**
     * @brief Destroy the connectivity object
     * 
     * Only used by \code ~singleton_holder \endcode.
     */
    ~connectivity_impl_t() = default;
    //**************************************************************************************************
    p4est_connectivity_t * pconn_ ;  //!< The p4est connectivity object 
    //**************************************************************************************************
    friend class utils::singleton_holder<connectivity_impl_t, memory::default_create> ; //!< Give access 
    friend class memory::new_delete_creator<connectivity_impl_t, memory::new_delete_allocator> ;
};
//**************************************************************************************************

//**************************************************************************************************
using connectivity = utils::singleton_holder< connectivity_impl_t > ; 

} } // namespace thunder::amr 

#endif 