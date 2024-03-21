/**
 * @file quadrant.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-01
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Discontinuous Galerkin
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

#include <thunder/utils/thunder_utils.hh>

#include <thunder/amr/p4est_headers.hh> 

#include <array>

namespace thunder { namespace amr { 
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief Thin wrapper around p4est_quadrant_t* 
 * \ingroup amr  
 */
class quadrant_t
{
 private:
    //*****************************************************************************************************
    p4est_quadrant_t * _pquad ; //!< Pointer to underlying p4est object
    //*****************************************************************************************************
 public: 
    //*****************************************************************************************************
    /**
     * @brief Construct a new quadrant_t object
     * 
     * @param pquad Pointer to p4est_quadrant_t
     */
    quadrant_t( p4est_quadrant_t * pquad) : _pquad(pquad) {
        ASSERT_DBG(_pquad!=nullptr, "Quadrant object initialized with dangling pointer.") ;
    } ; 
    //*****************************************************************************************************
    /**
     * @brief Destroy the quadrant_t object
     */
    ~quadrant_t() = default; 
    //*****************************************************************************************************
    /**
     * @brief Get integer quadrant coordinates.
     * 
     * @param use_current_level Return coordinates at quadrant's level
     *                          as opposed to P4EST_MAXLEVEL. 
     * @return Array containing the integer coordinates of the quadrant 
     *         in a uniform grid at its level or at P4EST_MAXLEVEL.
     */
    std::array< p4est_qcoord_t, THUNDER_NSPACEDIM > THUNDER_ALWAYS_INLINE 
    qcoords(bool use_current_level=true) const 
    {   
        std::array< p4est_qcoord_t, THUNDER_NSPACEDIM > ret ; 
        ret[0] = static_cast<p4est_qcoord_t>( _pquad -> x ) ; 
        ret[1] = static_cast<p4est_qcoord_t>( _pquad -> y ) ;  
        #ifdef THUNDER_3D
        ret[2] = static_cast<p4est_qcoord_t>( _pquad -> z ) ;  
        #endif  
        if ( use_current_level ) {
            for(auto& xx: ret) xx = xx >> ( P4EST_MAXLEVEL - (int) _pquad->level ) ; 
        }  
        return ret ; 
    }
    //*****************************************************************************************************
    /**
     * @brief Get quadrant's refinement level.
     * 
     * @return int The quadrant's level.
     */
    int THUNDER_ALWAYS_INLINE level() const { return static_cast<int>( _pquad->level ) ; }
    //*****************************************************************************************************
    /**
     * @brief Return linear (morton) index of the quadrant
     *        in a uniform grid at a certain level.
     * 
     * @param level The level of the grid where the Morton 
     *              index is computed.
     * @return uint64_t Morton index of the quadrant.
     */
    uint64_t THUNDER_ALWAYS_INLINE 
    linearid(int level) const {
        return p4est_quadrant_linear_id(_pquad, level) ; 
    }
    /**
     * @brief Get the index of tree containing this quadrant.
     */
    int64_t THUNDER_ALWAYS_INLINE 
    tree_index() const { return _pquad->p.which_tree; }
    //*****************************************************************************************************
    /**
     * @brief Set user data of this quadrant.
     * \cond thunder_detail
     * @tparam Type of user data.
     * The user data is used internally in Thunder to register amr 
     * information into the quadrants. This includes information about 
     * whether the quadrant has been modified by amr routines such as 
     * refinement and coarsening.
    */
    template< typename T > 
    void THUNDER_ALWAYS_INLINE 
    set_user_data(T const & data) 
    {
        memcpy(_pquad->p.user_data, (void*) &data, sizeof(T)) ; 
    }
    //*****************************************************************************************************
    //*****************************************************************************************************
    /**
     * @brief Get user data of this quadrant.
     * \cond thunder_detail
     * @tparam Type of user data.
     * NB: this is a simple <code>reinterpret_cast<T*></code>
     *     use at your own risk.
    */
    template< typename T > 
    THUNDER_ALWAYS_INLINE T*  
    get_user_data() 
    {
        return reinterpret_cast<T*>(_pquad->p.user_data); 
    }
    //*****************************************************************************************************
    /**
     * @brief For halo quadrants: get tree index.
     * \cond thunder_detail
    */
    THUNDER_ALWAYS_INLINE p4est_topidx_t  
    halo_owner_tree() 
    {
        return _pquad->p.piggy3.which_tree; 
    }
    //*****************************************************************************************************
    /**
     * @brief For halo quadrants: get owner's local index of this quadrant.
     * \cond thunder_detail
    */
    THUNDER_ALWAYS_INLINE p4est_topidx_t  
    halo_owner_locidx() 
    {
        return _pquad->p.piggy3.local_num; 
    }
    //*****************************************************************************************************
    /**
     * @brief For halo quadrants: get owner's rank.
     * \cond thunder_detail
    */
    THUNDER_ALWAYS_INLINE int  
    halo_owner_rank() 
    {
        return _pquad->p.piggy1.owner_rank; 
    }
    //*****************************************************************************************************

    //*****************************************************************************************************
    /**
     * @brief Get the raw pointer to the p4est_quadrant_t.
     * \cond thunder_detail
    */
    THUNDER_ALWAYS_INLINE p4est_quadrant_t*  
    get() 
    {
        return _pquad; 
    }
    //*****************************************************************************************************
} ; 
//*****************************************************************************************************
//*****************************************************************************************************
}} /* thunder::amr */
 
#endif /* THUNDER_AMR_QUADRANT_HH */