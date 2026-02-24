/**
 * @file quadrant.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-01
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Discontinuous Galerkin
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


#ifndef GRACE_AMR_QUADRANT_HH 
#define GRACE_AMR_QUADRANT_HH 

#include <grace/utils/grace_utils.hh>
#include <grace/amr/amr_flags.hh>
#include <grace/amr/p4est_headers.hh> 

#include <array>

namespace grace { namespace amr { 
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
    std::array< p4est_qcoord_t, GRACE_NSPACEDIM > GRACE_ALWAYS_INLINE 
    qcoords(bool use_current_level=true) const 
    {   
        std::array< p4est_qcoord_t, GRACE_NSPACEDIM > ret ; 
        ret[0] = static_cast<p4est_qcoord_t>( _pquad -> x ) ; 
        ret[1] = static_cast<p4est_qcoord_t>( _pquad -> y ) ;  
        #ifdef GRACE_3D
        ret[2] = static_cast<p4est_qcoord_t>( _pquad -> z ) ;  
        #endif  
        if ( use_current_level ) {
            for(auto& xx: ret) xx = xx >> ( P4EST_MAXLEVEL - (int) _pquad->level ) ; 
        }  
        return ret ; 
    }
    //*****************************************************************************************************
    /**
     * @brief Get quadrant's spacing in tree-logical ([0,1]) coordinates.
     * 
     * @return double The quadrant's spacing.
     */
     double GRACE_ALWAYS_INLINE spacing() {
        return 1.0 / static_cast<double>(1<<level()) ; 
     }
    //*****************************************************************************************************
    /**
     * @brief Get quadrant's refinement level.
     * 
     * @return int The quadrant's level.
     */
    int GRACE_ALWAYS_INLINE level() const { return static_cast<int>( _pquad->level ) ; }
    //*****************************************************************************************************
    /**
     * @brief Return linear (morton) index of the quadrant
     *        in a uniform grid at a certain level.
     * 
     * @param level The level of the grid where the Morton 
     *              index is computed.
     * @return uint64_t Morton index of the quadrant.
     */
    uint64_t GRACE_ALWAYS_INLINE 
    linearid(int level) const {
        return p4est_quadrant_linear_id(_pquad, level) ; 
    }
    /**
     * @brief Get the index of tree containing this quadrant.
     */
    int64_t GRACE_ALWAYS_INLINE 
    tree_index() const { return _pquad->p.which_tree; }
    //*****************************************************************************************************
    /**
     * @brief Set user data of this quadrant.
     * \cond grace_detail
     * @tparam Type of user data.
     * The user data is used internally in GRACE to register amr 
     * information into the quadrants. This includes information about 
     * whether the quadrant has been modified by amr routines such as 
     * refinement and coarsening.
    */
    template< typename T > 
    [[deprecated]] void GRACE_ALWAYS_INLINE 
    set_user_data(T const & data) 
    {
        ASSERT(_pquad, "Quadrant pointer is null.") ; 
        ASSERT(_pquad->p.user_data, "User data is null.") ; 
        memcpy(_pquad->p.user_data, (void*) &data, sizeof(T)) ; 
    }
    //*****************************************************************************************************
    //*****************************************************************************************************
    /**
     * @brief Get user data of this quadrant.
     * \cond grace_detail
     * @tparam Type of user data.
     * NB: this is a simple <code>reinterpret_cast<T*></code>
     *     use at your own risk.
    */
    template< typename T > 
    [[deprecated]] GRACE_ALWAYS_INLINE T*  
    get_user_data() 
    {
        return reinterpret_cast<T*>(_pquad->p.user_data); 
    }
    //*****************************************************************************************************
    /**
     * @brief Set the regrid flag in the internal data structure of the quad.
     * @param[in] flag The value to set the flag to
     */
    void GRACE_ALWAYS_INLINE 
    set_regrid_flag(quadrant_flags_t flag) {
        auto pd = reinterpret_cast<grace_quadrant_user_data_t*>(_pquad->p.user_data);
        pd->regrid_flag = flag ; 
    }
    //*****************************************************************************************************
    /**
     * @brief Get the current value of the regrid flag for this quad 
     */
    quadrant_flags_t GRACE_ALWAYS_INLINE 
    get_regrid_flag() {
        auto pd = reinterpret_cast<grace_quadrant_user_data_t*>(_pquad->p.user_data);
        return pd->regrid_flag ; 
    }
    //*****************************************************************************************************
    /**
     * @brief Set the minimum allowed refinement level for this quadrant
     * @param[in] min_level Minimum allowed refinement level
     */
    void GRACE_ALWAYS_INLINE 
    set_min_level(int min_level) {
        auto pd = reinterpret_cast<grace_quadrant_user_data_t*>(_pquad->p.user_data);
        pd->min_level = min_level ; 
    }
    //*****************************************************************************************************
    /**
     * @brief Get the minumum allowed refinement level for this quadrant
     */
    int GRACE_ALWAYS_INLINE 
    get_min_level() {
        auto pd = reinterpret_cast<grace_quadrant_user_data_t*>(_pquad->p.user_data);
        return pd->min_level ; 
    }
    //*****************************************************************************************************
    //*****************************************************************************************************
    /**
     * @brief Set the user int of this quadrant
     * 
     * @param data What to set the data to.
     */
    void GRACE_ALWAYS_INLINE 
    set_user_int(int const & data) 
    {
        ASSERT(0, "Never touch the user int.") ; 
    }
    //*****************************************************************************************************
    /**
     * @brief Set the user long of this quadrant
     * 
     * @param data What to set the data to.
     */
    void GRACE_ALWAYS_INLINE 
    set_user_long(long const & data) 
    {
        ASSERT(0, "Never touch the user long.") ; 
    }
    //*****************************************************************************************************
    /**
     * @brief Get the user int of this quadrant.
     * 
     * @return int quadrant->p.user_int
     */
    int GRACE_ALWAYS_INLINE
    get_user_int() const 
    {
        ASSERT(0, "Never touch the user int.") ; 
    }
    /**
     * @brief Get the user long of this quadrant.
     * 
     * @return int quadrant->p.user_long
     */
    long GRACE_ALWAYS_INLINE
    get_user_long() const 
    {
        ASSERT(0, "Never touch the user long.") ; 
    }
    //*****************************************************************************************************
    /**
     * @brief For halo quadrants: get tree index.
     * \cond grace_detail
    */
    GRACE_ALWAYS_INLINE p4est_topidx_t  
    halo_owner_tree() 
    {
        return _pquad->p.piggy3.which_tree; 
    }
    //*****************************************************************************************************
    /**
     * @brief For halo quadrants: get owner's local index of this quadrant.
     * \cond grace_detail
    */
    GRACE_ALWAYS_INLINE p4est_topidx_t  
    halo_owner_locidx() 
    {
        return _pquad->p.piggy3.local_num; 
    }
    //*****************************************************************************************************

    //*****************************************************************************************************
    /**
     * @brief Get the raw pointer to the p4est_quadrant_t.
     * \cond grace_detail
    */
    GRACE_ALWAYS_INLINE p4est_quadrant_t*  
    get() 
    {
        return _pquad; 
    }
    //*****************************************************************************************************
} ; 
//*****************************************************************************************************
//*****************************************************************************************************
}} /* grace::amr */
 
#endif /* GRACE_AMR_QUADRANT_HH */