/**
 * @file amr_flags.hh
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

#ifndef AMR_FLAGS_HH 
#define AMR_FLAGS_HH

#include <grace/amr/p4est_headers.hh> 
#include <grace/errors/error.hh>

namespace grace { namespace amr { 

/**
 * @brief Possible quadrant states.
 * \cond grace_detail
 */
enum quadrant_flags_t : int8_t 
{
    DEFAULT_STATE=0,
    REFINE,
    COARSEN, 
    NEED_PROLONGATION,
    NEED_RESTRICTION,
    INVALID_STATE=-1
} ; 

/**
 * @brief Quadrant user data. Used in 
 *        GRACE to retain information
 *        about amr operations that 
 *        need to be transmitted to 
 *        the Device. 
 * \cond grace_detail
 * \ingroup amr 
 */
struct grace_quadrant_user_data_t {
    quadrant_flags_t regrid_flag ; //!< Regrid status
    int min_level ;                //!< Minimuma allowed level
} ; 

/**
 * @brief Initialize a quadrant to default state.
 * \cond grace_detail 
 * \ingroup amr 
 * @param p4est        The oct-tree forest. 
 * @param which_tree   Id of the tree where the quadrants live.
 * @param quad         Quadrant being initialized 
 */
static void initialize_quadrant(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quad)
{
    using namespace grace::amr ; 
    grace_quadrant_user_data_t* pd = 
            static_cast<grace_quadrant_user_data_t*>(quad->p.user_data) ; 
    pd->regrid_flag     = DEFAULT_STATE ; 
    pd->min_level       = 0 ; 
}

/**
 * @brief Flag quadrants in need of prolongation and/or restriction
 *        after refinement and coarsening.
 * \cond grace_detail 
 * \ingroup amr 
 * @param p4est        The oct-tree forest. 
 * @param which_tree   Id of the tree where the quadrants live.
 * @param num_outgoing Number of outgoing quadrants (1 for refinement, P4EST_CHILDREN for coarsening)
 * @param outgoing     Array of outgoing quadrants 
 * @param num_incoming Number of incoming quadrants (P4EST_CHILDREN for refinement, 1 for coarsening)
 * @param incoming     Array of incoming quadrants
 */ 
static void set_quadrant_flag( p4est_t* p4est
                             , p4est_topidx_t which_tree
                             , int num_outgoing
                             , p4est_quadrant_t* outgoing[] 
                             , int num_incoming 
                             , p4est_quadrant_t* incoming[] )
{

    if( num_outgoing == 1 and num_incoming == P4EST_CHILDREN ) // refinement 
    {
        grace_quadrant_user_data_t* pdo = 
                static_cast<grace_quadrant_user_data_t*>(outgoing[0]->p.user_data) ; 
        auto prev_state = 
            pdo->regrid_flag ; 
        auto min_level = pdo->min_level ; 
        pdo->regrid_flag = INVALID_STATE ; 

        for(int iquad=0; iquad<P4EST_CHILDREN; ++iquad) {
            grace_quadrant_user_data_t* pd = 
                static_cast<grace_quadrant_user_data_t*>(incoming[iquad]->p.user_data) ; 
            pd->regrid_flag     = (prev_state == NEED_RESTRICTION ? DEFAULT_STATE : NEED_PROLONGATION);
            pd->min_level       = min_level ; 
        } 
    } else if ( (num_outgoing == P4EST_CHILDREN) && (num_incoming == 1) ) // coarsening 
    {
        int prev_state = -1 ;
        int min_level = 0 ;
        for(int iquad=0; iquad<P4EST_CHILDREN; ++iquad) {
            if( outgoing[iquad] != nullptr )
            {
                grace_quadrant_user_data_t* pdo = 
                    static_cast<grace_quadrant_user_data_t*>(outgoing[iquad]->p.user_data) ; 
                prev_state = 
                    pdo->regrid_flag ; 
                min_level = std::max(min_level, pdo->min_level) ; 

                pdo->regrid_flag = INVALID_STATE ; 
            }
        }
        grace_quadrant_user_data_t* pd = 
            static_cast<grace_quadrant_user_data_t*>(incoming[0]->p.user_data) ; 
        pd->regrid_flag =
            prev_state==NEED_PROLONGATION ? DEFAULT_STATE : NEED_RESTRICTION;
        pd->min_level   =  min_level;
    } else {
        ERROR( "In call to initialize_quadrant, num_incoming"
               "and num_outgoing incompatible with both refinement and coarsening. ") ; 
    } 

}
/**
 * @brief Callback for quadrant refinement.
 * 
 * @param p4est Pointer to the forest.
 * @param which_tree Index of tree containing the quadrant.
 * @param quadrant Quadrant being processed.
 * @return int Returns <code>true</code> if the quadrant is flagged for refinement,
 *         <code>false</code> otherwise.
 */
static int refine_cback( p4est_t* p4est 
                       , p4est_topidx_t which_tree 
                       , p4est_quadrant_t * quadrant )
{
    grace_quadrant_user_data_t* pd = 
            static_cast<grace_quadrant_user_data_t*>(quadrant->p.user_data) ; 
    return  pd->regrid_flag == REFINE ; 
}

/**
 * @brief Callback for quadrant coarsening.
 * 
 * @param p4est Pointer to the forest.
 * @param which_tree Index of tree containing the quadrants.
 * @param quadrant Quadrants being processed.
 * @return int Returns <code>true</code> if more than half of the children 
 *         quadrants are flagged for coarsening, <code>false</code> otherwise.
 */
static int coarsen_cback( p4est_t* p4est 
                       , p4est_topidx_t which_tree 
                       , p4est_quadrant_t * quadrants[] )
{
    int ncoarsen{0};
    for( int ichild=0; ichild<P4EST_CHILDREN; ++ichild)
    {
        if( quadrants[ichild] == nullptr ) {
            continue ; 
        } else {
        grace_quadrant_user_data_t* pd = 
            static_cast<grace_quadrant_user_data_t*>(quadrants[ichild]->p.user_data) ; 
        ncoarsen += 
            (pd->regrid_flag == COARSEN);
        }
    }
     
    
    return  ncoarsen == P4EST_CHILDREN  ; 
}

}} /* grace::amr */
 

#endif 