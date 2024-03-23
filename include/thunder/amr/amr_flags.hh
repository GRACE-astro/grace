/**
 * @file amr_flags.hh
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

#ifndef AMR_FLAGS_HH 
#define AMR_FLAGS_HH

#include <thunder/amr/p4est_headers.hh> 
#include <thunder/amr/quadrant.hh> 
#include <thunder/errors/error.hh>

namespace thunder { namespace amr { 

/**
 * @brief Possible quadrant states.
 * \cond thunder_detail
 */
enum quadrant_flags_t 
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
 *        Thunder to retain information
 *        about amr operations that 
 *        need to be transmitted to 
 *        the Device. 
 * \cond thunder_detail
 * \ingroup amr 
 */
struct amr_flags_t {
    quadrant_flags_t quadrant_status ; 
} ; 

/**
 * @brief Initialize a quadrant to default state.
 * \cond thunder_detail 
 * \ingroup amr 
 * @param p4est        The oct-tree forest. 
 * @param which_tree   Id of the tree where the quadrants live.
 * @param quad         Quadrant being initialized 
 */
static void initialize_quadrant(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quad)
{
    quadrant_t quadrant(quad) ; 
    quadrant.set_user_data( amr_flags_t{DEFAULT_STATE} ) ; 
}

/**
 * @brief Flag quadrants in need of prolongation and/or restriction
 *        after refinement and coarsening.
 * \cond thunder_detail 
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
        quadrant_t quadrant( outgoing[0] ) ;
        auto prev_state = 
            quadrant.get_user_data<amr_flags_t>()->quadrant_status ; 
        quadrant.set_user_data( amr_flags_t{INVALID_STATE} ) ;
        for(int iquad=0; iquad<P4EST_CHILDREN; ++iquad) {
            quadrant = quadrant_t(incoming[iquad] ) ; 
            quadrant.set_user_data( amr_flags_t{
                prev_state == NEED_RESTRICTION ? DEFAULT_STATE : NEED_PROLONGATION
                } ) ;
        } 
    } else if ( num_outgoing == P4EST_CHILDREN and num_incoming == 1 ) // coarsening 
    {
        int prev_state = -1 ;
        for(int iquad=0; iquad<P4EST_CHILDREN; ++iquad) {
            if( outgoing[iquad] != nullptr )
            {
                quadrant_t quadrant = quadrant_t(outgoing[iquad] ) ;
                prev_state = 
                    quadrant.get_user_data<amr_flags_t>()->quadrant_status ; 
                quadrant.set_user_data( amr_flags_t{INVALID_STATE} ) ;
            }
        }
        quadrant_t quadrant( incoming[0] ) ; 
        quadrant.set_user_data( amr_flags_t{
            prev_state==NEED_PROLONGATION ? DEFAULT_STATE : NEED_RESTRICTION
            } ) ;
    } else {
        ERROR( "In call to initialize_quadrant, num_incoming"
               "and num_outgoing incompatible with both refinement and coarsening. ") ; 
    } 

}
/**
 * @brief Callback for quadrant refinement.
 * 
 * @param p4est 
 * @param which_tree 
 * @param quadrant 
 * @return int 
 */
static int refine_cback( p4est_t* p4est 
                       , p4est_topidx_t which_tree 
                       , p4est_quadrant_t * quadrant )
{
    quadrant_t quad{quadrant} ; 
    return  quad.get_user_data<amr_flags_t>()->quadrant_status == REFINE ; 
}

/**
 * @brief Callback for quadrant coarsening.
 * 
 * @param p4est 
 * @param which_tree 
 * @param quadrant 
 * @return int 
 */
static int coarsen_cback( p4est_t* p4est 
                       , p4est_topidx_t which_tree 
                       , p4est_quadrant_t * quadrants[] )
{
    int ncoarsen ;
    for( int ichild=0; ichild<P4EST_CHILDREN; ++ichild)
    {
        if( quadrants[ichild] == nullptr ) {
            continue ; 
        } else {
        quadrant_t quad(quadrants[ichild]) ; 
        ncoarsen += 
            (quad.get_user_data<amr_flags_t>()->quadrant_status == COARSEN);
        }
    }
     
    
    return  ncoarsen > (P4EST_CHILDREN/2)  ; 
}

}} /* thunder::amr */
 

#endif 