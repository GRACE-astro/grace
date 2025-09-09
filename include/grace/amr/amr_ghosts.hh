/**
 * @file amr_ghosts.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Index fiesta.
 * @date 2025-09-05
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
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

#ifndef GRACE_AMR_AMR_GHOSTS_HH
#define GRACE_AMR_AMR_GHOSTS_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/config/config_parser.hh>
#include <grace/errors/assert.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <array> 
namespace grace {
/**************************************************************************************************/
enum interface_kind_t : uint8_t { PHYS, INTERNAL }  ;
/**************************************************************************************************/
struct full_face_t {
        std::size_t quad_id ; //!< Index of quadrant on the other side 
        bool is_remote      ; //!< Whether the quadrant is local or remote 
} ; 
struct hanging_face_t {
        std::size_t quad_id[P4EST_CHILDREN/2] ; //!< Indices of hanging quads (in coarse bufs)
        bool is_remote[P4EST_CHILDREN/2]      ; //!< Are the quads remote 
} ; 
union face_data_t {
    full_face_t full ; 
    hanging_face_t hanging ; 
    face_data_t() = default ;
} ; 
/**
* @brief This class describes a face of a quadrant 
 */
struct face_descriptor_t { 
    interface_kind_t kind ; 
    uint8_t face        ; //!< Face code 
    int8_t level_diff   ; //!< Ref level difference (+-1 or 0)
    face_data_t data    ; //!< Quadrant ids 
} ; 
/**************************************************************************************************/
struct edge_descriptor_t ; 
struct corner_descriptor_t ; 
/**************************************************************************************************/
struct quad_neighbors_descriptor_t {
    std::array< face_descriptor_t, P4EST_FACES > faces ; //!< Faces 
    #ifdef GRACE_3D 
    std::array< edge_descriptor_t, 12 > edges ; //!< edges 
    #endif 
    std::array< corner_descriptor_t, P4EST_CHILDREN> corners; //!< Corners
    std::size_t quad_id ; //!< Quadrant id 
    std::size_t cbuf_id ; //!< For fine quads only: index into coarse buffer array 
}  ;  
/**************************************************************************************************/
struct face_descriptor_t {
    using quadid_t = long long int ; 


    

    uint8_t faceA, faceB ; //!< Face codes 0,...,5 in z-order 

    // for hanging only:
    uint8_t coarse_remote     ; //!< Is the coarse side remote?
    uint8_t child_remote_mask ; //!< bitmask for remote children on fine side

    quadid_t child_rank[P4EST_CHILDREN/2] ; //!< Children rank if remote i

    quadid_t qid_a{-1}, qid_b{-1} ; //!< For simple only: quad ids 
    quadid_t qid_c{-1}, qid_f[P4EST_CHILDREN/2] ; //!< For hanging only: quad ids
    

    face_descriptor_t()
    {
        faceA = faceB = 0 ; 
        kind = INVALID ; loc = INVALID ; 
        qid_a = qid_b = -1 ; 
        child_rank = {
              -1, -1
            #ifdef GRACE_3D 
            , -1, -1 
            #endif 
        } ; 
        qid_f = {
              -1, -1
            #ifdef GRACE_3D 
            , -1, -1 
            #endif 
        } ;
    }
} ; 
/**************************************************************************************************/
struct edge_descriptor_t ;
/**************************************************************************************************/
struct corner_descriptor_t ; 
/**************************************************************************************************/
struct remote_data_descriptor_t {
    enum top_t : uint8_t { 
        FACE, EDGE, CORNER 
    } topology ; //!< Remote element topology 
    enum op_t : uint8_t {
        SEND, RECV 
    } op ; //!< Is this a send / receive data unit 
    std::size_t rank ;    //!< Rank this data comes from / goes to 
    std::size_t loc_qid ; //!< Local quadrant id
} ;  
/**************************************************************************************************/
struct neighbor_descriptor_t {
    std::vector<face_descriptor_t> faces     ; //!< Face descriptors 
    std::vector<edge_descriptor_t> edges     ; //!< Edge descriptors 
    std::vector<corner_descriptor_t> corners ; //!< Corner descriptors 
    //! Map rank -> remote descriptors 
    std::map< std::size_t, std::vector<remote_data_descriptor_t> > remote_descriptors ; 
} ; 
/**************************************************************************************************/
struct simple_face_info_t {
    
    using q_view_t = Kokkos::View<std::size_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>, Kokkos::LayoutLeft, grace::default_space> ; 
    using f_view_t = Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::RandomAccess>, Kokkos::LayoutLeft, grace::default_space> ; 

    q_view_t qid_a_d, qid_b_d   ; //!< Device views for quadrant_ids 
    f_view_t face_a_d, face_b_d ; //!< Device views for face_ids 

    std::vector<std::size_t> qid_a_h, qid_b_h ; //!< Host vecs for quadrant_ids
    std::vector<int> face_a_h, face_b_h ; //!< Host vecs for face_ids

    std::size_t n_faces ; 
} ; 
/**************************************************************************************************/
struct hanging_face_info_t {
    using q_view_t = Kokkos::View<std::size_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>, Kokkos::LayoutLeft, grace::default_space> ; 
    using f_view_t = Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::RandomAccess>, Kokkos::LayoutLeft, grace::default_space> ; 

    q_view_t qid_f_d, qid_c_d   ; //!< Fine and coarse quadrant ids
    f_view_t face_f_d, face_c_d ; //!< Fine and coarse face ids
    q_view_t qif_f_to_bid_d     ; //!< quadrant id to coarse buffer id 
    
    std::vector<std::size_t> qid_f_h, qid_c_h   ; //!< Fine and coarse quadrant ids (host) 
    std::vector<int>         face_f_h, face_c_h ; //!< Fine and coarse face ids (host)
    std::vector<std::size_t> qif_f_to_bid_h     ; //!< quadrant id to coarse buffer id (host)

    std::size_t n_coarse, n_fine ; //!< Number of fine quads is also number of coarse bufs
} ; 
/**************************************************************************************************/
/**
 * @brief Iterate through all the faces of grid quadrants to
 *        store boundary information.
 * \ingroup amr
 * @param info <code>p4est</code>'s struct containing information   
 *             regarding the quadrant face.
 * @param user_data Type erased <code>grace_face_info_t</code> 
 *                  where information is stored.
 * This function is used as callback in <code>p4est_iterate</code> to store 
 * all the necessary information to apply interior and exterior boundary conditions.
 * In particular, this function stores, for all faces, the quadrant id's which share 
 * this face, whether this face is hanging or simple, whether it's internal or external,
 * its face orientation code, the tree(s) containing the quadrants on each side, and whether
 * any of the quadrants on this face are in the halo.
 */
void grace_iterate_faces( p4est_iter_face_info_t* info 
                          , void* user_data ) ;
/**************************************************************************************************/
/**************************************************************************************************/
class amr_ghosts_impl_t {

    /**************************************************************************************************/
    public: 
    /**************************************************************************************************/
    simple_face_info_t local_face_info, remote_face_info ; 
    /**************************************************************************************************/
    void update() ; 
    /**************************************************************************************************/
    protected:

    //**************************************************************************************************
    static constexpr unsigned long longevity = unique_objects_lifetimes::AMR_GHOSTS ; 
    //**************************************************************************************************
    amr_ghosts_impl_t() = default ; 
    //**************************************************************************************************
    ~amr_ghosts_impl_t() = default ; 
    //**************************************************************************************************
    p4est_ghost_t * halos ; 
    //**************************************************************************************************
    friend class utils::singleton_holder<amr_ghosts_impl_t> ;
    friend class memory::new_delete_creator<amr_ghosts_impl_t, memory::new_delete_allocator> ; 
    //**************************************************************************************************
} ; 
//**************************************************************************************************
using amr_ghosts = utils::singleton_holder<amr_ghosts_impl_t> ; 

}

#endif /* GRACE_AMR_AMR_GHOSTS_HH */