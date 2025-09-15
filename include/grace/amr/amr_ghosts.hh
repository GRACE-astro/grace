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
#include <grace/utils/task_queue.hh>

#include <grace/data_structures/memory_defaults.hh>

#include <grace/amr/p4est_headers.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <array> 
#include <memory> 

namespace grace {

/**************************************************************************************************/
enum bc_t: uint8_t {BC_OUTFLOW=0, BC_LAGRANGE_EXTRAP, BC_NONE} ; 
/**************************************************************************************************/
enum interface_kind_t : uint8_t { PHYS, INTERNAL }  ;
/**************************************************************************************************/
enum level_diff_t : int8_t {FINER=-1, SAME=0, COARSER=+1} ; // The other one is ? 
/**************************************************************************************************/
struct full_face_t {
        std::size_t quad_id       ; //!< Index of quadrant on the other side 
        std::size_t recv_buffer_id ; //!< Index in receive array, if relevant
        bool is_remote            ; //!< Whether the quadrant is local or remote 
        std::size_t owner_rank    ; //!< Owner rank if remote
} ; 
struct hanging_face_t {
        std::size_t quad_id[P4EST_CHILDREN/2]    ; //!< Indices of hanging quads (in coarse bufs)
        std::size_t recv_buffer_id[P4EST_CHILDREN/2] ; //!< Indices in receive array, if relevant
        bool is_remote[P4EST_CHILDREN/2]         ; //!< Are the quads remote 
        std::size_t owner_rank[P4EST_CHILDREN/2] ; //!< owner ranks if remote 
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
    std::size_t send_buffer_id ; //!< Index in send buffer if applicable 
    int8_t level_diff   ; //!< Ref level difference (+-1 or 0)
    face_data_t data    ; //!< Quadrant ids 
    int8_t face ; //!< Face code as seen from other side 
    int8_t child_id ; //!< If level diff = + 1, which child is this? \in [0,...,4[
} ; 
/**************************************************************************************************/
struct full_edge_t {
    std::size_t quad_id       ; //!< Index of quadrant on the other side 
    std::size_t recv_buffer_id ; //!< Index in receive array, if relevant
    bool is_remote            ; //!< Whether the quadrant is local or remote 
    std::size_t owner_rank    ; //!< Owner rank if remote
} ; 
struct hanging_edge_t {
        std::size_t quad_id[2]    ; //!< Indices of hanging quads (in coarse bufs)
        std::size_t recv_buffer_id[2] ; //!< Indices in receive array, if relevant
        bool is_remote[2]         ; //!< Are the quads remote 
        std::size_t owner_rank[2] ; //!< owner ranks if remote 
} ;
union edge_data_t {
    full_edge_t full ; 
    hanging_edge_t hanging ; 
} ; 
struct edge_descriptor_t {
    interface_kind_t kind ; 
    std::size_t send_buffer_id ; 
    int8_t level_diff ; 
    edge_data_t data ;
    int8_t edge ; 
    int8_t child_id ;
}; 
/**************************************************************************************************/
struct corner_data_t {
    std::size_t quad_id        ; //!< Indices of hanging quads (in coarse bufs)
    std::size_t recv_buffer_id ; //!< Indices in receive array, if relevant
    bool is_remote             ; //!< Are the quads remote 
    std::size_t owner_rank     ; //!< owner ranks if remote 
} ; 
struct corner_descriptor_t {
    interface_kind_t kind ; 
    std::size_t send_buffer_id ; 
    int8_t level_diff ; 
    corner_data_t data ;
    int8_t corner ; 
}; 
/**************************************************************************************************/
struct quad_neighbors_descriptor_t {
    std::array< face_descriptor_t, P4EST_FACES > faces ; //!< Faces 
    #ifdef GRACE_3D 
    std::array< edge_descriptor_t, 12 > edges ; //!< edges 
    #endif 
    std::array< corner_descriptor_t, P4EST_CHILDREN> corners; //!< Corners
    std::size_t quad_id ; //!< Quadrant id 
    std::size_t cbuf_id ; //!< For fine quads only: index into coarse buffer array 

    //! Debug information
    int8_t n_registered_faces {0} ; 
    int8_t n_registered_edges {0} ; 
    int8_t n_registered_corners {0} ; 
}  ;  
/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/
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
#ifdef GRACE_3D
/**
 * @brief Iterate through all the edges of grid quadrants to
 *        store boundary information.
 * \ingroup amr
 * @param info <code>p4est</code>'s struct containing information   
 *             regarding the quadrant edge.
 * @param user_data Type erased <code>grace_face_info_t</code> 
 *                  where information is stored.
 * This function is used as callback in <code>p4est_iterate</code> to store 
 * all the necessary information to apply interior and exterior boundary conditions.
 * In particular, this function stores, for all edges, the quadrant id's which share 
 * this face, whether this face is hanging or simple, whether it's internal or external,
 * its face orientation code, the tree(s) containing the quadrants on each side, and whether
 * any of the quadrants on this face are in the halo.
 */
void grace_iterate_edges( p8est_iter_edge_info_t* info 
                          , void* user_data ) ;
#endif
/**************************************************************************************************/
/**
 * @brief Iterate through all the corners of grid quadrants to
 *        store boundary information.
 * \ingroup amr
 * @param info <code>p4est</code>'s struct containing information   
 *             regarding the quadrant corner.
 * @param user_data Type erased <code>grace_face_info_t</code> 
 *                  where information is stored.
 * This function is used as callback in <code>p4est_iterate</code> to store 
 * all the necessary information to apply interior and exterior boundary conditions.
 * In particular, this function stores, for all corners, the quadrant id's which share 
 * this face, whether this face is hanging or simple, whether it's internal or external,
 * its face orientation code, the tree(s) containing the quadrants on each side, and whether
 * any of the quadrants on this face are in the halo.
 */
void grace_iterate_corners( p4est_iter_corner_info_t* info 
                          , void* user_data ) ;
/**************************************************************************************************/
/**************************************************************************************************/
class amr_ghosts_impl_t {
    /**************************************************************************************************/
    static constexpr unsigned int BATCH_N_KERNELS = 64U ; 
    /**************************************************************************************************/
    public: 
    /**************************************************************************************************/
    std::vector<quad_neighbors_descriptor_t> const& get_ghost_layer() { return ghost_layer ; }
    /**************************************************************************************************/
    p4est_ghost_t* get_p4est_ghosts() { return p4est_ghost_layer ; }
    /**************************************************************************************************/
    void get_rank_offsets( std::vector<std::size_t>& send, std::vector<std::size_t>& receive ) {
        send = send_rank_offsets ; receive = recv_rank_offsets ; 
    }
    void get_rank_sizes(  std::vector<std::size_t>& send, std::vector<std::size_t>& receive ) {
        send = send_rank_sizes ; receive = recv_rank_sizes ; 
    }
    size_t get_send_buf_size() const { return _send_buffer.extent(0) ; }
    size_t get_recv_buf_size() const { return _recv_buffer.extent(0) ; }
    auto& get_task_list () {return task_list;}
    auto& get_task_executor () {return task_queue;}

    /**************************************************************************************************/
    void update() ; 
    /**************************************************************************************************/
    protected:
    /**************************************************************************************************/
    std::vector<quad_neighbors_descriptor_t> ghost_layer ; //!< Ghost layer used by GRACE
    p4est_ghost_t* p4est_ghost_layer = nullptr           ; //!< p4est data struct 

    std::vector<std::unique_ptr<task_t>> task_list ;
    executor task_queue ; 
    std::vector<std::size_t> send_rank_offsets, recv_rank_offsets ; //!< In # of elements
    std::vector<std::size_t> send_rank_sizes, recv_rank_sizes ; //!< In # of elements
    Kokkos::View<double*, grace::default_space> _send_buffer, _recv_buffer ;
    Kokkos::View<bc_t*> var_bc_kind ; //!< Boundary condition per-variable
    //**************************************************************************************************
    void build_task_list() ; 
    //**************************************************************************************************
    void build_remote_buffers() ; 
    //**************************************************************************************************
    void build_executor_runtime() ; 
    //**************************************************************************************************
    static constexpr unsigned long longevity = unique_objects_lifetimes::AMR_GHOSTS ; 
    //**************************************************************************************************
    amr_ghosts_impl_t()
    : _send_buffer("ghost_send_buf", 0),
      _recv_buffer("ghost_recv_buf", 0)
    {}
    //**************************************************************************************************
    ~amr_ghosts_impl_t() { 
        if (p4est_ghost_layer) p4est_ghost_destroy(p4est_ghost_layer) ; 
    } ; 
    //**************************************************************************************************
    
    //**************************************************************************************************
    friend class utils::singleton_holder<amr_ghosts_impl_t> ;
    friend class memory::new_delete_creator<amr_ghosts_impl_t, memory::new_delete_allocator> ; 
    //**************************************************************************************************
} ; 
//**************************************************************************************************
using amr_ghosts = utils::singleton_holder<amr_ghosts_impl_t> ; 

}

#endif /* GRACE_AMR_AMR_GHOSTS_HH */