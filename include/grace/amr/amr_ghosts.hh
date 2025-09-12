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

namespace grace {
/**************************************************************************************************/
enum interface_kind_t : uint8_t { PHYS, INTERNAL }  ;
/**************************************************************************************************/
struct full_face_t {
        std::size_t quad_id       ; //!< Index of quadrant on the other side 
        bool is_remote            ; //!< Whether the quadrant is local or remote 
        std::size_t owner_rank    ; //!< Owner rank if remote
} ; 
struct hanging_face_t {
        std::size_t quad_id[P4EST_CHILDREN/2]    ; //!< Indices of hanging quads (in coarse bufs)
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
} ; 
/**************************************************************************************************/
struct edge_descriptor_t {}; 
struct corner_descriptor_t {}; 
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
    std::vector<mpi_task_t> const& get_mpi_tasks () {return mpi_task_list;}
    std::vector<gpu_task_t> const& get_gpu_tasks () {return gpu_task_list;}
    std::vector<cpu_task_t> const& get_cpu_tasks () {return cpu_task_list;}
    /**************************************************************************************************/
    void update() ; 
    /**************************************************************************************************/
    protected:
    /**************************************************************************************************/
    std::vector<quad_neighbors_descriptor_t> ghost_layer ; //!< Ghost layer used by GRACE
    p4est_ghost_t* p4est_ghost_layer                     ; //!< p4est data struct 
    std::vector<gpu_task_t> gpu_task_list ; 
    std::vector<mpi_task_t> mpi_task_list ; 
    std::vector<cpu_task_t> cpu_task_list ; 
    executor task_queue ; 
    std::vector<std::size_t> send_rank_offsets, recv_rank_offsets ; //!< In # of elements
    std::vector<std::size_t> send_rank_sizes, recv_rank_sizes ; //!< In # of elements
    Kokkos::View<double*> _send_buffer, _recv_buffer ;
    //**************************************************************************************************
    void build_task_list() ; 
    //**************************************************************************************************
    void build_remote_buffers() ; 
    void generate_mpi_transfer_tasks(std::size_t rank, mpi_task_t& send, mpi_task_t& recv, task_id_t& task_counter) ; 
    //**************************************************************************************************
    static constexpr unsigned long longevity = unique_objects_lifetimes::AMR_GHOSTS ; 
    //**************************************************************************************************
    amr_ghosts_impl_t() = default ; 
    //**************************************************************************************************
    ~amr_ghosts_impl_t() { if (p4est_ghost_layer) p4est_ghost_destroy(p4est_ghost_layer) ; } ; 
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