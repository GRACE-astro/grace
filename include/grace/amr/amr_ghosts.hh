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
#include <grace/data_structures/variable_properties.hh>

#include <grace/amr/p4est_headers.hh>
#include <grace/amr/forest.hh>
#include <grace/amr/ghostzone_kernels/ghost_array.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <array> 
#include <memory> 
#include <variant>
#include <unordered_set>
#include <limits> 

// unset task_id guard 
#define UNSET_TASK_ID std::numeric_limits<size_t>::max()

namespace grace {

/**************************************************************************************************/
enum bc_t: uint8_t {BC_OUTFLOW=0, BC_LAGRANGE_EXTRAP, BC_NONE} ; 
/**************************************************************************************************/
enum interface_kind_t : uint8_t { PHYS, INTERNAL }  ;
/**************************************************************************************************/
enum level_diff_t : int8_t {FINER=-1, SAME=0, COARSER=+1} ; // The other one is ? 
/**************************************************************************************************/
// TODO_FLUX
// Face bitmask used by refluxing to mark which faces of a quadrant are next to coarser one
// 0:x-, 1:x+, 2:y-, 3:y+, (4:z-, 5:z+ in 3D).
enum face_mask_bits : uint8_t {
    FACE_XMIN = 1u << 0, FACE_XMAX = 1u << 1,
    FACE_YMIN = 1u << 2, FACE_YMAX = 1u << 3,
#ifdef GRACE_3D
    FACE_ZMIN = 1u << 4, FACE_ZMAX = 1u << 5,
#endif
};
/**************************************************************************************************/
// map a P4EST-style face index [0..2*dim-1] to its bit.
GRACE_INLINE inline uint8_t face_bit_from_index(int f) {
#ifdef GRACE_3D
    ASSERT(0 <= f && f < 6, "face index out of range");
#else
    ASSERT(0 <= f && f < 4, "face index out of range");
#endif
    return static_cast<uint8_t>(1u << f);
}

/**************************************************************************************************/
struct full_face_t {
        std::size_t quad_id       ; //!< Index of quadrant on the other side 
        std::size_t recv_buffer_id ; //!< Index in receive array, if relevant
        std::size_t send_buffer_id ; //!< Index in send array 
        bool is_remote            ; //!< Whether the quadrant is local or remote 
        std::size_t owner_rank    ; //!< Owner rank if remote
        task_id_t task_id         ;
} ; 
struct hanging_face_t {
        std::size_t quad_id[P4EST_CHILDREN/2]    ; //!< Indices of hanging quads (in coarse bufs)
        std::size_t recv_buffer_id[P4EST_CHILDREN/2] ; //!< Indices in receive array, if relevant
        std::size_t send_buffer_id[P4EST_CHILDREN/2] ; //!< Indices in send array
        bool is_remote[P4EST_CHILDREN/2]         ; //!< Are the quads remote 
        std::size_t owner_rank[P4EST_CHILDREN/2] ; //!< owner ranks if remote 
        task_id_t task_id[P4EST_CHILDREN/2]      ; 
} ; 
struct physical_face_t {
    int8_t dir[3] ; //!< Normal to the grid 
    bool in_cbuf  ; //!< Do we need to fill inside cbufs ? 
    amr::element_kind_t type ;
    task_id_t task_id          ; 
} ; 
union face_data_t {
    full_face_t full ; 
    hanging_face_t hanging ; 
    physical_face_t phys ; 
    face_data_t() = default ;
} ; 
/**
* @brief This class describes a face of a quadrant 
 */
struct face_descriptor_t { 
    interface_kind_t kind ; 
    int8_t level_diff   ; //!< Ref level difference (+-1 or 0)
    face_data_t data    ; //!< Quadrant ids 
    int8_t face ; //!< Face code as seen from other side 
    int8_t child_id ; //!< If level diff = + 1, which child is this? \in [0,...,4[
} ; 
/**************************************************************************************************/
struct full_edge_t {
    std::size_t quad_id       ; //!< Index of quadrant on the other side 
    std::size_t recv_buffer_id ; //!< Index in receive array, if relevant
    std::size_t send_buffer_id ; //!< Index in send array, if relevant
    bool is_remote            ; //!< Whether the quadrant is local or remote 
    std::size_t owner_rank    ; //!< Owner rank if remote
    task_id_t task_id ; 
} ; 
struct hanging_edge_t {
        std::size_t quad_id[2]    ; //!< Indices of hanging quads (in coarse bufs)
        std::size_t recv_buffer_id[2] ; //!< Indices in receive array, if relevant
        std::size_t send_buffer_id[2] ; //!< Indices in send array, if relevant
        bool is_remote[2]         ; //!< Are the quads remote 
        std::size_t owner_rank[2] ; //!< owner ranks if remote 
        task_id_t task_id[2] ; 
} ;
struct physical_edge_t {
    int8_t dir[3] ; //!< Normal to the grid 
    bool in_cbuf  ; //!< Do we need to fill inside cbufs ? 
    amr::element_kind_t type ;
    task_id_t task_id          ; 
} ; 
union edge_data_t {
    full_edge_t full ; 
    hanging_edge_t hanging ; 
    physical_edge_t phys ; 
} ; 
struct edge_descriptor_t {
    interface_kind_t kind ; 
    int8_t level_diff ; 
    edge_data_t data ;
    int8_t edge ; 
    int8_t child_id ;
    bool filled{false} ; 
}; 
/**************************************************************************************************/
struct corner_data_t {
    std::size_t quad_id        ; //!< Index of hanging quads (in coarse bufs)
    std::size_t recv_buffer_id ; //!< Index in receive array, if relevant
    std::size_t send_buffer_id ; //!< Index in send array, if relevant 
    bool is_remote             ; //!< Are the quads remote 
    std::size_t owner_rank     ; //!< owner ranks if remote 
    task_id_t task_id          ; 
} ; 
struct physical_corner_t {
    int8_t dir[3] ; //!< Normal to the grid 
    bool in_cbuf{false}  ; //!< Do we need to fill inside cbufs ? 
    amr::element_kind_t type ;
    task_id_t task_id          ; 
} ;
struct corner_descriptor_t {
    interface_kind_t kind ; 
    int8_t level_diff ; 
    corner_data_t data ;
    physical_corner_t phys ; 
    int8_t corner ; 
    bool filled{false} ; 
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
    //TODO_FLUX:
    std::size_t cflux_id { SIZE_MAX};
    uint8_t cflux_mask {0};

    //! Debug information
    int8_t n_registered_faces {0} ; 
    int8_t n_registered_edges {0} ; 
    int8_t n_registered_corners {0} ; 
}  ;  
/**************************************************************************************************/
template < amr::element_kind_t elem_kind > 
inline uint8_t 
get_adjacent_idx(uint8_t eid, const int8_t dir[3]) {return 0;};  // TODO changed interfacex
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
    size_t get_send_buf_size() const { return _send_buffer.size() ; }
    size_t get_recv_buf_size() const { return _recv_buffer.size() ; }
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

    amr::ghost_array_t _send_buffer, _recv_buffer ;

    grace::var_array_t<GRACE_NSPACEDIM> _coarse_buffers ; 
    grace::flux_array_t _coarse_flux_buffers ; //TODO
    Kokkos::View<bc_t*> var_bc_kind ; //!< Boundary condition per-variable
    //**************************************************************************************************

    /* TODO_FLUX ! */
    void build_flux_buffers(
        std::unordered_set<size_t> & ) ; 

    //**************************************************************************************************
    void reset() {
        ghost_layer.clear() ; 
        task_list.clear()  ; 
        send_rank_offsets.clear() ; 
        recv_rank_offsets.clear() ; 
        send_rank_sizes.clear() ;
        recv_rank_sizes.clear() ; 
        task_queue.clear() ; 
    }
    //**************************************************************************************************
    void build_task_list(
        bucket_t& ,
        bucket_t& ,
        hang_bucket_t& ,
        bucket_t& ,
        std::vector<bucket_t>& , 
        std::vector<bucket_t>& , 
        std::vector<hang_bucket_t>& , 
        std::vector<bucket_t>& ,
        std::vector<bucket_t>& ,
        std::vector<hang_bucket_t>&,
        bucket_t&,
        std::unordered_set<size_t> const&, 
        std::unordered_set<size_t> const& 
    ) ; 
    //**************************************************************************************************
    void build_remote_buffers(
        bucket_t&, bucket_t&,
        hang_bucket_t&, bucket_t&,
        std::vector<bucket_t>& , std::vector<bucket_t>& , std::vector<hang_bucket_t>& ,
        std::vector<bucket_t>& , std::vector<bucket_t>& ,
        std::vector<hang_bucket_t>&, bucket_t&
    ) ; 
    //**************************************************************************************************
    void build_coarse_buffers(
        std::unordered_set<size_t> & 
    ) ; 
    //**************************************************************************************************
    void build_executor_runtime() ; 
    //**************************************************************************************************
    static constexpr unsigned long longevity = unique_objects_lifetimes::AMR_GHOSTS ; 
    //**************************************************************************************************
    amr_ghosts_impl_t()
    : _send_buffer("MPI_send_buffer")
    , _recv_buffer("MPI_recv_buffer") 
    {} ; 
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
//**************************************************************************************************
//**************************************************************************************************
template <> 
inline uint8_t
get_adjacent_idx<amr::FACE>(uint8_t eid, const int8_t dir[3]) {
    using namespace amr::detail ; 
    
    int nz0=-1, nz1=-1;
    int sgn0=0, sgn1=0;
    int cnt=0;
    for(int i=0;i<3;++i){
        if(dir[i]!=0){
            if(cnt==0){ nz0=i; sgn0=dir[i]>0; }
            else { nz1=i; sgn1=dir[i]>0; }
            ++cnt;
        }
    }
    

    return f2e[eid][2*(face_axes[eid/2][0] != nz0)+sgn0] ; 
}; 


template <> 
inline uint8_t
get_adjacent_idx<amr::EDGE>(uint8_t eid, const int8_t dir[3]) {
    using namespace amr::detail ; 

    int nz0=-1, nz1=-1;
    int sgn0=0, sgn1=0;
    int cnt=0;
    for(int i=0;i<3;++i){
        if(dir[i]!=0){
            if(cnt==0){ nz0=i; sgn0=dir[i]>0; }
            else { nz1=i; sgn1=dir[i]>0; }
            ++cnt;
        }
    }

    int8_t edge_dir = eid/4 ; 

    if ( nz0 == edge_dir ) {
        // corner 
        return (sgn0) ? e2c[eid][1] : e2c[eid][0] ; 
    }  else {
        // face 
        if ( edge_dir == 0 ) { 
            // fixme here we're not checking the sign 
            // won't change the result but technically only one 
            // sign makes sense 
            
            return (nz0==1) ? e2f[eid][0] : e2f[eid][1] ;
        } else {
            return (nz0==0) ? e2f[eid][0] : e2f[eid][1] ; 
        } 
    }
}; 


template <> 
inline uint8_t 
get_adjacent_idx<amr::CORNER>(uint8_t eid, const int8_t dir[3]) {
    using namespace amr::detail ; 

    int nz0=-1, nz1=-1;
    int sgn0=0, sgn1=0;
    int cnt=0;
    for(int i=0;i<3;++i){
        if(dir[i]!=0){
            if(cnt==0){ nz0=i; sgn0=dir[i]>0; }
            else { nz1=i; sgn1=dir[i]>0; }
            ++cnt;
        }
    }
    ASSERT(cnt == 1, "Only along axes directions supported for now.") ; 

    

    return c2e[eid][nz0] ; 
}; 

//**************************************************************************************************
}

#endif /* GRACE_AMR_AMR_GHOSTS_HH */
