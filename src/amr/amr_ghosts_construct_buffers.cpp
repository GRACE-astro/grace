/**
 * @file amr_ghosts.cpp
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

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_utils.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/assert.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <grace/amr/amr_ghosts.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/ghostzone_kernels/copy_kernels.hh>
#include <grace/amr/ghostzone_kernels/phys_bc_kernels.hh>
#include <grace/amr/ghostzone_kernels/pack_unpack_kernels.hh>

#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <numeric>
#include <variant> 
#include <set> 
namespace grace {

enum sec_t : uint8_t {FACE=0, EDGE=1, CORNER=2, CBFACE=3, CBEDGE=4, CBCORNER=5} ; 


using desc_ptr_t = std::variant<
    face_descriptor_t*, 
    edge_descriptor_t*, 
    corner_descriptor_t*
>;


struct comm_key_t {
    //! Type-erased pointer to descriptor in neigbor array
    desc_ptr_t desc      ; 

    sec_t kind ; //!< Kind of interface
    size_t rank     ; //!< Other rank
    size_t quad_id  ; //!< Quadrant id 
    int8_t elem_id  ; //!< Element id
    int8_t elem_slot ; //!< If needed 

    bool operator==(const comm_key_t & other) const {
        return (rank == other.rank) && 
               (quad_id == other.quad_id) && 
               (elem_id == other.elem_id) && 
               (kind == other.kind) ;
    }
} ;


// comparison functor for sort
struct key_cmp {
    bool operator()(comm_key_t const& a, comm_key_t const& b) const {
        if (a.rank < b.rank) return true;
        if (a.rank > b.rank) return false;
        if (a.kind < b.kind) return true; 
        if (a.kind > b.kind) return false ; 
        if (a.quad_id < b.quad_id) return true ;
        if (a.quad_id > b.quad_id) return false ; 
        return (a.elem_id < b.elem_id) ; 
    }
};

using idx_key_t = std::tuple<size_t /*rank*/, size_t /*qid*/, int8_t /*element_id*/> ; 
struct idx_key_hash_t {
    std::size_t operator()(const idx_key_t &k) const noexcept {
        auto [rank, quad, face] = k;
        std::size_t h = std::hash<size_t>{}(rank);
        h ^= std::hash<size_t>{}(quad) + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int8_t>{}(face) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};


void register_index(
    desc_ptr_t const& desc, int8_t ic, size_t idx, bool send
) {
    std::visit([&](auto* d) {
        using T = std::decay_t<decltype(*d)>;
        if constexpr (std::is_same_v<T, face_descriptor_t>) {
            if (d->level_diff == level_diff_t::FINER) {
                if (send) d->data.hanging.send_buffer_id[ic] = idx;
                else      d->data.hanging.recv_buffer_id[ic] = idx;
            } else {
                if (send) d->data.full.send_buffer_id = idx;
                else      d->data.full.recv_buffer_id = idx;
            }
        } else if constexpr (std::is_same_v<T, edge_descriptor_t>) {
            if (d->level_diff == level_diff_t::FINER) {
                if (send) d->data.hanging.send_buffer_id[ic] = idx;
                else      d->data.hanging.recv_buffer_id[ic] = idx;
            } else {
                if (send) d->data.full.send_buffer_id = idx;
                else      d->data.full.recv_buffer_id = idx;
            }
        } else if constexpr (std::is_same_v<T, corner_descriptor_t>) {
            if (send) d->data.send_buffer_id = idx;
            else      d->data.recv_buffer_id = idx;
        }
    }, desc);
}


void process_key_arrays(
    std::vector<comm_key_t> & send_comm_keys,
    std::vector<comm_key_t> & recv_comm_keys,
    std::array<std::vector<size_t>,6> & send_counts,
    std::array<std::vector<size_t>,6> & recv_counts
)
{
    using idx_map_t = std::unordered_map<idx_key_t, size_t, idx_key_hash_t> ; 
    {
        // sort by rank, quadid, elem id 
        // note there might be duplicates here, we sort it out later
        std::sort(send_comm_keys.begin(), send_comm_keys.end(), key_cmp{}) ; 

        idx_map_t coarse_to_id; 
        for( auto& key: send_comm_keys ) {
            
            auto const rank = key.rank ; 
            auto const kind = key.kind ; 

            auto& count = send_counts[kind][rank] ; 

            auto [it, inserted] = coarse_to_id.try_emplace(
                std::make_tuple(key.rank,key.quad_id,key.elem_id), count
            ) ; 

            if (inserted) ++count; 

            register_index(
                key.desc, key.elem_slot, it->second, true /*send*/
            ) ; 
        }
    }

    {
        // sort by rank, quadid, elem id 
        // note there might be duplicates here, we sort it out later
        std::sort(recv_comm_keys.begin(), recv_comm_keys.end(), key_cmp{}) ; 

        idx_map_t coarse_to_id; 
        for( auto& key: recv_comm_keys ) {
            auto const rank = key.rank ; 
            auto const kind = key.kind ; 

            auto& count = recv_counts[kind][rank] ;

            auto [it, inserted] = coarse_to_id.try_emplace(
                std::make_tuple(key.rank,key.quad_id,key.elem_id), count 
            ) ; 

            if (inserted) ++count ; 

            register_index(
                key.desc, key.elem_slot, it->second, false /*recv*/
            ) ; 
        }

    }


}


void amr_ghosts_impl_t::build_remote_buffers(
    bucket_t& phys_bc_kernels,
    bucket_t& copy_kernels,
    hang_bucket_t& copy_from_cbuf_kernels,
    bucket_t& copy_to_cbuf_kernels,
    std::vector<bucket_t>& pack_kernels, 
    std::vector<bucket_t>& unpack_kernels, 
    std::vector<bucket_t>& pack_to_cbuf_kernels,
    std::vector<bucket_t>& unpack_to_cbuf_kernels,
    std::vector<hang_bucket_t>& unpack_from_cbuf_kernels
) {
    // goals of this function: 
    // 1. come up with a unique ordering of mirror and 
    // ghost datasets (faces / edges / corners ) that 
    // matches across ranks and store indices in neighbor 
    // struct 
    // 2. compute sizes / offsets per rank and per data type
    // 3. allocate MPI transfer buffers

    /****************************************************/
    // get mpi info
    auto rank = parallel::mpi_comm_rank() ; 
    auto nproc= parallel::mpi_comm_size() ;
    // get grid props 
    auto nq = amr::get_local_num_quadrants() ; 
    std::size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts() ; 
    // get n vars 
    std::size_t nvars = variables::get_n_evolved() ; 
    /****************************************************/
    pack_kernels.resize(nproc) ; unpack_kernels.resize(nproc) ; 
    pack_to_cbuf_kernels.resize(nproc) ;
    unpack_to_cbuf_kernels.resize(nproc) ;
    unpack_from_cbuf_kernels.resize(nproc) ;
    /****************************************************/
    // start: allocate a bunch of temp helpers 
    using svec_t = std::vector<size_t> ; 
    using arr_svec_t = std::array<svec_t,6> ; // one per face / edge /corner 
    auto make_vec = [nproc]() { return std::vector<size_t>(nproc, 0); };
    auto init_arr = [nproc, make_vec](arr_svec_t& arr) { std::generate(arr.begin(),arr.end(),make_vec); };

    std::vector<comm_key_t> mirror_keys, halo_keys; 

    // Step 1. we need a unique ordering of elements in the buffers 
    auto append_keys = [&] ( sec_t m_kind, /* mirror, send  */
                             sec_t h_kind, /* halo, receive */
                             size_t const& this_rank,
                             size_t const& other_rank,
                             size_t const& miq, 
                             size_t const& hiq,
                             int8_t this_ie,
                             int8_t other_ie, 
                             desc_ptr_t elem,
                             int8_t ic)
    {
        comm_key_t mkey, hkey ; 
        mkey.desc = elem ;
        hkey.desc = elem ; 

        mkey.elem_slot = ic ; 
        hkey.elem_slot = ic ; 

        mkey.kind = m_kind ; 
        hkey.kind = h_kind ; 

        mkey.rank = other_rank ;
        hkey.rank = other_rank ; 

        auto e_c = (other_rank < this_rank) 
                 ? other_ie :  this_ie ; 
        
        mkey.quad_id = miq  ;
        mkey.elem_id = e_c  ; 

        hkey.quad_id = hiq ;
        hkey.elem_id = e_c ; 

        mirror_keys.push_back(mkey) ; 
        halo_keys.push_back(hkey)   ;
    } ; 

    for( size_t iq=0UL; iq<nq; iq+=1UL) {
        for (uint8_t f = 0; f < P4EST_FACES; ++f) {
            auto& face = ghost_layer[iq].faces[f] ; 
            if ( face.kind ==  interface_kind_t::PHYS ) {
                phys_bc_kernels[amr::element_kind_t::FACE].emplace_back(iq,f) ; 
                continue ; 
            } 
            if ( face.level_diff == level_diff_t::COARSER ) {
                if ( !face.data.full.is_remote) {
                    copy_to_cbuf_kernels[amr::element_kind_t::FACE].emplace_back(iq,f) ; 
                } else {
                    append_keys(sec_t::CBFACE, sec_t::FACE, 
                            rank, face.data.full.owner_rank, 
                            iq /*should it be cbuf*/,face.data.full.quad_id, 
                            f, face.face,
                            &face, 0 /*not needed*/) ; 
                    // other side is coarser, this means we need to 
                    // pack - unpack a coarse buf 
                    pack_to_cbuf_kernels[face.data.full.owner_rank][amr::element_kind_t::FACE].emplace_back(iq, f) ;
                    unpack_to_cbuf_kernels[face.data.full.owner_rank][amr::element_kind_t::FACE].emplace_back(iq, f) ;
                }
                
            } else if (face.level_diff == level_diff_t::FINER) {
                for( int ic=0; ic<P4EST_CHILDREN/2; ++ic) {
                    if ( !face.data.hanging.is_remote[ic]) {
                        // copy 
                        copy_from_cbuf_kernels[amr::element_kind_t::FACE].emplace_back(iq,f,ic) ; 
                    } else {
                        append_keys(sec_t::FACE, sec_t::CBFACE, 
                            rank, face.data.hanging.owner_rank[ic], 
                            iq,face.data.hanging.quad_id[ic]/*should this be cbuf*/, 
                            f, face.face,
                            &face, ic) ; 
                        // pack into normal buf, unpack from cbuf
                        pack_kernels[face.data.hanging.owner_rank[ic]][amr::element_kind_t::FACE].emplace_back(iq,f) ; 
                        unpack_from_cbuf_kernels[face.data.hanging.owner_rank[ic]][amr::element_kind_t::FACE].emplace_back(iq,f,ic) ; 
                    }
                    
                }   
            } else {
                if ( !face.data.full.is_remote) {
                    /* local */
                    copy_kernels[amr::element_kind_t::FACE].emplace_back(iq,f) ; 
                } else {
                    append_keys(sec_t::FACE, sec_t::FACE, 
                            rank, face.data.full.owner_rank, 
                            iq,face.data.full.quad_id, 
                            f, face.face,
                            &face, 0 /*not needed*/) ; 
                    /* remote */
                    pack_kernels[face.data.full.owner_rank][amr::element_kind_t::FACE].emplace_back(iq,f) ; 
                    unpack_kernels[face.data.full.owner_rank][amr::element_kind_t::FACE].emplace_back(iq,f) ; 
                }
                
            } /* face not hanging */
        } /* for f .. nfaces */
        // edge loop 
        for( uint8_t e=0; e<12; ++e) {
            auto& edge = ghost_layer[iq].edges[e] ; 
            // we could be in a situation
            // where this edge sits in the 
            // middle of a coarser face, 
            // in which case the filling is 
            // taken care of already 
            if( !edge.filled) continue ; 
            if (edge.kind == interface_kind_t::PHYS) {
                phys_bc_kernels[amr::element_kind_t::EDGE].emplace_back(iq,e) ; 
                continue ; 
            } 
            if ( edge.level_diff == level_diff_t::COARSER ) {
                if ( !edge.data.full.is_remote)  {
                    copy_to_cbuf_kernels[amr::element_kind_t::EDGE].emplace_back(iq,e,edge.child_id) ; 
                }  else {
                    append_keys(sec_t::CBEDGE, sec_t::EDGE, 
                            rank, edge.data.full.owner_rank, 
                            iq /*should it be cbuf*/,edge.data.full.quad_id, 
                            e, edge.edge,
                            &edge, 0 /*not needed*/) ;
                    pack_to_cbuf_kernels[edge.data.full.owner_rank][amr::element_kind_t::EDGE].emplace_back(iq, e) ;
                    unpack_to_cbuf_kernels[edge.data.full.owner_rank][amr::element_kind_t::EDGE].emplace_back(iq, e) ;
                }
                
            } else if ( edge.level_diff == level_diff_t::FINER ) {
                for ( int ic=0; ic<2; ++ic){
                    if( ! edge.data.hanging.is_remote[ic]) {
                        copy_from_cbuf_kernels[amr::element_kind_t::EDGE].emplace_back(iq,e,edge.child_id) ;
                    } else {
                        append_keys(sec_t::EDGE, sec_t::CBEDGE, 
                            rank, edge.data.hanging.owner_rank[ic], 
                            iq /*should it be cbuf*/,edge.data.hanging.quad_id[ic], 
                            e, edge.edge,
                            &edge, ic) ;
                        pack_kernels[edge.data.hanging.owner_rank[ic]][amr::element_kind_t::EDGE].emplace_back(iq,e) ; 
                        unpack_from_cbuf_kernels[edge.data.hanging.owner_rank[ic]][amr::element_kind_t::EDGE].emplace_back(iq,e) ; 
                    }
                    
                }
            } else {
                if ( !edge.data.full.is_remote ) {
                    copy_kernels[amr::element_kind_t::EDGE].emplace_back(iq,e) ; 
                } else {
                    append_keys(sec_t::EDGE, sec_t::EDGE, 
                            rank, edge.data.full.owner_rank, 
                            iq,edge.data.full.quad_id, 
                            e, edge.edge,
                            &edge, 0/*not used*/) ;
                    pack_kernels[edge.data.full.owner_rank][amr::element_kind_t::EDGE].emplace_back(iq,e) ; 
                    unpack_kernels[edge.data.full.owner_rank][amr::element_kind_t::EDGE].emplace_back(iq,e) ; 
                }
                 
            }
        }
        // corner loop 
        for( uint8_t c=0; c<P4EST_CHILDREN; ++c) {
            auto& corner = ghost_layer[iq].corners[c] ;
            if( !corner.filled) continue ;  
            if (corner.kind == interface_kind_t::PHYS) {
                phys_bc_kernels[amr::element_kind_t::CORNER].emplace_back(iq,c) ; 
                continue ;
            } 
            if ( corner.level_diff == level_diff_t::COARSER ) {
                if ( !corner.data.is_remote ) {
                    copy_to_cbuf_kernels[amr::element_kind_t::CORNER].emplace_back(iq,c,0) ; 
                } else {
                    append_keys(sec_t::CBCORNER, sec_t::CORNER, 
                            rank, corner.data.owner_rank, 
                            iq,corner.data.quad_id, 
                            c, corner.corner,
                            &corner, 0 /*not used*/) ;
                    pack_to_cbuf_kernels[corner.data.owner_rank][amr::element_kind_t::CORNER].emplace_back(iq,c) ; 
                    unpack_to_cbuf_kernels[corner.data.owner_rank][amr::element_kind_t::CORNER].emplace_back(iq,c) ; 
                }
                 
            } else if (corner.level_diff == level_diff_t::FINER) {
                if ( !corner.data.is_remote ) {
                    copy_from_cbuf_kernels[amr::element_kind_t::CORNER].emplace_back(iq,c,0) ; 
                } else {
                    append_keys(sec_t::CORNER, sec_t::CBCORNER, 
                            rank, corner.data.owner_rank, 
                            iq,corner.data.quad_id, 
                            c, corner.corner,
                            &corner, 0 /*not used*/) ; 
                    pack_kernels[corner.data.owner_rank][amr::element_kind_t::CORNER].emplace_back(iq,c ) ; 
                    unpack_from_cbuf_kernels[corner.data.owner_rank][amr::element_kind_t::CORNER].emplace_back(iq,c,0) ; 
                }
                
            } else { 
                if ( !corner.data.is_remote ) {
                    copy_kernels[amr::element_kind_t::CORNER].emplace_back(iq,c) ; 
                } else {
                    append_keys(sec_t::CORNER, sec_t::CORNER, 
                            rank, corner.data.owner_rank, 
                            iq,corner.data.quad_id, 
                            c, corner.corner,
                            &corner, 0 /*not used*/) ;
                    pack_kernels[corner.data.owner_rank][amr::element_kind_t::CORNER].emplace_back(iq,c);
                    unpack_kernels[corner.data.owner_rank][amr::element_kind_t::CORNER].emplace_back(iq,c);
                }
                
            }
        }
    } /* for iq .. nquads */

    // counts of faces / edges / corners per rank (send & recv)
    arr_svec_t rank_send_counts, rank_recv_counts;
    init_arr(rank_send_counts);
    init_arr(rank_recv_counts) ; 
    // This function fills the recv / send_id's of all the 
    // descriptors in the neighbor struct. It also fills the 
    // count vectors 
    process_key_arrays(
        mirror_keys, halo_keys,
        rank_send_counts,
        rank_recv_counts
    ) ; 

    // Finally now we compute offsets and total sizes,
    // the hard part is over! 
    std::array<size_t,6> const elem_sizes {
        nx * nx * ngz * nvars,
        nx * ngz * ngz * nvars,
        ngz * ngz * ngz * nvars,
        nx * nx * ngz * nvars / 4,
        nx * ngz * ngz * nvars / 2,
        ngz * ngz * ngz * nvars
    } ; 

    // compute message sizes 
    send_rank_sizes = svec_t(nproc,0); 
    recv_rank_sizes = svec_t(nproc,0);
    
    arr_svec_t send_sizes, recv_sizes ; 
    init_arr(send_sizes);
    init_arr(recv_sizes) ; 
    std::set<size_t> active_send, active_recv ; ; 
    for( int r=0; r<nproc; ++r) {
        for( int ik=0; ik<6; ++ik) {
            send_sizes[ik][r] = elem_sizes[ik] * rank_send_counts[ik][r] ; 
            send_rank_sizes[r] += send_sizes[ik][r] ; 
            recv_sizes[ik][r] = elem_sizes[ik] * rank_recv_counts[ik][r] ; 
            recv_rank_sizes[r] += recv_sizes[ik][r] ;
	    if ( send_sizes[ik][r] > 0 ) {
	      active_send.insert(r) ; 
	    }
	    if ( recv_sizes[ik][r] > 0 ) {
	      active_recv.insert(r) ; 
	    }
        }
    }

    // Compute message offsets 
    arr_svec_t send_offsets, recv_offsets;
    init_arr(send_offsets);
    init_arr(recv_offsets) ;

    for (int r = 0; r < nproc; ++r) {
        size_t cur_send = 0, cur_recv = 0;
        for (int ik = 0; ik < 6; ++ik) {
            send_offsets[ik][r] = cur_send;
            recv_offsets[ik][r] = cur_recv;
            cur_send += send_sizes[ik][r];
            cur_recv += recv_sizes[ik][r];
        }
    }

    std::array<std::string,6> labels {
        "faces", "edges", "corners",
        "cbuf_faces", "cbuf_edges", "cbuf_corners"
    } ; 
    for( int r=0; r<nproc; ++r) {
        for( int ik=0; ik<6; ++ik){
            GRACE_TRACE(
                "Rank {} section {} send count {}, offset {}", r, labels[ik], rank_send_counts[ik][r], send_offsets[ik][r]
            ) ; 
            GRACE_TRACE(
                "Rank {} section {} receive count {}, offset {}", r, labels[ik], rank_recv_counts[ik][r], recv_offsets[ik][r]
            ) ; 
        }
    }
    // exclusive scan for rank offsets 
    send_rank_offsets.resize(nproc) ; 
    std::exclusive_scan( send_rank_sizes.begin(), send_rank_sizes.end()
                       , send_rank_offsets.begin(), 0) ; 
    recv_rank_offsets.resize(nproc) ; 
    std::exclusive_scan( recv_rank_sizes.begin(), recv_rank_sizes.end()
                       , recv_rank_offsets.begin(), 0) ;

    // reduce for total sizes
    size_t total_send_size = std::reduce(
        send_rank_sizes.begin(), send_rank_sizes.end()
    ); 
    size_t total_recv_size = std::reduce(
        recv_rank_sizes.begin(), recv_rank_sizes.end()
    ); 

    // allocate buffers
    _send_buffer.set_offsets(
        send_rank_offsets, send_offsets
    ) ; 
    _send_buffer.set_strides(nx,ny,nz,nvars,ngz);
    _send_buffer.realloc(total_send_size) ; 

    _recv_buffer.set_offsets(
        recv_rank_offsets, recv_offsets
    ) ; 
    _recv_buffer.set_strides(nx,ny,nz,nvars,ngz);
    _recv_buffer.realloc(total_recv_size) ;

    GRACE_INFO("Setup of remote buffers complete, total send/recv size [MB] {}/{}, avg message size per rank [MB] {}/{}",
           sizeof(double)*total_send_size/1e06,
           sizeof(double)*total_recv_size/1e06,
           sizeof(double)*total_send_size/1e06/active_send.size(),
           sizeof(double)*total_recv_size/1e06/active_recv.size() );

}

} /* namespace grace */
