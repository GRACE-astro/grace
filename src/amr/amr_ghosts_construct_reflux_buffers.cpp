/**
 * @file amr_ghosts_construct_buffers.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Helpers for 4th order prolongation and restriction
 * @date 2026-01-02
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
#include <grace/amr/ghostzone_kernels/index_helpers.hh>


#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <numeric>
#include <variant> 
#include <set> 
#include <unordered_map>

namespace grace {

// communication key 
struct comm_key_t {
    size_t qid ;
    int elem_id ; // face or edge 
    bool operator==(const comm_key_t & other) const {
    return (qid == other.qid) && 
            (elem_id == other.elem_id);
    }
} ;

// hasher 
struct comm_key_hash {
    std::size_t operator()(comm_key_t const& k) const noexcept {
        std::size_t h1 = std::hash<size_t>{}(k.qid);
        std::size_t h2 = std::hash<int>{}(k.elem_id);

        // Combine hashes (boost-like method)
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b97f4a7c15ULL + (seed<<6) + (seed>>2);
        return seed;
    }
};

// comparator 
struct key_cmp {
    bool operator()(auto const& a, auto const& b) const {
        return std::tie(a.qid,a.elem_id)
             < std::tie(b.qid,b.elem_id);
    }
};


// utility 
struct comm_patt_builder {
    int nproc ;

    auto sort_and_dedup(
        std::vector<std::vector<comm_key_t>>& keys 
    ) const 
    {
        std::vector<std::unordered_map<comm_key_t, size_t, comm_key_hash>> lookup(nproc);
        std::vector<size_t> counts(nproc, 0);
        for (int r = 0; r < nproc; ++r) {
            auto& vec = keys[r];
            std::sort(vec.begin(), vec.end(), key_cmp{});
            vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
            counts[r] = vec.size();
            
            std::unordered_map<comm_key_t, size_t, comm_key_hash> index;
            index.reserve(vec.size());
            for (size_t k = 0; k < vec.size(); ++k)
                index[vec[k]] = k;
            lookup[r] = std::move(index);
        }
        return std::make_pair(lookup, counts);
    }

    auto get_offsets_and_sizes(
        std::vector<size_t> const& counts,
        size_t size 
    ) const 
    {
        std::vector<size_t> sizes(nproc,0), offsets(nproc,0) ; 
        size_t total = 0;
        for (size_t r = 0; r < nproc; ++r) {
            size_t _loc_size = counts[r] * size ; 
            total+=_loc_size ; 
            sizes[r] = _loc_size ;
        }
        for( size_t r=1; r<nproc; ++r) {
            offsets[r] = offsets[r-1] + sizes[r-1];
        }
        return std::make_tuple(total,sizes,offsets) ; 
    }
} ; 

// order of cmoparison: 
// 1 rank 
// 2 buffer id 
// 3 quadrant id 
// 4 element id 
struct rdesc_cmp {
    bool operator() (const hanging_remote_reflux_desc_t& a, const hanging_remote_reflux_desc_t& b) const {
        return std::tie(
            a.rank, a.buf_id, a.qid, a.elem_id  
        ) < std::tie(
            b.rank, b.buf_id, b.qid, b.elem_id  
        ) ; 
    }
} ; 

struct rdesc_eq {
    bool operator() (const hanging_remote_reflux_desc_t& a, const hanging_remote_reflux_desc_t& b) const {
        return std::tie(
            a.rank, a.buf_id, a.qid, a.elem_id  
        ) == std::tie(
            b.rank, b.buf_id, b.qid, b.elem_id  
        ) ; 
    }
} ; 

template<typename cmp, typename eq, typename T>
void sort_and_dedup(std::vector<T>& v) {
    std::sort(v.begin(),v.end(), cmp{}) ; 

    v.erase(std::unique(v.begin(),v.end(),eq{}), v.end()) ; 
}

static hanging_remote_reflux_device_desc_t 
to_device(std::vector<hanging_remote_reflux_desc_t> const& v ) {
    using namespace Kokkos ; 
    hanging_remote_reflux_device_desc_t out{} ; 
    Kokkos::realloc(out.qid,v.size()) ; 
    Kokkos::realloc(out.buf_id,v.size()) ; 
    Kokkos::realloc(out.rank,v.size()) ; 
    Kokkos::realloc(out.elem_id,v.size()) ;    

    auto qidh = create_mirror_view(out.qid) ; 
    auto bidh = create_mirror_view(out.buf_id) ; 
    auto rh = create_mirror_view(out.rank) ; 
    auto eh = create_mirror_view(out.elem_id) ; 

    for( size_t i=0UL; i<v.size(); ++i) {
        qidh(i) = v[i].qid ; 
        bidh(i) = v[i].buf_id ; 
        rh(i) = v[i].rank ; 
        eh(i) = v[i].elem_id ; 
    }

    deep_copy(out.qid,qidh) ; 
    deep_copy(out.buf_id,bidh) ; 
    deep_copy(out.rank,rh) ; 
    deep_copy(out.elem_id,eh) ; 

    return out ; 
} 

static hanging_face_reflux_device_desc_t
to_device(std::vector<hanging_face_reflux_desc_t> const& v)
{
    using namespace Kokkos ; 
    auto const N = v.size() ; 

    hanging_face_reflux_device_desc_t out ; 

    Kokkos::realloc(out.coarse_qid,N) ; 
    Kokkos::realloc(out.coarse_owner_rank,N) ; 
    Kokkos::realloc(out.coarse_is_remote,N) ; 
    Kokkos::realloc(out.coarse_face_id,N) ; 

    Kokkos::realloc(out.fine_face_id,N) ; 
    Kokkos::realloc(out.fine_qid,N) ; 
    Kokkos::realloc(out.fine_bid,N) ; 
    Kokkos::realloc(out.fine_owner_rank,N) ; 
    Kokkos::realloc(out.fine_is_remote,N) ; 

    auto cqh = create_mirror_view(out.coarse_qid) ; 
    auto crh = create_mirror_view(out.coarse_owner_rank) ; 
    auto cremoteh = create_mirror_view(out.coarse_is_remote) ; 
    auto cfh = create_mirror_view(out.coarse_face_id) ; 

    auto fqh = create_mirror_view(out.fine_qid) ; 
    auto fbh = create_mirror_view(out.fine_bid) ; 
    auto frh = create_mirror_view(out.fine_owner_rank) ; 
    auto fremoteh = create_mirror_view(out.fine_is_remote) ; 
    auto ffh = create_mirror_view(out.fine_face_id) ; 

    for( int i=0; i<N; ++i ) {
        auto const& dsc = v[i] ; 
        cqh(i) = static_cast<int>(dsc.coarse_qid) ; 
        cremoteh(i) = static_cast<uint8_t>(dsc.coarse_is_remote)  ; 
        crh(i) = dsc.coarse_owner_rank ; 
        cfh(i) = dsc.coarse_face_id ; 
        ffh(i) = dsc.fine_face_id ; 
        for ( int ic=0; ic<4; ++ic) {
            fqh(i,ic) = static_cast<int>(dsc.fine_qid[ic]) ; 
            fbh(i,ic) = static_cast<int>(dsc.fine_bid[ic]) ; 
            frh(i,ic) = dsc.fine_owner_rank[ic] ; 
            fremoteh(i,ic) = static_cast<uint8_t>(dsc.fine_is_remote[ic]) ; 
        }
    }

    deep_copy(out.coarse_qid,cqh);
    deep_copy(out.coarse_owner_rank,crh);
    deep_copy(out.coarse_is_remote,cremoteh);
    deep_copy(out.coarse_face_id,cfh);

    deep_copy(out.fine_qid,fqh);
    deep_copy(out.fine_bid,fbh);
    deep_copy(out.fine_owner_rank,frh);
    deep_copy(out.fine_is_remote,fremoteh);
    deep_copy(out.fine_face_id,ffh);

    return out ; 
} 

static full_face_reflux_device_desc_t
to_device(std::vector<full_face_reflux_desc_t> const& v)
{
    using namespace Kokkos ; 
    auto const N = v.size() ; 

    full_face_reflux_device_desc_t out ; 

    Kokkos::realloc(out.qid,N) ; 
    Kokkos::realloc(out.bid,N) ; 
    Kokkos::realloc(out.owner_rank,N) ; 
    Kokkos::realloc(out.is_remote,N) ; 
    Kokkos::realloc(out.face_id,N) ; 


    auto cqh = create_mirror_view(out.qid) ; 
    auto cbh = create_mirror_view(out.bid) ; 
    auto crh = create_mirror_view(out.owner_rank) ; 
    auto cremoteh = create_mirror_view(out.is_remote) ; 
    auto cfh = create_mirror_view(out.face_id) ; 


    for( int i=0; i<N; ++i ) {
        auto const& dsc = v[i] ; 
        for ( int ic=0; ic<2; ++ic) {
            cqh(i,ic) = static_cast<int>(dsc.qid[ic]) ; 
            cbh(i,ic) = static_cast<int>(dsc.bid[ic]) ; 
            crh(i,ic) = dsc.owner_rank[ic] ; 
            cremoteh(i,ic) = static_cast<uint8_t>(dsc.is_remote[ic]) ; 
            cfh(i,ic) = static_cast<uint8_t>(dsc.face_id[ic]) ; 
        }
    }

    deep_copy(out.qid,cqh);
    deep_copy(out.bid,cbh);
    deep_copy(out.owner_rank,crh);
    deep_copy(out.is_remote,cremoteh);
    deep_copy(out.face_id,cfh);

    return out ; 
} 


hanging_edge_reflux_device_desc_t
to_device(std::vector<hanging_edge_reflux_desc_t> const& v)
{
    using namespace Kokkos ; 
    auto const N = v.size() ; 
    hanging_edge_reflux_device_desc_t out(N) ; 

    auto is_fineh = create_mirror_view(out.is_fine) ; 
    auto n_sidesh = create_mirror_view(out.n_sides) ; 

    auto fine_qidh = create_mirror_view(out.fine_qid) ; 
    auto fine_bidh = create_mirror_view(out.fine_bid) ; 
    auto fine_owner_rankh = create_mirror_view(out.fine_owner_rank) ; 
    auto fine_is_remoteh = create_mirror_view(out.fine_is_remote) ; 

    auto coarse_qidh = create_mirror_view(out.coarse_qid) ;
    auto coarse_bidh = create_mirror_view(out.coarse_bid) ;
    auto coarse_owner_rankh = create_mirror_view(out.coarse_owner_rank) ; 
    auto coarse_is_remoteh = create_mirror_view(out.coarse_is_remote) ;

    auto edge_idh = create_mirror_view(out.edge_id) ;
    auto off_ih = create_mirror_view(out.off_i) ;
    auto off_jh = create_mirror_view(out.off_j) ;

    for( int i=0; i<N; ++i) {
        auto const& dsc = v[i]  ; 
        n_sidesh(i) = dsc.n_sides ; 

        for( int iside=0; iside<dsc.n_sides; ++iside) {
            auto const& dthis = dsc.sides[iside] ; 
            is_fineh(i,iside) = dthis.is_fine ; 
            edge_idh(i,iside) = dthis.edge_id ; 
            off_ih(i,iside) = dthis.off_i ; 
            off_jh(i,iside) = dthis.off_j ; 
            if ( dthis.is_fine ) {
                for( int ic=0; ic<2; ++ic) {
                    fine_qidh(i,iside,ic) = dthis.octants.fine.quad_id[ic] ; 
                    fine_bidh(i,iside,ic) = dthis.octants.fine.buf_id[ic] ; 
                    fine_owner_rankh(i,iside,ic) = dthis.octants.fine.owner_rank[ic] ; 
                    fine_is_remoteh(i,iside,ic) = dthis.octants.fine.is_remote[ic];
                }
            } else {
                coarse_qidh(i,iside) = dthis.octants.coarse.quad_id ; 
                coarse_bidh(i,iside) = dthis.octants.coarse.buf_id ; 
                coarse_owner_rankh(i,iside) = dthis.octants.coarse.owner_rank ; 
                coarse_is_remoteh(i,iside) = dthis.octants.coarse.is_remote ; 
            }
        }
    }

    deep_copy(out.is_fine,is_fineh) ; 
    deep_copy(out.n_sides,n_sidesh) ; 

    deep_copy(out.fine_qid,fine_qidh) ; 
    deep_copy(out.fine_bid,fine_bidh) ; 
    deep_copy(out.fine_owner_rank,fine_owner_rankh) ; 
    deep_copy(out.fine_is_remote,fine_is_remoteh) ; 

    deep_copy(out.coarse_qid,coarse_qidh) ; 
    deep_copy(out.coarse_bid,coarse_bidh) ; 
    deep_copy(out.coarse_owner_rank,coarse_owner_rankh) ; 
    deep_copy(out.coarse_is_remote,coarse_is_remoteh) ; 

    deep_copy(out.edge_id,edge_idh) ; 
    deep_copy(out.off_i,off_ih) ; 
    deep_copy(out.off_j,off_jh) ; 

    return out ; 
}

void make_rflux_array(
    amr::reflux_array_t& arr,
    std::string const& name, 
    std::vector<size_t> const& offsets,
    size_t total_size,
    size_t s1, size_t s2
) 
{
    arr = amr::reflux_array_t(name) ; 
    arr.set_strides(s1,s2) ; 
    arr.set_offsets(offsets) ; 
    arr.realloc(total_size) ;  
}

void make_erflux_array(
    amr::reflux_edge_array_t& arr,
    std::string const& name, 
    std::vector<size_t> const& offsets,
    size_t total_size,
    size_t stride
) 
{
    arr = amr::reflux_edge_array_t(name) ; 
    arr.set_strides(stride) ; 
    arr.set_offsets(offsets) ; 
    arr.realloc(total_size) ;  
}

void amr_ghosts_impl_t::build_reflux_buffers() {
    /************************************************************************************************/
    // we need to construct the data structures for: 
    // 
    // reflux across amr faces 
    // re-circulation across amr and uniform faces 
    // re-circulation across amr and uniform edges 
    DECLARE_GRID_EXTENTS ; 
    /************************************************************************************************/
    // get mpi info
    auto rank = parallel::mpi_comm_rank() ; 
    auto nproc= parallel::mpi_comm_size() ;
    // nvars 
    auto nvars_hrsc = variables::get_n_hrsc() ;
    // helper 
    comm_patt_builder cpb{nproc} ; 
    // this we will use over and over 
    std::vector<std::vector<comm_key_t>> snd_keys(nproc), rcv_keys(nproc) ; 
    /************************************************************************************************/
    /************************************************************************************************/
    // 1) Faces with non-uniform resolution
    // This affects both refluxing and re-circulation
    for( int i=0; i<_reflux_face_descs.size(); ++i) {
        auto const& dsc =  _reflux_face_descs[i] ; 
        // if coarse side is remote we send our data 
        // if coarse side is local we receive from any remote children
        if ( dsc.coarse_is_remote ) {
            for( int ic=0; ic<P4EST_CHILDREN/2; ++ic) {
                if ( dsc.fine_is_remote[ic] ) continue ; 
                // for faces: we use the element id (face id) from the smaller 
                // rank. This ensures that the relative ordering is the same 
                // from the sender and receiver pov 
                int8_t elem_id = dsc.coarse_owner_rank < rank 
                                    ? dsc.coarse_face_id : dsc.fine_face_id ; 
                // create a send key 
                snd_keys[dsc.coarse_owner_rank].push_back(
                    comm_key_t{
                        dsc.fine_qid[ic], elem_id
                    }
                ) ;
            } // for ic 
        } else { // coarse is local 
            for( int ic=0; ic<P4EST_CHILDREN/2; ++ic) {
                if ( !dsc.fine_is_remote[ic]) continue ; // both local, nothing to report
                int8_t elem_id = dsc.fine_owner_rank[ic] < rank 
                    ? dsc.fine_face_id : dsc.coarse_face_id ; 
                rcv_keys[dsc.fine_owner_rank[ic]].push_back(
                    comm_key_t{
                        dsc.fine_qid[ic], elem_id
                    }
                ); 
            } // for ic 
        } // if coarse remote 
    } // for face descriptor 
    /************************************************************************************************/
    /************************************************************************************************/
    // TODO remove this 
    size_t tnsnd{0UL},tnrcv{0UL} ; 
    for( int iproc=0; iproc<nproc; ++iproc) {
        tnsnd += snd_keys[iproc].size() ;
        tnrcv += rcv_keys[iproc].size() ; 
    } 
    GRACE_TRACE("[REFLUX] Hanging Face Descs, pre-dedup send {} recv {}", tnsnd,tnrcv) ;

    // now we sort and de-duplicate keys 
    std::vector<std::unordered_map<comm_key_t, size_t, comm_key_hash>> send_lookup, recv_lookup ; 
    std::vector<size_t> rank_send_counts, rank_recv_counts ; 
    std::tie(send_lookup,rank_send_counts) = cpb.sort_and_dedup(snd_keys) ;
    std::tie(recv_lookup,rank_recv_counts) = cpb.sort_and_dedup(rcv_keys) ;

    tnsnd = 0UL ; tnrcv = 0UL ; 
    for( int iproc=0; iproc<nproc; ++iproc) {
        tnsnd += snd_keys[iproc].size() ;
        tnrcv += rcv_keys[iproc].size() ; 
    } 
    GRACE_TRACE("[REFLUX] Hanging Face Descs, post-dedup send {} recv {}", tnsnd,tnrcv) ;
    /************************************************************************************************/
    /************************************************************************************************/
    // phase 2, write back buffer ids and create a list of send descriptors 
     std::vector<hanging_face_reflux_desc_t> local_interfaces ; 
    _reflux_face_snd.clear() ;  

    for (int i = 0; i < _reflux_face_descs.size(); ++i) {
        auto &dsc = _reflux_face_descs[i];
        if (dsc.coarse_is_remote) {
            for (int ic = 0; ic < P4EST_CHILDREN/2; ++ic) {
                if (dsc.fine_is_remote[ic]) continue ; // both remote 
                // need to send data 
                hanging_remote_reflux_desc_t snd_desc{} ;
                // rank 
                int r = dsc.coarse_owner_rank;
                // element id for key 
                int8_t elem = (dsc.coarse_owner_rank < rank ?
                            dsc.coarse_face_id : dsc.fine_face_id);
                // construct key 
                comm_key_t key{ dsc.fine_qid[ic], elem };
                // construct desc 
                snd_desc.qid = dsc.fine_qid[ic] ; 
                snd_desc.rank = r ; 
                snd_desc.elem_id =  dsc.fine_face_id ; // note 
                // store buffer id 
                snd_desc.buf_id = send_lookup[r][key];
                // push 
                _reflux_face_snd.push_back(snd_desc) ; 
            } // for ic   
        } else { // coarse is local 
            for (int ic = 0; ic < P4EST_CHILDREN/2; ++ic) {
                if (!dsc.fine_is_remote[ic]) continue ;
                // rank 
                int r = dsc.fine_owner_rank[ic];
                // elem id for key 
                int8_t elem = (dsc.fine_owner_rank[ic] < rank ?
                            dsc.fine_face_id : dsc.coarse_face_id);
                comm_key_t key{ dsc.fine_qid[ic], elem };
                // no longer replace fine qid with the buf id 
                dsc.fine_bid[ic] = recv_lookup[r][key];
            } // for ic 
            // no need to keep interfaces where the coarse side 
            // is not local
            local_interfaces.push_back( dsc ); 
        } // if coarse is remote 
    } // for desc 
    _reflux_face_descs.swap(local_interfaces) ; // note, we discard the total array here and only keep the locals
    _reflux_face_descs_d = to_device(_reflux_face_descs) ; 
    // we need to dedup 
    sort_and_dedup<rdesc_cmp,rdesc_eq>(_reflux_face_snd) ; 
    // upload 
    _reflux_face_snd_d = to_device(_reflux_face_snd) ; 
    // -- 
    GRACE_VERBOSE("[REFLUX] We have {} hanging faces which need refluxing", _reflux_face_snd.size()) ; 
    /************************************************************************************************/
    // allocate the buffers 
    size_t send_size_flux = (nx/2)*(nx/2) * nvars_hrsc ; 
    // then emfs: size of each is (n/2+1) x (n/2+1) x 2 (the 2 because two edge dirs per face)
    size_t send_size_emf = (nx/2)*(nx/2) * 2  ;
    size_t total_snd_flux, total_rcv_flux, total_snd_emf, total_rcv_emf ; 
    std::tie(total_snd_flux,_reflux_snd_size,_reflux_snd_off) = cpb.get_offsets_and_sizes(rank_send_counts,send_size_flux) ; 
    std::tie(total_rcv_flux,_reflux_rcv_size,_reflux_rcv_off) = cpb.get_offsets_and_sizes(rank_recv_counts,send_size_flux) ; 

    std::tie(total_snd_emf,_reflux_snd_emf_size,_reflux_snd_emf_off) = cpb.get_offsets_and_sizes(rank_send_counts,send_size_emf) ; 
    std::tie(total_rcv_emf,_reflux_rcv_emf_size,_reflux_rcv_emf_off) = cpb.get_offsets_and_sizes(rank_recv_counts,send_size_emf) ;

    make_rflux_array(_reflux_snd_buf,"reflux_flux_send",_reflux_snd_off,total_snd_flux,nx/2,nvars_hrsc) ;
    make_rflux_array(_reflux_recv_buf,"reflux_flux_recv",_reflux_rcv_off,total_rcv_flux,nx/2,nvars_hrsc) ;

    make_rflux_array(_reflux_emf_snd_buf,"reflux_emf_send",_reflux_snd_emf_off,total_snd_emf,nx/2,2) ;
    make_rflux_array(_reflux_emf_recv_buf,"reflux_emf_recv",_reflux_rcv_emf_off,total_rcv_emf,nx/2,2) ;

    // Detailed per-rank breakdown
    for( int r=0; r<nproc; ++r) {
        if (_reflux_snd_size[r] > 0 || _reflux_rcv_size[r] > 0) {
            GRACE_VERBOSE("[REFLUX]: Fluxes: Rank {}: send[off={}, size={}] recv[off={}, size={}]", 
                r, 
                _reflux_snd_off[r], _reflux_snd_size[r],
                _reflux_rcv_off[r], _reflux_rcv_size[r]);
            GRACE_VERBOSE("[REFLUX]: EMFs face: Rank {}: send[off={}, size={}] recv[off={}, size={}]", 
                r, 
                _reflux_snd_emf_off[r], _reflux_snd_emf_size[r],
                _reflux_rcv_emf_off[r], _reflux_rcv_emf_size[r]);
        }
    }

    /************************************************************************************************/
    /************************************************************************************************/
    /************************************************************************************************/
    // next we handle coarse faces -- needed **only** for EMF recirculation 
    // reset 
    snd_keys = std::vector<std::vector<comm_key_t>>(nproc) ; 
    rcv_keys = std::vector<std::vector<comm_key_t>>(nproc) ;

    for( int i=0; i<_reflux_coarse_face_descs.size(); ++i) {
        auto const& dsc =  _reflux_coarse_face_descs[i] ; 
        // loop over sides 
        for( int is=0; is<2; ++is ) {
            // if remote we must exchange 
            if ( dsc.is_remote[is] ) {
                // send and receive 
                int8_t elem_id = dsc.owner_rank[is] < rank 
                    ? dsc.face_id[is] : dsc.face_id[1-is] ;
                // we send the other qid 
                snd_keys[dsc.owner_rank[is]].push_back(
                    comm_key_t(dsc.qid[1-is], elem_id)
                ) ; 
                // we receive this qid 
                rcv_keys[dsc.owner_rank[is]].push_back(
                    comm_key_t(dsc.qid[is], elem_id)
                ) ;
            } 
        } // loop over sides 
    } // coarse face desc 
    /************************************************************************************************/
    // sort and deduplicate the send and receive keys 
    std::tie(send_lookup,rank_send_counts) = cpb.sort_and_dedup(snd_keys) ;
    std::tie(recv_lookup,rank_recv_counts) = cpb.sort_and_dedup(rcv_keys) ;
    /************************************************************************************************/
    _reflux_coarse_face_snd.clear() ; 
    
    // second loop over coarse
    for( int i=0; i<_reflux_coarse_face_descs.size(); ++i) {
        auto& dsc =  _reflux_coarse_face_descs[i] ; 
         
        for( int is=0; is<2; ++is ) {
            
            if ( dsc.is_remote[is] ) {
                ASSERT(!dsc.is_remote[1-is], "both sides remote!") ; 
                // remote owner rank id 
                auto r = dsc.owner_rank[is] ;
                // send and receive 
                int8_t elem_id = r < rank 
                    ? dsc.face_id[is] : dsc.face_id[1-is] ; 
                // send: we need to register into the send list 
                // other qid (the local one!)
                comm_key_t snd_key{
                    dsc.qid[1-is], elem_id
                } ; 
                hanging_remote_reflux_desc_t snd_desc{} ;
                snd_desc.qid = dsc.qid[1-is] ;
                snd_desc.rank = r ; 
                snd_desc.elem_id =  dsc.face_id[1-is] ;
                snd_desc.buf_id = send_lookup[r][snd_key];
                _reflux_coarse_face_snd.push_back(snd_desc) ;

                // receive: we write back the bufid into the desc
                comm_key_t rcv_key{
                    dsc.qid[is], elem_id
                } ; 
                dsc.bid[is] = recv_lookup[r][rcv_key] ; 
            }
        } // loop over sides 
    } // loop over coarse 
    std::ofstream file("coarse_face_descs_"+std::to_string(rank)+".dat");
    for( int i=0; i<_reflux_coarse_face_descs.size(); ++i) {
        auto const dsc =  _reflux_coarse_face_descs[i] ; 
        for( int is=0; is<2; ++is ) {
            file << i << "\t" << is << "\t" << dsc.is_remote[is] << "\t" << dsc.owner_rank[is] << "\t" << dsc.face_id[is] << "\t" << dsc.qid[is] << '\t' << dsc.bid[is] << '\n'; 
        }
    }
    
    /************************************************************************************************/
    sort_and_dedup<rdesc_cmp,rdesc_eq>(_reflux_coarse_face_snd) ; 
    _reflux_coarse_face_snd_d = to_device(_reflux_coarse_face_snd) ; 
    _reflux_coarse_face_descs_d = to_device(_reflux_coarse_face_descs) ; 
    GRACE_VERBOSE("[REFLUX] We have {} coarse faces which need refluxing", _reflux_coarse_face_descs.size()) ; 
    std::ofstream file4("coarse_face_send_"+std::to_string(rank)+".dat");
    for( int i=0; i<_reflux_coarse_face_snd.size(); ++i) {
        auto const& dsc = _reflux_coarse_face_snd[i] ; 
        file4 << i << "\t" << dsc.rank << "\t" <<  dsc.elem_id << "\t" << dsc.qid << '\t' << dsc.buf_id << '\n'; 
    }
    /************************************************************************************************/
    // allocate buffers 
    send_size_emf = nx*nx*2 ; 
    size_t total_snd_emf_coarse, total_recv_emf_coarse ; 
    std::tie(total_snd_emf_coarse,_reflux_snd_emf_coarse_size,_reflux_snd_emf_coarse_off) = cpb.get_offsets_and_sizes(rank_send_counts,send_size_emf) ; 
    std::tie(total_recv_emf_coarse,_reflux_rcv_emf_coarse_size,_reflux_rcv_emf_coarse_off) = cpb.get_offsets_and_sizes(rank_recv_counts,send_size_emf) ;
    make_rflux_array(_reflux_emf_coarse_snd_buf,"reflux_emf_coarse_send",_reflux_snd_emf_coarse_off,total_snd_emf_coarse,nx,2) ; 
    make_rflux_array(_reflux_emf_coarse_recv_buf,"reflux_emf_coarse_receive",_reflux_rcv_emf_coarse_off,total_recv_emf_coarse,nx,2) ;
    // Detailed per-rank breakdown
    for( int r=0; r<nproc; ++r) {
        
        if (_reflux_snd_emf_coarse_size[r] > 0 || _reflux_rcv_emf_coarse_size[r] > 0) {
            GRACE_VERBOSE("[REFLUX]: Coarse faces: Rank {}: send[off={}, size={}] recv[off={}, size={}]", 
                r, 
                _reflux_snd_emf_coarse_off[r], _reflux_snd_emf_coarse_size[r],
                _reflux_rcv_emf_coarse_off[r], _reflux_rcv_emf_coarse_size[r]);
        }
    }
    /************************************************************************************************/
    /************************************************************************************************/
    /************************************************************************************************/
    // Edges 
    // reset 
    snd_keys = std::vector<std::vector<comm_key_t>>(nproc) ; 
    rcv_keys = std::vector<std::vector<comm_key_t>>(nproc) ; 
    for( int i=0; i<_reflux_edge_descs.size(); ++i) {
        auto& dsc = _reflux_edge_descs[i] ; 
        // loop over sides 
        for( int iside=0; iside<dsc.n_sides; ++iside) {
            auto const& dsc_this = dsc.sides[iside] ; 
            // we only send and receive fine data 
            if ( !dsc_this.is_fine ) continue ; 
            // loop over children 
            for( int ic=0; ic<2; ++ic) {
                // remote -> receive 
                if ( dsc_this.octants.fine.is_remote[ic] ) {
                    auto r =  dsc_this.octants.fine.owner_rank[ic] ; 
                    rcv_keys[r].push_back(
                        comm_key_t{
                            dsc_this.octants.fine.quad_id[ic],
                            dsc_this.edge_id
                        } 
                    ) ; 
                } else {
                    // local -> send 
                    for( int jside=0; jside<dsc.n_sides; ++jside) {
                        if ( iside == jside ) continue ; 
                        auto const& dsc_other = dsc.sides[jside] ; 
                        if (dsc_other.is_fine) {
                            // only need to consider 
                            // matching ic 
                            // this assumes coordinates align 
                            if ( !dsc_other.octants.fine.is_remote[ic] ) continue ; 
                            // send 
                            auto r =  dsc_other.octants.fine.owner_rank[ic] ; 
                            snd_keys[r].push_back(
                                comm_key_t{
                                    dsc_this.octants.fine.quad_id[ic],
                                    dsc_this.edge_id
                                } 
                            ) ;
                        } else { // other is coarse 
                            if ( !dsc_other.octants.coarse.is_remote ) continue ; 
                            // send 
                            auto r =  dsc_other.octants.coarse.owner_rank ; 
                            snd_keys[r].push_back(
                                comm_key_t{
                                    dsc_this.octants.fine.quad_id[ic],
                                    dsc_this.edge_id
                                } 
                            ) ; 
                        } // if other is fine 
                    } // loop over sides again 
                } // if this is remote 
            } // ic loop 
        } // loop over sides 
    } // loop over edge 
    /************************************************************************************************/
    // note: we need to dedup since if multiple remotes are on the same rank 
    // we send the data to each octant, which is redundant. 
    std::tie(send_lookup,rank_send_counts) = cpb.sort_and_dedup(snd_keys) ;
    std::tie(recv_lookup,rank_recv_counts) = cpb.sort_and_dedup(rcv_keys) ;
    /************************************************************************************************/
    // loop again to fill send desc 
    _reflux_edge_snd.clear() ;     
    for( int i=0; i<_reflux_edge_descs.size(); ++i) {
        auto& dsc = _reflux_edge_descs[i] ; 
        // only loop over fine, these are the ones we send and receive 
        for( int iside=0; iside<dsc.n_sides; ++iside) {
            auto& dsc_this = dsc.sides[iside] ; 
            // skip coarse 
            if (!dsc_this.is_fine) continue ; 

            for( int ic=0; ic<2; ++ic) {
                if ( dsc_this.octants.fine.is_remote[ic] ) {
                    // fine remote --> receive 
                    comm_key_t key {
                            dsc_this.octants.fine.quad_id[ic], dsc_this.edge_id
                    } ; 
                    auto r = dsc_this.octants.fine.owner_rank[ic] ; 
                    // modify in-place. This quad is remote anyway, 
                    // currently its quad-id is set to the index in the 
                    // mirror ghost array which is in z-order wrt the owner 
                    // tree. The only time we ever need that number is 
                    // to ensure stable ordering in the send / receive buffers,
                    // which we already constructed. FIXME is this right? 
                    dsc_this.octants.fine.buf_id[ic] = recv_lookup[r][key];
                } else { // local 
                    for( int jside=0; jside<dsc.n_sides; ++jside){ 
                        if ( jside==iside ) continue ; 
                        auto const& dsc_other = dsc.sides[jside] ; 
                        if ( dsc_other.is_fine) { // other is fine
                            // note we only check matching ic the other one
                            // is staggered (this changed from previous v)
                            if (!dsc_other.octants.fine.is_remote[ic]) continue ; 
                            // note, in send we use **our** qid and eid 
                            comm_key_t key {
                                dsc_this.octants.fine.quad_id[ic], dsc_this.edge_id
                            } ; 
                            hanging_remote_reflux_desc_t snd_desc{} ; 
                            snd_desc.qid = dsc_this.octants.fine.quad_id[ic] ; 
                            auto r = dsc_other.octants.fine.owner_rank[ic] ;
                            snd_desc.rank = r; 
                            snd_desc.elem_id = dsc_this.edge_id ; 
                            snd_desc.buf_id = send_lookup[r][key] ; 
                            _reflux_edge_snd.push_back(snd_desc) ; 
                        } else {
                            if ( !dsc_other.octants.coarse.is_remote) continue ; 
                            // note, in send we use **our** qid and eid 
                            comm_key_t key {
                                        dsc_this.octants.fine.quad_id[ic], dsc_this.edge_id
                            } ; 
                            hanging_remote_reflux_desc_t snd_desc{} ;
                            snd_desc.qid = dsc_this.octants.fine.quad_id[ic] ; 
                            auto r = dsc_other.octants.coarse.owner_rank ;
                            snd_desc.rank = r; 
                            snd_desc.elem_id = dsc_this.edge_id ; 
                            snd_desc.buf_id = send_lookup[r][key] ;
                            _reflux_edge_snd.push_back(snd_desc) ; 
                        } // other is coarse 
                    } // loop over other side 
                } // if remote 
            } // loop over ic 
        } // loop over sides 
    } // loop over edges 
    /************************************************************************************************/
    sort_and_dedup<rdesc_cmp,rdesc_eq>(_reflux_edge_snd) ; 
    _reflux_edge_snd_d = to_device(_reflux_edge_snd) ; 
    _reflux_edge_descs_d = to_device(_reflux_edge_descs) ; 
    /************************************************************************************************/
    send_size_emf = (nx) ; 

    size_t total_snd_emf_edge, total_rcv_emf_edge;
    std::tie(total_snd_emf_edge,_reflux_snd_emf_edge_size,_reflux_snd_emf_edge_off) = cpb.get_offsets_and_sizes(rank_send_counts,send_size_emf) ; 
    std::tie(total_rcv_emf_edge,_reflux_rcv_emf_edge_size,_reflux_rcv_emf_edge_off) = cpb.get_offsets_and_sizes(rank_recv_counts,send_size_emf) ; 

    make_erflux_array(_reflux_emf_edge_snd_buf,"reflux_emf_edge_send",_reflux_snd_emf_edge_off,total_snd_emf_edge,nx) ; 
    make_erflux_array(_reflux_emf_edge_recv_buf,"reflux_emf_edge_receive",_reflux_rcv_emf_edge_off,total_rcv_emf_edge,nx) ;

    _reflux_emf_edge_accumulation_buf = Kokkos::View<double***, grace::default_space>(
        "reflux_emf_edge_local_buffer", nx, 2, _reflux_edge_descs.size()
    ) ; 

    // Detailed per-rank breakdown
    for( int r=0; r<nproc; ++r) {
        if (_reflux_snd_emf_edge_size[r] > 0 || _reflux_rcv_emf_edge_size[r] > 0) {
            GRACE_VERBOSE("[REFLUX]: EMFs edge: Rank {}: send[off={}, size={}] recv[off={}, size={}]", 
                r, 
                _reflux_snd_emf_edge_off[r], _reflux_snd_emf_edge_size[r],
                _reflux_rcv_emf_edge_off[r], _reflux_rcv_emf_edge_size[r]);
        }
    }
    /************************************************************************************************/
    /************************************************************************************************/
    /************************************************************************************************/
    // now the coarse edge
    // reset 
    snd_keys = std::vector<std::vector<comm_key_t>>(nproc) ; 
    rcv_keys = std::vector<std::vector<comm_key_t>>(nproc) ; 
    for( int i=0; i<_reflux_coarse_edge_descs.size(); ++i) {
        auto& dsc = _reflux_coarse_edge_descs[i] ;
        // loop over sides 
        for( int iside=0; iside<dsc.n_sides; ++iside) {
            auto const& dsc_this = dsc.sides[iside] ; 
            ASSERT(!dsc_this.is_fine, "In coarse edges got fine side.") ; 
            // if remote 
            if ( dsc_this.octants.coarse.is_remote ) {
                GRACE_TRACE("Receive coarse, quadid {} rank {} edge {}",dsc_this.octants.coarse.quad_id,dsc_this.octants.coarse.owner_rank,dsc_this.edge_id );
                // receive 
                rcv_keys[dsc_this.octants.coarse.owner_rank].push_back(
                        comm_key_t{
                            dsc_this.octants.coarse.quad_id, dsc_this.edge_id
                        }
                    ) ;
            // if local
            } else { 
                // send 
                for( int jside=0; jside<dsc.n_sides; ++jside){ 
                    if ( jside==iside ) continue ;
                    auto const& dsc_other = dsc.sides[jside] ; 
                    if ( !dsc_other.octants.coarse.is_remote ) continue ; 
                    GRACE_TRACE("Send coarse, quadid {} rank {} edge {}",dsc_this.octants.coarse.quad_id,dsc_other.octants.coarse.owner_rank,dsc_this.edge_id );
                    snd_keys[dsc_other.octants.coarse.owner_rank].push_back(
                                    comm_key_t{
                                        dsc_this.octants.coarse.quad_id, dsc_this.edge_id
                                    }
                                ) ;
                }
            } // is remote 
        } // loop over sides 
    } // loop over coarse edges 
    /************************************************************************************************/
    std::tie(send_lookup,rank_send_counts) = cpb.sort_and_dedup(snd_keys) ;
    std::tie(recv_lookup,rank_recv_counts) = cpb.sort_and_dedup(rcv_keys) ;
    /************************************************************************************************/
    _reflux_coarse_edge_snd.clear() ;
    // loop again 
    for( int i=0; i<_reflux_coarse_edge_descs.size(); ++i) {
        auto& dsc = _reflux_coarse_edge_descs[i] ;

        for( int iside=0; iside<dsc.n_sides; ++iside) {
            auto& dsc_this = dsc.sides[iside] ; 
            if ( dsc_this.octants.coarse.is_remote ) {
                // receive 
                comm_key_t key {
                            dsc_this.octants.coarse.quad_id, dsc_this.edge_id
                        } ;
                // fixme, ensure this is never used again 
                auto r = dsc_this.octants.coarse.owner_rank ; 
                GRACE_TRACE("Receive coarse, quadid {} rank {} edge {} buf idx {}",dsc_this.octants.coarse.quad_id,dsc_this.octants.coarse.owner_rank,dsc_this.edge_id, recv_lookup[r][key] );
                dsc_this.octants.coarse.buf_id = recv_lookup[r][key];
            } else {
                // send 
                for( int jside=0; jside<dsc.n_sides; ++jside){ 
                    if ( jside==iside ) continue ;
                    auto const& dsc_other = dsc.sides[jside] ; 
                    if ( !dsc_other.octants.coarse.is_remote ) continue ; 
                    comm_key_t key {
                                    dsc_this.octants.coarse.quad_id, dsc_this.edge_id
                                } ; 
                    hanging_remote_reflux_desc_t snd_desc{} ; 
                    snd_desc.qid = dsc_this.octants.coarse.quad_id ; 
                    auto r = dsc_other.octants.coarse.owner_rank ;
                    snd_desc.rank = r; 
                    snd_desc.elem_id = dsc_this.edge_id ; 
                    snd_desc.buf_id = send_lookup[r][key] ; 
                    _reflux_coarse_edge_snd.push_back(snd_desc) ;
                    GRACE_TRACE("Send coarse, quadid {} rank {} edge {} buf idx {}",dsc_this.octants.coarse.quad_id,dsc_other.octants.coarse.owner_rank,dsc_this.edge_id, send_lookup[r][key]  );
                }
            }
        }
    }
    
    // we need to dedup 
    sort_and_dedup<rdesc_cmp,rdesc_eq>(_reflux_coarse_edge_snd) ; 
    // upload 
    _reflux_coarse_edge_snd_d = to_device(_reflux_coarse_edge_snd) ; 
    _reflux_coarse_edge_descs_d = to_device(_reflux_coarse_edge_descs) ; 
    std::ofstream file2("coarse_edge_descs_"+std::to_string(rank)+".dat");
    for( int i=0; i<_reflux_coarse_edge_descs.size(); ++i) {
        auto const dsc =  _reflux_coarse_edge_descs[i] ; 
        for( int is=0; is<4; ++is ) {
            auto& dsc_this = dsc.sides[is] ; 
            file2 << i << "\t" << is << "\t" << dsc_this.octants.coarse.is_remote << "\t" << dsc_this.octants.coarse.owner_rank << "\t" <<  dsc_this.edge_id << "\t" << dsc_this.octants.coarse.quad_id << '\t' << dsc_this.octants.coarse.buf_id << '\n'; 
        }
    }
    std::ofstream file3("coarse_edge_send_"+std::to_string(rank)+".dat");
    for( int i=0; i<_reflux_coarse_edge_snd.size(); ++i) {
        auto const& dsc = _reflux_coarse_edge_snd[i] ; 
        file3 << i << "\t" << dsc.rank << "\t" <<  dsc.elem_id << "\t" << dsc.qid << '\t' << dsc.buf_id << '\n'; 
    }
    /************************************************************************************************/
    // allocate the buffers 
    send_size_emf = (nx) ; 

    size_t total_snd_emf_coarse_edge, total_rcv_emf_coarse_edge;
    std::tie(total_snd_emf_coarse_edge,_reflux_snd_emf_coarse_edge_size,_reflux_snd_emf_coarse_edge_off) = cpb.get_offsets_and_sizes(rank_send_counts,send_size_emf) ; 
    std::tie(total_rcv_emf_coarse_edge,_reflux_rcv_emf_coarse_edge_size,_reflux_rcv_emf_coarse_edge_off) = cpb.get_offsets_and_sizes(rank_recv_counts,send_size_emf) ; 
    
    make_erflux_array(_reflux_emf_coarse_edge_snd_buf,"reflux_emf_coarse_edge_send",_reflux_snd_emf_coarse_edge_off,total_snd_emf_coarse_edge,nx) ; 
    make_erflux_array(_reflux_emf_coarse_edge_recv_buf,"reflux_emf_coarse_edge_receive",_reflux_rcv_emf_coarse_edge_off,total_rcv_emf_coarse_edge,nx) ;

    _reflux_emf_coarse_edge_accumulation_buf = Kokkos::View<double**, grace::default_space>(
        "reflux_emf_coarse_edge_local_buffer", nx, _reflux_coarse_edge_descs.size()
    ) ;

    // Detailed per-rank breakdown
    for( int r=0; r<nproc; ++r) {
        if (_reflux_snd_emf_coarse_edge_size[r] > 0 || _reflux_rcv_emf_coarse_edge_size[r] > 0) {
            GRACE_VERBOSE("[REFLUX]: EMFs coarse-edge: Rank {}: send[off={}, size={}] recv[off={}, size={}]", 
                r, 
                _reflux_snd_emf_coarse_edge_off[r], _reflux_snd_emf_coarse_edge_size[r],
                _reflux_rcv_emf_coarse_edge_off[r], _reflux_rcv_emf_coarse_edge_size[r]);
        }
    }

}

} /* namespace grace */