/**
 * @file task_factories.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
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

#include <grace/amr/regrid/regrid_transaction.hh>


#include <grace/utils/device_stream_pool.hh>

#include <grace/amr/amr_functions.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/amr/ghostzone_kernels/ghost_array.hh>

#include <grace/amr/forest.hh>
#include <grace/amr/quadrant.hh>
#include <grace/amr/amr_flags.hh>
#include <grace/amr/regrid/regrid_helpers.tpp>
#include <grace/amr/regrid/regridding_policy_kernels.tpp>
#include <grace/amr/regrid/partition.hh>
#include <grace/amr/amr_ghosts.hh>

#include <grace/coordinates/coordinates.hh>

namespace grace { namespace amr {

void regrid_transaction_t::build_buffers() {
    GRACE_TRACE("[REGRID] Entering build buffers") ; 
    /******************************************************************************************/
    auto nprocs = parallel::mpi_comm_size() ; 
    /******************************************************************************************/
    auto& ghosts = grace::amr_ghosts::get() ; 
    auto const& ghost_array = ghosts.get_ghost_layer() ;
    auto const p4est_ghosts = ghosts.get_p4est_ghosts() ;
    /******************************************************************************************/
    auto get_piggy3 = [&] (size_t gid) -> std::tuple<size_t,size_t> {
        sc_array_view_t<p4est_quadrant_t> _garr{&(p4est_ghosts->ghosts)} ; 
        return std::make_tuple(
            _garr[gid].p.piggy3.local_num,
            _garr[gid].p.piggy3.which_tree
        ) ; 
    } ;
    /******************************************************************************************/
    have_fine_data_x.resize(refine_incoming.size(), {{0,0}}) ; 
    have_fine_data_y.resize(refine_incoming.size(), {{0,0}}) ; 
    have_fine_data_z.resize(refine_incoming.size(), {{0,0}}) ; 
    // we collect descriptors of the data we need to receive
    // from each rank. Later we call allgatherv and figure 
    // out what we need to send. From that we can construct 
    // the buffers. 
    std::vector<std::vector<fine_face_data_desc_t>> recv_x(nprocs), recv_y(nprocs), recv_z(nprocs) ; 
    // loop over refine_outgoing --> quadrants that will be replaced
    // by prolonged finer quads. For each of them we check if any 
    // of the face neighbors are already fine, in which case we need 
    // to copy the available fine data before the div free prolong.
    for( int iquad=0; iquad< refine_outgoing.size(); ++iquad ) {
        auto iq = refine_outgoing[iquad] ; 
        for( int8_t iface=0; iface<P4EST_FACES; ++iface) {
            auto& face = ghost_array[iq].faces[iface] ; 
            int8_t axis = iface/2 ;
            int side = (iface%2) ;
            if ( face.kind == interface_kind_t::PHYS ) continue ; 
            if ( face.level_diff == level_diff_t::FINER ) {
                for( int icx=0; icx<2; ++icx) {
                    for ( int icy=0; icy<2; ++icy) {
                        int ic = icx + (icy<<1) ; // child id in nbor face, 0,..,3 
                        if ( face.data.hanging.is_remote[ic] ) {
                            auto owner_rank = face.data.hanging.owner_rank[ic] ; 
                            fine_face_data_desc_t desc {} ; 
                            desc.axis = axis ; 
                            desc.side = side ; 
                            desc.fid_local = iface ; 
                            desc.fid_remote = face.face ; 
                            auto ghost_id = face.data.hanging.quad_id[ic] ; 
                            auto [quad_id_remote, which_tree] = get_piggy3(ghost_id) ;
                            desc.qid_remote = quad_id_remote ; 
                            desc.which_tree = which_tree ; 
                            desc.qid_ghost = ghost_id ; 
                            if ( axis == 0 ) {
                                // x face, cross directions are y and z
                                int ic_vol = side + (icx<<1) + (icy<<2) ; 
                                auto quad_id_local = refine_incoming[P4EST_CHILDREN*iquad + ic_vol];
                                desc.qid_local = quad_id_local ; 
                                
                                recv_x[owner_rank].push_back(desc) ; 
                            } else if ( axis == 1 ) { 
                                // y face, cross directions are x and z
                                int ic_vol = (icx<<0) + (side<<1) + (icy<<2) ;
                                auto quad_id_local = refine_incoming[P4EST_CHILDREN*iquad + ic_vol];
                                desc.qid_local = quad_id_local ; 
                                recv_y[owner_rank].push_back(desc) ; 
                            } else { 
                                // z face, cross directions are x and y
                                int ic_vol = (icx<<0) + (icy<<1) + (side<<2);
                                auto quad_id_local = refine_incoming[P4EST_CHILDREN*iquad + ic_vol];
                                desc.qid_local = quad_id_local ; 
                                recv_z[owner_rank].push_back(desc) ; 
                            }
                        } else {
                            fine_interface_desc_t desc ; 
                            desc.qid_src = face.data.hanging.quad_id[ic];
                            desc.fid_src = face.face ;
                            desc.fid_dst = iface ;
                            if ( axis == 0 ) {
                                // x face, cross directions are y and z
                                int ic_vol = side + (icx<<1) + (icy<<2) ; 
                                desc.qid_dst = (refine_incoming[P4EST_CHILDREN*iquad + ic_vol]) ; 
                                local_fine_face_x.push_back(desc);
                                // logic: iface%2 == 0 -> lower face, iface%2==1 -> upper face
                                // this selects whether the fine data is in the upper or lower 
                                // index range
                                have_fine_data_x[P4EST_CHILDREN*iquad + ic_vol][side] = 1 ; 
                            } else if ( axis == 1 ) {
                                // y face, cross directions are x and z
                                int ic_vol = (icx<<0) + (side<<1) + (icy<<2) ; 
                                desc.qid_dst = (refine_incoming[P4EST_CHILDREN*iquad + ic_vol]) ; 
                                local_fine_face_y.push_back(desc);
                                have_fine_data_y[P4EST_CHILDREN*iquad + ic_vol][side] = 1 ; 
                            } else {
                                // z face, cross directions are x and y
                                int ic_vol = (icx<<0) + (icy<<1) + (side<<2); 
                                desc.qid_dst = (refine_incoming[P4EST_CHILDREN*iquad + ic_vol]) ; 
                                local_fine_face_z.push_back(desc);
                                have_fine_data_z[P4EST_CHILDREN*iquad + ic_vol][side] = 1 ; 
                            }
                        } // if local
                    } // loop icy 
                } // loop icx 
            } // if level diff is finer
        } // loop faces
    } // loop quadrants 

    // we are now done with local data. 
    // For remote buffers: 
    // now we make a call to alltoallv 
    // to figure out what we need to send.
    // we will receive the data from the 
    // descriptors above.
    
    
    // 
    // Nomenclature : counts means numbers of faces, sizes means number of doubles 
    // 
    // Each face's size (in doubles is)
    int const send_size_face = nx * nx * nvars_fs ; 

    std::vector<int> sendcounts_x(nprocs), sendcounts_y(nprocs), sendcounts_z(nprocs) ; 
    std::vector<int> recvcounts_x(nprocs), recvcounts_y(nprocs), recvcounts_z(nprocs) ; 
    std::vector<int> sendbcounts_x(nprocs), sendbcounts_y(nprocs), sendbcounts_z(nprocs) ; 
    std::vector<int> recvbcounts_x(nprocs), recvbcounts_y(nprocs), recvbcounts_z(nprocs) ; 

    send_size_x.resize(nprocs);
    send_size_y.resize(nprocs); 
    send_size_z.resize(nprocs);
    recv_size_x.resize(nprocs); 
    recv_size_y.resize(nprocs); 
    recv_size_z.resize(nprocs);

    // 1) exchange counts: send our receive counts, receive other ranks' receive counts
    #define MPI_EXCHANGE_COUNTS(axis) \
    do { \
        for (int r = 0; r < nprocs; ++r) { \
            recvcounts_##axis[r]  =  recv_##axis[r].size()                                ; \
            recvbcounts_##axis[r] =  recvcounts_##axis[r] * sizeof(fine_face_data_desc_t) ; \
            recv_size_##axis[r]   =  recvcounts_##axis[r] * send_size_face                ; \
            GRACE_TRACE("[REGRID] Rank {} receive size {}",r,recv_##axis[r].size());\
        }\
        MPI_Alltoall(recvcounts_##axis.data(), 1, MPI_INT, \
                     sendcounts_##axis.data(), 1, MPI_INT, \
                     MPI_COMM_WORLD ); \
        for (int r = 0; r < nprocs; ++r)\
            GRACE_TRACE("[REGRID] Rank {} send size {}", r, sendcounts_##axis[r]) ; \
    } while(0)
    MPI_EXCHANGE_COUNTS(x);
    MPI_EXCHANGE_COUNTS(y);
    MPI_EXCHANGE_COUNTS(z);

    // 2) Compute byte counts on send 
    #define CALC_BCOUNT_AND_SIZE(axis)\
    do {\
        for (int r = 0; r < nprocs; ++r) { \
            sendbcounts_##axis[r] = sendcounts_##axis[r] * sizeof(fine_face_data_desc_t) ;\
            send_size_##axis[r]   = sendcounts_##axis[r] * send_size_face                ;\
        }\
    } while(false)
    
    CALC_BCOUNT_AND_SIZE(x);
    CALC_BCOUNT_AND_SIZE(y);
    CALC_BCOUNT_AND_SIZE(z);



    // 3) Compute displacements 
    std::vector<int> sdispls_x(nprocs,0), sdispls_y(nprocs,0), sdispls_z(nprocs,0) ; 
    std::vector<int> rdispls_x(nprocs,0), rdispls_y(nprocs,0), rdispls_z(nprocs,0) ; 
    std::vector<int> sbdispls_x(nprocs,0), sbdispls_y(nprocs,0), sbdispls_z(nprocs,0) ; 
    std::vector<int> rbdispls_x(nprocs,0), rbdispls_y(nprocs,0), rbdispls_z(nprocs,0) ; 

    send_off_x.resize(nprocs,0) ; 
    send_off_y.resize(nprocs,0) ; 
    send_off_z.resize(nprocs,0) ; 
    recv_off_x.resize(nprocs,0) ; 
    recv_off_y.resize(nprocs,0) ; 
    recv_off_z.resize(nprocs,0) ; 


    #define MPI_COMPUTE_DISPLACEMENTS(axis) \
    for (int r = 1; r < nprocs; ++r) {\
        sdispls_##axis[r] = sdispls_##axis[r-1] + sendcounts_##axis[r-1];\
        rdispls_##axis[r] = rdispls_##axis[r-1] + recvcounts_##axis[r-1];\
        sbdispls_##axis[r] = sbdispls_##axis[r-1] + sendbcounts_##axis[r-1] ;\
        rbdispls_##axis[r] = rbdispls_##axis[r-1] + recvbcounts_##axis[r-1] ;\
        send_off_##axis[r] = send_off_##axis[r-1] + send_size_##axis[r-1];\
        recv_off_##axis[r] = recv_off_##axis[r-1] + recv_size_##axis[r-1];\
    }
    MPI_COMPUTE_DISPLACEMENTS(x);
    MPI_COMPUTE_DISPLACEMENTS(y);
    MPI_COMPUTE_DISPLACEMENTS(z);

    // 4) total sizes (in bytes)
    #define GET_TOTAL_SIZE(axis)\
    size_t recv_total_size_##axis{0UL}, send_total_size_##axis{0UL} ; \
    size_t recv_total_count_##axis{0UL}, send_total_count_##axis{0UL}; \
    size_t recv_total_bcount_##axis{0UL}, send_total_bcount_##axis{0UL}; \
    for (int r = 0; r < nprocs; ++r){\
        recv_total_size_##axis += recv_size_##axis[r] ; \
        send_total_size_##axis += send_size_##axis[r] ; \
        send_total_count_##axis += sendcounts_##axis[r] ; \
        recv_total_count_##axis += recvcounts_##axis[r] ; \
        send_total_bcount_##axis += sendbcounts_##axis[r] ; \
        recv_total_bcount_##axis += recvbcounts_##axis[r] ; \
    }
    GET_TOTAL_SIZE(x);
    GET_TOTAL_SIZE(y);
    GET_TOTAL_SIZE(z);

    // 5) flatten the receive array, which we are about to **send**
    #define MPI_FLATTEN_ARRAY(axis)\
    std::vector<fine_face_data_desc_t> recvbuf_##axis, sendbuf_##axis; \
    sendbuf_##axis.resize(send_total_count_##axis);\
    for (int r = 0; r < nprocs; ++r){\
        recvbuf_##axis.insert(recvbuf_##axis.end(), recv_##axis[r].begin(), recv_##axis[r].end());\
    }
    MPI_FLATTEN_ARRAY(x);
    MPI_FLATTEN_ARRAY(y);
    MPI_FLATTEN_ARRAY(z);

    // 6) finally we are ready for alltoallv
    // we send our receive buffer, since this means it's what 
    // we need to receive data-wise. Likewise we receive into send buf 
    // since that is what we will need to send data-wise
    #define MPI_EXCHANGE_ALLTOALL(axis)\
    MPI_Alltoallv(recvbuf_##axis.data(), recvbcounts_##axis.data(), rbdispls_##axis.data(), MPI_BYTE,\
            sendbuf_##axis.data(), sendbcounts_##axis.data(), sbdispls_##axis.data(), MPI_BYTE,\
            MPI_COMM_WORLD)
    MPI_EXCHANGE_ALLTOALL(x);
    MPI_EXCHANGE_ALLTOALL(y);
    MPI_EXCHANGE_ALLTOALL(z);

    

    #define REALLOC_BUF(axis)\
    _recv_fbuf_##axis.realloc(recv_total_size_##axis) ;\
    _send_fbuf_##axis.realloc(send_total_size_##axis) ;\
    _recv_fbuf_##axis.set_strides({nx,nvars_fs});\
    _send_fbuf_##axis.set_strides({nx,nvars_fs});\
    _recv_fbuf_##axis.set_offsets(recv_off_##axis);\
    _send_fbuf_##axis.set_offsets(send_off_##axis)

    REALLOC_BUF(x);
    REALLOC_BUF(y);
    REALLOC_BUF(z);
    // for send data: src is local dst is buffer (pack)
    // for reecv data: reverse (unpack)
    // Note on send: The local source of data was recorder by
    // remote partner under qid_remote and which_tree. 
    // NOTE: docs of p4est_ghosts say that the piggy3 local number
    // is CUMULATIVE over trees! 
    #define FILL_BUF_DESC(axis)\
    remote_fine_face_recv_##axis.resize(nprocs);\
    remote_fine_face_send_##axis.resize(nprocs);\
    for( int r=0; r<nprocs; ++r) {\
        for( int ircv=0; ircv<recvcounts_##axis[r]; ++ircv) {\
            fine_interface_desc_t desc; \
            desc.qid_src = ircv ; \
            desc.qid_dst = recvbuf_##axis[ircv+rdispls_##axis[r]].qid_local ;\
            desc.fid_src = recvbuf_##axis[ircv+rdispls_##axis[r]].fid_remote ;\
            desc.fid_dst = recvbuf_##axis[ircv+rdispls_##axis[r]].fid_local ;\
            remote_fine_face_recv_##axis[r].push_back(desc);\
            GRACE_TRACE("[REGRID] Remote face recv rank r {} qsrc {} qdst {} fsrc {} fdst {}", r, desc.qid_src, desc.qid_dst, desc.fid_src, desc.fid_dst  );\
        }\
        for( int isnd=0; isnd<sendcounts_##axis[r]; ++isnd) {\
            auto const& info = sendbuf_##axis[isnd+sdispls_##axis[r]];\
            fine_interface_desc_t desc; \
            desc.qid_dst = isnd ; \
            desc.qid_src = info.qid_remote ;/*+ grace::amr::get_local_quadrants_offset(info.which_tree);*/\
            desc.fid_src = info.fid_remote ;\
            desc.fid_dst = info.fid_local  ;\
            remote_fine_face_send_##axis[r].push_back(desc);\
            GRACE_TRACE("[REGRID] Remote face send rank r {} qsrc {} qdst {} fsrc {} fdst {}", r, desc.qid_src, desc.qid_dst, desc.fid_src, desc.fid_dst  );\
        }\
    }

    FILL_BUF_DESC(x);
    FILL_BUF_DESC(y);
    FILL_BUF_DESC(z);

    parallel::mpi_barrier() ; 
};

}}