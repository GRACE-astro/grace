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
    GRACE_TRACE_DBG("Entering build buffers") ; 
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
                        int ic = icx + (icy<<1) ; 
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
    
    // 1) exchange counts: send our receive counts, receive other ranks' receive counts
    sendcounts_x.resize(nprocs);
    sendcounts_y.resize(nprocs); 
    sendcounts_z.resize(nprocs);
    recvcounts_x.resize(nprocs); 
    recvcounts_y.resize(nprocs); 
    recvcounts_z.resize(nprocs);
    #define MPI_EXCHANGE_COUNTS(axis) \
    do { \
        for (int r = 0; r < nprocs; ++r) \
            recvcounts_##axis[r] = sizeof(fine_face_data_desc_t) * recv_##axis[r].size(); \
        MPI_Alltoall(recvcounts_##axis.data(), 1, MPI_INT, \
                    sendcounts_##axis.data(), 1, MPI_INT, \
                    MPI_COMM_WORLD); \
    } while(0)
    MPI_EXCHANGE_COUNTS(x);
    MPI_EXCHANGE_COUNTS(y);
    MPI_EXCHANGE_COUNTS(z);

    // 2) Compute displacements
    #define MPI_COMPUTE_DISPLACEMENTS(axis) \
    sdispls_##axis.resize(nprocs,0); rdispls_##axis.resize(nprocs, 0);\
    for (int r = 1; r < nprocs; ++r) {\
        sdispls_##axis[r] = sdispls_##axis[r-1] + sendcounts_##axis[r-1];\
        rdispls_##axis[r] = rdispls_##axis[r-1] + recvcounts_##axis[r-1];\
    }
    MPI_COMPUTE_DISPLACEMENTS(x);
    MPI_COMPUTE_DISPLACEMENTS(y);
    MPI_COMPUTE_DISPLACEMENTS(z);

    // total sizes (in bytes)
    #define GET_TOTAL_SIZE(axis)\
    size_t recv_size_##axis{0UL}, send_size_##axis{0UL} ; \
    for (int r = 0; r < nprocs; ++r){\
        recv_size_##axis += recvcounts_##axis[r] ; \
        send_size_##axis += sendcounts_##axis[r] ; \
    }
    GET_TOTAL_SIZE(x);
    GET_TOTAL_SIZE(y);
    GET_TOTAL_SIZE(z);

    // flatten the receive array, which we are about to **send**
    #define MPI_FLATTEN_ARRAY(axis)\
    std::vector<fine_face_data_desc_t> recvbuf_##axis, sendbuf_##axis; \
    sendbuf_##axis.resize(send_size_##axis);\
    for (int r = 0; r < nprocs; ++r){\
        recvbuf_##axis.insert(recvbuf_##axis.end(), recv_##axis[r].begin(), recv_##axis[r].end());\
    }
    MPI_FLATTEN_ARRAY(x);
    MPI_FLATTEN_ARRAY(y);
    MPI_FLATTEN_ARRAY(z);

    // finally we are ready for alltoallv
    // we send our receive buffer, since this means it's what 
    // we need to receive data-wise. Likewise we receive into send buf 
    // since that is what we will need to send data-wise
    #define MPI_EXCHANGE_ALLTOALL(axis)\
    MPI_Alltoallv(recvbuf_##axis.data(), recvcounts_##axis.data(), rdispls_##axis.data(), MPI_BYTE,\
            sendbuf_##axis.data(), sendcounts_##axis.data(), sdispls_##axis.data(), MPI_BYTE,\
            MPI_COMM_WORLD)
    MPI_EXCHANGE_ALLTOALL(x);
    MPI_EXCHANGE_ALLTOALL(y);
    MPI_EXCHANGE_ALLTOALL(z);


    // now we know what to send and what to recveive.
    // These guys are also ordered since we received 
    // the lists from all ranks just now.
    // next step is to divide the counts and offsets by 
    // the size in bytes of the struct and finally 
    // allocate the buffers.
    #define NBYTES_TO_NELEM(axis)\
    recv_size_##axis /= sizeof(fine_face_data_desc_t) ; \
    send_size_##axis /= sizeof(fine_face_data_desc_t) ; \
    for (int r = 0; r < nprocs; ++r){\
        recvcounts_##axis[r] /= sizeof(fine_face_data_desc_t) ; \
        rdispls_##axis[r] /= sizeof(fine_face_data_desc_t) ; \
        sendcounts_##axis[r] /= sizeof(fine_face_data_desc_t) ; \
        sdispls_##axis[r] /= sizeof(fine_face_data_desc_t) ; \
    }
    NBYTES_TO_NELEM(x);
    NBYTES_TO_NELEM(y);
    NBYTES_TO_NELEM(z);

    

    #define REALLOC_BUF(axis)\
    _recv_fbuf_##axis.realloc(nx*nx*nvars_fs*recv_size_##axis) ;\
    _send_fbuf_##axis.realloc(nx*nx*nvars_fs*send_size_##axis) ;\
    _recv_fbuf_##axis.set_strides({nx,nvars_fs});\
    _send_fbuf_##axis.set_strides({nx,nvars_fs});\
    _recv_fbuf_##axis.set_offsets(rdispls_##axis);\
    _send_fbuf_##axis.set_offsets(sdispls_##axis)

    REALLOC_BUF(x);
    REALLOC_BUF(y);
    REALLOC_BUF(z);
    // for send data: src is local dst is buffer (pack)
    // for reecv data: reverse (unpack)
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
        }\
        for( int isnd=0; isnd<sendcounts_##axis[r]; ++isnd) {\
            fine_interface_desc_t desc; \
            desc.qid_dst = isnd ; \
            desc.qid_src = sendbuf_##axis[isnd+sdispls_##axis[r]].qid_local ;\
            desc.fid_src = sendbuf_##axis[isnd+sdispls_##axis[r]].fid_local ;\
            desc.fid_dst = sendbuf_##axis[isnd+sdispls_##axis[r]].fid_remote ;\
            remote_fine_face_send_##axis[r].push_back(desc);\
        }\
    }

    FILL_BUF_DESC(x);
    FILL_BUF_DESC(y);
    FILL_BUF_DESC(z);
};

}}