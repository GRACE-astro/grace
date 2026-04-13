/**
 * @file refluxing.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-10-16
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

#include <grace/evolution/refluxing.hh>
#include <grace/amr/amr_ghosts.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/grace_system.hh>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cmath>

namespace grace {

parallel::grace_transfer_context_t reflux_fill_flux_buffers() 
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ; 
    int nvars_hrsc = variables::get_n_hrsc() ;
    //**************************************************************************************************/
    // and some more 
    auto& ghost_layer = grace::amr_ghosts::get();
    auto sbuf = ghost_layer.get_reflux_send_buffer() ; 
    auto rbuf = ghost_layer.get_reflux_recv_buffer() ; 
    auto info = ghost_layer.get_reflux_face_send_list() ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<4>> (
            {0
            ,0
            ,0
            ,0},
            {static_cast<long>(nx/2)
            ,static_cast<long>(nx/2)
            ,static_cast<long>(nvars_hrsc)
            ,static_cast<long>(info.qid.size())}
        ) ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_fill_flux_buffers")
            , policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& ivar, int const& iq) {

                auto const iface = info.elem_id(iq) ; 
                auto const rank = info.rank(iq)     ; 
                auto const idir = iface / 2    ; 
                auto const iside = iface%2     ; 
                auto const qid = info.qid(iq)      ; 

                size_t ijk_s[3]                        ; 
                ijk_s[idir] = iside ? nx + ngz : ngz   ; 
                ijk_s[other_dirs[idir][0]] = ngz + 2*i ; 
                ijk_s[other_dirs[idir][1]] = ngz + 2*j ; 
                
                double flux = 0 ; 
                for( int ii=0; ii<=(idir!=0); ++ii) {
                    for( int jj=0; jj<=(idir!=1); ++jj) {
                        for( int kk=0; kk<=(idir!=2); ++kk) {
                            flux += fluxes(ijk_s[0]+ii, ijk_s[1]+jj, ijk_s[2]+kk, ivar, idir, qid) ; 
                        }
                    }
                }
                
                auto bid = info.buf_id(iq) ; 
                sbuf(i,j,ivar,bid,rank) = 0.25*flux ; 
            }
        ) ; 
    Kokkos::fence() ; 
    /* now we send and receive */
    auto soffsets = ghost_layer.get_reflux_buffer_rank_send_offsets() ; 
    auto ssizes = ghost_layer.get_reflux_buffer_rank_send_sizes() ;
    
    auto roffsets = ghost_layer.get_reflux_buffer_rank_recv_offsets() ; 
    auto rsizes = ghost_layer.get_reflux_buffer_rank_recv_sizes() ;

    parallel::grace_transfer_context_t context ; 
    auto nprocs = parallel::mpi_comm_size() ;
    auto proc = parallel::mpi_comm_rank() ; 
    for( int iproc=0; iproc<nprocs; ++iproc) {
        if ( iproc == proc ) continue ; 
        // send 
        if ( ssizes[iproc] > 0 ) {
            context._send_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_isend(
                sbuf.data() + soffsets[iproc],
                ssizes[iproc],
                iproc,
                parallel::GRACE_REFLUX_TAG,
                MPI_COMM_WORLD,
                &context._send_requests.back()
            );  
        }
        if ( rsizes[iproc] > 0 ) {
            context._recv_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_irecv(
                rbuf.data() + roffsets[iproc],
                rsizes[iproc],
                iproc,
                parallel::GRACE_REFLUX_TAG,
                MPI_COMM_WORLD,
                &context._recv_requests.back()
            );  
        }
    }

    return context ;
}

parallel::grace_transfer_context_t reflux_fill_emf_buffers() 
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& emf  = grace::variable_list::get().getemfarray() ; 
    int nvars_hrsc = variables::get_n_hrsc() ;
    //**************************************************************************************************/
    // and some more 
    auto& ghost_layer = grace::amr_ghosts::get();
    auto sbuf = ghost_layer.get_reflux_emf_send_buffer() ; 
    auto rbuf = ghost_layer.get_reflux_emf_recv_buffer() ; 
    auto info = ghost_layer.get_reflux_face_send_list() ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<3>> (
            {0,0,0},
            {static_cast<long>(nx/2)
            ,static_cast<long>(nx/2)
            ,static_cast<long>(info.qid.extent(0))}
        ) ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_fill_emf_buffers")
            , policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                 
                auto const iface = info.elem_id(iq) ; 
                auto const rank = info.rank(iq) ; 

                auto const fdir = iface / 2;
                auto const idir = other_dirs[fdir][0]; 
                auto const jdir = other_dirs[fdir][1]; 
                auto const iside = iface % 2 ;

                auto const qid = info.qid(iq) ; 

                size_t ijk_s[3] ; 
                ijk_s[fdir] = iside ? nx + ngz : ngz ; 
                ijk_s[idir] = 2*i + ngz ; 
                ijk_s[jdir] = 2*j + ngz ; 
                // note that the range is n+1 in both dirs
                // for each, one iteration is out of bounds.
                // however the arrays have padding of ngz and the garbage
                // value will be unused
                double emf_i = 0.5 * (
                    emf(ijk_s[0],ijk_s[1],ijk_s[2],idir,qid) + 
                    emf(ijk_s[0] + (idir==0),ijk_s[1] + (idir==1),ijk_s[2] + (idir==2),idir,qid)
                );
                double emf_j = 0.5 * (
                    emf(ijk_s[0],ijk_s[1],ijk_s[2],jdir,qid) + 
                    emf(ijk_s[0] + (jdir==0),ijk_s[1] + (jdir==1),ijk_s[2] + (jdir==2),jdir,qid)
                );
                
                auto bid = info.buf_id(iq) ; 
                sbuf(i,j,0,bid,rank) = emf_i ; 
                sbuf(i,j,1,bid,rank) = emf_j ; 
            }
        ) ; 
    Kokkos::fence() ; 
    // send - receive face buffers
    auto soffsets = ghost_layer.get_reflux_buffer_rank_send_emf_offsets() ; 
    auto ssizes = ghost_layer.get_reflux_buffer_rank_send_emf_sizes() ;
    
    auto roffsets = ghost_layer.get_reflux_buffer_rank_recv_emf_offsets() ; 
    auto rsizes = ghost_layer.get_reflux_buffer_rank_recv_emf_sizes() ;

    parallel::grace_transfer_context_t context ; 
    auto nprocs = parallel::mpi_comm_size() ;
    auto proc = parallel::mpi_comm_rank() ; 
    for( int iproc=0; iproc<nprocs; ++iproc) {
        if ( ssizes[iproc] > 0 ) {
            context._send_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_isend(
                sbuf.data() + soffsets[iproc],
                ssizes[iproc],
                iproc,
                parallel::GRACE_REFLUX_EMF_FACE_TAG,
                MPI_COMM_WORLD,
                &context._send_requests.back()
            );  
        }
        if ( rsizes[iproc] > 0 ) {
            context._recv_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_irecv(
                rbuf.data() + roffsets[iproc],
                rsizes[iproc],
                iproc,
                parallel::GRACE_REFLUX_EMF_FACE_TAG,
                MPI_COMM_WORLD,
                &context._recv_requests.back()
            );  
        }
    }
    //**************************************************************************************************/
    auto coarse_sbuf = ghost_layer.get_reflux_emf_coarse_send_buffer() ; 
    auto coarse_rbuf = ghost_layer.get_reflux_emf_coarse_recv_buffer() ; 
    auto coarse_info = ghost_layer.get_reflux_coarse_face_send_list() ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto coarse_policy = 
        MDRangePolicy<Rank<3>> (
            {0,0,0},
            {static_cast<long>(nx)
            ,static_cast<long>(nx)
            ,static_cast<long>(coarse_info.qid.extent(0))}
        ) ; 
    //**************************************************************************************************/ 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_fill_emf_coarse_buffers")
            , coarse_policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {                 
                auto const iface = coarse_info.elem_id(iq) ; 
                auto const rank = coarse_info.rank(iq) ; 

                auto const fdir = iface / 2;
                auto const idir = other_dirs[fdir][0]; 
                auto const jdir = other_dirs[fdir][1]; 
                auto const iside = iface % 2 ;

                auto const qid = coarse_info.qid(iq) ; 

                size_t ijk_s[3] ; 
                ijk_s[fdir] = iside ? nx + ngz : ngz ; 
                ijk_s[idir] = i + ngz ; 
                ijk_s[jdir] = j + ngz ; 
                
                
                auto bid = coarse_info.buf_id(iq) ; 
                coarse_sbuf(i,j,0,bid,rank) = emf(ijk_s[0],ijk_s[1],ijk_s[2],idir,qid) ; 
                coarse_sbuf(i,j,1,bid,rank) = emf(ijk_s[0],ijk_s[1],ijk_s[2],jdir,qid) ; 
            }
        ) ; 
    Kokkos::fence() ; 
    //**************************************************************************************************/
    // send - receive face buffers
    auto coarse_soffsets = ghost_layer.get_reflux_buffer_rank_send_emf_coarse_offsets() ; 
    auto coarse_ssizes = ghost_layer.get_reflux_buffer_rank_send_emf_coarse_sizes() ;
    
    auto coarse_roffsets = ghost_layer.get_reflux_buffer_rank_recv_emf_coarse_offsets() ; 
    auto coarse_rsizes = ghost_layer.get_reflux_buffer_rank_recv_emf_coarse_sizes() ;
    //**************************************************************************************************/
    for( int iproc=0; iproc<nprocs; ++iproc) {
        if ( coarse_ssizes[iproc] > 0 ) {
            context._send_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_isend(
                coarse_sbuf.data() + coarse_soffsets[iproc],
                coarse_ssizes[iproc],
                iproc,
                parallel::GRACE_REFLUX_EMF_COARSE_FACE_TAG,
                MPI_COMM_WORLD,
                &context._send_requests.back()
            );  
        }
        if ( coarse_rsizes[iproc] > 0 ) {
            context._recv_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_irecv(
                coarse_rbuf.data() + coarse_roffsets[iproc],
                coarse_rsizes[iproc],
                iproc,
                parallel::GRACE_REFLUX_EMF_COARSE_FACE_TAG,
                MPI_COMM_WORLD,
                &context._recv_requests.back()
            );  
        }
    }

    //**************************************************************************************************/
    auto sbuf_edge = ghost_layer.get_reflux_emf_edge_send_buffer() ; 
    auto rbuf_edge = ghost_layer.get_reflux_emf_edge_recv_buffer() ; 
    auto info_edge = ghost_layer.get_reflux_edge_send_list() ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto edge_policy = 
        MDRangePolicy<Rank<2>> (
            {0,0},
            {static_cast<long>(nx),static_cast<long>(info_edge.qid.extent(0))}
        ) ; 
    // fill edge buffers 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_fill_emf_edge_buffers")
            , edge_policy 
            , KOKKOS_LAMBDA (int const& i, int const& iq) {                
                auto const iedge = info_edge.elem_id(iq); 
                // edge direction 
                int idir  = (iedge/4)          ; 
                // upper or lower gz?
                int jside = (iedge>>0)&1       ; 
                int kside = (iedge>>1)&1       ;
                // orthogonal directions (z-order)
                int jdir = other_dirs[idir][0] ; 
                int kdir = other_dirs[idir][1] ; 
                // quad-id (fine)
                auto const qid = info_edge.qid(iq) ; 
                // indices of edge 
                size_t ijk_s[3] ;
                ijk_s[idir] = ngz + i ; 
                ijk_s[jdir] = jside ? nx + ngz : ngz ; 
                ijk_s[kdir] = kside ? nx + ngz : ngz ; 

                auto const rank = info_edge.rank(iq) ;
                auto bid = info_edge.buf_id(iq);
                // write to buffer
                sbuf_edge(i, bid, rank) = emf(ijk_s[0],ijk_s[1],ijk_s[2],idir,qid) ;
            }
        ) ;
    Kokkos::fence() ;
    // ---- DEBUG: dump edge send buffer at i=0 ----
    {
        auto n_send = info_edge.qid.extent(0) ;
        for(size_t iq=0; iq<n_send; ++iq) {
            auto iedge = info_edge.elem_id(iq) ;
            int idir = iedge / 4 ;
            int jside = (iedge>>0)&1 ;
            int kside = (iedge>>1)&1 ;
            auto qid = info_edge.qid(iq) ;
            auto rank = info_edge.rank(iq) ;
            auto bid = info_edge.buf_id(iq) ;
            double val = sbuf_edge(0, bid, rank) ;
            GRACE_TRACE("[EDGE_SEND] iq={} iedge={} dir={} jside={} kside={} qid={} bid={} dest_rank={} val={:.15e}",
                iq, iedge, idir, jside, kside, (int)qid, (int)bid, (int)rank, val) ;
        }
    }
    // ---- END DEBUG ----
        // todo maybe edge bufs can be separate, this seems wasteful
    // send - receive edge buffers
    auto soffsets_edge = ghost_layer.get_reflux_buffer_rank_send_emf_edge_offsets() ; 
    auto ssizes_edge   = ghost_layer.get_reflux_buffer_rank_send_emf_edge_sizes()   ;
    
    auto roffsets_edge = ghost_layer.get_reflux_buffer_rank_recv_emf_edge_offsets() ; 
    auto rsizes_edge   = ghost_layer.get_reflux_buffer_rank_recv_emf_edge_sizes()   ;
    for( int iproc=0; iproc<nprocs; ++iproc) {
        if ( ssizes_edge[iproc] > 0 ) {
            GRACE_TRACE("Proc {} send {} offset {}",iproc, ssizes_edge[iproc], soffsets_edge[iproc]);
            context._send_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_isend(
                sbuf_edge.data() + soffsets_edge[iproc],
                ssizes_edge[iproc],
                iproc,
                parallel::GRACE_REFLUX_EMF_EDGE_TAG,
                MPI_COMM_WORLD,
                &context._send_requests.back()
            );  
        }
        if ( rsizes_edge[iproc] > 0 ) {
            GRACE_TRACE("Proc {} receive {} offset {}",iproc, rsizes_edge[iproc], roffsets_edge[iproc]);
            context._recv_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_irecv(
                rbuf_edge.data() + roffsets_edge[iproc],
                rsizes_edge[iproc],
                iproc,
                parallel::GRACE_REFLUX_EMF_EDGE_TAG,
                MPI_COMM_WORLD,
                &context._recv_requests.back()
            );  
        }
    }

    // coarse edges 
    auto sbuf_cedge = ghost_layer.get_reflux_emf_coarse_edge_send_buffer() ; 
    auto rbuf_cedge = ghost_layer.get_reflux_emf_coarse_edge_recv_buffer() ; 
    auto info_cedge = ghost_layer.get_reflux_coarse_edge_send_list() ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto cedge_policy = 
        MDRangePolicy<Rank<2>> (
            {0,0},
            {static_cast<long>(nx),static_cast<long>(info_cedge.qid.extent(0))}
        ) ; 
    // fill edge buffers 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_fill_emf_coarse_edge_buffers")
            , cedge_policy 
            , KOKKOS_LAMBDA (int const& i, int const& iq) {                
                auto const iedge = info_cedge.elem_id(iq); 
                // edge direction 
                int idir  = (iedge/4)          ; 
                // upper or lower gz?
                int jside = (iedge>>0)&1       ; 
                int kside = (iedge>>1)&1       ;
                // orthogonal directions (z-order)
                int jdir = other_dirs[idir][0] ; 
                int kdir = other_dirs[idir][1] ; 
                // quad-id (fine)
                auto const qid = info_cedge.qid(iq) ; 
                // indices of edge 
                size_t ijk_s[3] ;
                ijk_s[idir] = ngz + i ; 
                ijk_s[jdir] = jside ? nx + ngz : ngz ; 
                ijk_s[kdir] = kside ? nx + ngz : ngz ; 

                auto const rank = info_cedge.rank(iq) ; 
                auto bid = info_cedge.buf_id(iq);
                // write to buffer
                sbuf_cedge(i, bid, rank) = emf(ijk_s[0],ijk_s[1],ijk_s[2],idir,qid) ; 
            }
        ) ;
    Kokkos::fence() ; 
    // todo maybe edge bufs can be separate, this seems wasteful 
    // send - receive edge buffers 
    auto soffsets_cedge = ghost_layer.get_reflux_buffer_rank_send_emf_coarse_edge_offsets() ; 
    auto ssizes_cedge   = ghost_layer.get_reflux_buffer_rank_send_emf_coarse_edge_sizes()   ;
    
    auto roffsets_cedge = ghost_layer.get_reflux_buffer_rank_recv_emf_coarse_edge_offsets() ; 
    auto rsizes_cedge   = ghost_layer.get_reflux_buffer_rank_recv_emf_coarse_edge_sizes()   ;
    for( int iproc=0; iproc<nprocs; ++iproc) {
        if ( ssizes_cedge[iproc] > 0 ) {
            GRACE_TRACE("Proc {} send {} offset {}",iproc, ssizes_cedge[iproc], soffsets_cedge[iproc]);
            context._send_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_isend(
                sbuf_cedge.data() + soffsets_cedge[iproc],
                ssizes_cedge[iproc],
                iproc,
                parallel::GRACE_REFLUX_EMF_COARSE_EDGE_TAG,
                MPI_COMM_WORLD,
                &context._send_requests.back()
            );  
        }
        if ( rsizes_cedge[iproc] > 0 ) {
            GRACE_TRACE("Proc {} receive {} offset {}",iproc, rsizes_cedge[iproc], roffsets_cedge[iproc]);
            context._recv_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_irecv(
                rbuf_cedge.data() + roffsets_cedge[iproc],
                rsizes_cedge[iproc],
                iproc,
                parallel::GRACE_REFLUX_EMF_COARSE_EDGE_TAG,
                MPI_COMM_WORLD,
                &context._recv_requests.back()
            );  
        }
    }

    return context ; 
}

void reflux_correct_fluxes(
    parallel::grace_transfer_context_t& context,
    double t, double dt, double dtfact,
    var_array_t & new_state 
)
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ; 
    int nvars_hrsc = variables::get_n_hrsc() ;
    //**************************************************************************************************/
    auto& ghost_layer = grace::amr_ghosts::get();
    auto rbuf = ghost_layer.get_reflux_recv_buffer() ; 
    auto desc = ghost_layer.get_reflux_face_descriptors() ; 
    //**************************************************************************************************/    
    //**************************************************************************************************/
    parallel::mpi_waitall(context) ; 
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<4>> (
            {0,0,0,0},
            {static_cast<long>(nx)
            ,static_cast<long>(nx)
            ,static_cast<long>(nvars_hrsc)
            ,static_cast<long>(desc.coarse_qid.extent(0))}
        ) ;
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_apply")
            , policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& ivar, int const& iq) {
                auto const qid_c  = desc.coarse_qid(iq)     ; 
                auto const iface_c = desc.coarse_face_id(iq) ; 

                auto const idir = iface_c / 2; 
                auto const side = iface_c % 2;

                size_t ijk_fs[3], ijk_cc[3] ; 
                // face-staggered index
                ijk_fs[idir] = (iface_c % 2)    
                            ? ngz + nx 
                            : ngz ;
                // cell centered index 
                ijk_cc[idir] = (iface_c % 2)    
                            ? ngz + nx - 1 
                            : ngz ;
                ijk_fs[other_dirs[idir][0]] = ijk_cc[other_dirs[idir][0]] = ngz + i ; 
                ijk_fs[other_dirs[idir][1]] = ijk_cc[other_dirs[idir][1]] = ngz + j ; 

                // compute child id 
                int8_t ichild = (2*i>=nx) + 2 * (2*j>=nx) ; 

                double flux_correction = 0 ; 
                if ( desc.fine_is_remote(iq,ichild) ) {
                    flux_correction = rbuf(i%(nx/2),j%(nx/2),ivar,desc.fine_bid(iq,ichild),desc.fine_owner_rank(iq,ichild)) ; 
                } else {
                    // compute flux correction 
                    size_t qid_f = desc.fine_qid(iq,ichild);
                    size_t ijk_f[3] ; 
                    // on fine side the side is opposite 
                    ijk_f[idir] = (iface_c % 2)    
                                ? ngz  
                                : ngz + nx ;
                    ijk_f[other_dirs[idir][0]] = ngz + (2*i%nx) ; 
                    ijk_f[other_dirs[idir][1]] = ngz + (2*j%nx) ; 

                    for( int ii=0; ii<=(idir!=0); ++ii) {
                        for( int jj=0; jj<=(idir!=1); ++jj) {
                            for( int kk=0; kk<=(idir!=2); ++kk) {
                                flux_correction += fluxes(ijk_f[0]+ii, ijk_f[1]+jj, ijk_f[2]+kk, ivar, idir, qid_f) ; 
                            }
                        }
                    }
                    flux_correction *= 0.25 ; 
                }
                int sign = side ? -1 : +1 ; 
                new_state(ijk_cc[0],ijk_cc[1],ijk_cc[2],ivar,qid_c) += sign * dt * dtfact * idx(idir,qid_c) * (
                    flux_correction - fluxes(ijk_fs[0],ijk_fs[1],ijk_fs[2],ivar,idir,qid_c)
                ) ; 
            }
        ) ;
    //**************************************************************************************************/
    //**************************************************************************************************/
    //**************************************************************************************************/ 

}

void reflux_correct_emfs(parallel::grace_transfer_context_t& context)
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    auto myproc = parallel::mpi_comm_rank() ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& emf  = grace::variable_list::get().getemfarray() ;  
    //**************************************************************************************************/
    auto& ghost_layer = grace::amr_ghosts::get();
    auto rbuf = ghost_layer.get_reflux_emf_recv_buffer() ; 
    auto desc = ghost_layer.get_reflux_face_descriptors() ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    parallel::mpi_waitall(context) ;
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<3>> (
            {0,0,0},
            {static_cast<long>(nx/2)
            ,static_cast<long>(nx/2)
            ,static_cast<long>(desc.coarse_qid.extent(0))}
        ) ;
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 
    //**************************************************************************************************/
    #if 1
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_emf_apply_face")
            , policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                // coarse face 
                auto const iface_c = desc.coarse_face_id(iq) ; 
                // coarse face direction 
                auto const fdir = iface_c / 2 ; 
                // other directions (z-order)
                auto const idir = other_dirs[fdir][0]; 
                auto const jdir = other_dirs[fdir][1]; 
                // side of the face 
                auto const iside = iface_c % 2 ;
                // qid of coarse side 
                auto const qid_c = desc.coarse_qid(iq) ; 

                // indices of face center (coarse)
                // edge center (coarse)
                // edge center (fine)
                size_t ijk_c[3], ijk_f[3] ; 
                for( int ichild=0; ichild<P4EST_CHILDREN/2; ++ichild) {
                     
                    // emf correction 
                    double emf_corr_i{0}, emf_corr_j{0} ; 
                    if ( desc.fine_is_remote(iq,ichild) ) {
                        // fine quadid 
                        auto const qid_f = desc.fine_bid(iq,ichild) ;
                        auto rank = desc.fine_owner_rank(iq,ichild) ; 
                        emf_corr_i = rbuf(i,j,0,qid_f,rank) ; 
                        emf_corr_j = rbuf(i,j,1,qid_f,rank) ; 
                    } else {
                        // fine quadid 
                        auto const qid_f = desc.fine_qid(iq,ichild) ;
                        // fine side so iside is opposite
                        ijk_f[fdir] = iside ? ngz : nx + ngz ; 
                        ijk_f[idir] = 2*i + ngz ; 
                        ijk_f[jdir] = 2*j + ngz ; 
                        emf_corr_i = 0.5 * (
                            emf(ijk_f[0],ijk_f[1],ijk_f[2],idir,qid_f) + 
                            emf(ijk_f[0] + (idir==0),ijk_f[1] + (idir==1),ijk_f[2] + (idir==2),idir,qid_f)
                        );
                        emf_corr_j = 0.5 * (
                            emf(ijk_f[0],ijk_f[1],ijk_f[2],jdir,qid_f) + 
                            emf(ijk_f[0] + (jdir==0),ijk_f[1] + (jdir==1),ijk_f[2] + (jdir==2),jdir,qid_f)
                        ) ; 
                    }
                    // child based offset into coarse view 
                    int ichild_i = (ichild>>0)&1 ; 
                    int ichild_j = (ichild>>1)&1 ; 
                    int off_i = ichild_i ? nx/2 : 0 ; 
                    int off_j = ichild_j ? nx/2 : 0 ; 
                    // edge indices --> emfs to be corrected 
                    // coarse side so iside is correct
                    ijk_c[fdir] = iside ? nx + ngz : ngz ; 
                    ijk_c[idir] = i + off_i + ngz ; 
                    ijk_c[jdir] = j + off_j + ngz ; 
                    // a few things to check: 
                    // 1) we don't want to write on the edges of 
                    //    the quadrant nor in the middle since this
                    //    is taken care of by the edge corrector
                    // 2) Eˆd is **not** staggered in d-direction, 
                    //    so we avoid the very last iteration in 
                    //    the d index.
                    if ( ijk_c[jdir] != nx/2+ngz and 
                         ijk_c[jdir] != ngz     ) 
                    { 
                        emf(ijk_c[0], ijk_c[1], ijk_c[2], idir, qid_c) = emf_corr_i ;
                    } 
                    if ( ijk_c[idir] != nx/2+ngz and 
                         ijk_c[idir] != ngz      ) 
                    { 
                        emf(ijk_c[0], ijk_c[1], ijk_c[2], jdir, qid_c) = emf_corr_j ; 
                    }
                }    
            }
                
        ) ;  
    #endif 
    #if 1
    //**************************************************************************************************/
    auto coarse_rbuf = ghost_layer.get_reflux_emf_coarse_recv_buffer() ; 
    auto coarse_desc = ghost_layer.get_reflux_coarse_face_descriptors() ; 
    //**************************************************************************************************/ 
    //**************************************************************************************************/
    auto coarse_policy = 
        MDRangePolicy<Rank<3>> (
            {0,0,0},
            {static_cast<long>(nx)
            ,static_cast<long>(nx)
            ,static_cast<long>(coarse_desc.qid.extent(0))}
        ) ;
    GRACE_TRACE("About to run reflux {}", coarse_desc.qid.extent(0) ) ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_emf_apply_coarse_face")
            , coarse_policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                // step 1 compute 
                size_t ijk[3] ; 
                double emf_corr[2] = {0,0}; 
                for( int is=0; is<2; ++is) {
                    auto const fid = coarse_desc.face_id(iq,is) ;
                    auto const fdir = fid / 2 ; 
                    // other directions (z-order)
                    auto const idir = other_dirs[fdir][0]; 
                    auto const jdir = other_dirs[fdir][1]; 
                    // iside 
                    auto const iside = fid % 2 ;
                     
                    if ( coarse_desc.is_remote(iq,is) ) {
                        // id 
                        auto const bid = coarse_desc.bid(iq,is);
                        int const r = coarse_desc.owner_rank(iq,is) ; 
                        emf_corr[0] += coarse_rbuf(i,j,0,bid,r) ; 
                        emf_corr[1] += coarse_rbuf(i,j,1,bid,r) ; 
                    } else {
                        // qid 
                        auto const qid = coarse_desc.qid(iq,is);
                        ijk[fdir] = iside ? nx + ngz : ngz ; 
                        ijk[idir] = ngz + i ; 
                        ijk[jdir] = ngz + j ; 
                        emf_corr[0] += emf(ijk[0],ijk[1],ijk[2],idir,qid) ; 
                        emf_corr[1] += emf(ijk[0],ijk[1],ijk[2],jdir,qid) ; 
                    }
                } // iside 
                emf_corr[0] *= 0.5 ; 
                emf_corr[1] *= 0.5 ; 
                // step two correct 
                for( int is=0; is<2; ++is) {
                    auto const fid = coarse_desc.face_id(iq,is) ;
                    auto const fdir = fid / 2 ; 
                    // other directions (z-order)
                    auto const idir = other_dirs[fdir][0]; 
                    auto const jdir = other_dirs[fdir][1]; 
                    // iside 
                    auto const iside = fid % 2 ;
                    if ( !coarse_desc.is_remote(iq,is) ) {
                        // qid 
                        auto const qid = coarse_desc.qid(iq,is); 
                        ijk[fdir] = iside ? nx + ngz : ngz ; 
                        ijk[idir] = ngz + i ; 
                        ijk[jdir] = ngz + j ; 
                        if ( ijk[jdir] > ngz ) emf(ijk[0],ijk[1],ijk[2],idir,qid) = emf_corr[0]; 
                        if ( ijk[idir] > ngz ) emf(ijk[0],ijk[1],ijk[2],jdir,qid) = emf_corr[1];  
                    } 
                } // iside 
            }
                
        ) ; 
    #endif 
    //**************************************************************************************************/
    auto edge_rbuf = ghost_layer.get_reflux_emf_edge_recv_buffer() ; 
    auto edge_desc = ghost_layer.get_reflux_edge_descriptors() ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto edge_policy = 
        MDRangePolicy<Rank<2>> (
            {0,0},
            {static_cast<long>(nx),static_cast<long>(edge_desc.coarse_qid.extent(0))}
        ) ;
    //**************************************************************************************************/
    #if 1
    // two phases, first we need to compute the correction, then we apply
    auto emf_edge_correction = ghost_layer.get_reflux_edge_emf_accumulation_buffer() ;  
    parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "reflux_emf_compute_edge"),
        edge_policy,
        KOKKOS_LAMBDA (int const& i, int const& iq) {
            auto n_sides = edge_desc.n_sides(iq);

            size_t ijk[3] ;
            // Collect fine contributions with their edge_id for canonical ordering.
            // Accumulating in edge_id order guarantees FP-identical results on all
            // ranks, regardless of local/remote descriptor layout.
            double v0[4], v1[4] ;
            int keys[4] ;
            int cnt = 0 ;
            for( int iside=0; iside<n_sides; ++iside) {
                if ( ! edge_desc.is_fine(iq,iside) ) continue ;
                // edge index
                auto edge_id = edge_desc.edge_id(iq,iside) ;
                keys[cnt] = edge_id ;
                // direction and side
                int edge_dir = edge_id / 4 ;
                int side_i = (edge_id>>0)&1;
                int side_j = (edge_id>>1)&1;
                // child id loop
                for( int ichild=0; ichild<2; ++ichild ) {
                    double val = 0.0;
                    if ( edge_desc.fine_is_remote(iq,iside,ichild) ) {
                        auto bid = edge_desc.fine_bid(iq,iside,ichild);
                        auto rank = edge_desc.fine_owner_rank(iq,iside,ichild) ;
                        val = edge_rbuf(i,bid,rank) ;
                    } else {
                        auto qid = edge_desc.fine_qid(iq,iside,ichild);
                        ijk[edge_dir] = ngz + i ;
                        ijk[other_dirs[edge_dir][0]] = side_i ? nx + ngz : ngz ;
                        ijk[other_dirs[edge_dir][1]] = side_j ? nx + ngz : ngz ;
                        val = emf(ijk[0],ijk[1],ijk[2],edge_dir,qid);
                    }
                    if (ichild == 0) v0[cnt] = val ;
                    else             v1[cnt] = val ;
                }
                cnt++ ;
            }
            // insertion sort by edge_id (≤4 elements)
            for( int a=1; a<cnt; ++a ) {
                int k = keys[a] ; double t0 = v0[a], t1 = v1[a] ;
                int b = a - 1 ;
                while( b >= 0 && keys[b] > k ) {
                    keys[b+1] = keys[b] ; v0[b+1] = v0[b] ; v1[b+1] = v1[b] ;
                    b-- ;
                }
                keys[b+1] = k ; v0[b+1] = t0 ; v1[b+1] = t1 ;
            }
            // Use first sorted value (smallest edge_id) instead of averaging.
            // All fine EMFs at a shared edge are the same physical field, so
            // any single value is as good as the mean.  Unlike an average,
            // this is deterministic regardless of how many fine sides each
            // rank's descriptor happens to list.
            emf_edge_correction(i,0,iq) = cnt ? v0[0] : 0.0 ;
            emf_edge_correction(i,1,iq) = cnt ? v1[0] : 0.0 ; 
        }
    );
    //**************************************************************************************************/
    Kokkos::fence() ;
    // ---- DEBUG: dump Phase 3 edge correction inputs at i=0 ----
    {
        auto n_edges = edge_desc.n_sides.extent(0) ;
        for(size_t iq=0; iq<n_edges; ++iq) {
            int ns = edge_desc.n_sides(iq) ;
            for(int iside=0; iside<ns; ++iside) {
                if ( !edge_desc.is_fine(iq,iside) ) continue ;
                int eid = edge_desc.edge_id(iq,iside) ;
                int edir = eid / 4 ;
                int si = (eid>>0)&1, sj = (eid>>1)&1 ;
                for(int ichild=0; ichild<2; ++ichild) {
                    double val = 0.0 ;
                    bool remote = edge_desc.fine_is_remote(iq,iside,ichild) ;
                    if (remote) {
                        auto bid = edge_desc.fine_bid(iq,iside,ichild) ;
                        auto rank = edge_desc.fine_owner_rank(iq,iside,ichild) ;
                        val = edge_rbuf(0, bid, rank) ;
                    } else {
                        auto qid = edge_desc.fine_qid(iq,iside,ichild) ;
                        size_t ijk[3] ;
                        ijk[edir] = ngz ;
                        ijk[other_dirs[edir][0]] = si ? nx + ngz : ngz ;
                        ijk[other_dirs[edir][1]] = sj ? nx + ngz : ngz ;
                        val = emf(ijk[0],ijk[1],ijk[2],edir,qid) ;
                    }
                    if (remote) {
                        auto bid = edge_desc.fine_bid(iq,iside,ichild) ;
                        auto rank = edge_desc.fine_owner_rank(iq,iside,ichild) ;
                        GRACE_TRACE("[EDGE_DBG] iq={} iside={} eid={} dir={} si={} sj={} ichild={} REMOTE from_rank={} bid={} val={:.15e}",
                            iq, iside, eid, edir, si, sj, ichild, rank, bid, val) ;
                    } else {
                        auto qid = edge_desc.fine_qid(iq,iside,ichild) ;
                        GRACE_TRACE("[EDGE_DBG] iq={} iside={} eid={} dir={} si={} sj={} ichild={} LOCAL qid={} val={:.15e}",
                            iq, iside, eid, edir, si, sj, ichild, qid, val) ;
                    }
                }
            }
            // also dump the computed correction
            GRACE_TRACE("[EDGE_DBG] iq={} n_sides={} correction child0={:.15e} child1={:.15e}",
                iq, ns, emf_edge_correction(0,0,iq), emf_edge_correction(0,1,iq)) ;
        }
    }
    // ---- END DEBUG ----
    // apply
    parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "reflux_emf_apply_edge"),
        edge_policy,
        KOKKOS_LAMBDA (int const& i, int const& iq) {
            // information about the edge we are correcting 
            auto const n_sides = edge_desc.n_sides(iq); 
            // pre-allocate indices 
            size_t ijk[3] ; 
            // loop over 4 sides of the edge
            for( int iside=0; iside<n_sides; ++iside) {
                // edge index 
                auto edge_id = edge_desc.edge_id(iq,iside) ;  
                // direction along and orthogonal to edge (z-order)
                int edge_dir = edge_id / 4 ; 
                int side_i = (edge_id>>0)&1;
                int side_j = (edge_id>>1)&1;

                // if coarse we need to correct with - emf + 1/n_fine 1/2 sum( fine emfs )
                if ( ! edge_desc.is_fine(iq,iside) ) {
                    // Remote: nothing to do 
                    if ( edge_desc.coarse_is_remote(iq,iside) ) continue ;
                    // quad-id 
                    auto qid = edge_desc.coarse_qid(iq,iside) ; 
                    // we need to figure out if it's the upper or lower
                    // child we are reading from 
                    // indices of edge 
                    // TODO offsets need to be figured out.
                    // When we register i and j are wrt the face dir
                    // here they are wrt the edge dir which lies inside 
                    // the face. So they are not consistent.. Essentially 
                    // here we need to just take the side for the direction
                    // orthogonal to the coarse face and offset the other if 
                    // the child_id is 0...
                    int ichild = (2*i)>=nx ; 

                    ijk[edge_dir] = ngz + i ; 
                    ijk[other_dirs[edge_dir][0]] = edge_desc.off_i(iq,iside) ? nx/2 + ngz : ( side_i ? nx + ngz : ngz ) ;  
                    ijk[other_dirs[edge_dir][1]] = edge_desc.off_j(iq,iside) ? nx/2 + ngz : ( side_j ? nx + ngz : ngz ) ;
                    
                    emf(ijk[0],ijk[1],ijk[2],edge_dir,qid) = 
                        +0.5*(emf_edge_correction((2*i)%nx,ichild,iq) + emf_edge_correction((2*i)%nx+1,ichild,iq));
                } else {
                    for( int ichild=0; ichild<2; ++ichild) {
                        // Remote: nothing to do
                        if ( edge_desc.fine_is_remote(iq,iside,ichild) ) continue ;
                        // quad-id
                        auto qid = edge_desc.fine_qid(iq,iside,ichild) ;
                        // indices of emf to be corrected
                        ijk[edge_dir] = ngz + i ;  
                        ijk[other_dirs[edge_dir][0]] = side_i ? nx + ngz : ngz ;  
                        ijk[other_dirs[edge_dir][1]] = side_j ? nx + ngz : ngz ;
                        emf(ijk[0],ijk[1],ijk[2],edge_dir,qid) = emf_edge_correction(i,ichild,iq);
                    }
                } // if fine 
            }
        }
       
    ) ; 
    #endif 
    auto coarse_edge_rbuf = ghost_layer.get_reflux_emf_coarse_edge_recv_buffer() ; 
    auto coarse_edge_desc = ghost_layer.get_reflux_coarse_edge_descriptors() ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto emf_coarse_edge_correction = ghost_layer.get_reflux_coarse_edge_emf_accumulation_buffer() ; 
    //**************************************************************************************************/
     auto coarse_edge_policy = 
        MDRangePolicy<Rank<2>> (
            {0,0},
            {static_cast<long>(nx),static_cast<long>(coarse_edge_desc.n_sides.extent(0))}
        ) ;
    //**************************************************************************************************/
    parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "reflux_coarse_emf_compute_coarse_edge"),
        coarse_edge_policy,
        KOKKOS_LAMBDA (int const& i, int const& iq) {
            auto n_sides = coarse_edge_desc.n_sides(iq);
            size_t ijk[3] ;
            // Collect all coarse contributions with their edge_id for canonical ordering.
            double vals[4] ;
            int keys[4] ;
            for( int iside=0; iside<n_sides; ++iside) {
                // edge index
                auto edge_id = coarse_edge_desc.edge_id(iq,iside) ;
                keys[iside] = edge_id ;
                // direction and side
                int edge_dir = edge_id / 4 ;
                int side_i = (edge_id>>0)&1;
                int side_j = (edge_id>>1)&1;
                // coarse quadid
                if ( coarse_edge_desc.coarse_is_remote(iq,iside) ) {
                    auto bid = coarse_edge_desc.coarse_bid(iq,iside);
                    auto rank = coarse_edge_desc.coarse_owner_rank(iq,iside) ;
                    vals[iside] = coarse_edge_rbuf(i,bid,rank) ;
                } else {
                    auto qid = coarse_edge_desc.coarse_qid(iq,iside);
                    ijk[edge_dir] = ngz + i ;
                    ijk[other_dirs[edge_dir][0]] = side_i ? nx + ngz : ngz ;
                    ijk[other_dirs[edge_dir][1]] = side_j ? nx + ngz : ngz ;
                    vals[iside] = emf(ijk[0],ijk[1],ijk[2],edge_dir,qid);
                }
            }
            // insertion sort by edge_id (≤4 elements)
            for( int a=1; a<n_sides; ++a ) {
                int k = keys[a] ; double tv = vals[a] ;
                int b = a - 1 ;
                while( b >= 0 && keys[b] > k ) {
                    keys[b+1] = keys[b] ; vals[b+1] = vals[b] ;
                    b-- ;
                }
                keys[b+1] = k ; vals[b+1] = tv ;
            }
            // Use first sorted value (smallest edge_id) — deterministic
            // regardless of how many coarse sides each rank sees.
            emf_coarse_edge_correction(i,iq) = n_sides ? vals[0] : 0.0 ; 
        }
    );
    //**************************************************************************************************/
    parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "reflux_emf_apply_coarse_edge"),
        coarse_edge_policy,
        KOKKOS_LAMBDA (int const& i, int const& iq) {
            auto n_sides = coarse_edge_desc.n_sides(iq); 
            // pre-allocate indices 
            size_t ijk[3] ; 
            // loop over 4 sides of the edge
            for( int iside=0; iside<n_sides; ++iside) {
                // edge index 
                auto edge_id = coarse_edge_desc.edge_id(iq,iside) ; 
                // direction along and orthogonal to edge (z-order)
                int edge_dir = edge_id / 4 ; 
                int side_i = (edge_id>>0)&1;
                int side_j = (edge_id>>1)&1;

                // coarse remote nothing to do 
                if ( coarse_edge_desc.coarse_is_remote(iq,iside) ) continue ;
                // quad-id 
                auto qid = coarse_edge_desc.coarse_qid(iq,iside);
                // we need to figure out if it's the upper or lower
                // child we are reading from 
                // indices of edge 
                ijk[edge_dir] = ngz + i ; 
                ijk[other_dirs[edge_dir][0]] = side_i ? nx + ngz : ngz ;  
                ijk[other_dirs[edge_dir][1]] = side_j ? nx + ngz : ngz ;
                // for coarse-only we store it here 
                emf(ijk[0],ijk[1],ijk[2],edge_dir,qid) = emf_coarse_edge_correction(i,iq) ; 
            }
        }
    ) ; 
}



//**************************************************************************************************/
//**************************************************************************************************/
// DIAGNOSTIC: check face-staggered B conservation at all interfaces
// Reports per-(level, direction) max error and integrated flux imbalance
// for both hanging (coarse-fine) and same-level faces.
//**************************************************************************************************/

namespace {

constexpr int DIAG_MAX_LEVELS = 16 ;
constexpr int DIAG_NDIRS      = 3 ;

// CAS-based atomic max for doubles on device
KOKKOS_INLINE_FUNCTION
void atomic_max_d(double* ptr, double val) {
    double old = *ptr ;
    while (val > old) {
        double prev = Kokkos::atomic_compare_exchange(ptr, old, val) ;
        if (prev == old) return ;
        old = prev ;
    }
}

// Derive relative refinement level from inverse spacing
KOKKOS_INLINE_FUNCTION
int level_from_idx(double idx_q, double idx_min) {
    double ratio = idx_q / idx_min ;
    // ratio is an exact power of 2 in oct-tree AMR
    int lev = 0 ;
    while (ratio > 1.5) { ratio *= 0.5 ; lev++ ; }
    return (lev < DIAG_MAX_LEVELS) ? lev : DIAG_MAX_LEVELS - 1 ;
}

// File output helpers (rank 0 only)
void init_Bflux_dat(std::filesystem::path const& fpath, std::string const& header) {
    if (!std::filesystem::exists(fpath)) {
        std::ofstream f(fpath.string()) ;
        f << header << '\n' ;
    }
}

} // anonymous namespace

void diagnose_face_B_conservation()
{
    using namespace grace ;
    using namespace Kokkos ;
    DECLARE_GRID_EXTENTS ;
    //**************************************************************************************************/
    auto& stag = grace::variable_list::get().getstaggeredstate() ;
    auto Bx = stag.face_staggered_fields_x ;
    auto By = stag.face_staggered_fields_y ;
    auto Bz = stag.face_staggered_fields_z ;
    //**************************************************************************************************/
    auto& ghost_layer = grace::amr_ghosts::get();
    auto nprocs = parallel::mpi_comm_size() ;
    auto proc   = parallel::mpi_comm_rank() ;
    //**************************************************************************************************/
    auto idx = grace::variable_list::get().getinvspacings() ;
    auto dx  = grace::variable_list::get().getspacings() ;
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ;
    //**************************************************************************************************/
    // Helper: derive 1-var offsets/sizes from existing 2-var EMF offsets/sizes
    auto halve_offsets = [](std::vector<size_t> const& v) {
        std::vector<size_t> out(v.size()) ;
        for(size_t i=0; i<v.size(); ++i) out[i] = v[i] / 2 ;
        return out ;
    } ;
    //**************************************************************************************************/
    // Find coarsest inverse spacing to derive relative levels
    //**************************************************************************************************/
    double idx_min_local = std::numeric_limits<double>::max() ;
    Kokkos::parallel_reduce("diag_B_idx_min",
        RangePolicy<>(0, static_cast<long>(nq)),
        KOKKOS_LAMBDA(int q, double& lmin) {
            double v = idx(0, q) ;
            if (v < lmin) lmin = v ;
        },
        Kokkos::Min<double>(idx_min_local)
    ) ;
    Kokkos::fence() ;
    double idx_min = 0.0 ;
    parallel::mpi_allreduce(&idx_min_local, &idx_min, 1, sc_MPI_MIN) ;
    //**************************************************************************************************/
    // Per-(level, dir) accumulation views on device (LayoutRight for consistent flat indexing)
    // Split into local-only (_loc) and MPI-involving (_mpi) contributions
    //**************************************************************************************************/
    using diag_view_t = Kokkos::View<double**, Kokkos::LayoutRight> ;
    diag_view_t hang_max_loc_d("diag_hang_max_loc", DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    diag_view_t hang_sum_loc_d("diag_hang_sum_loc", DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    diag_view_t hang_max_mpi_d("diag_hang_max_mpi", DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    diag_view_t hang_sum_mpi_d("diag_hang_sum_mpi", DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    diag_view_t same_max_loc_d("diag_same_max_loc", DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    diag_view_t same_sum_loc_d("diag_same_sum_loc", DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    diag_view_t same_max_mpi_d("diag_same_max_mpi", DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    diag_view_t same_sum_mpi_d("diag_same_sum_mpi", DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    // Edge vs interior split for hanging faces:
    // "edge" = B-face cell whose EMF stencil touches a Phase-1-skipped edge
    //          (block boundary at 0/nx, or quadrant boundary at nx/2)
    // "interior" = everything else (all 4 surrounding EMFs corrected by Phase 1)
    diag_view_t hang_max_loc_int_d ("diag_hang_max_loc_int",  DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    diag_view_t hang_max_mpi_int_d ("diag_hang_max_mpi_int",  DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    diag_view_t hang_max_loc_edge_d("diag_hang_max_loc_edge", DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    diag_view_t hang_max_mpi_edge_d("diag_hang_max_mpi_edge", DIAG_MAX_LEVELS, DIAG_NDIRS) ;
    Kokkos::deep_copy(hang_max_loc_d, 0.0) ;
    Kokkos::deep_copy(hang_sum_loc_d, 0.0) ;
    Kokkos::deep_copy(hang_max_mpi_d, 0.0) ;
    Kokkos::deep_copy(hang_sum_mpi_d, 0.0) ;
    Kokkos::deep_copy(same_max_loc_d, 0.0) ;
    Kokkos::deep_copy(same_sum_loc_d, 0.0) ;
    Kokkos::deep_copy(same_max_mpi_d, 0.0) ;
    Kokkos::deep_copy(same_sum_mpi_d, 0.0) ;
    Kokkos::deep_copy(hang_max_loc_int_d,  0.0) ;
    Kokkos::deep_copy(hang_max_mpi_int_d,  0.0) ;
    Kokkos::deep_copy(hang_max_loc_edge_d, 0.0) ;
    Kokkos::deep_copy(hang_max_mpi_edge_d, 0.0) ;
    //**************************************************************************************************/
    //**************************************************************************************************/
    // PART 1: HANGING FACES (coarse-fine)
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto desc = ghost_layer.get_reflux_face_descriptors() ;
    auto info = ghost_layer.get_reflux_face_send_list() ;
    //**************************************************************************************************/
    // Build 1-var send/recv buffers reusing the hanging-face EMF comm pattern
    auto diag_snd_off  = halve_offsets(ghost_layer.get_reflux_buffer_rank_send_emf_offsets()) ;
    auto diag_snd_size = halve_offsets(ghost_layer.get_reflux_buffer_rank_send_emf_sizes()) ;
    auto diag_rcv_off  = halve_offsets(ghost_layer.get_reflux_buffer_rank_recv_emf_offsets()) ;
    auto diag_rcv_size = halve_offsets(ghost_layer.get_reflux_buffer_rank_recv_emf_sizes()) ;
    size_t total_snd = 0, total_rcv = 0 ;
    for(int r=0; r<nprocs; ++r) { total_snd += diag_snd_size[r] ; total_rcv += diag_rcv_size[r] ; }
    //**************************************************************************************************/
    amr::reflux_array_t sbuf("diag_B_hang_snd") ;
    sbuf.set_strides(nx/2, 1) ;
    sbuf.set_offsets(diag_snd_off) ;
    sbuf.realloc(total_snd) ;
    amr::reflux_array_t rbuf("diag_B_hang_rcv") ;
    rbuf.set_strides(nx/2, 1) ;
    rbuf.set_offsets(diag_rcv_off) ;
    rbuf.realloc(total_rcv) ;
    //**************************************************************************************************/
    // Fill send buffer: fine side packs restricted face B = 0.25 * sum(2x2 fine B)
    {
        auto policy =
            MDRangePolicy<Rank<3>> (
                {0,0,0},
                {static_cast<long>(nx/2)
                ,static_cast<long>(nx/2)
                ,static_cast<long>(info.qid.extent(0))}
            ) ;
        parallel_for( GRACE_EXECUTION_TAG("DIAG", "diag_B_hang_fill")
                , policy
                , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                    auto const iface = info.elem_id(iq) ;
                    auto const rank  = info.rank(iq) ;
                    auto const fdir  = iface / 2 ;
                    auto const iside = iface % 2 ;
                    auto const qid   = info.qid(iq) ;
                    auto const bid   = info.buf_id(iq) ;
                    // face position on the fine side
                    size_t ijk[3] ;
                    ijk[fdir] = iside ? nx + ngz : ngz ;
                    ijk[other_dirs[fdir][0]] = 2*i + ngz ;
                    ijk[other_dirs[fdir][1]] = 2*j + ngz ;
                    // accumulate 2x2 fine B values
                    auto idir = other_dirs[fdir][0] ;
                    auto jdir = other_dirs[fdir][1] ;
                    double sum = 0.0 ;
                    for(int di=0; di<2; ++di) {
                        for(int dj=0; dj<2; ++dj) {
                            size_t ijk_f[3] ;
                            ijk_f[fdir] = ijk[fdir] ;
                            ijk_f[idir] = 2*i + di + ngz ;
                            ijk_f[jdir] = 2*j + dj + ngz ;
                            if      (fdir==0) sum += Bx(ijk_f[0],ijk_f[1],ijk_f[2],0,qid) ;
                            else if (fdir==1) sum += By(ijk_f[0],ijk_f[1],ijk_f[2],0,qid) ;
                            else              sum += Bz(ijk_f[0],ijk_f[1],ijk_f[2],0,qid) ;
                        }
                    }
                    sbuf(i,j,0,bid,rank) = 0.25 * sum ;
                }
            ) ;
    }
    Kokkos::fence() ;
    //**************************************************************************************************/
    // MPI exchange for hanging faces
    parallel::grace_transfer_context_t ctx_hang ;
    for(int iproc=0; iproc<nprocs; ++iproc) {
        if ( iproc == proc ) continue ;
        if ( diag_snd_size[iproc] > 0 ) {
            ctx_hang._send_requests.push_back(MPI_Request{}) ;
            parallel::mpi_isend(
                sbuf.data() + diag_snd_off[iproc],
                diag_snd_size[iproc], iproc,
                parallel::GRACE_DIAG_FACE_B_HANGING_TAG,
                MPI_COMM_WORLD, &ctx_hang._send_requests.back()
            ) ;
        }
        if ( diag_rcv_size[iproc] > 0 ) {
            ctx_hang._recv_requests.push_back(MPI_Request{}) ;
            parallel::mpi_irecv(
                rbuf.data() + diag_rcv_off[iproc],
                diag_rcv_size[iproc], iproc,
                parallel::GRACE_DIAG_FACE_B_HANGING_TAG,
                MPI_COMM_WORLD, &ctx_hang._recv_requests.back()
            ) ;
        }
    }
    parallel::mpi_waitall(ctx_hang) ;
    //**************************************************************************************************/
    // Compare: on the coarse side, per-(level, dir) via atomics
    //**************************************************************************************************/
    {
        auto policy =
            MDRangePolicy<Rank<3>> (
                {0,0,0},
                {static_cast<long>(nx/2)
                ,static_cast<long>(nx/2)
                ,static_cast<long>(desc.coarse_qid.extent(0))}
            ) ;
        parallel_for( GRACE_EXECUTION_TAG("DIAG", "diag_B_hang_check")
                , policy
                , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                    if ( desc.coarse_is_remote(iq) ) return ;
                    auto const iface_c = desc.coarse_face_id(iq) ;
                    auto const fdir  = iface_c / 2 ;
                    auto const idir  = other_dirs[fdir][0] ;
                    auto const jdir  = other_dirs[fdir][1] ;
                    auto const iside = iface_c % 2 ;
                    auto const qid_c = desc.coarse_qid(iq) ;
                    int const lev = level_from_idx(idx(0, qid_c), idx_min) ;

                    for(int ichild=0; ichild<P4EST_CHILDREN/2; ++ichild) {
                        int ichild_i = (ichild>>0)&1 ;
                        int ichild_j = (ichild>>1)&1 ;
                        int off_i = ichild_i ? nx/2 : 0 ;
                        int off_j = ichild_j ? nx/2 : 0 ;
                        // coarse position
                        size_t ijk_c[3] ;
                        ijk_c[fdir] = iside ? nx + ngz : ngz ;
                        ijk_c[idir] = i + off_i + ngz ;
                        ijk_c[jdir] = j + off_j + ngz ;
                        // read coarse B
                        double Bc = 0.0 ;
                        if      (fdir==0) Bc = Bx(ijk_c[0],ijk_c[1],ijk_c[2],0,qid_c) ;
                        else if (fdir==1) Bc = By(ijk_c[0],ijk_c[1],ijk_c[2],0,qid_c) ;
                        else              Bc = Bz(ijk_c[0],ijk_c[1],ijk_c[2],0,qid_c) ;
                        // fine side restricted B
                        double Bf_avg = 0.0 ;
                        if ( desc.fine_is_remote(iq,ichild) ) {
                            auto bid = desc.fine_bid(iq,ichild) ;
                            auto rank = desc.fine_owner_rank(iq,ichild) ;
                            Bf_avg = rbuf(i,j,0,bid,rank) ;
                        } else {
                            auto qid_f = desc.fine_qid(iq,ichild) ;
                            size_t ijk_f[3] ;
                            ijk_f[fdir] = iside ? ngz : nx + ngz ;
                            double sum = 0.0 ;
                            for(int di=0; di<2; ++di) {
                                for(int dj=0; dj<2; ++dj) {
                                    ijk_f[idir] = 2*i + di + ngz ;
                                    ijk_f[jdir] = 2*j + dj + ngz ;
                                    if      (fdir==0) sum += Bx(ijk_f[0],ijk_f[1],ijk_f[2],0,qid_f) ;
                                    else if (fdir==1) sum += By(ijk_f[0],ijk_f[1],ijk_f[2],0,qid_f) ;
                                    else              sum += Bz(ijk_f[0],ijk_f[1],ijk_f[2],0,qid_f) ;
                                }
                            }
                            Bf_avg = 0.25 * sum ;
                        }
                        double err = fabs(Bc - Bf_avg) ;
                        bool is_mpi = desc.fine_is_remote(iq,ichild) ;
                        // Classify: is this B-face cell adjacent to an
                        // edge-corrected EMF?  Phase 1 skips EMFs at
                        // idir-index ∈ {0, nx/2} and jdir-index ∈ {0, nx/2}
                        // (block and quadrant boundaries).  Phase 1 also
                        // never reaches index nx (upper block boundary).
                        // A B-cell at (ic,jc) uses E_jdir at ic and ic+1,
                        // and E_idir at jc and jc+1, so it is edge-affected
                        // when any of those falls in the skipped set.
                        int ic = i + off_i ;   // 0-based coarse idir coord
                        int jc = j + off_j ;   // 0-based coarse jdir coord
                        bool is_edge =
                            (ic == 0 || ic == (int)nx/2 - 1 || ic == (int)nx/2 || ic == (int)nx - 1 ||
                             jc == 0 || jc == (int)nx/2 - 1 || jc == (int)nx/2 || jc == (int)nx - 1) ;
                        if (is_mpi) {
                            atomic_max_d(&hang_max_mpi_d(lev, fdir), err) ;
                            Kokkos::atomic_add(&hang_sum_mpi_d(lev, fdir), err) ;
                            if (is_edge) atomic_max_d(&hang_max_mpi_edge_d(lev, fdir), err) ;
                            else         atomic_max_d(&hang_max_mpi_int_d(lev, fdir), err) ;
                        } else {
                            atomic_max_d(&hang_max_loc_d(lev, fdir), err) ;
                            Kokkos::atomic_add(&hang_sum_loc_d(lev, fdir), err) ;
                            if (is_edge) atomic_max_d(&hang_max_loc_edge_d(lev, fdir), err) ;
                            else         atomic_max_d(&hang_max_loc_int_d(lev, fdir), err) ;
                        }
                    }
                }
            ) ;
    }
    Kokkos::fence() ;
    //**************************************************************************************************/
    //**************************************************************************************************/
    // PART 2: SAME-LEVEL FACES
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto coarse_desc = ghost_layer.get_reflux_coarse_face_descriptors() ;
    auto coarse_info = ghost_layer.get_reflux_coarse_face_send_list() ;
    //**************************************************************************************************/
    // Build 1-var send/recv buffers reusing the coarse-face EMF comm pattern
    auto cdiag_snd_off  = halve_offsets(ghost_layer.get_reflux_buffer_rank_send_emf_coarse_offsets()) ;
    auto cdiag_snd_size = halve_offsets(ghost_layer.get_reflux_buffer_rank_send_emf_coarse_sizes()) ;
    auto cdiag_rcv_off  = halve_offsets(ghost_layer.get_reflux_buffer_rank_recv_emf_coarse_offsets()) ;
    auto cdiag_rcv_size = halve_offsets(ghost_layer.get_reflux_buffer_rank_recv_emf_coarse_sizes()) ;
    size_t ctotal_snd = 0, ctotal_rcv = 0 ;
    for(int r=0; r<nprocs; ++r) { ctotal_snd += cdiag_snd_size[r] ; ctotal_rcv += cdiag_rcv_size[r] ; }
    //**************************************************************************************************/
    amr::reflux_array_t csbuf("diag_B_same_snd") ;
    csbuf.set_strides(nx, 1) ;
    csbuf.set_offsets(cdiag_snd_off) ;
    csbuf.realloc(ctotal_snd) ;
    amr::reflux_array_t crbuf("diag_B_same_rcv") ;
    crbuf.set_strides(nx, 1) ;
    crbuf.set_offsets(cdiag_rcv_off) ;
    crbuf.realloc(ctotal_rcv) ;
    //**************************************************************************************************/
    // Fill send buffer: pack face B
    {
        auto policy =
            MDRangePolicy<Rank<3>> (
                {0,0,0},
                {static_cast<long>(nx)
                ,static_cast<long>(nx)
                ,static_cast<long>(coarse_info.qid.extent(0))}
            ) ;
        parallel_for( GRACE_EXECUTION_TAG("DIAG", "diag_B_same_fill")
                , policy
                , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                    auto const iface = coarse_info.elem_id(iq) ;
                    auto const rank  = coarse_info.rank(iq) ;
                    auto const fdir  = iface / 2 ;
                    auto const idir  = other_dirs[fdir][0] ;
                    auto const jdir  = other_dirs[fdir][1] ;
                    auto const iside = iface % 2 ;
                    auto const qid   = coarse_info.qid(iq) ;
                    auto const bid   = coarse_info.buf_id(iq) ;
                    size_t ijk[3] ;
                    ijk[fdir] = iside ? nx + ngz : ngz ;
                    ijk[idir] = i + ngz ;
                    ijk[jdir] = j + ngz ;
                    double B = 0.0 ;
                    if      (fdir==0) B = Bx(ijk[0],ijk[1],ijk[2],0,qid) ;
                    else if (fdir==1) B = By(ijk[0],ijk[1],ijk[2],0,qid) ;
                    else              B = Bz(ijk[0],ijk[1],ijk[2],0,qid) ;
                    csbuf(i,j,0,bid,rank) = B ;
                }
            ) ;
    }
    Kokkos::fence() ;
    //**************************************************************************************************/
    // MPI exchange for same-level faces
    parallel::grace_transfer_context_t ctx_same ;
    for(int iproc=0; iproc<nprocs; ++iproc) {
        if ( iproc == proc ) continue ;
        if ( cdiag_snd_size[iproc] > 0 ) {
            ctx_same._send_requests.push_back(MPI_Request{}) ;
            parallel::mpi_isend(
                csbuf.data() + cdiag_snd_off[iproc],
                cdiag_snd_size[iproc], iproc,
                parallel::GRACE_DIAG_FACE_B_SAMELEVEL_TAG,
                MPI_COMM_WORLD, &ctx_same._send_requests.back()
            ) ;
        }
        if ( cdiag_rcv_size[iproc] > 0 ) {
            ctx_same._recv_requests.push_back(MPI_Request{}) ;
            parallel::mpi_irecv(
                crbuf.data() + cdiag_rcv_off[iproc],
                cdiag_rcv_size[iproc], iproc,
                parallel::GRACE_DIAG_FACE_B_SAMELEVEL_TAG,
                MPI_COMM_WORLD, &ctx_same._recv_requests.back()
            ) ;
        }
    }
    parallel::mpi_waitall(ctx_same) ;
    //**************************************************************************************************/
    // Compare same-level faces, per-(level, dir) via atomics
    //**************************************************************************************************/
    {
        auto policy =
            MDRangePolicy<Rank<3>> (
                {0,0,0},
                {static_cast<long>(nx)
                ,static_cast<long>(nx)
                ,static_cast<long>(coarse_desc.qid.extent(0))}
            ) ;
        parallel_for( GRACE_EXECUTION_TAG("DIAG", "diag_B_same_check")
                , policy
                , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                    auto const fid0  = coarse_desc.face_id(iq,0) ;
                    auto const fdir  = fid0 / 2 ;
                    auto const idir  = other_dirs[fdir][0] ;
                    auto const jdir  = other_dirs[fdir][1] ;
                    // Derive level from whichever side is local
                    int lev = 0 ;
                    for(int is=0; is<2; ++is) {
                        if ( !coarse_desc.is_remote(iq,is) ) {
                            auto qid = coarse_desc.qid(iq,is) ;
                            lev = level_from_idx(idx(0, qid), idx_min) ;
                            break ;
                        }
                    }
                    // read local B from both sides (remote from recv buffer)
                    double B[2] ;
                    for(int is=0; is<2; ++is) {
                        auto const fid   = coarse_desc.face_id(iq,is) ;
                        auto const iside = fid % 2 ;
                        if ( coarse_desc.is_remote(iq,is) ) {
                            auto bid  = coarse_desc.bid(iq,is) ;
                            int  rank = coarse_desc.owner_rank(iq,is) ;
                            B[is] = crbuf(i,j,0,bid,rank) ;
                        } else {
                            auto qid = coarse_desc.qid(iq,is) ;
                            size_t ijk[3] ;
                            ijk[fdir] = iside ? nx + ngz : ngz ;
                            ijk[idir] = i + ngz ;
                            ijk[jdir] = j + ngz ;
                            if      (fdir==0) B[is] = Bx(ijk[0],ijk[1],ijk[2],0,qid) ;
                            else if (fdir==1) B[is] = By(ijk[0],ijk[1],ijk[2],0,qid) ;
                            else              B[is] = Bz(ijk[0],ijk[1],ijk[2],0,qid) ;
                        }
                    }
                    double err = fabs(B[0] - B[1]) ;
                    bool is_mpi = coarse_desc.is_remote(iq,0) || coarse_desc.is_remote(iq,1) ;
                    if (is_mpi) {
                        atomic_max_d(&same_max_mpi_d(lev, fdir), err) ;
                        Kokkos::atomic_add(&same_sum_mpi_d(lev, fdir), err) ;
                    } else {
                        atomic_max_d(&same_max_loc_d(lev, fdir), err) ;
                        Kokkos::atomic_add(&same_sum_loc_d(lev, fdir), err) ;
                    }
                }
            ) ;
    }
    Kokkos::fence() ;
    //**************************************************************************************************/
    //**************************************************************************************************/
    // HOST-SIDE POST-PROCESSING
    //**************************************************************************************************/
    //**************************************************************************************************/
    // Copy device views to host
    constexpr int NN = DIAG_MAX_LEVELS * DIAG_NDIRS ;
    auto hang_max_loc_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hang_max_loc_d) ;
    auto hang_sum_loc_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hang_sum_loc_d) ;
    auto hang_max_mpi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hang_max_mpi_d) ;
    auto hang_sum_mpi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hang_sum_mpi_d) ;
    auto same_max_loc_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, same_max_loc_d) ;
    auto same_sum_loc_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, same_sum_loc_d) ;
    auto same_max_mpi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, same_max_mpi_d) ;
    auto same_sum_mpi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, same_sum_mpi_d) ;
    auto hang_max_loc_int_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hang_max_loc_int_d) ;
    auto hang_max_mpi_int_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hang_max_mpi_int_d) ;
    auto hang_max_loc_edge_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hang_max_loc_edge_d) ;
    auto hang_max_mpi_edge_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hang_max_mpi_edge_d) ;
    //**************************************************************************************************/
    // MPI allreduce: max for max_err, sum for sum_err
    std::array<double, NN> hang_max_loc_g{}, hang_sum_loc_g{} ;
    std::array<double, NN> hang_max_mpi_g{}, hang_sum_mpi_g{} ;
    std::array<double, NN> same_max_loc_g{}, same_sum_loc_g{} ;
    std::array<double, NN> same_max_mpi_g{}, same_sum_mpi_g{} ;
    parallel::mpi_allreduce(hang_max_loc_h.data(), hang_max_loc_g.data(), NN, sc_MPI_MAX) ;
    parallel::mpi_allreduce(hang_sum_loc_h.data(), hang_sum_loc_g.data(), NN, sc_MPI_SUM) ;
    parallel::mpi_allreduce(hang_max_mpi_h.data(), hang_max_mpi_g.data(), NN, sc_MPI_MAX) ;
    parallel::mpi_allreduce(hang_sum_mpi_h.data(), hang_sum_mpi_g.data(), NN, sc_MPI_SUM) ;
    parallel::mpi_allreduce(same_max_loc_h.data(), same_max_loc_g.data(), NN, sc_MPI_MAX) ;
    parallel::mpi_allreduce(same_sum_loc_h.data(), same_sum_loc_g.data(), NN, sc_MPI_SUM) ;
    parallel::mpi_allreduce(same_max_mpi_h.data(), same_max_mpi_g.data(), NN, sc_MPI_MAX) ;
    parallel::mpi_allreduce(same_sum_mpi_h.data(), same_sum_mpi_g.data(), NN, sc_MPI_SUM) ;
    std::array<double, NN> hang_max_loc_int_g{}, hang_max_mpi_int_g{} ;
    std::array<double, NN> hang_max_loc_edge_g{}, hang_max_mpi_edge_g{} ;
    parallel::mpi_allreduce(hang_max_loc_int_h.data(),  hang_max_loc_int_g.data(),  NN, sc_MPI_MAX) ;
    parallel::mpi_allreduce(hang_max_mpi_int_h.data(),  hang_max_mpi_int_g.data(),  NN, sc_MPI_MAX) ;
    parallel::mpi_allreduce(hang_max_loc_edge_h.data(), hang_max_loc_edge_g.data(), NN, sc_MPI_MAX) ;
    parallel::mpi_allreduce(hang_max_mpi_edge_h.data(), hang_max_mpi_edge_g.data(), NN, sc_MPI_MAX) ;
    //**************************************************************************************************/
    // Compute per-level face area for integrated flux: dA = dx_coarse^2
    // dx at level L = 1/(idx_min * 2^L)
    std::array<double, DIAG_MAX_LEVELS> dA_level{} ;
    for(int l=0; l<DIAG_MAX_LEVELS; ++l) {
        double dx_l = 1.0 / (idx_min * (1 << l)) ;
        dA_level[l] = dx_l * dx_l ;
    }
    //**************************************************************************************************/
    // Backward-compatible global max
    double global_hanging = 0.0, global_samelevel = 0.0 ;
    for(int k=0; k<NN; ++k) {
        double hm = std::max(hang_max_loc_g[k], hang_max_mpi_g[k]) ;
        double sm = std::max(same_max_loc_g[k], same_max_mpi_g[k]) ;
        if (hm > global_hanging)   global_hanging   = hm ;
        if (sm > global_samelevel) global_samelevel  = sm ;
    }
    GRACE_TRACE("[DIAG] face B conservation: hanging max|Bc-avg(Bf)|={:.6e}  same-level max|B0-B1|={:.6e}",
                global_hanging, global_samelevel) ;
    //**************************************************************************************************/
    // Per-level detail split by LOCAL vs MPI
    constexpr char const* dname[3] = {"x","y","z"} ;
    for(int l=0; l<DIAG_MAX_LEVELS; ++l) {
        for(int d=0; d<DIAG_NDIRS; ++d) {
            int k = l * DIAG_NDIRS + d ;
            if (hang_max_loc_g[k] > 0.0 || hang_max_mpi_g[k] > 0.0) {
                GRACE_TRACE("[DIAG]   hanging L{}/L{} dir={}: LOCAL max={:.6e} sum*dA={:.6e}  |  MPI max={:.6e} sum*dA={:.6e}",
                    l, l+1, dname[d],
                    hang_max_loc_g[k], hang_sum_loc_g[k] * dA_level[l],
                    hang_max_mpi_g[k], hang_sum_mpi_g[k] * dA_level[l]) ;
                GRACE_TRACE("[DIAG]     -> interior: LOC={:.6e}  MPI={:.6e}  |  edge: LOC={:.6e}  MPI={:.6e}",
                    hang_max_loc_int_g[k], hang_max_mpi_int_g[k],
                    hang_max_loc_edge_g[k], hang_max_mpi_edge_g[k]) ;
            }
        }
    }
    for(int l=0; l<DIAG_MAX_LEVELS; ++l) {
        for(int d=0; d<DIAG_NDIRS; ++d) {
            int k = l * DIAG_NDIRS + d ;
            if (same_max_loc_g[k] > 0.0 || same_max_mpi_g[k] > 0.0) {
                GRACE_TRACE("[DIAG]   same-level L{} dir={}: LOCAL max={:.6e} sum*dA={:.6e}  |  MPI max={:.6e} sum*dA={:.6e}",
                    l, dname[d],
                    same_max_loc_g[k], same_sum_loc_g[k] * dA_level[l],
                    same_max_mpi_g[k], same_sum_mpi_g[k] * dA_level[l]) ;
            }
        }
    }
    //**************************************************************************************************/
    // .dat file output (rank 0 only, at scalar output frequency)
    //**************************************************************************************************/
    auto& grace_runtime = grace::runtime::get() ;
    auto iter = grace_runtime.iteration() ;
    auto time = grace_runtime.time() ;
    int scalar_every = grace_runtime.scalar_output_every() ;
    if ( proc == 0 && scalar_every > 0 && (iter % scalar_every == 0) ) {
        std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ;
        std::string base = grace_runtime.scalar_io_basename() ;
        //**************************************************************************************************/
        // Helper lambda: write one .dat file with local|mpi columns
        auto write_dat = [&](std::string const& suffix,
                             std::array<double,NN> const& loc_data,
                             std::array<double,NN> const& mpi_data,
                             double const* scale, /* if non-null, multiply by scale[l] */
                             bool& initialized)
        {
            auto fpath = bdir / (base + suffix) ;
            if (!initialized) {
                std::string hdr = "# Iteration\tTime" ;
                for(int l=0; l<DIAG_MAX_LEVELS; ++l)
                    for(int d=0; d<DIAG_NDIRS; ++d) {
                        std::string tag = "L" + std::to_string(l) + "_" + dname[d] ;
                        hdr += "\t" + tag + "_loc\t" + tag + "_mpi" ;
                    }
                init_Bflux_dat(fpath, hdr) ;
                initialized = true ;
            }
            std::ofstream f(fpath.string(), std::ios::app) ;
            f << std::scientific << std::setprecision(6) ;
            f << iter << '\t' << time ;
            for(int l=0; l<DIAG_MAX_LEVELS; ++l) {
                double s = scale ? scale[l] : 1.0 ;
                for(int d=0; d<DIAG_NDIRS; ++d) {
                    int k = l * DIAG_NDIRS + d ;
                    f << '\t' << loc_data[k] * s << '\t' << mpi_data[k] * s ;
                }
            }
            f << '\n' ;
        } ;
        //**************************************************************************************************/
        static bool i1=false, i2=false, i3=false, i4=false, i5=false, i6=false ;
        write_dat("Bflux_hanging_max.dat",      hang_max_loc_g, hang_max_mpi_g, nullptr,          i1) ;
        write_dat("Bflux_hanging_integral.dat",  hang_sum_loc_g, hang_sum_mpi_g, dA_level.data(),  i2) ;
        write_dat("Bflux_samelevel_max.dat",     same_max_loc_g, same_max_mpi_g, nullptr,          i3) ;
        write_dat("Bflux_samelevel_integral.dat", same_sum_loc_g, same_sum_mpi_g, dA_level.data(), i4) ;
        write_dat("Bflux_hanging_max_interior.dat", hang_max_loc_int_g,  hang_max_mpi_int_g,  nullptr, i5) ;
        write_dat("Bflux_hanging_max_edge.dat",     hang_max_loc_edge_g, hang_max_mpi_edge_g, nullptr, i6) ;
    }
}

} /* namespace grace */