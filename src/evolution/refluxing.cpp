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

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

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
    auto desc = ghost_layer.get_reflux_face_send_list() ; 
    //**************************************************************************************************/
    View<hanging_remote_reflux_desc_t*> info ; 
    grace::deep_copy_vec_to_const_view(info,desc) ; 
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
            ,static_cast<long>(desc.size())}
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
                auto const dsc = info(iq) ; 

                auto const iface = dsc.elem_id ; 
                auto const rank = dsc.rank     ; 
                auto const idir = iface / 2    ; 
                auto const iside = iface%2     ; 
                auto const qid = dsc.qid       ; 

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
                
                auto bid = dsc.buf_id ; 
                sbuf(i,j,ivar,bid,rank) = 0.25*flux ; 
            }
        ) ; 
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
    auto desc = ghost_layer.get_reflux_face_send_list() ; 
    //**************************************************************************************************/
    View<hanging_remote_reflux_desc_t*> info ; 
    grace::deep_copy_vec_to_const_view(info,desc) ; 
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<3>> (
            {0,0,0},
            {static_cast<long>(nx/2)
            ,static_cast<long>(nx/2)
            ,static_cast<long>(desc.size())}
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
                auto const dsc = info(iq) ; 
                 
                auto const iface = dsc.elem_id ; 
                auto const rank = dsc.rank ; 

                auto const fdir = iface / 2;
                auto const idir = other_dirs[fdir][0]; 
                auto const jdir = other_dirs[fdir][1]; 
                auto const iside = iface % 2 ;

                auto const qid = dsc.qid ; 

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
                
                auto bid = dsc.buf_id ; 
                sbuf(i,j,0,bid,rank) = emf_i ; 
                sbuf(i,j,1,bid,rank) = emf_j ; 
            }
        ) ; 
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
    auto coarse_desc = ghost_layer.get_reflux_coarse_face_send_list() ; 
    //**************************************************************************************************/
    View<hanging_remote_reflux_desc_t*> coarse_info ; 
    grace::deep_copy_vec_to_const_view(coarse_info,coarse_desc) ; 
    //**************************************************************************************************/
    auto coarse_policy = 
        MDRangePolicy<Rank<3>> (
            {0,0,0},
            {static_cast<long>(nx)
            ,static_cast<long>(nx)
            ,static_cast<long>(coarse_desc.size())}
        ) ; 
    //**************************************************************************************************/ 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_fill_emf_coarse_buffers")
            , coarse_policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                auto const dsc = coarse_info(iq) ; 
                 
                auto const iface = dsc.elem_id ; 
                auto const rank = dsc.rank ; 

                auto const fdir = iface / 2;
                auto const idir = other_dirs[fdir][0]; 
                auto const jdir = other_dirs[fdir][1]; 
                auto const iside = iface % 2 ;

                auto const qid = dsc.qid ; 

                size_t ijk_s[3] ; 
                ijk_s[fdir] = iside ? nx + ngz : ngz ; 
                ijk_s[idir] = i + ngz ; 
                ijk_s[jdir] = j + ngz ; 
                
                
                auto bid = dsc.buf_id ; 
                coarse_sbuf(i,j,0,bid,rank) = emf(ijk_s[0],ijk_s[1],ijk_s[2],idir,qid) ; 
                coarse_sbuf(i,j,1,bid,rank) = emf(ijk_s[0],ijk_s[1],ijk_s[2],jdir,qid) ; 
            }
        ) ; 
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
    auto desc_edge = ghost_layer.get_reflux_edge_send_list() ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    View<hanging_remote_reflux_desc_t*> info_edge ; 
    grace::deep_copy_vec_to_const_view(info_edge,desc_edge) ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto edge_policy = 
        MDRangePolicy<Rank<2>> (
            {0,0},
            {static_cast<long>(nx),static_cast<long>(desc_edge.size())}
        ) ; 
    // fill edge buffers 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_fill_emf_edge_buffers")
            , edge_policy 
            , KOKKOS_LAMBDA (int const& i, int const& iq) {
                auto const dsc = info_edge(iq) ; 
                
                auto const iedge = dsc.elem_id ; 
                // edge direction 
                int idir  = (iedge/4)          ; 
                // upper or lower gz?
                int jside = (iedge>>0)&1       ; 
                int kside = (iedge>>1)&1       ;
                // orthogonal directions (z-order)
                int jdir = other_dirs[idir][0] ; 
                int kdir = other_dirs[idir][1] ; 
                // quad-id (fine)
                auto const qid = dsc.qid ; 
                // indices of edge 
                size_t ijk_s[3] ;
                ijk_s[idir] = ngz + i ; 
                ijk_s[jdir] = jside ? nx + ngz : ngz ; 
                ijk_s[kdir] = kside ? nx + ngz : ngz ; 

                auto const rank = dsc.rank ; 
                auto bid = dsc.buf_id ;
                // write to buffer
                sbuf_edge(i, bid, rank) = emf(ijk_s[0],ijk_s[1],ijk_s[2],idir,qid) ; 
            }
        ) ;
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
    View<hanging_face_reflux_desc_t*> info ; 
    grace::deep_copy_vec_to_const_view(info,desc) ; 
    //**************************************************************************************************/
    parallel::mpi_waitall(context) ; 
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<4>> (
            {0,0,0,0},
            {static_cast<long>(nx)
            ,static_cast<long>(nx)
            ,static_cast<long>(nvars_hrsc)
            ,static_cast<long>(desc.size())}
        ) ;
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_apply")
            , policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& ivar, int const& iq) {
                auto const dsc = info(iq) ; 

                auto const qid_c  = dsc.coarse_qid     ; 
                auto const iface_c = dsc.coarse_face_id ; 

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
                if ( dsc.fine_is_remote[ichild] ) {
                    flux_correction = rbuf(i%(nx/2),j%(nx/2),ivar,dsc.fine_qid[ichild],dsc.fine_owner_rank[ichild]) ; 
                } else {
                    // compute flux correction 
                    size_t qid_f = dsc.fine_qid[ichild] ; 
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
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& emf  = grace::variable_list::get().getemfarray() ;  
    //**************************************************************************************************/
    auto& ghost_layer = grace::amr_ghosts::get();
    auto rbuf = ghost_layer.get_reflux_emf_recv_buffer() ; 
    auto desc = ghost_layer.get_reflux_face_descriptors() ; 
    //**************************************************************************************************/
    View<hanging_face_reflux_desc_t*> info ; 
    grace::deep_copy_vec_to_const_view(info,desc) ; 
    //**************************************************************************************************/
    parallel::mpi_waitall(context) ;
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<3>> (
            {0,0,0},
            {static_cast<long>(nx/2)
            ,static_cast<long>(nx/2)
            ,static_cast<long>(desc.size())}
        ) ;
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_emf_apply_face")
            , policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                auto const dsc = info(iq) ; 

                // coarse face 
                auto const iface_c = dsc.coarse_face_id ; 
                // coarse face direction 
                auto const fdir = iface_c / 2 ; 
                // other directions (z-order)
                auto const idir = other_dirs[fdir][0]; 
                auto const jdir = other_dirs[fdir][1]; 
                // side of the face 
                auto const iside = iface_c % 2 ;
                // qid of coarse side 
                auto const qid_c = dsc.coarse_qid ; 

                // indices of face center (coarse)
                // edge center (coarse)
                // edge center (fine)
                size_t ijk_c[3], ijk_f[3] ; 
                for( int ichild=0; ichild<P4EST_CHILDREN/2; ++ichild) {
                    // fine quadid 
                    auto const qid_f = dsc.fine_qid[ichild] ; 
                    // emf correction 
                    double emf_corr_i{0}, emf_corr_j{0} ; 
                    if ( dsc.fine_is_remote[ichild] ) {
                        auto rank = dsc.fine_owner_rank[ichild] ; 
                        emf_corr_i = rbuf(i,j,0,qid_f,rank) ; 
                        emf_corr_j = rbuf(i,j,1,qid_f,rank) ; 
                    } else {
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
    //**************************************************************************************************/
    auto coarse_rbuf = ghost_layer.get_reflux_emf_coarse_recv_buffer() ; 
    auto coarse_desc = ghost_layer.get_reflux_coarse_face_descriptors() ; 
    //**************************************************************************************************/
    View<full_face_reflux_desc_t*> coarse_info ; 
    grace::deep_copy_vec_to_const_view(coarse_info,coarse_desc) ; 
    //**************************************************************************************************/
    auto coarse_policy = 
        MDRangePolicy<Rank<3>> (
            {0,0,0},
            {static_cast<long>(nx)
            ,static_cast<long>(nx)
            ,static_cast<long>(coarse_desc.size())}
        ) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_emf_apply_coarse_face")
            , coarse_policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                auto const dsc = coarse_info(iq) ; 
                // step 1 compute 
                size_t ijk[3] ; 
                double emf_corr[2] = {0,0}; 
                for( int is=0; is<2; ++is) {
                    auto const fid = dsc.face_id[is] ;
                    auto const fdir = fid / 2 ; 
                    // other directions (z-order)
                    auto const idir = other_dirs[fdir][0]; 
                    auto const jdir = other_dirs[fdir][1]; 
                    // iside 
                    auto const iside = fid % 2 ;
                    // qid 
                    auto const qid = dsc.qid[is]; 
                    if ( dsc.is_remote[is] ) {
                        emf_corr[0] += coarse_rbuf(i,j,0,qid,dsc.owner_rank[is]) ; 
                        emf_corr[1] += coarse_rbuf(i,j,1,qid,dsc.owner_rank[is]) ; 
                    } else {
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
                    auto const fid = dsc.face_id[is] ;
                    auto const fdir = fid / 2 ; 
                    // other directions (z-order)
                    auto const idir = other_dirs[fdir][0]; 
                    auto const jdir = other_dirs[fdir][1]; 
                    // iside 
                    auto const iside = fid % 2 ;
                    // qid 
                    auto const qid = dsc.qid[is]; 
                    if ( !dsc.is_remote[is] ) {
                        ijk[fdir] = iside ? nx + ngz : ngz ; 
                        ijk[idir] = ngz + i ; 
                        ijk[jdir] = ngz + j ; 
                        if ( ijk[jdir] > ngz ) emf(ijk[0],ijk[1],ijk[2],idir,qid) = emf_corr[0]; 
                        if ( ijk[idir] > ngz ) emf(ijk[0],ijk[1],ijk[2],jdir,qid) = emf_corr[1];  
                    } 
                } // iside 
            }
                
        ) ; 
    //**************************************************************************************************/
    auto edge_rbuf = ghost_layer.get_reflux_emf_edge_recv_buffer() ; 
    auto edge_desc = ghost_layer.get_reflux_edge_descriptors() ; 
    auto coarse_edge_desc = ghost_layer.get_reflux_coarse_edge_descriptors() ; 
    //**************************************************************************************************/
    View<hanging_edge_reflux_desc_t*> edge_info ; 
    grace::deep_copy_vec_to_const_view(edge_info,edge_desc) ; 
    View<hanging_edge_reflux_desc_t*> coarse_edge_info ; 
    grace::deep_copy_vec_to_const_view(coarse_edge_info,coarse_edge_desc) ;
    //**************************************************************************************************/
    auto edge_policy = 
        MDRangePolicy<Rank<2>> (
            {0,0},
            {static_cast<long>(nx),static_cast<long>(edge_desc.size())}
        ) ;
    auto coarse_edge_policy = 
        MDRangePolicy<Rank<2>> (
            {0,0},
            {static_cast<long>(nx),static_cast<long>(coarse_edge_desc.size())}
        ) ;
    //**************************************************************************************************/
    // two phases, first we need to compute the correction, then we apply
    auto emf_edge_correction = ghost_layer.get_reflux_edge_emf_accumulation_buffer() ;  
    parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "reflux_emf_compute_edge"),
        edge_policy,
        KOKKOS_LAMBDA (int const& i, int const& iq) {
            auto& desc = edge_info(iq) ; 
            auto n_sides = desc.n_sides; 
            auto n_fine = desc.n_fine ; 
            double norm =  1.0/static_cast<double>(desc.n_fine) ;
            size_t ijk[3] ; 
            double emf_correction[2] = {0,0} ; // accumulate here 
            int cnt = 0 ; 
            for( int iside=0; iside<n_sides; ++iside) {
                auto& side = desc.sides[iside] ; 
                if ( ! side.is_fine) continue ; 
                cnt ++ ; 
                // edge index 
                auto edge_id = side.edge_id ; 
                // direction and side
                int edge_dir = edge_id / 4 ; 
                int side_i = (edge_id>>0)&1;
                int side_j = (edge_id>>1)&1;
                // child id loop 
                for( int ichild=0; ichild<2; ++ichild ) {
                    // fine quadid
                    auto qid = side.octants.fine.quad_id[ichild];
                    double val = 0.0;
                    if ( side.octants.fine.is_remote[ichild] ) {
                        auto rank = side.octants.fine.owner_rank[ichild] ; 
                        val = edge_rbuf(i,qid,rank) ; 
                    } else {
                        ijk[edge_dir] = ngz + i ; 
                        ijk[other_dirs[edge_dir][0]] = side_i ? nx + ngz : ngz ; 
                        ijk[other_dirs[edge_dir][1]] = side_j ? nx + ngz : ngz ; 
                        val = emf(ijk[0],ijk[1],ijk[2],edge_dir,qid); 
                    }
                    emf_correction[ichild] += val ; 
                }
            }
            
            emf_edge_correction(i,0,iq) = cnt ? emf_correction[0] / static_cast<double>(cnt) : 0.0 ; 
            emf_edge_correction(i,1,iq) = cnt ? emf_correction[1] / static_cast<double>(cnt) : 0.0 ; 
        }
    );
    auto emf_coarse_edge_correction = ghost_layer.get_reflux_coarse_edge_emf_accumulation_buffer() ; 
    parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "reflux_coarse_emf_compute_edge"),
        coarse_edge_policy,
        KOKKOS_LAMBDA (int const& i, int const& iq) {
            auto& desc = coarse_edge_info(iq) ; 
            auto n_sides = desc.n_sides; 
            size_t ijk[3] ; 
            double emf_correction{0} ; // accumulate here 
            for( int iside=0; iside<4; ++iside) {
                auto& side = desc.sides[iside] ; 
                // edge index 
                auto edge_id = side.edge_id ; 
                // direction and side
                int edge_dir = edge_id / 4 ; 
                int side_i = (edge_id>>0)&1;
                int side_j = (edge_id>>1)&1;
                // coarse quadid
                auto qid = side.octants.coarse.quad_id;
                if ( side.octants.coarse.is_remote ) {
                    auto rank = side.octants.coarse.owner_rank ; 
                    emf_correction += edge_rbuf(i,qid,rank) ; 
                } else {
                    ijk[edge_dir] = ngz + i ; 
                    ijk[other_dirs[edge_dir][0]] = side_i ? nx + ngz : ngz ; 
                    ijk[other_dirs[edge_dir][1]] = side_j ? nx + ngz : ngz ; 
                    emf_correction += emf(ijk[0],ijk[1],ijk[2],edge_dir,qid); 
                }
            }
            emf_coarse_edge_correction(i,iq) = emf_correction * 0.25 ; 
        }
    );
    Kokkos::fence() ; 
    // apply 
    parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "reflux_emf_apply_edge"),
        edge_policy,
        KOKKOS_LAMBDA (int const& i, int const& iq) {
            // information about the edge we are correcting 
            auto const& desc = edge_info(iq) ; 
            // pre-allocate indices 
            size_t ijk[3] ; 
            // loop over 4 sides of the edge
            for( int iside=0; iside<desc.n_sides; ++iside) {
                // side descriptor 
                auto const& side = desc.sides[iside] ;
                // edge index 
                auto edge_id = side.edge_id ;  
                // direction along and orthogonal to edge (z-order)
                int edge_dir = edge_id / 4 ; 
                int side_i = (edge_id>>0)&1;
                int side_j = (edge_id>>1)&1;

                // if coarse we need to correct with - emf + 1/n_fine 1/2 sum( fine emfs )
                if ( ! side.is_fine ) {
                    // Remote: nothing to do 
                    if ( side.octants.coarse.is_remote ) continue ;
                    // quad-id 
                    auto qid = side.octants.coarse.quad_id ; 
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
                    ijk[other_dirs[edge_dir][0]] = side.off_i ? nx/2 + ngz : ( side_i ? nx + ngz : ngz ) ;  
                    ijk[other_dirs[edge_dir][1]] = side.off_j ? nx/2 + ngz : ( side_j ? nx + ngz : ngz ) ;
                    
                    emf(ijk[0],ijk[1],ijk[2],edge_dir,qid) = 
                        +0.5*(emf_edge_correction((2*i)%nx,ichild,iq) + emf_edge_correction((2*i)%nx+1,ichild,iq));
                } else {
                    for( int ichild=0; ichild<2; ++ichild) {
                        // Remote: nothing to do
                        if ( side.octants.fine.is_remote[ichild] ) continue ;
                        // quad-id
                        auto qid = side.octants.fine.quad_id[ichild] ;
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

    parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "reflux_emf_apply_coarse_edge"),
        coarse_edge_policy,
        KOKKOS_LAMBDA (int const& i, int const& iq) {
            // information about the edge we are correcting 
            auto const& desc = coarse_edge_info(iq) ; 
            // pre-allocate indices 
            size_t ijk[3] ; 
            // loop over 4 sides of the edge
            for( int iside=0; iside<4; ++iside) {
                // side descriptor 
                auto const& side = desc.sides[iside] ;
                // edge index 
                auto edge_id = side.edge_id ;  
                // direction along and orthogonal to edge (z-order)
                int edge_dir = edge_id / 4 ; 
                int side_i = (edge_id>>0)&1;
                int side_j = (edge_id>>1)&1;

                // coarse remote nothing to do 
                if ( side.octants.coarse.is_remote ) continue ;
                // quad-id 
                auto qid = side.octants.coarse.quad_id ; 
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



} /* namespace grace */