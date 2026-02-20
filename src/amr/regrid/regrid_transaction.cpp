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

#include <unordered_set>

#include <grace/amr/regrid/regrid_transaction.hh>

#include <grace/utils/task_queue.hh>
#include <grace/utils/sc_wrappers.hh>
#include <grace/utils/limiters.hh>
#include <grace/utils/device_stream_pool.hh>

#include <grace/amr/amr_functions.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/amr/ghostzone_kernels/ghost_array.hh>
#include <grace/amr/ghostzone_kernels/pr_ho_coeffs.hh>


#include <grace/amr/regrid/copy_kernels.hh>
#include <grace/amr/regrid/restrict_kernels.hh>
#include <grace/amr/regrid/prolong_kernels.hh>
#include <grace/amr/regrid/mpi_kernels.hh>

#include <grace/amr/forest.hh>
#include <grace/amr/quadrant.hh>
#include <grace/amr/amr_flags.hh>
#include <grace/amr/regrid/regrid_helpers.tpp>
#include <grace/amr/regrid/regridding_policy_kernels.tpp>
#include <grace/amr/regrid/partition.hh>
#include <grace/amr/amr_ghosts.hh>

#include <grace/coordinates/coordinates.hh>

namespace grace { namespace amr {
/**
 * @brief Set all quadrants to default state.
 * \cond grace_detail 
 * \ingroup amr 
 */
static void set_quadrants_to_default()  
{
    for(int itree=forest::get().first_local_tree();
            itree<=forest::get().last_local_tree();
            ++itree) 
    {
        auto quadrants = forest::get().tree(itree).quadrants() ; 
        for( int iquad=0; iquad<quadrants.size(); ++iquad) {
            quadrant_t quad{ &(quadrants[iquad]) } ;
            quad.set_user_int( static_cast<int>(DEFAULT_STATE) ) ; 
        }
    }
}


void regrid_transaction_t::evaluate_criterion() {
    auto& params = config_parser::get()             ; 
    auto& state  = variable_list::get().getstate()  ; 
    auto& aux = variable_list::get().getaux()       ; 
    int nvars = state.extent(GRACE_NSPACEDIM)   ; 
    size_t nx,ny,nz                                        ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents()       ; 
    auto ngz = amr::get_n_ghosts()                         ; 
    size_t nq = amr::get_local_num_quadrants()             ; 
    /* create host and device views to hold refinement / coarsening flags */ 
    Kokkos::View<int *, default_space> d_regrid_flags("regrid_flags", nq) ; 
    auto h_regrid_flags = Kokkos::create_mirror_view(d_regrid_flags)      ;
    
    std::string ref_criterion = 
        params["amr"]["refinement_criterion"].as<std::string>() ;
    auto varname = params["amr"]["refinement_criterion_var"].as<std::string>() ;
    bool var_is_aux = get_param<bool>("amr", "refinement_criterion_var_is_aux"); 
    auto varidx = get_variable_index(varname, var_is_aux) ; 
    ASSERT(varidx>=0, "Index of variable " << varname << " not found.") ; 
    auto& criterion_view = var_is_aux ? aux : state ; 
    auto u = Kokkos::subview(criterion_view, VEC( Kokkos::ALL() 
                                                , Kokkos::ALL() 
                                                , Kokkos::ALL() )
                                        , varidx
                                        , Kokkos::ALL() ) ; 
    if( ref_criterion == "FLASH_second_deriv") {
        double eps = params["amr"]["FLASH_criterion_eps"].as<double>() ; 
        amr::flash_second_deriv_criterion<decltype(u)> kernel{ u, eps } ; 
        evaluate_regrid_criterion(
                d_regrid_flags
                , kernel ) ;
    } else if ( ref_criterion == "simple_threshold" ) {
        amr::simple_threshold_criterion<decltype(u)> kernel{ u } ; 
        evaluate_regrid_criterion(
                d_regrid_flags
                , kernel) ;
    } else if ( ref_criterion == "gradient" ) {
        amr::gradient_criterion<decltype(u)> kernel{ u } ; 
        evaluate_regrid_criterion(
                d_regrid_flags
                , kernel) ;
    } else if ( ref_criterion == "shear" ) { 
        auto vx = Kokkos::subview(aux, VEC( Kokkos::ALL() 
                                        , Kokkos::ALL() 
                                        , Kokkos::ALL() )
                                        , static_cast<size_t>(ZVECX_)
                                        , Kokkos::ALL() ) ; 
        auto vy = Kokkos::subview(aux, VEC( Kokkos::ALL() 
                                        , Kokkos::ALL() 
                                        , Kokkos::ALL() )
                                        , static_cast<size_t>(ZVECY_)
                                        , Kokkos::ALL() ) ; 
    #ifdef GRACE_3D
        auto vz = Kokkos::subview(aux, VEC( Kokkos::ALL() 
                                        , Kokkos::ALL() 
                                        , Kokkos::ALL() )
                                        , static_cast<size_t>(ZVECZ_)
                                        , Kokkos::ALL() ) ; 
    #endif 
        amr::shear_criterion<decltype(vx)> kernel{ VEC(vx,vy,vz) } ; 
        evaluate_regrid_criterion( d_regrid_flags
                                , kernel) ;
    } else {
        ERROR("Unsupported refinement criterion.") ; 
    }
    /* copy flags from device to host */ 
    Kokkos::deep_copy(h_regrid_flags, d_regrid_flags) ; 
    for( size_t iq=0UL; iq<amr::get_local_num_quadrants(); ++iq)
    {
        auto quad = amr::get_quadrant(iq) ;
        int flag = INVALID_STATE ; 
        if ( h_regrid_flags(iq) == REFINE ) { 
            flag = REFINE ; 
        } else if ( h_regrid_flags(iq) == COARSEN ) {
            flag = (quad.level() - 1 >= quad.get_user_long()) ? COARSEN : DEFAULT_STATE ; 
        } else {
            flag = DEFAULT_STATE ; 
        }
        quad.set_user_int(
            flag            
        ) ; 
    }
}

void regrid_transaction_t::execute_host_side_regrid() {
    auto const grace_maxlevel = get_param<int>("amr","max_refinement_level") ; 
    /******************************************************************************************/
    /* Call to p4est_refine                                                                   */  
    /* The arguments are:                                                                     */
    /* p4est_t* p4est   --> The forest object pointer.                                        */
    /* refine_recursive --> Wether we allow for recursive refinement (never).                 */
    /* maxlevel         --> Maximum allowed refinement level (parameter).                     */
    /* refine_fn        --> Function called on each quadrant to determine                     */
    /*                      whether it should be refined (see amr_flags.hh).                  */
    /* init_fn          --> Function to initialize new quadrants.                             */
    /* replace_fn       --> Function to modify the new quadrants.                             */
    /******************************************************************************************/ 
    p4est_refine_ext( amr::forest::get().get() 
                    , 0, grace_maxlevel 
                    , amr::refine_cback
                    , amr::initialize_quadrant 
                    , amr::set_quadrant_flag ) ; 
    /******************************************************************************************/
    /* Call to p4est_coarsen                                                                  */
    /* The arguments are:                                                                     */
    /* p4est_t* p4est    --> The forest object pointer.                                       */
    /* coarsen_recursive --> Wether we allow for recursive coarsening (never).                */
    /* callback_orphans  --> Allow passing orphan nodes into coarsen_fn.                      */
    /* coarsen_fn        --> Function called on each quadrant family to determine             */
    /*                       whether it should be coarsened (see amr_flags.hh).               */
    /* init_fn           --> Function to initialize new quadrants.                            */
    /* replace_fn        --> Function to modify the new quadrants.                            */
    /******************************************************************************************/
    p4est_coarsen_ext( amr::forest::get().get() 
                    , 0, 0 
                    , amr::coarsen_cback
                    , amr::initialize_quadrant 
                    , amr::set_quadrant_flag ) ; 
    /******************************************************************************************/
    /* Call to p4est_balance                                                                  */
    /* This ensures the grid is 2:1 balanced.                                                 */
    /******************************************************************************************/
    p4est_balance_ext( amr::forest::get().get() 
                    , P4EST_CONNECT_FULL
                    , amr::initialize_quadrant 
                    , amr::set_quadrant_flag ) ;
    /******************************************************************************************/
    /* Now we know the quad count                                                             */
    /******************************************************************************************/
    nq_regrid = amr::get_local_num_quadrants() ; 
    /******************************************************************************************/
    /* Finally we fill the *outgoing *incoming vectors                                        */
    /******************************************************************************************/
    coarsen_incoming.clear() ; coarsen_outgoing.clear() ; 
    refine_incoming.clear() ; refine_outgoing.clear() ; 
    keep_incoming.clear(); keep_outgoing.clear() ; 
    size_t iq_new{0UL}, iq_old{0UL} ; 
    while( iq_new < nq_regrid ) {
        quadrant_t quadrant = amr::get_quadrant(iq_new) ; 
        int flag = 
            quadrant.get_user_int() ; 
        if ( (flag == DEFAULT_STATE) or (flag==REFINE) or (flag==COARSEN) ) {
            keep_incoming.push_back(iq_new++) ;  keep_outgoing.push_back(iq_old++) ; 
        } else if ( flag == NEED_PROLONGATION ) {
            refine_outgoing.push_back(iq_old) ; 
            iq_old++ ; 
            for( int ichild=0; ichild<P4EST_CHILDREN; ++ichild){
                refine_incoming.push_back(static_cast<int>(iq_new)) ; 
                iq_new++ ;
            }
        } else if ( flag == NEED_RESTRICTION ) {
            coarsen_incoming.push_back(iq_new) ; 
            iq_new++ ; 
            for( int ichild=0; ichild<P4EST_CHILDREN; ++ichild){
                coarsen_outgoing.push_back(static_cast<int>(iq_old)) ; 
                iq_old++ ;
            }
        } else {
            ERROR("Invalid state " << flag << " for quadrant " << iq_new << '\n') ;
        }

    }
    /******************************************************************************************/
    /*                       Resize variable arrays, we use state_p                           */
    /*                       as scratch space                                                 */
    /******************************************************************************************/
    auto& state_swap = variable_list::get().getscratch() ;
    Kokkos::realloc(state_swap   , VEC( nx + 2*ngz 
                                ,      ny + 2*ngz 
                                ,      nz + 2*ngz )
                                , nvars_cc
                                , nq_regrid                          
    ) ; 
    auto& sstate_swap = variable_list::get().getstaggeredscratch() ;
    sstate_swap.realloc(nx,ny,nz,ngz,nq_regrid,nvars_fs,nvars_es,nvars_cs);
}

void regrid_transaction_t::execute_partition() {
    auto& state = variable_list::get().getstate() ; 
    auto& state_swap = variable_list::get().getscratch() ;
    auto& sstate = variable_list::get().getstaggeredstate() ; 
    auto& sstate_swap = variable_list::get().getstaggeredscratch() ;
    /******************************************************************************************/
    /* Start transfer                                                                         */
    /******************************************************************************************/
    Kokkos::realloc( state, VEC(  nx + 2*ngz 
                                , ny + 2*ngz 
                                , nz + 2*ngz )
                            ,   nvars_cc
                            ,   nq_final 
                    ) ;
    sstate.realloc(nx,ny,nz,ngz,nq_final,nvars_fs,nvars_es,nvars_cs) ; 
    size_t  quadrant_data_size {(nx+2*ngz)*(ny+2*ngz)*(nz+2*ngz)*nvars_cc}
          , quadrant_data_size_fs{(nx+2*ngz+1)*(ny+2*ngz)*(nz+2*ngz)*nvars_fs} ; 
    auto context = grace_transfer_fixed_begin<decltype(state)> (
                      new_glob_qoffsets.data() 
                    , old_glob_qoffsets.data()
                    , parallel::get_comm_world() 
                    , parallel::GRACE_PARTITION_TAG 
                    , state_swap
                    , state
                    , sstate_swap 
                    , sstate 
                    , quadrant_data_size 
                    , quadrant_data_size_fs
            ) ;
    // now we can refill the coordinates and so on while we wait 
    auto& idx = variable_list::get().getinvspacings()  ;
    auto& dx = variable_list::get().getspacings()    ;
    auto& coords = variable_list::get().getcoords()  ; 
    auto& vol = variable_list::get().getvolumes()    ;
    auto& staggered_coords = variable_list::get().getstaggeredcoords() ; 
    Kokkos::resize( coords      ,   GRACE_NSPACEDIM
                                ,   nq_final 
                                 ) ;
    Kokkos::realloc( idx        , GRACE_NSPACEDIM
                                ,   nq_final 
                                 ) ;
    Kokkos::realloc(  dx        , GRACE_NSPACEDIM
                                ,   nq_final 
                                 ) ;
    Kokkos::realloc( vol        , VEC(  nx + 2*ngz 
                                      , ny + 2*ngz 
                                      , nz + 2*ngz )
                                ,  nq_final 
                                 ) ;
    staggered_coords.realloc(VEC(nx,ny,nz),ngz,nq_final) ; 
    fill_cell_coordinates(coords, idx, dx, vol,staggered_coords) ;
    // realloc aux as well 
    int nvars_aux = grace::variables::get_n_auxiliary() ; 
    auto& aux = variable_list::get().getaux() ; 
    GRACE_TRACE("Resizing aux array {} aux vars registered, new quad count {}", nvars_aux, nq_final) ; 
    Kokkos::realloc( aux        ,   VEC(  nx + 2*ngz 
                                        , ny + 2*ngz 
                                        , nz + 2*ngz )
                                ,   nvars_aux
                                ,   nq_final 
                                ) ;  
    // and also the fluxes 
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ; 
    int nvars_hrsc = grace::variables::get_n_hrsc() ; 
    GRACE_TRACE("Resizing flux array {} flux vars registered", nvars_hrsc) ; 
    Kokkos::realloc( fluxes
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , nvars_hrsc 
                   , GRACE_NSPACEDIM
                   , nq_final 
                   ) ; 
    // and the emf 
    auto& emf = grace::variable_list::get().getemfarray() ; 
    auto& vbar = grace::variable_list::get().getvbararray() ; 
    Kokkos::realloc( emf, VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz), GRACE_NSPACEDIM, nq_final) ;
    Kokkos::realloc( vbar, VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz), 4, GRACE_NSPACEDIM, nq_final) ;
    // now wait 
    grace_transfer_fixed_end(context) ;  
}; 

void regrid_transaction_t::cleanup() {
    // fixme is this necessary? 
    auto& state = variable_list::get().getstate() ; 
    auto& state_swap = variable_list::get().getscratch() ;
    auto& sstate = variable_list::get().getstaggeredstate() ; 
    auto& sstate_swap = variable_list::get().getstaggeredscratch() ;
    /******************************************************************************************/
    /*                                Copy state to scratch                                   */
    /******************************************************************************************/
    Kokkos::realloc( state_swap ,   VEC(  nx + 2*ngz 
                                        , ny + 2*ngz 
                                        , nz + 2*ngz )
                                ,   nvars_cc
                                ,   nq_final 
                                ) ;
    Kokkos::deep_copy(state_swap, state) ; 
    sstate_swap.realloc(nx,ny,nz,ngz,nq_final,nvars_fs,nvars_es,nvars_cs) ; 
    deep_copy(sstate_swap,sstate) ;  

    set_quadrants_to_default(); 
}; 

void regrid_transaction_t::partition_grid() {
    /******************************************************************************************/
    /*                      Partition the new forest in parallel                              */
    /*                      we store global quadrant offsets, then                            */
    /*                      partition the forest, transfer state data                         */
    /*                      asynchronously, and realloc other fields                          */
    /*                      in the meanwhile. Coordinates are recomputed                      */
    /*                      but the auxiliary fields are left empty.                          */
    /******************************************************************************************/
    old_glob_qoffsets = amr::get_global_quadrant_offsets() ;
    /******************************************************************************************/
    /*                                    Partition forest                                    */
    /******************************************************************************************/
    size_t transfer_count = p4est_partition_ext( forest::get().get()
                                            , 0
                                            , nullptr  ) ; 
    new_glob_qoffsets = amr::get_global_quadrant_offsets() ; 
    /******************************************************************************************/
    /* Now we know the quad count                                                             */
    /******************************************************************************************/
    nq_final = amr::get_local_num_quadrants() ; 
}

void regrid_transaction_t::build_task_list() {
    using namespace Kokkos ; 
    /*****************************************************************/
    // get high order p/r coefficients 
    std::vector<double> ho_prolong_coefficients,ho_restrict_coefficients ;
    grace::detail::fill_fifth_order_prolongation_coefficients(ho_prolong_coefficients) ; 
    grace::detail::fill_fifth_order_restriction_coefficients(ho_restrict_coefficients) ; 
    // get list of vars for lo and ho p/r
    std::vector<size_t> high_order_interp_varlist, low_order_interp_varlist; 
    for(int ivar=0; ivar<N_EVOL_VARS; ++ivar){
        auto interp_kind = variables::get_interp_type(ivar) ; 
        if ( interp_kind ==  var_amr_interp_t::INTERP_SECOND_ORDER) {
            low_order_interp_varlist.push_back(static_cast<size_t>(ivar)) ; 
        } else if (  interp_kind ==  var_amr_interp_t::INTERP_FOURTH_ORDER ) {
            high_order_interp_varlist.push_back(static_cast<size_t>(ivar)) ; 
        } else {
            ERROR("Unrecognised prolongation/restriction operator requested for var " << ivar) ; 
        }
    }
    /*****************************************************************/
    task_id_t task_counter{0UL} ; 
    std::unordered_set<task_id_t> prolong_fs_dependencies ;
    /*****************************************************************/
    // scratch space has been realloc'd and is now empty,
    // we use state as the source of the data to be copied / interpolated
    auto& state_src = variable_list::get().getstate() ; 
    auto& sstate_src = variable_list::get().getstaggeredstate() ; 
    auto& state_dst = variable_list::get().getscratch() ; 
    auto& sstate_dst = variable_list::get().getstaggeredscratch() ;
    /*****************************************************************/ 
    auto& stream_pool = device_stream_pool::get() ; 
    auto& copy_stream = stream_pool.next() ; 
    auto& interp_stream = stream_pool.next() ;
    /*****************************************************************/
    auto nprocs = parallel::mpi_comm_size() ; 
    auto rank = parallel::mpi_comm_rank() ; 
    /*****************************************************************/
    // first: mpi transfers
    // these vectors contain for each rank tid_x, tid_y, tid_z 
    /*****************************************************************/
    #define INSERT_MPI_TASKS(axis,idx)\
    if ( recvcounts_##axis[r] > 0 ){\
            task_list.push_back(\
                std::make_unique<mpi_task_t>(\
                    make_mpi_recv_task_regrid(\
                        r, _recv_fbuf_##axis, rdispls_##axis, recvcounts_##axis, task_counter\
                    )\
                ) \
            ) ; \
            mpi_recv_tid[r][idx] = task_list.back()->task_id ;\
    }\
    if ( sendcounts_##axis[r] > 0) {\
        task_list.push_back(\
                std::make_unique<mpi_task_t>(\
                    make_mpi_send_task_regrid(\
                        r, _send_fbuf_##axis, sdispls_##axis, sendcounts_##axis, task_counter\
                    )\
                ) \
            ) ; \
            mpi_send_tid[r][idx] = task_list.back()->task_id ;\
    }
    std::vector<std::array<task_id_t,3>> mpi_send_tid(nprocs), mpi_recv_tid(nprocs) ; 
    for( int r=0; r<nprocs; ++r) {
        INSERT_MPI_TASKS(x,0);
        INSERT_MPI_TASKS(y,1);
        INSERT_MPI_TASKS(z,2);
    }

    // local face copies
    #define INSERT_FCOPY_TASKS(axis)\
    task_list.push_back(\
        std::make_unique<gpu_task_t>(\
            make_copy_face<decltype(sstate_dst.face_staggered_fields_##axis)>(\
                sstate_dst.face_staggered_fields_##axis,\
                sstate_src.face_staggered_fields_##axis,\
                local_fine_face_##axis,\
                copy_stream,nvars_fs,task_counter\
            )\
        )\
    );\
    prolong_fs_dependencies.insert(task_list.back()->task_id);
    /*****************************************************************/
    /*****************************************************************/
    INSERT_FCOPY_TASKS(x);
    INSERT_FCOPY_TASKS(y);
    INSERT_FCOPY_TASKS(z);
    /*****************************************************************/
    // remote face copies: pack and unpack
    #define INSERT_FPACK_TASKS(axis,idx)\
    if( remote_fine_face_send_##axis[r].size() > 0)\
    task_list.push_back(\
        std::make_unique<gpu_task_t>(\
            make_pack_face<decltype(sstate_src.face_staggered_fields_##axis)>(\
                sstate_src.face_staggered_fields_##axis,\
                _send_fbuf_##axis,\
                remote_fine_face_send_##axis[r],\
                copy_stream,nvars_fs,r,mpi_send_tid[r][idx],task_counter,task_list\
            )\
        )\
    )
    #define INSERT_FUPACK_TASKS(axis,idx)\
    if( remote_fine_face_recv_##axis[r].size() > 0)\
    task_list.push_back(\
        std::make_unique<gpu_task_t>(\
            make_unpack_face<decltype(sstate_dst.face_staggered_fields_##axis)>(\
                sstate_dst.face_staggered_fields_##axis,\
                _recv_fbuf_##axis,\
                remote_fine_face_recv_##axis[r],\
                copy_stream,nvars_fs,r,mpi_recv_tid[r][idx],task_counter,task_list\
            )\
        )\
    );\
    prolong_fs_dependencies.insert(task_list.back()->task_id)

    for( int r=0; r<nprocs; ++r) {
        INSERT_FPACK_TASKS(x,0);
        INSERT_FPACK_TASKS(y,1);
        INSERT_FPACK_TASKS(z,2);
        INSERT_FUPACK_TASKS(x,0);
        INSERT_FUPACK_TASKS(y,1);
        INSERT_FUPACK_TASKS(z,2);
    }
    /********************************************************************************/
    #define INSERT_COPY(stag,dst,src,nv) \
    task_list.push_back(\
        std::make_unique<gpu_task_t>(\
            make_copy<stag,decltype(src)>(\
                dst,src,\
                keep_incoming,keep_outgoing,\
                copy_stream,nv,task_counter\
            )\
        )\
    ) 
    INSERT_COPY(STAG_CENTER,state_dst,state_src,nvars_cc) ; 
    INSERT_COPY(STAG_FACEX,sstate_dst.face_staggered_fields_x,sstate_src.face_staggered_fields_x,nvars_fs);
    INSERT_COPY(STAG_FACEY,sstate_dst.face_staggered_fields_y,sstate_src.face_staggered_fields_y,nvars_fs);
    INSERT_COPY(STAG_FACEZ,sstate_dst.face_staggered_fields_z,sstate_src.face_staggered_fields_z,nvars_fs);
    /**********************************************************************************/
    /**********************************************************************************/
    task_list.push_back(
        std::make_unique<gpu_task_t>(
            make_restrict(
                state_dst, state_src,
                coarsen_incoming, coarsen_outgoing,
                low_order_interp_varlist,
                high_order_interp_varlist,
                ho_restrict_coefficients,
                interp_stream, nvars_cc, task_counter 
            )
        )
    ) ; 
    // then: restrict 
    #define INSERT_RESTRICT(stag,dst,src) \
    task_list.push_back(\
        std::make_unique<gpu_task_t>(\
            make_div_free_restrict<stag,decltype(src)>(\
                dst,src,\
                coarsen_incoming,coarsen_outgoing,\
                interp_stream,nvars_fs,task_counter\
            )\
        )\
    ) ; 
    task_list.push_back(
        std::make_unique<gpu_task_t>(
            make_div_free_restrict<STAG_FACEX,decltype(sstate_dst.face_staggered_fields_x)>(
                sstate_dst.face_staggered_fields_x,
                sstate_src.face_staggered_fields_x,
                coarsen_incoming,coarsen_outgoing,
                interp_stream,nvars_fs,task_counter
            )
        )
    ) ;
    task_list.push_back(
        std::make_unique<gpu_task_t>(
            make_div_free_restrict<STAG_FACEY,decltype(sstate_dst.face_staggered_fields_x)>(
                sstate_dst.face_staggered_fields_y,
                sstate_src.face_staggered_fields_y,
                coarsen_incoming,coarsen_outgoing,
                interp_stream,nvars_fs,task_counter
            )
        )
    ) ;
    task_list.push_back(
        std::make_unique<gpu_task_t>(
            make_div_free_restrict<STAG_FACEZ,decltype(sstate_dst.face_staggered_fields_x)>(
                sstate_dst.face_staggered_fields_z,
                sstate_src.face_staggered_fields_z,
                coarsen_incoming,coarsen_outgoing,
                interp_stream,nvars_fs,task_counter
            )
        )
    ) ;

    // prolong cc 
    task_list.push_back(
        std::make_unique<gpu_task_t>(
            make_prolong(
                state_dst,
                state_src,
                refine_incoming,
                refine_outgoing,
                low_order_interp_varlist,
                high_order_interp_varlist,
                ho_prolong_coefficients,
                interp_stream,
                task_counter
            )
        )
    ); 
    for( auto tid: prolong_fs_dependencies ) GRACE_TRACE_DBG("Prolong dep {}", tid) ; 
    // prolong fs 
    task_list.push_back(
        std::make_unique<gpu_task_t>(
                make_div_free_prolong(
                sstate_dst,
                sstate_src,
                refine_incoming,
                refine_outgoing,
                have_fine_data_x,
                have_fine_data_y,
                have_fine_data_z,
                interp_stream,
                nvars_fs,
                task_counter,
                task_list,
                prolong_fs_dependencies
            )
        )
    ); 
    
}; 

}}