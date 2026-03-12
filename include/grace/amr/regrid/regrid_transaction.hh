/**
 * @file regrid_transaction.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief This file contains the class responsible for orchestrating a regrid.
 * @version 0.1
 * @date 2025-10-29
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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
#ifndef GRACE_AMR_REGRID_TRANSACTION_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/task_queue.hh>
#include <grace/amr/ghostzone_kernels/ghost_array.hh>
#include <grace/data_structures/variables.hh>

#include <grace/amr/amr_ghosts.hh>
#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

struct fine_face_data_desc_t {
    int axis ; //!< Face axis, also stag direction
    int side ; //!< 0 lower 1 upper 
    size_t qid_ghost  ; //!< ghost idx
    size_t qid_local  ; //!< local index in this forest 
    size_t qid_remote ; //!< local index in owner rank's forest
    size_t which_tree ; //!< tree index in owner rank's forest
    int8_t fid_local ;  //!< Local face index 
    int8_t fid_remote ; //<! Remote face index 
} ;

struct fine_interface_desc_t {
    size_t qid_src  ; //!< ghost idx
    size_t qid_dst  ; //!< local index in this forest 
    uint8_t fid_src ; //!< face idx local
    uint8_t fid_dst ; //!< face idx remote
    bool is_remote  ; //!< other side is remote   
} ; 

struct fine_interface_desc_device_t {
    readonly_view_t<int> qid_src       ; //!< ghost idx
    readonly_view_t<int> qid_dst       ; //!< local index in this forest 
    readonly_view_t<uint8_t> fid_src   ; //!< face idx local
    readonly_view_t<uint8_t> fid_dst   ; //!< face idx remote
    readonly_view_t<uint8_t> is_remote ; //!< other side is remote 
} ; 

struct regrid_transaction_t {

    regrid_transaction_t() : nq_init(amr::get_local_num_quadrants()) 
    {
        std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
        ngz = grace::amr::get_n_ghosts() ; 
        nvars_cc = variables::get_n_evolved() ; 
        nvars_fs = variables::get_n_evolved_face_staggered() ; 
        nvars_es = nvars_cs = 0 ; 

        evaluate_criterion() ; 
        changed = execute_host_side_regrid() ; 

        // if nothing changed we can finish here
        // but first we reset quads to default 
        // status 
        if (!changed) {
            set_quadrants_to_default() ; 
            return ; 
        }

        // otherwise we 
        // 1) tag quads for prolong / restrict and resize scratch space 
        prepare_device_side() ; 
        // 2) build MPI buffers for magnetic field transfer 
        build_buffers() ;
        // 3) build task list for device side regrid 
        build_task_list() ; 

        task_queue.clear() ; 
        task_queue.reserve(task_list.size()) ; 

        for( auto& t: task_list) {
            
            runtime_task_view rtv ; 
            rtv.t = t.get() ; 
            ASSERT( rtv.t != nullptr, "Dangling pointer! ") ; 
            rtv.pending = t->_dependencies.size() ; 
            if ( rtv.pending == 0 ) {
                t -> status = status_id_t::READY ;
                task_queue.ready.push_back(t -> task_id) ;  
            }

            task_queue.rt.push_back(std::move(rtv)) ; 
        }
        GRACE_VERBOSE("Task queue constructed. {} tasks are ready to run.", task_queue.ready.size()) ;
    } 

    void execute() {
        // nothing to do if the grid 
        // hasn't changed 
        if (!changed) return ; 
        /* first run the data tasks */
        task_queue.run(view_alias_t{}/*dummy argument*/) ; 
        Kokkos::fence() ; 
        parallel::mpi_barrier() ; 
        // now we can partition the grid 
        partition_grid() ; 
        execute_partition() ; 
        Kokkos::fence() ; 
        parallel::mpi_barrier() ; 
        cleanup() ; 
        /* all done! */
    }; 

    size_t get_nq_init() {return nq_init;}
    size_t get_nq_final() {return nq_final;}

    bool grid_has_changed() { return changed ; }
    
    private:
    
    //! Task list for the regrid
    std::vector<std::unique_ptr<task_t>> task_list ;
    executor task_queue ; 

    //! Send / receive buffers for staggered data
    face_buffer_t _send_fbuf_x, _recv_fbuf_x
                , _send_fbuf_y, _recv_fbuf_y
                , _send_fbuf_z, _recv_fbuf_z ; 

    //! Quad ids of incoming and outgoing quads 
    std::vector<size_t> refine_incoming, coarsen_incoming, keep_incoming ; 
    std::vector<size_t> refine_outgoing, coarsen_outgoing, keep_outgoing ;
    std::vector<int64_t> old_glob_qoffsets, new_glob_qoffsets ;  
    //! For MPI: indices of local and remote fine faces that need to be exchanged 
    std::vector<fine_interface_desc_t> local_fine_face_x, local_fine_face_y, local_fine_face_z;
    std::vector<std::vector<fine_interface_desc_t>> remote_fine_face_send_x, remote_fine_face_send_y, remote_fine_face_send_z
                                                  , remote_fine_face_recv_x, remote_fine_face_recv_y, remote_fine_face_recv_z;
    //! For MPI: counts and displacements of sends and receives 
    std::vector<int> send_off_x, send_off_y, send_off_z, recv_off_x, recv_off_y, recv_off_z ; 
    std::vector<int> send_size_x, send_size_y, send_size_z, recv_size_x, recv_size_y, recv_size_z ; 
    //! Flags indicating whether fine data is available on faces, if available, it will be copied instead of prolonged.
    std::vector<std::array<int8_t,2>> have_fine_data_x, have_fine_data_y, have_fine_data_z ; 
    fine_interface_desc_device_t fine_face_descs ; 
    //! Minimum allowed refinement level for each quadrant 
    std::vector<long> min_levels ; 
    //! Number of quadrants: before regrid, after, and after partition
    size_t nq_init, nq_regrid, nq_final ; 
    //! Number of cells and ghost cells 
    size_t nx,ny,nz, ngz ; 
    //! Number of variables in each staggering 
    size_t nvars_cc, nvars_fs, nvars_es, nvars_cs ; 
    //! Has the grid changed? 
    bool changed ; 


    //! Evaluate criterion and write flags into 
    //! quad's user_int 
    void evaluate_criterion() ;
    //! Using the info stored in quadrants 
    //! execute the regrid on the p4est, then
    //! resize scratch states in preparation 
    //! for data operations and extract quad 
    //! indices of outgoing and incoming quads.
    //! Returns true or false depending on whether
    //! the grid was modified. 
    bool execute_host_side_regrid() ;
    //! Prepare structures necessary to perform
    //! the actual regrid 
    void prepare_device_side() ; 
    //! Resize state and transfer data for parallel
    //! partition
    void execute_partition() ;
    //! Call p4est_partition, store qoffsets
    void partition_grid();
    //! Figure out what local and remote 
    //! fine face data to copy, allocate 
    //! transfer buffers
    void build_buffers();
    //! Construct the task list 
    void build_task_list() ;
    //! Reset quadrant flags, reallocate scratch space
    void cleanup() ;
    /**
     * @brief Set all quadrants to default state.
     * \cond grace_detail 
     * \ingroup amr 
     */
    void set_quadrants_to_default()  
    {
        for(int itree=forest::get().first_local_tree();
                itree<=forest::get().last_local_tree();
                ++itree) 
        {
            auto quadrants = forest::get().tree(itree).quadrants() ; 
            for( int iquad=0; iquad<quadrants.size(); ++iquad) {
                quadrant_t quad{ &(quadrants[iquad]) } ;
                quad.set_regrid_flag(DEFAULT_STATE) ; 
            }
        }
    }
} ; 

} } /* namespace grace::amr */

#endif /* GRACE_AMR_REGRID_TRANSACTION_HH */
