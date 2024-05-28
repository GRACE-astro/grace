/**
 * @file scalar_output.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-22
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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

#include <thunder_config.h>

#include <thunder/IO/scalar_output.hh>
#include <thunder/system/thunder_system.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/parallel/mpi_wrappers.hh>

#include <Kokkos_Core.hpp>

#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace thunder { namespace IO {

namespace detail {

std::map<std::string,minmax_res_t<double>> _minmax_reduction_vars_results ;
std::map<std::string,minmax_res_t<double>> _minmax_reduction_aux_results  ;
std::map<std::string,double> _norm2_reduction_vars_results    ;
std::map<std::string,double> _norm2_reduction_aux_results     ;
std::map<std::string,double> _integral_reduction_vars_results ;
std::map<std::string,double> _integral_reduction_aux_results  ;

}


void compute_reductions() {
    Kokkos::Profiling::pushRegion("Scalar reductions") ; 
    using namespace thunder ; 
    using namespace Kokkos  ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nq = amr::get_local_num_quadrants() ;

    auto& state = variable_list::get().getstate()   ; 
    auto& aux   = variable_list::get().getaux()     ; 
    auto& cvols = variable_list::get().getvolumes() ; 

    auto& thunder_runtime = thunder::runtime::get() ; 

    /* First: compute minmax reductions */  
    auto const minmax_vars = thunder_runtime.minmax_reduction_vars() ; 
    auto const minmax_aux  = thunder_runtime.minmax_reduction_aux()  ; 

    for( auto const& vname: minmax_vars ) {
        THUNDER_TRACE("Performing minmax reduction of variable {}", vname) ; 
        auto const vidx = get_variable_index(vname, false) ; 
        auto policy =
            MDRangePolicy<Rank<THUNDER_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(state, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        MinMaxScalar<double> res ; 
        parallel_reduce( THUNDER_EXECUTION_TAG("IO","minmax_reduction_vars") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, MinMaxScalar<double>& lres)
        {
            lres.min_val = lres.min_val < u(VEC(i,j,k),q) ? lres.min_val    : u(VEC(i,j,k),q) ; 
            lres.max_val = lres.max_val < u(VEC(i,j,k),q) ? u(VEC(i,j,k),q) : lres.max_val    ; 
        }, MinMax<double>(res)) ; 
        
        parallel::mpi_allreduce( &res.min_val
                               , &detail::_minmax_reduction_vars_results[vname].min_val
                               , 1
                               , sc_MPI_MIN) ; 
        parallel::mpi_allreduce( &res.max_val
                               , &detail::_minmax_reduction_vars_results[vname].max_val
                               , 1
                               , sc_MPI_MAX) ; 
        THUNDER_TRACE("Min {}, Max {}", res.min_val, res.max_val) ; 
    }
    for( auto const& vname: minmax_aux ) {
        auto const vidx = get_variable_index(vname, true) ; 
        auto policy =
            MDRangePolicy<Rank<THUNDER_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(aux, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        MinMaxScalar<double> res ; 
        parallel_reduce( THUNDER_EXECUTION_TAG("IO","minmax_reduction_aux") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, MinMaxScalar<double>& lres)
        {
            lres.min_val = lres.min_val < u(VEC(i,j,k),q) ? lres.min_val    : u(VEC(i,j,k),q) ; 
            lres.max_val = lres.max_val < u(VEC(i,j,k),q) ? u(VEC(i,j,k),q) : lres.max_val    ; 
        }, MinMax<double>(res)) ; 
        
        parallel::mpi_allreduce( &res.min_val
                               , &detail::_minmax_reduction_aux_results[vname].min_val
                               , 1
                               , sc_MPI_MIN) ; 
        parallel::mpi_allreduce( &res.max_val
                               , &detail::_minmax_reduction_aux_results[vname].max_val
                               , 1
                               , sc_MPI_MAX) ; 
    }

    /* Then: compute norm2 reductions */ 
    auto const norm2_vars = thunder_runtime.norm2_reduction_vars() ; 
    auto const norm2_aux  = thunder_runtime.norm2_reduction_aux()  ; 

    for( auto const& vname: norm2_vars ) {
        THUNDER_TRACE("Performing norm2 reduction of variable {}", vname) ; 
        auto const vidx = get_variable_index(vname, false) ; 
        auto policy =
            MDRangePolicy<Rank<THUNDER_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(state, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        double res ; 
        parallel_reduce( THUNDER_EXECUTION_TAG("IO","norm2_reduction_vars") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, double& lres)
        {
            lres += math::int_pow<2>(u(VEC(i,j,k),q)) ; 
        }, Sum<double>(res)) ; 
        
        parallel::mpi_allreduce( &res
                               , &detail::_norm2_reduction_vars_results[vname]
                               , 1
                               , sc_MPI_SUM) ; 
        detail::_norm2_reduction_vars_results[vname] = std::sqrt(detail::_norm2_reduction_vars_results[vname]) ; 
        THUNDER_TRACE("norm {}", std::sqrt(res)) ; 
    } 
    for( auto const& vname: norm2_aux ) {
        auto const vidx = get_variable_index(vname, true) ; 
        THUNDER_TRACE("Performing norm2 reduction of variable {} index {}", vname, vidx) ; 
        auto policy =
            MDRangePolicy<Rank<THUNDER_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(aux, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        double res ; 
        parallel_reduce( THUNDER_EXECUTION_TAG("IO","norm2_reduction_aux") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, double& lres)
        {
            lres += math::int_pow<2>(u(VEC(i,j,k),q)) ; 
        }, Sum<double>(res)) ; 
        
        parallel::mpi_allreduce( &res
                               , &detail::_norm2_reduction_aux_results[vname]
                               , 1
                               , sc_MPI_SUM) ; 
        detail::_norm2_reduction_aux_results[vname] = std::sqrt(detail::_norm2_reduction_aux_results[vname]) ; 
        THUNDER_TRACE("norm {}", std::sqrt(res)) ; 
    }
    /* Then: compute integral reductions */ 
    auto const integral_vars = thunder_runtime.integral_reduction_vars() ; 
    auto const integral_aux  = thunder_runtime.integral_reduction_aux()  ; 

    for( auto const& vname: integral_vars ) {
        THUNDER_TRACE("Performing integral reduction of variable {}", vname) ; 
        auto const vidx = get_variable_index(vname, false) ; 
        auto policy =
            MDRangePolicy<Rank<THUNDER_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(state, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        double res ; 
        parallel_reduce( THUNDER_EXECUTION_TAG("IO","integral_reduction_vars") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, double& lres)
        {
            lres += u(VEC(i,j,k),q) * cvols(VEC(i,j,k),q) ; 
        }, Sum<double>(res)) ; 
        
        parallel::mpi_allreduce( &res
                               , &detail::_integral_reduction_vars_results[vname]
                               , 1
                               , sc_MPI_SUM) ; 
        THUNDER_TRACE("Integral {}", res) ; 
    } 
    for( auto const& vname: integral_aux ) {
        auto const vidx = get_variable_index(vname, true) ; 
        auto policy =
            MDRangePolicy<Rank<THUNDER_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(aux, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        double res ; 
        parallel_reduce( THUNDER_EXECUTION_TAG("IO","integral_reduction_aux") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, double& lres)
        {
            lres += u(VEC(i,j,k),q) * cvols(VEC(i,j,k),q) ; 
        }, Sum<double>(res)) ; 
        
        parallel::mpi_allreduce( &res
                               , &detail::_integral_reduction_aux_results[vname]
                               , 1
                               , sc_MPI_SUM) ; 
    }
    Kokkos::Profiling::popRegion() ; 
}

void write_scalar_output() {
    Kokkos::Profiling::pushRegion("Scalar output") ;
    using namespace thunder ; 
    using namespace Kokkos  ; 

    THUNDER_VERBOSE("Performing scalar output iteration at {}.", thunder::get_iteration() ) ; 

    if( parallel::mpi_comm_rank() == 0 ) {
        
    auto& thunder_runtime = thunder::runtime::get() ; 

    std::filesystem::path bdir = thunder_runtime.scalar_io_basepath() ; 

    size_t const iter = thunder_runtime.iteration() ; 
    double const time = thunder_runtime.time()      ; 

    static constexpr size_t width = 20 ;

    auto const out_minmax_vars = thunder_runtime.scalar_output_minmax_vars() ; 
    auto const out_minmax_aux  = thunder_runtime.scalar_output_minmax_aux()  ; 
    for( auto const& vname: out_minmax_vars ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_min.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::fixed << std::setprecision(15) ; 
        outfile << std::left << std::setw(width) << iter 
                << std::left << std::setw(width) << time 
                << std::left << std::setw(width) << detail::_minmax_reduction_vars_results[vname].min_val << '\n' ; 
        std::string const pfname_max = thunder_runtime.scalar_io_basename() + vname + "_max.dat" ;
        std::filesystem::path fname_max = bdir /  pfname_max ; 
        std::ofstream outfile_max(fname_max.string(),std::ios::app) ; 
        outfile_max << std::fixed << std::setprecision(15) ; 
        outfile_max << std::left << std::setw(width) << iter 
                    << std::left << std::setw(width) << time 
                    << std::left << std::setw(width) << detail::_minmax_reduction_vars_results[vname].max_val << '\n' ;
    }
    for( auto const& vname: out_minmax_aux ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_min.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::fixed << std::setprecision(15) ; 
        outfile << std::left << std::setw(width) << iter 
                << std::left << std::setw(width) << time 
                << std::left << std::setw(width) << detail::_minmax_reduction_aux_results[vname].min_val << '\n' ; 
        std::string const pfname_max = thunder_runtime.scalar_io_basename() + vname + "_max.dat" ;
        std::filesystem::path fname_max = bdir /  pfname_max ; 
        std::ofstream outfile_max(fname_max.string(),std::ios::app) ; 
        outfile_max << std::fixed << std::setprecision(15) ; 
        outfile_max << std::left << std::setw(width) << iter 
                    << std::left << std::setw(width) << time 
                    << std::left << std::setw(width) << detail::_minmax_reduction_aux_results[vname].max_val << '\n' ; 
    }

    auto const out_norm2_vars = thunder_runtime.scalar_output_norm2_vars() ; 
    auto const out_norm2_aux  = thunder_runtime.scalar_output_norm2_aux()  ; 
    for( auto const& vname: out_norm2_vars ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_norm2.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::fixed << std::setprecision(15) ; 
        outfile << std::left << std::setw(width) << iter 
                << std::left << std::setw(width) << time 
                << std::left << std::setw(width) << detail::_norm2_reduction_vars_results[vname] << '\n' ; 
    }
    for( auto const& vname: out_norm2_aux ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_norm2.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ;
        outfile << std::fixed << std::setprecision(15) ;  
        outfile << std::left << std::setw(width) << iter 
                << std::left << std::setw(width) << time 
                << std::left << std::setw(width) << detail::_norm2_reduction_aux_results[vname] << '\n' ; 
    }

    auto const out_integral_vars = thunder_runtime.scalar_output_integral_vars() ; 
    auto const out_integral_aux  = thunder_runtime.scalar_output_integral_aux()  ; 
    for( auto const& vname: out_integral_vars ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_integral.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::fixed << std::setprecision(15) ; 
        outfile << std::left << std::setw(width) << iter 
                << std::left << std::setw(width) << time 
                << std::left << std::setw(width) << detail::_integral_reduction_vars_results[vname] << '\n' ; 
    }
    for( auto const& vname: out_integral_aux ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_integral.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::fixed << std::setprecision(15) ; 
        outfile << std::left << std::setw(width) << iter 
                << std::left << std::setw(width) << time 
                << std::left << std::setw(width) << detail::_integral_reduction_aux_results[vname] << '\n' ; 
    }

    }
    parallel::mpi_barrier() ;
    Kokkos::Profiling::popRegion() ; 
}


void initialize_output_files() {
    if( parallel::mpi_comm_rank() == 0 ) {
    auto& thunder_runtime = thunder::runtime::get() ; 
    static constexpr const size_t width = 20 ; 
    std::filesystem::path bdir = thunder_runtime.scalar_io_basepath() ; 
    auto const out_minmax_vars = thunder_runtime.scalar_output_minmax_vars() ; 
    auto const out_minmax_aux  = thunder_runtime.scalar_output_minmax_aux()  ; 
    for( auto const& vname: out_minmax_vars ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_min.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ; 
        std::string const pfname_max = thunder_runtime.scalar_io_basename() + vname + "_max.dat" ;
        std::filesystem::path fname_max = bdir /  pfname_max ; 
        std::ofstream outfile_max(fname_max.string(),std::ios::app) ;
        outfile_max << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ; 
    }
    for( auto const& vname: out_minmax_aux ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_min.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ; 
        std::string const pfname_max = thunder_runtime.scalar_io_basename() + vname + "_max.dat" ;
        std::filesystem::path fname_max = bdir /  pfname_max ; 
        std::ofstream outfile_max(fname_max.string(),std::ios::app) ;
        outfile_max << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ; 
    }
    auto const out_norm2_vars = thunder_runtime.scalar_output_norm2_vars() ; 
    auto const out_norm2_aux  = thunder_runtime.scalar_output_norm2_aux()  ; 
    for( auto const& vname: out_norm2_vars ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_norm2.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;
    }
    for( auto const& vname: out_norm2_aux ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_norm2.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;
    }
    auto const out_integral_vars = thunder_runtime.scalar_output_integral_vars() ; 
    auto const out_integral_aux  = thunder_runtime.scalar_output_integral_aux()  ; 
    for( auto const& vname: out_integral_vars ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_integral.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ;
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;
    }
    for( auto const& vname: out_integral_aux ) {
        std::string const pfname = thunder_runtime.scalar_io_basename() + vname + "_integral.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ;
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;
    }
    }
    parallel::mpi_barrier() ;
}

void info_output() {

    using namespace thunder ;

    int64_t iter = thunder::get_iteration() ; 
    double  time = thunder::get_simulation_time() ;
    double walltime = thunder::get_total_runtime() ; 
    double rate = time / walltime * 3.6e03 ; 

    int64_t outinfo_every = thunder::config_parser::get()["IO"]["info_output_every"].as<int64_t>() * 10 ; 

    std::ostringstream os ; 
    os << std::fixed << std::setprecision(5) ; 

    auto& thunder_runtime = thunder::runtime::get() ; 
    auto const max_vars = thunder_runtime.info_output_max_vars() ; 
    auto const max_aux  = thunder_runtime.info_output_max_aux()  ; 
    auto const min_vars = thunder_runtime.info_output_min_vars() ; 
    auto const min_aux  = thunder_runtime.info_output_min_aux()  ;
    auto const norm_vars = thunder_runtime.info_output_norm2_vars() ; 
    auto const norm_aux  = thunder_runtime.info_output_norm2_aux()  ; 

    static constexpr const size_t width = 15 ; 

    #define APPEND_OUTPUT(vvect,s)    \
    for(auto const& x: vvect) {       \
        std::string tmp = x + " " ;   \
        tmp += s ;                    \
        os << std::left << std::setw(width) << tmp ; \
    }

    
    if( iter%outinfo_every == 0 ) {
        os << std::left << std::setw(width) << "Iteration" 
           << std::left << std::setw(width) <<  "Time" 
           << std::left << std::setw(width) << "M/h" ; 
        APPEND_OUTPUT(max_vars,"[max]") ;
        APPEND_OUTPUT(max_aux, "[max]") ; 
        APPEND_OUTPUT(min_vars,"[min]") ;
        APPEND_OUTPUT(min_aux, "[min]") ; 
        APPEND_OUTPUT(norm_vars,"[norm2]") ;
        APPEND_OUTPUT(norm_aux, "[norm2]") ; 
        os << '\n' ; 
        THUNDER_INFO(os.str().c_str()) ;
        os.str("");  // Reset content to empty string
        os.clear();
    }

    #undef APPEND_OUTPUT
    #define APPEND_OUTPUT(vvect,dvect)\
    for(auto const& x: vvect) {       \
        os << std::left << std::setw(width) << dvect[x] ;      \
    }
    #define APPEND_MIN(vvect,dvect)      \
    for(auto const& x: vvect) {          \
        os << std::left << std::setw(width) << dvect[x].min_val ; \
    }
    #define APPEND_MAX(vvect,dvect)      \
    for(auto const& x: vvect) {          \
        os << std::left << std::setw(width) << dvect[x].max_val ; \
    }

    os << std::left << std::setw(width) << iter << std::left << std::setw(width) << time << std::left << std::setw(width) << rate ; 
    APPEND_MAX(max_vars,detail::_minmax_reduction_vars_results) ; 
    APPEND_MAX(max_aux,detail::_minmax_reduction_aux_results)   ; 
    APPEND_MIN(min_vars,detail::_minmax_reduction_vars_results) ; 
    APPEND_MIN(min_aux,detail::_minmax_reduction_aux_results)   ;
    APPEND_OUTPUT(norm_vars,detail::_norm2_reduction_vars_results) ; 
    APPEND_OUTPUT(norm_aux,detail::_norm2_reduction_aux_results) ; 
    os << '\n' ; 
    #undef APPEND_MAX
    #undef APPEND_MIN 
    #undef APPEND_OUTPUT
    THUNDER_INFO(os.str().c_str()) ; 
}

}}