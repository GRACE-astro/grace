/**
 * @file scalar_output.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-22
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

#include <grace/IO/scalar_output.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/config/config_parser.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <Kokkos_Core.hpp>

#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace grace { namespace IO {

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
    using namespace grace ; 
    using namespace Kokkos  ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nq = amr::get_local_num_quadrants() ;

    size_t global_ncells = math::int_pow<3>(nx) * grace::amr::forest::get().get()->global_num_quadrants ;

    auto& state = variable_list::get().getstate()   ; 
    auto& aux   = variable_list::get().getaux()     ; 
    auto& dx = variable_list::get().getspacings() ; 

    auto& grace_runtime = grace::runtime::get() ; 

    /* First: compute minmax reductions */  
    auto const minmax_vars = grace_runtime.minmax_reduction_vars() ; 
    auto const minmax_aux  = grace_runtime.minmax_reduction_aux()  ; 

    for( auto const& vname: minmax_vars ) {
        GRACE_TRACE("Performing minmax reduction of variable {}", vname) ; 
        auto const vidx = get_variable_index(vname, false) ; 
        auto policy =
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(state, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        MinMaxScalar<double> res ; 
        parallel_reduce( GRACE_EXECUTION_TAG("IO","minmax_reduction_vars") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, MinMaxScalar<double>& lres)
        {
            double v = u(VEC(i,j,k), q);
            if (v < lres.min_val) lres.min_val = v;
            if (v > lres.max_val) lres.max_val = v;
        }, MinMax<double>(res)) ; 
        
        parallel::mpi_allreduce( &res.min_val
                               , &detail::_minmax_reduction_vars_results[vname].min_val
                               , 1
                               , sc_MPI_MIN) ; 
        parallel::mpi_allreduce( &res.max_val
                               , &detail::_minmax_reduction_vars_results[vname].max_val
                               , 1
                               , sc_MPI_MAX) ; 
        GRACE_TRACE("Min {}, Max {}", detail::_minmax_reduction_vars_results[vname].min_val, detail::_minmax_reduction_vars_results[vname].max_val) ; 
    }
    for( auto const& vname: minmax_aux ) {
        auto const vidx = get_variable_index(vname, true) ; 
        auto policy =
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(aux, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        MinMaxScalar<double> res ; 
        parallel_reduce( GRACE_EXECUTION_TAG("IO","minmax_reduction_aux") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, MinMaxScalar<double>& lres)
        {
            double v = u(VEC(i,j,k), q);
            if (v < lres.min_val) lres.min_val = v;
            if (v > lres.max_val) lres.max_val = v;
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
    auto const norm2_vars = grace_runtime.norm2_reduction_vars() ; 
    auto const norm2_aux  = grace_runtime.norm2_reduction_aux()  ; 

    for( auto const& vname: norm2_vars ) {
        GRACE_TRACE("Performing norm2 reduction of variable {}", vname) ; 
        auto const vidx = get_variable_index(vname, false) ; 
        auto policy =
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(state, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        double res{0.}; 
        parallel_reduce( GRACE_EXECUTION_TAG("IO","norm2_reduction_vars") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, double& lres)
        {
            lres += SQR(u(VEC(i,j,k),q)) ; 
        }, Sum<double>(res)) ; 
        
        parallel::mpi_allreduce( &res
                               , &detail::_norm2_reduction_vars_results[vname]
                               , 1
                               , sc_MPI_SUM) ; 
        
        detail::_norm2_reduction_vars_results[vname] = std::sqrt(detail::_norm2_reduction_vars_results[vname]/global_ncells) ; 
        GRACE_TRACE("norm {}", detail::_norm2_reduction_vars_results[vname]) ; 
    } 
    for( auto const& vname: norm2_aux ) {
        auto const vidx = get_variable_index(vname, true) ; 
        GRACE_TRACE("Performing norm2 reduction of variable {} index {}", vname, vidx) ; 
        auto policy =
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(aux, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        double res{0.}; 
        parallel_reduce( GRACE_EXECUTION_TAG("IO","norm2_reduction_aux") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, double& lres)
        {
            lres += SQR(u(VEC(i,j,k),q)) ; 
        }, Sum<double>(res)) ; 
        
        parallel::mpi_allreduce( &res
                               , &detail::_norm2_reduction_aux_results[vname]
                               , 1
                               , sc_MPI_SUM) ; 
        detail::_norm2_reduction_aux_results[vname] = std::sqrt(detail::_norm2_reduction_aux_results[vname]/global_ncells) ; 
        GRACE_TRACE("norm {}", std::sqrt(detail::_norm2_reduction_aux_results[vname])) ; 
    }
    /* Then: compute integral reductions */ 
    auto const integral_vars = grace_runtime.integral_reduction_vars() ; 
    auto const integral_aux  = grace_runtime.integral_reduction_aux()  ; 

    for( auto const& vname: integral_vars ) {
        GRACE_TRACE("Performing integral reduction of variable {}", vname) ; 
        auto const vidx = get_variable_index(vname, false) ; 
        auto policy =
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(state, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        double res{0.}; 
        parallel_reduce( GRACE_EXECUTION_TAG("IO","integral_reduction_vars") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, double& lres)
        {
            double cvol = dx(0,q) * dx(1,q) * dx(2,q) ; 
            lres += u(VEC(i,j,k),q) * cvol ; 
        }, Sum<double>(res)) ; 
        
        parallel::mpi_allreduce( &res
                               , &detail::_integral_reduction_vars_results[vname]
                               , 1
                               , sc_MPI_SUM) ; 
        GRACE_TRACE("Integral {}", res) ; 
    } 
    for( auto const& vname: integral_aux ) {
        auto const vidx = get_variable_index(vname, true) ; 
        auto policy =
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        auto const u = subview(aux, VEC(ALL(),ALL(),ALL()), vidx, ALL()) ; 
        double res{0.}; 
        parallel_reduce( GRACE_EXECUTION_TAG("IO","integral_reduction_aux") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, double& lres)
        {
            double cvol = dx(0,q) * dx(1,q) * dx(2,q) ; 
            lres += u(VEC(i,j,k),q) * cvol ; 
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
    using namespace grace ; 
    using namespace Kokkos  ; 

    GRACE_VERBOSE("Performing scalar output iteration at {}.", grace::get_iteration() ) ; 

    if( parallel::mpi_comm_rank() == 0 ) {
        
    auto& grace_runtime = grace::runtime::get() ; 

    std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ; 

    size_t const iter = grace_runtime.iteration() ; 
    double const time = grace_runtime.time()      ; 

    static constexpr size_t width = 20 ;

    auto const out_minmax_vars = grace_runtime.scalar_output_minmax_vars() ; 
    auto const out_minmax_aux  = grace_runtime.scalar_output_minmax_aux()  ; 
    for( auto const& vname: out_minmax_vars ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_min.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::scientific << std::setprecision(15) ; 
        outfile << std::left << iter << '\t'
                << std::left << time << '\t' 
                << std::left << detail::_minmax_reduction_vars_results[vname].min_val << '\n' ; 
        std::string const pfname_max = grace_runtime.scalar_io_basename() + vname + "_max.dat" ;
        std::filesystem::path fname_max = bdir /  pfname_max ; 
        std::ofstream outfile_max(fname_max.string(),std::ios::app) ; 
        outfile_max << std::scientific << std::setprecision(15) ; 
        outfile_max << std::left << iter << '\t'
                    << std::left << time << '\t' 
                    << std::left << detail::_minmax_reduction_vars_results[vname].max_val << '\n' ;
    }
    for( auto const& vname: out_minmax_aux ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_min.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::scientific << std::setprecision(15) ; 
        outfile << std::left << iter << '\t'
                << std::left << time << '\t' 
                << std::left << detail::_minmax_reduction_aux_results[vname].min_val << '\n' ; 
        std::string const pfname_max = grace_runtime.scalar_io_basename() + vname + "_max.dat" ;
        std::filesystem::path fname_max = bdir /  pfname_max ; 
        std::ofstream outfile_max(fname_max.string(),std::ios::app) ; 
        outfile_max << std::scientific << std::setprecision(15) ; 
        outfile_max << std::left << iter << '\t'
                    << std::left << time << '\t' 
                    << std::left << detail::_minmax_reduction_aux_results[vname].max_val << '\n' ; 
    }

    auto const out_norm2_vars = grace_runtime.scalar_output_norm2_vars() ; 
    auto const out_norm2_aux  = grace_runtime.scalar_output_norm2_aux()  ; 
    for( auto const& vname: out_norm2_vars ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_norm2.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::scientific << std::setprecision(15) ; 
        outfile << std::left << iter << '\t'
                << std::left << time << '\t'
                << std::left << detail::_norm2_reduction_vars_results[vname] << '\n' ; 
    }
    for( auto const& vname: out_norm2_aux ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_norm2.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ;
        outfile << std::scientific << std::setprecision(15) ;  
        outfile << std::left << iter << '\t'
                << std::left << time << '\t' 
                << std::left << detail::_norm2_reduction_aux_results[vname] << '\n' ; 
    }

    auto const out_integral_vars = grace_runtime.scalar_output_integral_vars() ; 
    auto const out_integral_aux  = grace_runtime.scalar_output_integral_aux()  ; 
    for( auto const& vname: out_integral_vars ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_integral.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::scientific << std::setprecision(15) ; 
        outfile << std::left << iter << '\t'
                << std::left << time << '\t' 
                << std::left << detail::_integral_reduction_vars_results[vname] << '\n' ; 
    }
    for( auto const& vname: out_integral_aux ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_integral.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::scientific << std::setprecision(15) ; 
        outfile << std::left << iter << '\t'
                << std::left << time << '\t'
                << std::left << detail::_integral_reduction_aux_results[vname] << '\n' ; 
    }

    }
    parallel::mpi_barrier() ;
    Kokkos::Profiling::popRegion() ; 
}


void initialize_output_files() {
    if( parallel::mpi_comm_rank() == 0 ) {
    auto& grace_runtime = grace::runtime::get() ; 
    static constexpr const size_t width = 20 ; 
    std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ; 
    auto const out_minmax_vars = grace_runtime.scalar_output_minmax_vars() ; 
    auto const out_minmax_aux  = grace_runtime.scalar_output_minmax_aux()  ; 
    for( auto const& vname: out_minmax_vars ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_min.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ; 
        std::string const pfname_max = grace_runtime.scalar_io_basename() + vname + "_max.dat" ;
        std::filesystem::path fname_max = bdir /  pfname_max ; 
        std::ofstream outfile_max(fname_max.string(),std::ios::app) ;
        outfile_max << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ; 
    }
    for( auto const& vname: out_minmax_aux ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_min.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ; 
        std::string const pfname_max = grace_runtime.scalar_io_basename() + vname + "_max.dat" ;
        std::filesystem::path fname_max = bdir /  pfname_max ; 
        std::ofstream outfile_max(fname_max.string(),std::ios::app) ;
        outfile_max << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ; 
    }
    auto const out_norm2_vars = grace_runtime.scalar_output_norm2_vars() ; 
    auto const out_norm2_aux  = grace_runtime.scalar_output_norm2_aux()  ; 
    for( auto const& vname: out_norm2_vars ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_norm2.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;
    }
    for( auto const& vname: out_norm2_aux ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_norm2.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ; 
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;
    }
    auto const out_integral_vars = grace_runtime.scalar_output_integral_vars() ; 
    auto const out_integral_aux  = grace_runtime.scalar_output_integral_aux()  ; 
    for( auto const& vname: out_integral_vars ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_integral.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ;
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;
    }
    for( auto const& vname: out_integral_aux ) {
        std::string const pfname = grace_runtime.scalar_io_basename() + vname + "_integral.dat" ;
        std::filesystem::path fname = bdir /  pfname ; 
        std::ofstream outfile(fname.string(),std::ios::app) ;
        outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;
    }
    auto const perf_metric = grace_runtime.performance_metric() ;
    if ( perf_metric != perf_metric_display::PERF_NONE ) {
        auto write_header = [&](const std::string& fname_suffix) {
            std::string pfname = grace_runtime.scalar_io_basename() + fname_suffix ;
            std::filesystem::path fname = bdir / pfname ;
            std::ofstream outfile(fname.string(), std::ios::app) ;
            outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;
        } ;
        if ( perf_metric == perf_metric_display::PERF_MZPH || perf_metric == perf_metric_display::PERF_BOTH )
            write_header("performance_mzph.dat") ;
        if ( perf_metric == perf_metric_display::PERF_ZCPS || perf_metric == perf_metric_display::PERF_BOTH )
            write_header("performance_zcps.dat") ;
    }
    }
    parallel::mpi_barrier() ;
}

void info_output() {

    using namespace grace ;

    int64_t iter = grace::get_iteration() ;
    int64_t initial_iter = grace::get_initial_iteration() ;
    double  time = grace::get_simulation_time() ;
    double  initial_time = grace::get_initial_simulation_time() ;
    double walltime = grace::get_evol_runtime() ;
    double rate = (time-initial_time) / walltime * 3.6e03 ;

    int64_t outinfo_every = grace::config_parser::get()["IO"]["info_output_every"].as<int64_t>() * 10 ;

    auto& grace_runtime = grace::runtime::get() ;
    auto const perf_metric = grace_runtime.performance_metric() ;

    // compute performance metrics
    int64_t nx,ny,nz ;
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ;
    size_t global_ncells = math::int_pow<3>(nx) * grace::amr::forest::get().get()->global_num_quadrants ;
    int64_t iters_done = iter - initial_iter ;
    double mzph = 0.0, zcps = 0.0 ;
    if ( walltime > 0.0 && iters_done > 0 ) {
        zcps = static_cast<double>(iters_done) * static_cast<double>(global_ncells) / walltime ;
        mzph = zcps * 3.6e03 / 1.0e06 ;
    }

    std::ostringstream os ;
    os << std::scientific << std::setprecision(5) ;

    auto const max_vars = grace_runtime.info_output_max_vars() ;
    auto const max_aux  = grace_runtime.info_output_max_aux()  ;
    auto const min_vars = grace_runtime.info_output_min_vars() ;
    auto const min_aux  = grace_runtime.info_output_min_aux()  ;
    auto const norm_vars = grace_runtime.info_output_norm2_vars() ;
    auto const norm_aux  = grace_runtime.info_output_norm2_aux()  ;

    // compute column width from longest label
    auto label_width = [](const std::vector<std::string>& v, const std::string& suffix) -> size_t {
        size_t w = 0 ;
        for(auto const& x: v)
            w = std::max(w, x.size() + 1 + suffix.size()) ;
        return w ;
    } ;
    size_t width = 15 ;
    width = std::max(width, label_width(max_vars, "[max]")) ;
    width = std::max(width, label_width(max_aux,  "[max]")) ;
    width = std::max(width, label_width(min_vars, "[min]")) ;
    width = std::max(width, label_width(min_aux,  "[min]")) ;
    width = std::max(width, label_width(norm_vars,"[norm2]")) ;
    width = std::max(width, label_width(norm_aux, "[norm2]")) ;
    width += 2 ; // padding

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
        if ( perf_metric == perf_metric_display::PERF_MZPH || perf_metric == perf_metric_display::PERF_BOTH )
            os << std::left << std::setw(width) << "Mzc/h" ;
        if ( perf_metric == perf_metric_display::PERF_ZCPS || perf_metric == perf_metric_display::PERF_BOTH )
            os << std::left << std::setw(width) << "zc/s" ;
        APPEND_OUTPUT(max_vars,"[max]") ;
        APPEND_OUTPUT(max_aux, "[max]") ;
        APPEND_OUTPUT(min_vars,"[min]") ;
        APPEND_OUTPUT(min_aux, "[min]") ;
        APPEND_OUTPUT(norm_vars,"[norm2]") ;
        APPEND_OUTPUT(norm_aux, "[norm2]") ;
        os << '\n' ;
        GRACE_INFO(os.str().c_str()) ;
        os.str("");
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
    if ( perf_metric == perf_metric_display::PERF_MZPH || perf_metric == perf_metric_display::PERF_BOTH )
        os << std::left << std::setw(width) << mzph ;
    if ( perf_metric == perf_metric_display::PERF_ZCPS || perf_metric == perf_metric_display::PERF_BOTH )
        os << std::left << std::setw(width) << zcps ;
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
    GRACE_INFO(os.str().c_str()) ;

    // write performance scalar output files
    if ( perf_metric != perf_metric_display::PERF_NONE && parallel::mpi_comm_rank() == 0 ) {
        std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ;
        if( not std::filesystem::exists(bdir) ) {
            std::filesystem::create_directories(bdir) ;
        }
        auto write_perf_scalar = [&](const std::string& fname_suffix, double value) {
            std::string pfname = grace_runtime.scalar_io_basename() + fname_suffix ;
            std::filesystem::path fname = bdir / pfname ;
            std::ofstream outfile(fname.string(), std::ios::app) ;
            outfile << std::scientific << std::setprecision(15) ;
            outfile << std::left << iter << '\t'
                    << std::left << time << '\t'
                    << std::left << value << '\n' ;
        } ;
        if ( perf_metric == perf_metric_display::PERF_MZPH || perf_metric == perf_metric_display::PERF_BOTH )
            write_perf_scalar("performance_mzph.dat", mzph) ;
        if ( perf_metric == perf_metric_display::PERF_ZCPS || perf_metric == perf_metric_display::PERF_BOTH )
            write_perf_scalar("performance_zcps.dat", zcps) ;
    }
}

}}