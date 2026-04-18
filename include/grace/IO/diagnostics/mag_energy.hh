/**
 * @file black_hole_diagnostics.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-17
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
#ifndef GRACE_IO_EM_ENERGY_DIAGNOSTICS_HH
#define GRACE_IO_EM_ENERGY_DIAGNOSTICS_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_indices.hh>

#include <grace/utils/metric_utils.hh>

#include <grace/utils/device_vector.hh>

#include <grace/system/grace_runtime.hh>
#include <grace/coordinates/coordinate_systems.hh>

#include <grace/config/config_parser.hh>
#include <grace/utils/reductions.hh>

#include <grace/physics/grmhd_helpers.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>

namespace grace {


struct em_energy_diagnostic {

    //**************************************************************************************************
    em_energy_diagnostic() {
        out_every = get_param<int>("mhd_diagnostics", "compute_energy_every") ;
        auto& grace_runtime = grace::runtime::get() ; 
        std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ;  
        std::string pfname = "E_em.dat"; 
        fpath = bdir / pfname ; 
    }
    //**************************************************************************************************
    void compute_and_write() {
        auto& grace_runtime = grace::runtime::get() ; 
        size_t const iter = grace_runtime.iteration() ; 
        if ( (out_every > 0) && (iter%out_every == 0) ) {
            compute() ; 
            output() ; 
        }
    }
    //**************************************************************************************************
    void initialize_files() {
        static constexpr const size_t width = 20 ; 
        int proc = parallel::mpi_comm_rank() ; 
        if ( !std::filesystem::exists(fpath) and (proc == 0) and (out_every > 0)) {
            std::ofstream outfile(fpath.string());
            outfile << std::fixed << std::setprecision(15) ; 
            outfile << std::left << std::setw(width) << "Iteration" 
                    << std::left << std::setw(width) << "Time" 
                    << std::left << std::setw(width) << "E_tot" 
                    << std::left << std::setw(width) << "E_pol"
                    << std::left << std::setw(width) << "E_tor" << '\n' ;  
        }
        
        parallel::mpi_barrier() ; 
    }
    //**************************************************************************************************
    void compute()
    {
        DECLARE_GRID_EXTENTS ; 

        using namespace grace ;
        using namespace Kokkos ; 

        auto& state = grace::variable_list::get().getstate() ; 
        auto& aux = grace::variable_list::get().getaux() ; 
        auto& dx = grace::variable_list::get().getspacings() ; 
        auto dc = coordinate_system::get().get_device_coord_system() ; 

        array_sum_t<double,3> em_integrals ; 

        MDRangePolicy<Rank<4>> policy(
            {ngz,ngz,ngz,0},
            {nx+ngz,ny+ngz,nz+ngz,nq}
        ) ; 

        parallel_reduce(
            GRACE_EXECUTION_TAG("DIAG", "compute_magnetic_energy"),
            policy,
            KOKKOS_LAMBDA(int const i, int const j, int const k, int q, array_sum_t<double,3>& intloc) {

                metric_array_t metric ; 
                FILL_METRIC_ARRAY(metric,state,q,VEC(i,j,k)) ;

                grmhd_prims_array_t prims ;     
                FILL_PRIMS_ARRAY_ZVEC(
                    prims, aux, q, i,j,k
                ) ; 

                double xyz[3] ; 
                dc.get_physical_coordinates(i,j,k,q,xyz) ; 

                double const sqrtg         = metric.sqrtg()      ;

                std::array<double,3> z{{prims[ZXL],prims[ZYL],prims[ZZL]}} ; 
                std::array<double,3> B{{prims[BXL],prims[BYL],prims[BZL]}} ; 

                double const z2 = metric.contract_vec_vec(z,z);
                double const W2 = 1 + z2 ;
                double const W  = Kokkos::sqrt(W2) ;

                std::array<double,3> v = {
                    z[0]/W, z[1]/W, z[2]/W 
                } ; 

                double Bv = metric.contract_vec_vec(
                    v,B
                ) ; 
                double B2 = metric.contract_vec_vec(
                    B,B
                ) ; 

                // construct e_phi 
                std::array<double,4> lU = {0,-xyz[1],xyz[0],0} ;

                double u0U = W/metric.alp();
                std::array<double,4> uU = {
                    u0U, 
                    u0U*(metric.alp()*v[0] - metric.beta(0)),
                    u0U*(metric.alp()*v[1] - metric.beta(1)),
                    u0U*(metric.alp()*v[2] - metric.beta(2))
                } ; 
                // 1) make orthogonal to u 
                auto const ldotu = metric.contract_4dvec_4dvec(lU,uU) ; 
                for( int ii=0; ii<4; ++ii ) lU[ii] += ldotu * uU[ii] ; 
                // 2) normalize 
                auto lnorm = Kokkos::sqrt(
                    metric.contract_4dvec_4dvec(lU,lU) 
                ) ; 
                if (lnorm < 1e-15) lnorm = 1e-15 ; 
                for( int ii=0; ii<4; ++ii ) lU[ii] /= lnorm ; 


                // Now construct smallb 
                std::array<double,4> smallb ; 
                //B^i u_i / alpha 
                smallb[0] = u0U * Bv ; 
                // b^i = (B^i + (B^i u_i) u^i) / (W)
                for( int ii=0; ii<3; ++ii) {
                    smallb[ii+1] = (B[ii]/W + Bv * uU[ii+1]) ; 
                }
                // smallb^2 
                double const smallb2 = B2/SQR(W) + SQR(Bv) ;  

                // and now project onto ephi 
                auto const smallbphi = metric.contract_4dvec_4dvec(lU,smallb) ;
                // and poloidal  
                std::array<double,4> smallb_pol ; 
                for( int ii=0; ii<4; ++ii) smallb_pol[ii] = smallb[ii] - smallbphi * lU[ii] ;
                double const smallb_pol2 =  metric.contract_4dvec_4dvec(smallb_pol,smallb_pol) ;
                // compute energies 
                /***************************************************************/
                // Fluid frame energy 
                // T^{\mu\nu}_{\rm MHD} u_\mu u_\nu = 1/2 b^2
                /***************************************************************/
                double cell_vol = dx(0,q) * dx(1,q) * dx(2,q) ; 
                intloc.data[0] += cell_vol * metric.sqrtg() * W * 0.5 * smallb2        ; 
                intloc.data[1] += cell_vol * metric.sqrtg() * W * 0.5 * smallb_pol2    ; 
                intloc.data[2] += cell_vol * metric.sqrtg() * W * 0.5 * SQR(smallbphi) ; 
            }, Kokkos::Sum<array_sum_t<double,3>>(em_integrals)
        ) ;
        double em_integrals_glob[3] ; 
        MPI_Allreduce(
            em_integrals.data,          // sendbuf
            em_integrals_glob,          // recvbuf
            3,                          // count
            MPI_DOUBLE,                 // datatype
            MPI_SUM,                    // op
            MPI_COMM_WORLD              // comm
        );
        E     = em_integrals_glob[0] ; 
        Epol  = em_integrals_glob[1] ; 
        Etor  = em_integrals_glob[2] ; 
    }
    //**************************************************************************************************
    private: 
    //**************************************************************************************************
    void output() {
        int proc = parallel::mpi_comm_rank() ; 
        if ( proc == 0 ) { 
            auto& grace_runtime = grace::runtime::get() ; 
            size_t const iter = grace_runtime.iteration() ; 
            double const time = grace_runtime.time()      ;
            std::ofstream outfile(fpath.string(), std::ios::app) ;
            outfile << std::fixed << std::setprecision(15) ; 
            outfile << std::left << iter << '\t'
                << std::left << time << '\t' 
                << std::left << E << '\t'
                << std::left << Epol << '\t'
                << std::left << Etor << '\n' ; 
            
        }
        parallel::mpi_barrier() ; 
    }
    
    //**************************************************************************************************
    double E, Epol, Etor    ;        //!< Magnetic energies 
    //double Ef, Epolf, Etorf ;        //!< Magnetic energies wrt fluid frame
    int out_every ;               //!< Output frequency 
    std::filesystem::path fpath ; //!< Output file path 
    //**************************************************************************************************
} ; 

}


#endif 