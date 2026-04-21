/**
 * @file grmhd_B_from_A.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2026-04-08
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <grace/physics/id/Avec_id.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/evolution/hrsc_evolution_system.hh>
#include <grace/evolution/refluxing.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/grmhd.hh>

#include <grace/config/config_parser.hh>
#include <Kokkos_Core.hpp>

#include <string>
namespace grace {
double compute_B_max(std::array<double,3> const& center, double const& radius)
{
    DECLARE_GRID_EXTENTS;
    using namespace grace ; 
    using namespace Kokkos ; 

    // Avec is stored here! 
    auto& stag_state = grace::variable_list::get().getstaggeredstate() ; 
    auto& idx     = grace::variable_list::get().getinvspacings() ;
    auto& aux = variable_list::get().getaux() ;
    auto& state = variable_list::get().getstate() ;
    auto& csys = grace::coordinate_system::get() ;
    auto dev_coords = csys.get_device_coord_system() ;

    double Bmax ; 
    double Bmax_glob ; 
    // Find B max within a certain radius of a certain point 
    parallel_reduce( GRACE_EXECUTION_TAG("ID","compute_Bmax")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q, double& BmaxL)
        {
            metric_array_t metric ; 
            FILL_METRIC_ARRAY(metric, state, q, VEC(i,j,k)) ;

            auto BnL = Kokkos::sqrt(metric.square_vec(
                {aux(VEC(i,j,k),BX_,q),aux(VEC(i,j,k),BY_,q),aux(VEC(i,j,k),BZ_,q)}
            ));
            // cell centered coords! 
            double xyz[3] ; 
            dev_coords.get_physical_coordinates(i,j,k,q,xyz) ; 
            double d = Kokkos::sqrt(
                SQR(xyz[0]-center[0]) + SQR(xyz[1]-center[1]) + SQR(xyz[2]-center[2])
            ) ; 
            if ( (d < radius) && (BnL > BmaxL) ) {
                BmaxL = BnL; 
            }
        }, Kokkos::Max<double>(Bmax) 
    ) ; 
    Kokkos::fence() ; // never know 
    parallel::mpi_allreduce(  &Bmax
                            , &Bmax_glob
                            , 1
                            , sc_MPI_MAX) ; 

    return Bmax_glob ; 
}

void compute_B_from_A() {
    DECLARE_GRID_EXTENTS;
    using namespace grace ; 
    using namespace Kokkos ; 
    // Avec is stored here! 
    auto& emf = grace::variable_list::get().getemfarray() ; 
    auto& stag_state = grace::variable_list::get().getstaggeredstate() ; 
    auto& idx     = grace::variable_list::get().getinvspacings() ;
    auto& aux = variable_list::get().getaux() ;
    auto& state = variable_list::get().getstate() ;
    auto& csys = grace::coordinate_system::get() ;
    auto dev_coords = csys.get_device_coord_system() ;
    
    // Bx 
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BX")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    // B^x = d/dy A^z - d/dz A^y
                    stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) = (
                          (emf(VEC(i  ,j+1,k  ),2,q) - emf(VEC(i  ,j  ,k  ),2,q)) * idx(1,q)
                        + (emf(VEC(i  ,j  ,k  ),1,q) - emf(VEC(i  ,j  ,k+1),1,q)) * idx(2,q)
                    ) ; 
                });
    // By
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BY")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {  
                    // B^y = d/dz A^x - d/dx A^z
                    stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) = (
                          (emf(VEC(i  ,j  ,k+1),0,q) - emf(VEC(i  ,j  ,k  ),0,q)) * idx(2,q)
                        + (emf(VEC(i  ,j  ,k  ),2,q) - emf(VEC(i+1,j  ,k  ),2,q)) * idx(0,q)
                    ) ; 
                });
    // Bz 
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_BZ")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz+1),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    // B^z = d/dx A^y - d/dy A^x
                    stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) = (
                          (emf(VEC(i+1,j  ,k  ),1,q) - emf(VEC(i  ,j  ,k  ),1,q)) * idx(0,q)
                        + (emf(VEC(i  ,j  ,k  ),0,q) - emf(VEC(i  ,j+1,k  ),0,q)) * idx(1,q)
                    ) ; 
                });
    // Set cell centered B in aux 
    parallel_for( GRACE_EXECUTION_TAG("ID","set_conservs_from_prims")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
        {
            metric_array_t metric ; 
            FILL_METRIC_ARRAY(metric, state, q, VEC(i,j,k)) ;
            /*************************************************/
            /*                    Set B                      */
            /*************************************************/ 
            // note here we reset B-center since it is outdated 
            auto Bx = Kokkos::subview(stag_state.face_staggered_fields_x,
                                    VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), static_cast<size_t>(BSX_), q) ; 
            auto By = Kokkos::subview(stag_state.face_staggered_fields_y,
                                    VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), static_cast<size_t>(BSY_), q) ; 
            auto Bz = Kokkos::subview(stag_state.face_staggered_fields_z,
                                    VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), static_cast<size_t>(BSZ_), q) ;
            aux(VEC(i,j,k),BX_,q) = 0.5 * (stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) + stag_state.face_staggered_fields_x(VEC(i+1,j,k),BSX_,q)) / metric.sqrtg();
            aux(VEC(i,j,k),BY_,q) = 0.5 * (stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) + stag_state.face_staggered_fields_y(VEC(i,j+1,k),BSY_,q)) / metric.sqrtg();
            aux(VEC(i,j,k),BZ_,q) = 0.5 * (stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) + stag_state.face_staggered_fields_z(VEC(i,j,k+1),BSZ_,q)) / metric.sqrtg();

            aux(VEC(i,j,k),BDIV_,q) = ( (Bx(VEC(i+1,j,k)) - Bx(VEC(i,j,k))) * idx(0,q) 
                                      + (By(VEC(i,j+1,k)) - By(VEC(i,j,k))) * idx(1,q)
                                      + (Bz(VEC(i,j,k+1)) - Bz(VEC(i,j,k))) * idx(2,q))/metric.sqrtg() ; 
        }
    );

}

template< typename Avec_id_t >
void set_A(
    std::array<double,3> const& center, double radius, size_t varidx,
    Avec_id_t A_id 
) 
{
    DECLARE_GRID_EXTENTS;
    using namespace grace ; 
    using namespace Kokkos ; 
    // Avec is stored here! 
    auto& emf = grace::variable_list::get().getemfarray() ; 
    auto& stag_state = grace::variable_list::get().getstaggeredstate() ; 
    auto& idx     = grace::variable_list::get().getinvspacings() ;
    auto& aux = variable_list::get().getaux() ;
    auto& csys = grace::coordinate_system::get() ;
    auto dev_coords = csys.get_device_coord_system() ;

    // Ax 
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_AX")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz+1),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    auto varv = Kokkos::subview(aux, Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(), varidx, q ) ; 
                    double val = 0. ;
                    int cnt = 0 ; 
                    for( int ii=0; ii<2; ++ii ) {
                        for( int jj=0; jj<2; ++jj) {
                            int i_idx = j - ii ; 
                            int j_idx = k - jj ; 
                            if ( (i_idx > 0) && (i_idx < nx+2*ngz) && (j_idx > 0) && (j_idx < ny+2*ngz) ){
                                val += varv(i,i_idx,j_idx) ; 
                                cnt ++ ; 
                            }
                                
                        }
                    }
                    // coords of edge! 
                    double ccoords[3] = {0,0.5,0.5} ; 
                    double xyz[3] ;
                    dev_coords.get_physical_coordinates(i,j,k,q,ccoords,xyz,1/*count gzs*/) ; 
                    xyz[0] -= center[0] ;
                    xyz[1] -= center[1] ; 
                    xyz[2] -= center[2] ; 
                    // check if we are within the ball 
                    double d = Kokkos::sqrt(
                        SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2]) 
                    ) ;
                    if ( d <= radius )
                        emf(VEC(i,j,k),0,q) = A_id.template get<0>({xyz[0],xyz[1],xyz[2]}, val/cnt); 
                }
    );

    // Ay
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_AY")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz+1),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    auto varv = Kokkos::subview(aux, Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(), varidx, q ) ; 
                    double val = 0. ;
                    int cnt = 0 ; 
                    for( int ii=0; ii<2; ++ii ) {
                        for( int jj=0; jj<2; ++jj) {
                            int i_idx = i - ii ; 
                            int j_idx = k - jj ; 
                            if ( (i_idx > 0) && (i_idx < nx+2*ngz) && (j_idx > 0) && (j_idx < ny+2*ngz) ){
                                val += varv(i_idx,j,j_idx) ; 
                                cnt ++ ; 
                            }
                                
                        }
                    }
                    // coords of edge! 
                    double ccoords[3] = {0.5,0.,0.5} ; 
                    double xyz[3] ;
                    dev_coords.get_physical_coordinates(i,j,k,q,ccoords,xyz,1/*count gzs*/) ; 
                    xyz[0] -= center[0] ;
                    xyz[1] -= center[1] ; 
                    xyz[2] -= center[2] ; 
                    // check if we are within the ball 
                    double d = Kokkos::sqrt(
                        SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2]) 
                    ) ;
                    if ( d <= radius )
                        emf(VEC(i,j,k),1,q) = A_id.template get<1>({xyz[0],xyz[1],xyz[2]}, val/cnt); 
                }
    );

    // Az
    parallel_for( GRACE_EXECUTION_TAG("ID","grmhd_ID_AZ")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz+1,ny+2*ngz+1,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    auto varv = Kokkos::subview(aux, Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(), varidx, q ) ; 
                    double val = 0. ;
                    int cnt = 0 ; 
                    for( int ii=0; ii<2; ++ii ) {
                        for( int jj=0; jj<2; ++jj) {
                            int i_idx = i - ii ; 
                            int j_idx = j - jj ; 
                            if ( (i_idx > 0) && (i_idx < nx+2*ngz) && (j_idx > 0) && (j_idx < ny+2*ngz) ){
                                val += varv(i_idx,j_idx,k) ; 
                                cnt ++ ; 
                            }
                                
                        }
                    }
                    // coords of edge! 
                    double ccoords[3] = {0.5,0.5,0.} ; 
                    double xyz[3] ;
                    dev_coords.get_physical_coordinates(i,j,k,q,ccoords,xyz,1/*count gzs*/) ; 
                    xyz[0] -= center[0] ;
                    xyz[1] -= center[1] ; 
                    xyz[2] -= center[2] ; 
                    // check if we are within the ball 
                    double d = Kokkos::sqrt(
                        SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2]) 
                    ) ;
                    if ( d <= radius )
                        emf(VEC(i,j,k),2,q) = A_id.template get<2>({xyz[0],xyz[1],xyz[2]}, val/cnt);
                }
    );

    // Now we exchange EMFs to ensure they are fully consistent at quadrant 
    // interfaces 
    auto context = reflux_fill_emf_buffers() ; 
    reflux_correct_emfs(context) ; 
    Kokkos::fence() ;

}

double find_auxmax(size_t vidx)
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ;
    using namespace Kokkos ;

    auto& aux   = variable_list::get().getaux() ; 
    double vmax_loc ; 
    double vmax ; 
    auto policy =
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
    parallel_reduce( GRACE_EXECUTION_TAG("IO","find_max_press_loc") 
                       , policy 
                       , KOKKOS_LAMBDA(VEC(int i, int j, int k), int q, double& vmaxl)
        {
            auto v = aux(i,j,k,vidx,q) ; 
            if ( v > vmaxl ) vmaxl = v ; 
        }, Max<double>(vmax_loc)) ; 
    parallel::mpi_allreduce( &vmax_loc
                            , &vmax
                            , 1
                            , sc_MPI_MAX) ; 
    return vmax ; 
}

void setup_confined_poloidal_B_field_single(std::array<double,3> const& center, double radius, double Btarget) 
{

    auto cutoff_var = get_param<std::string>("grmhd", "Avec_ID","cutoff_var") ;
    ASSERT(cutoff_var=="press" or cutoff_var=="rho", "Only pressure and density-based cutoff supported.") ; 

    size_t vidx = (cutoff_var=="rho") ? RHO_ : PRESS_ ; 
    auto vmax = find_auxmax(vidx) ; 

    double cut = get_param<double>("grmhd", "Avec_ID","cutoff_fact") * vmax ; 
    double n   = get_param<double>("grmhd", "Avec_ID","A_n") ;

    double Aphi_guess = 1 ; 
    Avec_poloidal_id_t Avec_id(Aphi_guess, cut, n) ; 

    set_A<Avec_poloidal_id_t>(center,radius,vidx,Avec_id) ; 
    compute_B_from_A(); 

    auto Bmax = compute_B_max(center,radius) ;

    auto corr_fact = Btarget/Bmax ; 
    Aphi_guess *= corr_fact ; 
    Avec_id._A_phi = Aphi_guess ; 
    set_A<Avec_poloidal_id_t>(center,radius,vidx,Avec_id) ; 
    compute_B_from_A(); 

    Bmax = compute_B_max(center,radius) ;
    GRACE_INFO("B field initialized, local maximum norm: {} at cell center", Bmax) ; 
}; 

void setup_confined_poloidal_B_field() {
    using namespace grace ;

    bool is_binary = get_param<bool>(
        "grmhd", "Avec_ID", "is_binary"
    ) ; 

    std::array<double,3> c1 ; 
    c1[0] = get_param<double>("grmhd", "Avec_ID", "x_c_1") ; 
    c1[1] = get_param<double>("grmhd", "Avec_ID", "y_c_1") ; 
    c1[2] = get_param<double>("grmhd", "Avec_ID", "z_c_1") ; 
    double r1 = get_param<double>("grmhd", "Avec_ID", "radius_1") ; 

    double B_target = get_param<double>("grmhd", "Avec_ID", "Bmax_target") ; 

    if (! is_binary ) {
        // could be star, could be disk, we don't care 
        GRACE_INFO("Setting up single domain poloidal field") ; 
        setup_confined_poloidal_B_field_single(c1,r1,B_target) ; 
    } else {
        GRACE_INFO("Setting up binary domain poloidal field") ; 
        std::array<double,3> c2 ; 
        c2[0] = get_param<double>("grmhd", "Avec_ID", "x_c_2") ; 
        c2[1] = get_param<double>("grmhd", "Avec_ID", "y_c_2") ; 
        c2[2] = get_param<double>("grmhd", "Avec_ID", "z_c_2") ; 
        double r2 = get_param<double>("grmhd", "Avec_ID", "radius_2") ;
        setup_confined_poloidal_B_field_single(c1,r1,B_target) ;  
        setup_confined_poloidal_B_field_single(c2,r2,B_target) ; 
    }
}
}