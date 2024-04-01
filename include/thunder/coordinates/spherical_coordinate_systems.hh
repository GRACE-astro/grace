/**
 * @file coordinate_systems.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-26
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

#ifndef THUNDER_AMR_SPHERICAL_COORDINATE_SYSTEMS_HH 
#define THUNDER_AMR_SPHERICAL_COORDINATE_SYSTEMS_HH

#include <thunder_config.h>

#include <thunder/amr/p4est_headers.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/coordinates/spherical_device_inlines.hh>

#include<array>

namespace thunder { 
typedef void (*coord_transform_t) (double,double*,double*,double*,double*,double*) ; 
typedef void (*coord_transfer_t) ( double          // L 
                                 , double*,double* // Fa Sa
                                 , double*,double* // Fb Sb 
                                 , double*,double* // Ra Rb 
                                 , double*,double* ) ; 


namespace detail {
THUNDER_DEVICE coord_transform_t l2p[2*P4EST_FACES+1], p2l[2*P4EST_FACES+1] ; 
THUNDER_DEVICE coord_transfer_t gl2l[(2*P4EST_FACES+1)*P4EST_FACES] ;
}

struct spherical_device_coordinate_system_impl_t 
{
    spherical_device_coordinate_system_impl_t(
          Kokkos::View<double*, thunder::default_space> _params
        , Kokkos::View<double**, thunder::default_space> _rotmat
        , Kokkos::View<double**, thunder::default_space> _invrotmat
    ) : grid_params_(_params)
      , rotation_matrices_(_rotmat)
      , inverse_rotation_matrices_(_invrotmat)
    {} ; 

    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    get_physical_coordinates( int itree, double * l_coords, double * p_coords) const  
    {
        double F[2] = {
            ((itree-1)/P4EST_FACES==0) * grid_params_(1) + ((itree-1)/P4EST_FACES==1) * grid_params_(5),
            ((itree-1)/P4EST_FACES==0) * grid_params_(2) + ((itree-1)/P4EST_FACES==1) * grid_params_(6)
        } ; 
        double S[2] = {
            ((itree-1)/P4EST_FACES==0) * grid_params_(3) + ((itree-1)/P4EST_FACES==1) * grid_params_(7),
            ((itree-1)/P4EST_FACES==0) * grid_params_(4) + ((itree-1)/P4EST_FACES==1) * grid_params_(8)
        } ;
        int const midx = (itree>0) * (itree-1)%P4EST_FACES; // just to prevent indexing at -1 
        auto const R = Kokkos::subview(rotation_matrices_, Kokkos::ALL(),midx) ;  
        detail::l2p[itree](grid_params_(0),F,S,R.data(),l_coords,p_coords) ; 
    }

    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    get_logical_coordinates( int itree, double * p_coords, double * l_coords) const 
    {
        double F[2] = {
            ((itree-1)/P4EST_FACES==0) * grid_params_(1) + ((itree-1)/P4EST_FACES==1) * grid_params_(5),
            ((itree-1)/P4EST_FACES==0) * grid_params_(2) + ((itree-1)/P4EST_FACES==1) * grid_params_(6)
        } ; 
        double S[2] = {
            ((itree-1)/P4EST_FACES==0) * grid_params_(3) + ((itree-1)/P4EST_FACES==1) * grid_params_(7),
            ((itree-1)/P4EST_FACES==0) * grid_params_(4) + ((itree-1)/P4EST_FACES==1) * grid_params_(8)
        } ;
        int const midx = (itree>0) * (itree-1)%P4EST_FACES; // just to prevent indexing at -1 
        auto const R = Kokkos::subview(inverse_rotation_matrices_, Kokkos::ALL(),midx) ;  
        detail::p2l[itree](grid_params_(0),F,S,R.data(),p_coords,l_coords) ; 
 
    }

    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    transfer_coordinates( int itree_a, int itree_b, 
                          int face_a, int face_b,
                          double * l_coords_a, double * l_coords_b  ) const 
    {
        if( itree_a==itree_b ){
            EXPR(
                l_coords_b[0] = l_coords_a[0];,
                l_coords_b[1] = l_coords_a[1];,
                l_coords_b[2] = l_coords_a[2];
            )
            return ; 
        }
        int const midx_a = (itree_a>0) * (itree_a-1)%P4EST_FACES;
        int const midx_b = (itree_b>0) * (itree_b-1)%P4EST_FACES;

        double Fa[2] = {
            ((itree_a-1)/P4EST_FACES==0) * grid_params_(1) + ((itree_a-1)/P4EST_FACES==1) * grid_params_(5),
            ((itree_a-1)/P4EST_FACES==0) * grid_params_(2) + ((itree_a-1)/P4EST_FACES==1) * grid_params_(6)
        } ; 
        double Fb[2] = {
            ((itree_b-1)/P4EST_FACES==0) * grid_params_(1) + ((itree_b-1)/P4EST_FACES==1) * grid_params_(5),
            ((itree_b-1)/P4EST_FACES==0) * grid_params_(2) + ((itree_b-1)/P4EST_FACES==1) * grid_params_(6)
        } ;
        double Sa[2] = {
            ((itree_a-1)/P4EST_FACES==0) * grid_params_(3) + ((itree_a-1)/P4EST_FACES==1) * grid_params_(7),
            ((itree_a-1)/P4EST_FACES==0) * grid_params_(4) + ((itree_a-1)/P4EST_FACES==1) * grid_params_(8)
        } ;
        double Sb[2] = {
            ((itree_b-1)/P4EST_FACES==0) * grid_params_(3) + ((itree_b-1)/P4EST_FACES==1) * grid_params_(7),
            ((itree_b-1)/P4EST_FACES==0) * grid_params_(4) + ((itree_b-1)/P4EST_FACES==1) * grid_params_(8)
        } ;

        auto const Ra = Kokkos::subview(rotation_matrices_, Kokkos::ALL(),midx_a) ;  
        auto const Rb = Kokkos::subview(inverse_rotation_matrices_, Kokkos::ALL(),midx_b) ;  
        
        detail::gl2l[itree_a * P4EST_FACES + face_a](
            grid_params_(0),Fa,Sa,Fb,Sb,Ra.data(),Rb.data(),l_coords_a,l_coords_b
        ) ; 

    }
    #ifdef THUNDER_3D 
    static constexpr size_t ntrees = 13UL;
    #else 
    static constexpr size_t ntrees = 5UL ; 
    #endif 
 private: 

    Kokkos::View<double* , thunder::default_space> grid_params_ ;
    Kokkos::View<double**, thunder::default_space> rotation_matrices_ ;
    Kokkos::View<double**, thunder::default_space> inverse_rotation_matrices_ ;

} ;

class spherical_coordinate_system_impl_t 
{
 public:
    spherical_coordinate_system_impl_t() 
    {
        using namespace thunder ; 
        auto config = config_parser::get()["amr"] ; 

        _L = config["inner_region_side"].as<double>() ; 
        _Ri = config["inner_region_radius"].as<double>() ; 
        _Ro = config["outer_region_radius"].as<double>() ;
        _use_logr = config["use_logarithmic_radial_zone"].as<bool>() ; 
        double s0_in{0.}, s0_out{1.}, s1_in{1.}, s1_out{1.} ; 

        auto const get_F0
            = [&] (double const sin, double const sout,
                double const rin, double const rout, bool log_radius)
            {
                return log_radius ? 0. : (1-sin)*rin ; 
            }; 
        auto const get_Fr
            = [&] (double const sin, double const sout,
                double const rin, double const rout, bool log_radius)
            {
                return log_radius ? 0. : (-(1-sin)*rin + (1-sout)*rout) ; 
            };
        auto const get_S0
            = [&] (double const sin, double const sout,
                double const rin, double const rout, bool log_radius)
            {
                return log_radius ? .5*log(rin*rout) : sin*rin ; 
            }; 
        auto const get_Sr
            = [&] (double const sin, double const sout,
                double const rin, double const rout, bool log_radius)
            {
                return log_radius ? .5*log(rout/rin) : (-sin*rin + sout*rout) ; 
            };
        _F0 =   get_F0(0.,1.,_L,_Ri,false)  ; _F1  = get_F0(1.,1.,_Ri,_Ro,_use_logr) ;  
        _Fr =   get_Fr(0.,1.,_L,_Ri,false)  ; _Fr1 = get_Fr(1.,1.,_Ri,_Ro,_use_logr) ; 
        _S0 =   get_S0(0.,1.,_L,_Ri,false)  ; _S1  = get_S0(1.,1.,_Ri,_Ro,_use_logr) ; 
        _Sr =   get_Sr(0.,1.,_L,_Ri,false)  ; _Sr1 = get_Sr(1.,1.,_Ri,_Ro,_use_logr) ;
        using namespace thunder ;
        using namespace Kokkos ; 
        
        // This contains: _L, F0, Fr, S0, Sr, F1, Fr1, S1, Sr1 
        grid_params_ = 
            View<double*, default_space>( "device_coords_grid_parameters"
                                        , 1 + 2 + 2 + 2 + 2) ; 
        rotation_matrices_ =
            View<double**, default_space>( "device_coords_rotation_matrices"
                                        , THUNDER_NSPACEDIM*THUNDER_NSPACEDIM, P4EST_FACES ) ;
        inverse_rotation_matrices_ =
            View<double**, default_space>( "device_coords_inverse_rotation_matrices"
                                        , THUNDER_NSPACEDIM*THUNDER_NSPACEDIM, P4EST_FACES ) ;

        auto const h_params_ = create_mirror_view(grid_params_) ; 
        h_params_(0) = _L ; 
        h_params_(1) = _F0 ; h_params_(2) = _Fr ; 
        h_params_(3) = _S0 ; h_params_(4) = _Sr ; 
        h_params_(5) = _F1 ; h_params_(6) = _Fr1 ; 
        h_params_(7) = _S1 ; h_params_(8) = _Sr1 ; 
        /* Set up discrete rotation matrices on host and transfer to device */
        auto const h_rot_mat_ = create_mirror_view(rotation_matrices_) ; 
        auto const h_inv_rot_mat_ = create_mirror_view(inverse_rotation_matrices_) ; 
        #ifndef THUNDER_3D 
        double rot_mat_tmp_[P4EST_FACES][THUNDER_NSPACEDIM*THUNDER_NSPACEDIM]
            = {
                {-1,0,0,1},
                {1,0,0,1},
                {0,1,-1,0},
                {0,1,1,0}
            } ; 
        double inv_rot_mat_tmp_[P4EST_FACES][THUNDER_NSPACEDIM*THUNDER_NSPACEDIM]
            =
            {
                {-1,0,0,1},
                {1,0,0,1},
                {0,-1,1,0},
                {0,1,1,0}
            } ; 
        #else 

        #endif 
        for(int iface=0; iface<P4EST_FACES; ++iface){
            for(int i=0;i<THUNDER_NSPACEDIM;++i){
                for(int j=0;j<THUNDER_NSPACEDIM;++j){
                    auto const idx = j + THUNDER_NSPACEDIM*i ;
                    h_rot_mat_(idx,iface) = rot_mat_tmp_[iface][idx] ; 
                    h_inv_rot_mat_(idx,iface)= inv_rot_mat_tmp_[iface][idx] ;
                }  
            }
        }
        deep_copy(grid_params_, h_params_) ; 
        deep_copy(rotation_matrices_, h_rot_mat_) ; 
        deep_copy(inverse_rotation_matrices_, h_inv_rot_mat_) ; 
        bool use_logr = _use_logr ; 
        Kokkos::parallel_for("fill_device_coord_function_pointers", 1,
            KOKKOS_LAMBDA (int const& _dummy)
            {
                int ii=0; 
                int jj=0; 
                detail::p2l[ii] = physical_to_logical_cart ; ii++;
                detail::l2p[jj] = logical_to_physical_cart ; jj++;
                for( int i=0; i<P4EST_FACES; ++i){ 
                    detail::p2l[ii] = physical_to_logical_sph  ; ii++; 
                    detail::l2p[jj] = logical_to_physical_sph  ; jj++; 
                } 
                for( int i=0; i<P4EST_FACES; ++i){ 
                    detail::p2l[ii] = use_logr ? physical_to_logical_sph_log
                                               : physical_to_logical_sph  ; ii++; 
                    detail::l2p[jj] = use_logr ? logical_to_physical_sph_log
                                               : logical_to_physical_sph  ; jj++;
                } 
                /* Fill coordinate transfer pointers */
                /* Cartesian grid */
                int itree = 0 ; 
                for( int iface=0; iface<P4EST_FACES; ++iface){
                    detail::gl2l[P4EST_FACES*itree + iface] = cart_to_sph_transfer; 
                }
                itree++ ; 
                for(int sph_tree=0; sph_tree<P4EST_FACES; ++sph_tree){
                    detail::gl2l[P4EST_FACES*itree + 0] = sph_to_cart_transfer ; 
                    detail::gl2l[P4EST_FACES*itree + 1] = sph_to_sph_positive_r_transfer ;
                    for( int iface=2; iface<P4EST_FACES; ++iface){
                        detail::gl2l[P4EST_FACES*itree + iface] = sph_to_sph_angular_transfer ;
                    } 
                    itree++ ; 
                }
                for(int sph_tree=0; sph_tree<P4EST_FACES; ++sph_tree){
                    detail::gl2l[P4EST_FACES*itree + 0] = use_logr ? sph_to_sph_negative_r_transfer_log 
                                                                   : sph_to_sph_negative_r_transfer ; 
                    detail::gl2l[P4EST_FACES*itree + 1] = use_logr ? sph_to_sph_positive_r_transfer_log 
                                                                   : sph_to_sph_positive_r_transfer ;
                    for( int iface=2; iface<P4EST_FACES; ++iface){
                        detail::gl2l[P4EST_FACES*itree + iface] = use_logr ? sph_to_sph_angular_transfer_log
                                                                           : sph_to_sph_angular_transfer ;
                    } 
                    itree++ ; 
                }
            }
        ) ; 
    }; 

    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_physical_coordinates(
          int const itree 
        , std::array<double,THUNDER_NSPACEDIM> const& logical_coordinates
    ) ; 

    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_physical_coordinates(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk
        , int64_t q 
        , std::array<double, THUNDER_NSPACEDIM> const& cell_coordinates
        , bool use_ghostzones
    ) ;

    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_physical_coordinates(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk
        , int64_t q 
        , bool use_ghostzones
    ) ;

    std::array<double,THUNDER_NSPACEDIM> 
    THUNDER_HOST get_logical_coordinates(
          int itree 
        , std::array<double,THUNDER_NSPACEDIM> const& physical_coordinates
    ) ; 

    std::array<double,THUNDER_NSPACEDIM> 
    THUNDER_HOST get_logical_coordinates(
        std::array<double,THUNDER_NSPACEDIM> const& physical_coordinates
    ) ; 

    std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>
    THUNDER_HOST get_jacobian(
        std::array<double,THUNDER_NSPACEDIM> const& physical_coordinates 
    ) ; 

    spherical_device_coordinate_system_impl_t
    get_device_coord_system(){
        return spherical_device_coordinate_system_impl_t {
              grid_params_
            , rotation_matrices_
            , inverse_rotation_matrices_
        } ; 
    }

 private:  

    double THUNDER_HOST 
    get_zeta( double const& z
            , double const& one_over_rho
            , std::array<double,2> const& F
            , std::array<double,2> const& S
            , bool use_logr) const ; 

    bool   _use_logr ; 
    double _L,_Ri,_Ro,_F0,_F1,_Fr,_Fr1,_S0,_S1,_Sr,_Sr1 ;

    Kokkos::View<double* , thunder::default_space> grid_params_ ;
    Kokkos::View<double**, thunder::default_space> rotation_matrices_ ;
    Kokkos::View<double**, thunder::default_space> inverse_rotation_matrices_ ;

    static constexpr size_t longevity = THUNDER_COORDINATE_SYSTEM ; 

    friend class utils::singleton_holder<spherical_coordinate_system_impl_t, memory::default_create> ; 
    friend class memory::new_delete_creator<spherical_coordinate_system_impl_t,memory::new_delete_allocator> ;
} ; 
 

} /* namespace thunder::amr */

#endif /* THUNDER_AMR_SPHERICAL_COORDINATE_SYSTEMS_HH */