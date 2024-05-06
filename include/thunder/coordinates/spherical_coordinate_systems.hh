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

#include <array>
#include <tuple>

namespace thunder { 
typedef void (*coord_transform_t) (double,double*,double*,double*,double*,double*) ; 
typedef void (*coord_transfer_t) ( double          // L 
                                 , double*,double* // Fa Sa
                                 , double*,double* // Fb Sb 
                                 , double*,double* // Ra Rb 
                                 , double*,double* ) ; 


namespace detail {
THUNDER_DEVICE coord_transform_t l2p[2*P4EST_FACES+1], p2l[2*P4EST_FACES+1] ; 
__device__ coord_transfer_t gl2l[(2*P4EST_FACES+1)*P4EST_FACES] ;
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
    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    transform_indices( int itree_a, int itree_b, 
                       int face_a, int face_b,
                       VEC(int  i_a, int  j_a, int  k_a),
                       VEC(int* i_b, int* j_b, int* k_b) ) const 
    {
        int const midx_a = (itree_a>0) * (itree_a-1)%P4EST_FACES;
        int const midx_b = (itree_b>0) * (itree_b-1)%P4EST_FACES;

        auto const Ra = Kokkos::subview(rotation_matrices_, Kokkos::ALL(),midx_a) ;  
        auto const Rb = Kokkos::subview(inverse_rotation_matrices_, Kokkos::ALL(),midx_b) ;  
        
        

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
//**************************************************************************************************
/**
 * @brief Implementation of coordinate system class for spherical grids.
 * \ingroup thunder_coordinates
 * 
 * Spherical coordinates in thunder are implemented using a central patch of Cartesian coordinates
 * which corresponds with tree 0 in the p4est connectivity. The cartesian patch is surrounded 
 * by P4EST_FACES (4 in 2D, 6 in 3D) wedges (each of which is a tree) where the coordinates go 
 * from Cartesian to Spherical along the logical radius. The logical coordinates in each tree 
 * are as usual indicated by \f$(\zeta,\eta,\xi)\f$ and each span \f$[0,1]\f$. There is then a 
 * second layer of wedges where the coordinates are purely spherical in each patch, with the option
 * of using a logaritmhmic distribution for the radial coordinate. By convention, the first logical
 * coordinate (\f$\zeta\f$) is the radial one in each tree. The other two logical coordinates are then
 * chosen as to form a right handed basis where the angular logical coordinates align with the axes 
 * of the physical coordinates in the embedding space. 
 * In particular, the coordinate transformation is constructed as follows. Let \f$L\f$ be half of 
 * the length of the side of the central Cartesian patch, \f$R_i\f$ be the radius of the inner spherical
 * face and \f$R_o\f$ be the radius of the outer grid boundary. These variables can be chosen from the 
 * <code>Thunder</code> config file. We can then define the sphere and frustum functions as follows:
 * \f{eqnarray*}{
 *   S &=& S_0 + \zeta~S_r, \\
 *   F &=& F_0 + \zeta~F_r.
 * \f}
 * Here the frustum rates \f$F_0,F_r\f$, and the sphere rates \f$S_0,S_r\f$ are defined as
 * \f{eqnarray*}{
 *   F_0 &=& -r {\left(\mathit{s_i} - 1\right)}, \\
 *   F_r &=&  r {\left(\mathit{s_i} - 1\right)} - R {\left(\mathit{s_o} - 1\right)}, \\
 *   S_0 &=&  r \mathit{s_i}, \\
 *   S_r &=& -r \mathit{s_i} + R \mathit{s_o}.
 * \f}
 * Where \f$r\f$ is \f$L\f$ [[resp. \f$R_i\f$]] for the inner [[resp. outer]] and 
 * \f$R=R_i\f$ [[resp. \f$R_o\f$]], whereas \f$\mathit{s_i},\mathit{s_o}\f$ are the 
 * sphericities of the inner and outer face which are \f$0,1\f$ for a flat or spherical face.
 * We can now define the coordinate transformation which reads (in the case of the \f$+x\f$ wedge)
 * \f{eqnarray*}{
 *  x &=& F + \frac{S}{\rho}~, \\
 *  y &=& x~\left(2~\eta-1\right)~,  \\
 *  z &=& x~\left(2~\xi-1 \right)~.
 * \f}
 * Where the function \f$\rho\f$ is given by
 * \f[
 *   \rho=\sqrt{1+(2~\eta-1)^2+(2~\xi-1)^2}~.
 * \f]
 * When the radial points are radially sampled, the frustum rates are zero and the sphere rates 
 * change to 
 * \f{eqnarray*}{
 *   S_0 &=&  \frac{1}{2}\log(r~R)~, \\
 *   S_r &=&  \frac{1}{2}\log(R/r)~.
 * \f}
 * And the radial coordinate is computed as
 * \f[
 *  x = \frac{1}{\rho}~e^{S_0+S_r~\left(2~\zeta-1\right)}~.
 * \f] 
 * 
 */ 
//**************************************************************************************************
class spherical_coordinate_system_impl_t 
{
 private:
    //**************************************************************************************************
    /**
     * @brief Construct a new spherical coordinate system.
     */
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
        double rot_mat_tmp_[P4EST_FACES][THUNDER_NSPACEDIM*THUNDER_NSPACEDIM]
            = {
                {-1,0,0,
                0,0,1,
                0,1,0},
                {1,0,0,
                0,1,0,
                0,0,1},
                {0,-1,0,
                1,0,0,
                0,0,1},
                {0,1,0,
                0,0,1,
                0,1,0},
                {0,0,-1,
                0,1,0,
                1,0,0},
                {0,0,1,
                1,0,0,
                0,1,0}
            } ; 
        double inv_rot_mat_tmp_[P4EST_FACES][THUNDER_NSPACEDIM*THUNDER_NSPACEDIM]
            =
            {
                {-1,0,0,
                0,0,1,
                0,1,0},
                {1,0,0,
                0,1,0,
                0,0,1},
                {0,1,0,
                -1,0,0,
                0,0,1},
                {0,0,1,
                1,0,0,
                0,1,},
                {0,0,1,
                0,1,0,
                -1,0,0},
                {0,1,0,
                0,0,1,
                1,0,0}
            } ; 
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
    //**************************************************************************************************
    /**
     * @brief Destroy the spherical coordinate system
     */
    ~spherical_coordinate_system_impl_t() = default ; 
    //**************************************************************************************************
 public:
    //**************************************************************************************************
    /**
     * @brief Get the physical coordinates of a point
     * 
     * @param itree Index of the tree containing the point
     * @param logical_coordinates Logical coordinates of the point within the tree
     * @return std::array<double, THUNDER_NSPACEDIM> An array containing the point's 
     *                                               physical coordinates.
     */
    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_physical_coordinates(
          int const itree 
        , std::array<double,THUNDER_NSPACEDIM> const& logical_coordinates
    ) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the physical coordinates of a point
     * 
     * @param ijk Indices of the cell containing the point.
     * @param q Local quadrant index
     * @param cell_coordinates Coordinates of point within the cell (should be in \f$[0,1]^N_d\f$)
     * @param use_ghostzones Set to true if the indices are zero-offset, false if they are 
     *                       ngz-offset
     * @return std::array<double, THUNDER_NSPACEDIM> An array containing the point's 
     *                                               physical coordinates.
     */
    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_physical_coordinates(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk
        , int64_t q 
        , std::array<double, THUNDER_NSPACEDIM> const& cell_coordinates
        , bool use_ghostzones
    ) const ;
    //**************************************************************************************************
    /**
     * @brief Get the physical coordinates of a cell centre
     * 
     * @param ijk Cell indices
     * @param q Local quadrant index
     * @param use_ghostzones Set to true if the indices are zero-offset, false if they are 
     *                       ngz-offset
     * @return std::array<double, THUNDER_NSPACEDIM> An array containing the point's 
     *                                               physical coordinates.
     */
    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_physical_coordinates(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk
        , int64_t q 
        , bool use_ghostzones
    ) const ;
    //**************************************************************************************************
    /**
     * @brief Get the logical coordinates of a point
     * 
     * @param ijk Indices of the cell containing the point.
     * @param q Local quadrant index
     * @param cell_coordinates Coordinates of point within the cell (should be in \f$[0,1]^N_d\f$)
     * @param use_ghostzones Set to true if the indices are zero-offset, false if they are 
     *                       ngz-offset
     * @return std::array<double, THUNDER_NSPACEDIM> An array containing the point's 
     *                                               logical coordinates.
     */
    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_logical_coordinates(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk
        , int64_t q 
        , std::array<double, THUNDER_NSPACEDIM> const& cell_coordinates
        , bool use_ghostzones
    ) const ;
    //**************************************************************************************************
    /**
     * @brief Get the logical coordinates of a point
     * 
     * @param itree Index of tree containing the point
     * @param physical_coordinates Physical coordinates of requested point
     * @return std::array<double, THUNDER_NSPACEDIM> An array containing the point's 
     *                                               logical coordinates.
     */
    std::array<double,THUNDER_NSPACEDIM> 
    THUNDER_HOST get_logical_coordinates(
          int itree 
        , std::array<double,THUNDER_NSPACEDIM> const& physical_coordinates
    ) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the determinant of the Jacobian matrix of the coordinate transformation at a given point
     * 
     * @param ijk Indices of cell containing the point 
     * @param q   Local index of quadrant containing the point
     * @param cell_coordinates Coordinates of point within the cell 
     * @param use_ghostzones True if indices are zero-offset, false if ngz-offset
     * @return double The Jacobian matrix determinant.
     */
    double
    THUNDER_HOST get_jacobian(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
        , int64_t q 
        , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
        , bool use_ghostzones 
    ) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the determinant of the Jacobian matrix of the coordinate transformation at a given point
     * 
     * @param ijk Indices of cell containing the point 
     * @param q   Local index of quadrant containing the point
     * @param itree Index of tree containing the point
     * @param cell_coordinates Coordinates of point within the cell 
     * @param use_ghostzones True if indices are zero-offset, false if ngz-offset
     * @return double The Jacobian matrix determinant.
     */
    double
    THUNDER_HOST get_jacobian(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
        , int64_t q 
        , int itree
        , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
        , bool use_ghostzones 
    ) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the determinant of the Jacobian matrix of the coordinate transformation at a given point
     * 
     * @param itree Index of tree containing the point
     * @param lcoords Logical coordinates of point
     * @return double The Jacobian matrix determinant.
     * NB: This function checks for tree boundaries.
     */
    double
    THUNDER_HOST get_jacobian(
          int itree
        , std::array<double,THUNDER_NSPACEDIM> const& lcoords 
    ) const;
    //**************************************************************************************************
    /**
     * @brief Get the determinant of the Jacobian matrix 
     *        of the inverse coordinate transformation at a given point
     * 
     * @param ijk Indices of cell containing the point 
     * @param q   Local index of quadrant containing the point
     * @param cell_coordinates Coordinates of point within the cell 
     * @param use_ghostzones True if indices are zero-offset, false if ngz-offset
     * @return double The inverse Jacobian matrix determinant.
     */
    double
    THUNDER_HOST get_inverse_jacobian(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
        , int64_t q 
        , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
        , bool use_ghostzones 
    ) const; 
    //**************************************************************************************************
    /**
     * @brief Get the determinant of the Jacobian matrix 
     *        of the inverse coordinate transformation at a given point      
     * @param ijk Indices of cell containing the point 
     * @param q   Local index of quadrant containing the point
     * @param itree Index of tree containing the point
     * @param cell_coordinates Coordinates of point within the cell 
     * @param use_ghostzones True if indices are zero-offset, false if ngz-offset
     * @return double The inverse Jacobian matrix determinant.
     */
    double
    THUNDER_HOST get_inverse_jacobian(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
        , int64_t q 
        , int itree
        , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
        , bool use_ghostzones 
    ) const; 
    //**************************************************************************************************
    /**
     * @brief Get the determinant of the Jacobian matrix 
     *        of the inverse coordinate transformation at a given point  
     * @param itree Index of tree containing the point
     * @param lcoords Logical coordinates of point
     * @return double The invesre Jacobian matrix determinant.
     * NB: This function checks for tree boundaries.
     */
    double
    THUNDER_HOST get_inverse_jacobian(
          int itree
        , std::array<double,THUNDER_NSPACEDIM> const& lcoords 
    ) const;
    //**************************************************************************************************
    /**
     * @brief Get the Jacobian matrix of the coordinate transformation at a given point
     * 
     * @param ijk Indices of cell containing the point 
     * @param q   Local index of quadrant containing the point
     * @param cell_coordinates Coordinates of point within the cell 
     * @param use_ghostzones True if indices are zero-offset, false if ngz-offset
     * @return std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> The Jacobian matrix.
     */
    std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>
    THUNDER_HOST get_jacobian_matrix(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
        , int64_t q 
        , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
        , bool use_ghostzones 
    ) const; 
    //**************************************************************************************************
    /**
     * @brief Get the Jacobian matrix of the coordinate transformation at a given point
     * 
     * @param ijk Indices of cell containing the point 
     * @param q   Local index of quadrant containing the point
     * @param itree Index of tree containing the point
     * @param cell_coordinates Coordinates of point within the cell 
     * @param use_ghostzones True if indices are zero-offset, false if ngz-offset
     * @return std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> The Jacobian matrix.
     */
    std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>
    THUNDER_HOST get_jacobian_matrix(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
        , int64_t q 
        , int itree
        , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
        , bool use_ghostzones 
    ) const; 
    //**************************************************************************************************
    /**
     * @brief Get the Jacobian matrix of the coordinate transformation at a given point
     * 
     * @param itree Index of tree containing the point
     * @param lcoords Logical coordinates of point
     * @return std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> The Jacobian matrix.
     * NB: This function checks for tree boundaries.
     */
    std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>
    THUNDER_HOST get_jacobian_matrix(
          int itree
        , std::array<double,THUNDER_NSPACEDIM> const& lcoords 
    ) const;
    //**************************************************************************************************
    /**
     * @brief Get the Jacobian matrix of the inverse coordinate transformation at a given point
     * 
     * @param ijk Indices of cell containing the point 
     * @param q   Local index of quadrant containing the point
     * @param cell_coordinates Coordinates of point within the cell 
     * @param use_ghostzones True if indices are zero-offset, false if ngz-offset
     * @return std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> The inverse Jacobian matrix.
     */
    std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>
    THUNDER_HOST get_inverse_jacobian_matrix(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
        , int64_t q 
        , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
        , bool use_ghostzones 
    ) const; 
    //**************************************************************************************************
    /**
     * @brief Get the Jacobian matrix of the inverse coordinate transformation at a given point
     * 
     * @param ijk Indices of cell containing the point 
     * @param q   Local index of quadrant containing the point
     * @param itree Index of tree containing the point
     * @param cell_coordinates Coordinates of point within the cell 
     * @param use_ghostzones True if indices are zero-offset, false if ngz-offset
     * @return std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> The inverse Jacobian matrix.
     */
    std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>
    THUNDER_HOST get_inverse_jacobian_matrix(
          std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
        , int64_t q 
        , int itree
        , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
        , bool use_ghostzones 
    ) const; 
    //**************************************************************************************************
    /**
     * @brief Get the Jacobian matrix of the inverse coordinate transformation at a given point
     * 
     * @param itree Index of tree containing the point
     * @param lcoords Logical coordinates of point
     * @return std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> The inverse Jacobian matrix.
     * NB: This function checks for tree boundaries.
     */
    std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>
    THUNDER_HOST get_inverse_jacobian_matrix(
          int itree
        , std::array<double,THUNDER_NSPACEDIM> const& lcoords 
    ) const; 
    //**************************************************************************************************
    /**
     * @brief Get the volume of a cell
     * 
     * @param ijk Indices of the cell.
     * @param q Local quadrant index.
     * @param use_ghostzones Set to true if the indices are zero-offset, false if they are 
     *                       ngz-offset
     * @return double The volume of the requested cell.
     */
    double  
    THUNDER_HOST get_cell_volume(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q
    , bool use_ghostzones) const ;
    //**************************************************************************************************
    /**
     * @brief Get the volume of a cell
     * 
     * @param ijk Indices of the cell.
     * @param q Local quadrant index.
     * @param itree Index of the tree containing the cell.
     * @param dxl Cell spacing in logical coordinates
     * @param use_ghostzones Set to true if the indices are zero-offset, false if they are 
     *                       ngz-offset
     * @return double The volume of the requested cell.
     */
    double
    THUNDER_HOST get_cell_volume(
        std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
        , int64_t q
        , int itree
        , std::array<double, THUNDER_NSPACEDIM> const& dxl 
        , bool use_ghostzones
    ) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the volume of a cell
     * 
     * @param lcoords Logical coordinates of cell lower left corner.
     * @param itree Index of the tree containing the cell.
     * @param dxl Cell spacing in logical coordinates
     * @param use_ghostzones Set to true if the indices are zero-offset, false if they are 
     *                       ngz-offset
     * @return double The volume of the requested cell.
     */
    double
    THUNDER_HOST get_cell_volume(
          std::array<double,THUNDER_NSPACEDIM> const& lcoords
        , int itree
        , std::array<double, THUNDER_NSPACEDIM> const& dxl 
        , bool use_ghostzones
    ) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the suface of a cell face.
     * 
     * @param ijk     Cell indices.
     * @param q       Local quadrant index.
     * @param face    Cell face index.
     * @param dxl     (Logical) Cell coordinate spacing.
     * @param use_ghostzones Set to false if coordinates are always physical.
     * @return double The surface of the cell face.
     * NB: By convention, cell face indices are staggered backwards, meaning that given an index \f$i_f\f$
     * of a face, this routine returns the surface of the face whose center is located at index:
     * \f[
     *   (I,J,K) = (i-\frac{\delta_{i,i_f}}{2}, j-\frac{\delta_{j,i_f}}{2}, k-\frac{\delta_{k,i_f}}{2})
     * \f]
     */
    double 
    THUNDER_HOST 
    get_cell_face_surface(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q
    , int8_t face 
    , bool use_ghostzones) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the suface of a cell face.
     * 
     * @param ijk     Cell indices.
     * @param q       Local quadrant index.
     * @param face    Cell face index.
     * @param itree   Source tree id.
     * @param dxl     (Logical) Cell coordinate spacing.
     * @param use_ghostzones Set to false if coordinates are always physical.
     * @return double The surface of the cell face.
     * NB: By convention, cell face indices are staggered backwards, meaning that given an index \f$i_f\f$
     * of a face, this routine returns the surface of the face whose center is located at index:
     * \f[
     *   (I,J,K) = (i-\frac{\delta_{i,i_f}}{2}, j-\frac{\delta_{j,i_f}}{2}, k-\frac{\delta_{k,i_f}}{2})
     * \f]
     */
    double 
    THUNDER_HOST 
    get_cell_face_surface(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q
    , int8_t face 
    , int itree
    , std::array<double, THUNDER_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the suface of a cell face.
     * 
     * @param lcoords Logical coordinates of cell's lowest corner (z-ordering) in 
     *                tree <code>itree</code>'s coordinate system.
     * @param face    Cell face index.
     * @param itree   Source tree id.
     * @param dxl     (Logical) Cell coordinate spacing.
     * @param use_ghostzones Set to false if coordinates are always physical.
     * @return double The surface of the cell face.
     * NB: By convention, cell face indices are staggered backwards, meaning that given an index \f$i_f\f$
     * of a face, this routine returns the surface of the face whose center is located at index:
     * \f[
     *   (I,J,K) = (i-\frac{\delta_{i,i_f}}{2}, j-\frac{\delta_{j,i_f}}{2}, k-\frac{\delta_{k,i_f}}{2})
     * \f]
     */
    double 
    THUNDER_HOST 
    get_cell_face_surface(
      std::array<double, THUNDER_NSPACEDIM> const& lcoords 
    , int8_t face 
    , int itree
    , std::array<double, THUNDER_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const ;
    #ifdef THUNDER_3D 
    //**************************************************************************************************
    /**
     * @brief Get the length of a cell edge.
     * 
     * @param ijk     Cell indices.
     * @param q       Local quadrant index.
     * @param edge    Cell edge index (between 0 and <code>THUNDER_NSPACEDIM</code>).
     * @param dxl     (Logical) Cell coordinate spacing.
     * @param use_ghostzones Set to true if indices are 0-offset, false if they are ngz-offset
     * @return double The length of the cell edge. 
     * NB: By convention, cell edge indices are staggered backwards, meaning that given an index \f$i_e\f$
     * of an edge, this routine returns the length of the edge whose center is located at index:
     * \f[
     *   (I,J,K) = (i-\frac{1-\delta_{i,i_f}}{2}, j-\frac{1-\delta_{j,i_f}}{2}, k-\frac{1-\delta_{k,i_f}}{2})
     * \f]
     */
    double THUNDER_HOST 
    get_cell_edge_length(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , int64_t q 
    , int8_t edge
    , bool use_ghostzones) const ;
    //**************************************************************************************************
    /**
     * @brief Get the length of a cell edge.
     * 
     * @param ijk     Cell indices.
     * @param q       Local quadrant index.
     * @param itree   Source tree id.
     * @param edge    Cell edge index (between 0 and <code>THUNDER_NSPACEDIM</code>).
     * @param dxl     (Logical) Cell coordinate spacing.
     * @param use_ghostzones Set to true if indices are 0-offset, false if they are ngz-offset
     * @return double The length of the cell edge. 
     * NB: By convention, cell edge indices are staggered backwards, meaning that given an index \f$i_e\f$
     * of an edge, this routine returns the length of the edge whose center is located at index:
     * \f[
     *   (I,J,K) = (i-\frac{1-\delta_{i,i_f}}{2}, j-\frac{1-\delta_{j,i_f}}{2}, k-\frac{1-\delta_{k,i_f}}{2})
     * \f]
     */
    double THUNDER_HOST 
    get_cell_edge_length(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , int64_t q 
    , int8_t edge
    , int itree
    , std::array<double, THUNDER_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const ;
    //**************************************************************************************************
    /**
     * @brief Get the length of a cell edge.
     * 
     * @param lcoords Logical coordinates of cell's lowest corner (z-ordering) in 
     *                tree <code>itree</code>'s coordinate system.
     * @param itree   Source tree id.
     * @param edge    Cell edge index (between 0 and <code>THUNDER_NSPACEDIM</code>).
     * @param dxl     (Logical) Cell coordinate spacing.
     * @param use_ghostzones Set to false if coordinates are always physical.
     * @return double The length of the cell edge. 
     * NB: By convention, cell edge indices are staggered backwards, meaning that given an index \f$i_e\f$
     * of an edge, this routine returns the length of the edge whose center is located at index:
     * \f[
     *   (I,J,K) = (i-\frac{1-\delta_{i,i_f}}{2}, j-\frac{1-\delta_{j,i_f}}{2}, k-\frac{1-\delta_{k,i_f}}{2})
     * \f]
     */
    double
    THUNDER_HOST get_cell_edge_length(
      std::array<double, THUNDER_NSPACEDIM> const& lcoords
    , int8_t edge
    , int itree
    , std::array<double, THUNDER_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const ;
    #endif 
    //**************************************************************************************************
    /**
     * @brief Get the device coord system object
     * 
     * @return spherical_device_coordinate_system_impl_t A lightweight coordinate system object
     *                                                   whose methods are accessible from device.
     */
    spherical_device_coordinate_system_impl_t
    get_device_coord_system() const {
        return spherical_device_coordinate_system_impl_t {
              grid_params_
            , rotation_matrices_
            , inverse_rotation_matrices_
        } ; 
    }
    //**************************************************************************************************

 private:  
    //**************************************************************************************************
    /**
     * @brief Check whether a point is in the buffer zone.
     * 
     * @param lcoords  Logical coordinates
     * @param check_exact_boundary Does the upper boundary count as first point in the next tree?
     * @return true If the point is in the buffer zone.
     * @return false If the point is within the tree boundaries.
     */
    static bool THUNDER_HOST 
    is_outside_tree(std::array<double,THUNDER_NSPACEDIM> const& lcoords, bool check_exact_boundary=false) ; 
    //**************************************************************************************************
    /**
     * @brief Check whether a tree boundary is physical or internal.
     * 
     * @param lcoords  Logical coordinates w.r.t. source tree.
     * @param itree    Source tree id.
     * @param check_exact_boundary Does the upper boundary count as first point in the next tree?
     * @return true If the tree boundary faces the outside of the grid.
     * @return false If the tree boundary faces another tree in the grid.
     */
    bool THUNDER_HOST 
    is_physical_boundary(std::array<double,THUNDER_NSPACEDIM> const& lcoords, int itree, bool check_exact_boundary=false) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the neighbor tree and face indices in buffer zone.
     * 
     * @param itree    Source tree index
     * @param ijk      Cell indices w.r.t. source tree
     * @param check_exact_boundary Does the upper boundary count as first point in the next tree?
     * @return std::tuple<int, int8_t, int8_t> The index <code>itree_b</code> of the tree across 
     *         the boundary, the face <code>iface_b</code> of that tree and the face <code>iface</code>
     *         of this tree.
     * NB: this function does not check that the quadrant actually touches the face. It's the caller's
     *     responsibility to ensure this.
     */
    std::tuple<int, int8_t, int8_t>
    THUNDER_HOST get_neighbor_tree_and_face(
          int itree
        , std::array<size_t,THUNDER_NSPACEDIM> const& ijk 
    ) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the neighbor tree and face indices in buffer zone.
     * 
     * @param itree    Source tree index
     * @param lcoords  Logical coordinates w.r.t. source tree
     * @param check_exact_boundary Does the upper boundary count as first point in the next tree?
     * @return std::tuple<int, int8_t, int8_t> The index <code>itree_b</code> of the tree across 
     *         the boundary, the face <code>iface_b</code> of that tree and the face <code>iface</code>
     *         of this tree.
     */
    std::tuple<int, int8_t, int8_t>
    THUNDER_HOST get_neighbor_tree_and_face(
          int itree
        , std::array<double,THUNDER_NSPACEDIM> const& lcoords
        , bool check_exact_boundary=false
    ) const ;
    //**************************************************************************************************
    /**
     * @brief Get the logical coordinates of a point in the buffer zone.
     * 
     * @param itree   Source tree id.
     * @param itree_b Id of tree across face.
     * @param iface   Face id. 
     * @param iface_b Face id from tree_b's perspective.
     * @param lcoords Logical coordinates of cell's lowest corner (z-ordering) in 
     *                tree <code>itree</code>'s coordinate system.
     * @return double The logical coordinates w.r.t the tree across the boundary
     *                of the cell in the buffer zone of tree <code>itree</code>.
     * NB: By buffer zone we mean the ghost zone layer that crosses a tree boundary.
     */
    static std::array<double, THUNDER_NSPACEDIM> 
    THUNDER_HOST get_logical_coordinates_buffer_zone(
        int itree, int itree_b, int8_t iface, int8_t iface_b
      , std::array<double, THUNDER_NSPACEDIM> const& lcoords ) ;
    //**************************************************************************************************
    /**
     * @brief Get the volume of a cell in the buffer zone.
     * 
     * @param itree   Source tree id
     * @param q       Quadrant index
     * @param lcoords Logical coordinates of cell's lowest corner (z-ordering) in 
     *                tree <code>itree</code>'s coordinate system.
     * @param dxl     (Logical) Cell coordinate spacing 
     * @return double The volume of the cell in the buffer zone of tree <code>itree</code>.
     * NB: By buffer zone we mean the ghost zone layer that crosses a tree boundary.
     */
    double
    THUNDER_HOST get_cell_volume_buffer_zone(
      int itree
    , std::array<double, THUNDER_NSPACEDIM> const& lcoords
    , std::array<double, THUNDER_NSPACEDIM> const& dxl ) const ;
    //**************************************************************************************************
    /**
     * @brief Get the suface of a cell face in the buffer zone.
     * 
     * @param face    Cell face index.
     * @param itree   Source tree id.
     * @param lcoords Logical coordinates of cell's lowest corner (z-ordering) in 
     *                tree <code>itree</code>'s coordinate system.
     * @param dxl     (Logical) Cell coordinate spacing .
     * @return double The surface of the cell face in the buffer zone of tree <code>itree</code>.
     * NB: By buffer zone we mean the ghost zone layer that crosses a tree boundary.
     * By convention, cell face indices are staggered backwards, meaning that given an index \f$i_f\f$
     * of a face, this routine returns the surface of the face whose center is located at index:
     * \f[
     *   (I,J,K) = (i-\frac{\delta_{i,i_f}}{2}, j-\frac{\delta_{j,i_f}}{2}, k-\frac{\delta_{k,i_f}}{2})
     * \f]
     */
    double
    THUNDER_HOST get_cell_face_surface_buffer_zone(
      int8_t face
    , int itree
    , std::array<double, THUNDER_NSPACEDIM> const& lcoords
    , std::array<double, THUNDER_NSPACEDIM> const& dxl ) const ;
    #ifdef THUNDER_3D 
    //**************************************************************************************************
    /**
     * @brief Get the length of a cell edge in the buffer zone.
     * 
     * @param edge    Cell edge index.
     * @param itree   Source tree id.
     * @param lcoords Logical coordinates of cell's lowest corner (z-ordering) in 
     *                tree <code>itree</code>'s coordinate system.
     * @param dxl     (Logical) Cell coordinate spacing .
     * @return double The length of the cell edge in the buffer zone of tree <code>itree</code>.
     * NB: By buffer zone we mean the ghost zone layer that crosses a tree boundary.
     * By convention, cell edge indices are staggered backwards, meaning that given an index \f$i_e\f$
     * of an edge, this routine returns the length of the edge whose center is located at index:
     * \f[
     *   (I,J,K) = (i-\frac{1-\delta_{i,i_f}}{2}, j-\frac{1-\delta_{j,i_f}}{2}, k-\frac{1-\delta_{k,i_f}}{2})
     * \f]
     */
    double
    THUNDER_HOST get_cell_edge_length_buffer_zone(
      int8_t edge
    , int itree
    , std::array<double, THUNDER_NSPACEDIM> const& lcoords
    , std::array<double, THUNDER_NSPACEDIM> const& dxl ) const ;
    #endif 
    //**************************************************************************************************
    /**
     * @brief Compute physical coordinates in the cartesian coordinate patch
     * 
     * @param L       Half length of inner face's side
     * @param lcoords Logical coordinates \f$(\zeta,\eta,\xi)\f$
     * @return std::array<double, THUNDER_NSPACEDIM> Array containing physical coordinates of the point.
     */
    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_physical_coordinates_cart(
        double L,
        std::array<double, THUNDER_NSPACEDIM> const& lcoords
    ) const ; 
    //**************************************************************************************************
    /**
     * @brief Compute physical coordinates in a spherical coordinate patch
     * 
     * @param irot Rotation index, determines which coordinate is radial.
     * @param Ri   Inner radius
     * @param Ro   Outer radius
     * @param F    Frustum and scaled frustum rates
     * @param S    Sphere and scaled sphere rates
     * @param lcoords Logical coordinates \f$(\zeta,\eta,\xi)\f$
     * @param logr Is the radial point distribution logarithmic?
     * @return std::array<double, THUNDER_NSPACEDIM> Array containing physical coordinates of the point.
     */
    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_physical_coordinates_sph(
        int irot,
        double Ri,
        double Ro, 
        std::array<double,2> const& F, 
        std::array<double,2> const& S ,
        std::array<double, THUNDER_NSPACEDIM> const& lcoords,
        bool logr=false
    ) const ;
    //**************************************************************************************************
    /**
     * @brief Get physical zeta given the logical one.
     * 
     * @param z Zeroth logical coordinate in a spherical grid patch
     * @param one_over_rho Inverse of rho
     * @param F Frustum and scaled frustum rates
     * @param S Sphere and scaled sphere rates
     * @param use_logr Is the radial point distribution logarithmic?
     * @return double Physical radial coordinate
     */
    double THUNDER_HOST 
    get_zeta( double const& z
            , double const& one_over_rho
            , std::array<double,2> const& F
            , std::array<double,2> const& S
            , bool use_logr) const ; 
    //**************************************************************************************************
    //**************************************************************************************************
    bool   _use_logr ;                                      //!< Is the outer patch logarithmic in radius?
    double _L,_Ri,_Ro,_F0,_F1,_Fr,_Fr1,_S0,_S1,_Sr,_Sr1 ;   //!< Sphere and frustum rates
    //**************************************************************************************************
    Kokkos::View<double* , thunder::default_space> grid_params_ ;               //!< Grid parameters
    Kokkos::View<double**, thunder::default_space> rotation_matrices_ ;         //!< Discrete rotation matrices
    Kokkos::View<double**, thunder::default_space> inverse_rotation_matrices_ ; //!< Inverses of discrete rotation matrices
    //**************************************************************************************************
    static constexpr size_t longevity = THUNDER_COORDINATE_SYSTEM ; //!< Longevity of coordinate system
    //**************************************************************************************************
    friend class utils::singleton_holder<spherical_coordinate_system_impl_t, memory::default_create> ;         //!< Give access 
    friend class memory::new_delete_creator<spherical_coordinate_system_impl_t,memory::new_delete_allocator> ; //!< Give access
    //**************************************************************************************************
    static constexpr double eps_volume = 1e-12 ;
    //**************************************************************************************************
} ; 
 

} /* namespace thunder::amr */

#endif /* THUNDER_AMR_SPHERICAL_COORDINATE_SYSTEMS_HH */