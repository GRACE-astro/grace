/**
 * @file cartesian_coordinate_systems.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-26
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

#ifndef GRACE_AMR_CARTESIAN_COORDINATES_SYSTEMS_HH 
#define GRACE_AMR_CARTESIAN_COORDINATES_SYSTEMS_HH

#include <grace_config.h>

#include <Kokkos_Core.hpp>

#include <grace/utils/grace_utils.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>

#include<array>

namespace grace { 

struct cartesian_device_coordinate_system_impl_t
{
    cartesian_device_coordinate_system_impl_t( Kokkos::View<double*,grace::default_space> vertices
                                             , Kokkos::View<double*,grace::default_space> spacings )
        : tree_vertices_(vertices), tree_spacings_(spacings) 
    {} ; 
    
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    get_physical_coordinates( int itree, double * l_coords, double * p_coords) const
    {
        EXPR(
        p_coords[0] = 
            tree_vertices_(GRACE_NSPACEDIM * itree + 0UL) + l_coords[0] * tree_spacings_(GRACE_NSPACEDIM * itree + 0UL);,
        p_coords[1] = 
            tree_vertices_(GRACE_NSPACEDIM * itree + 1UL) + l_coords[1] * tree_spacings_(GRACE_NSPACEDIM * itree + 1UL);,
        p_coords[2] = 
            tree_vertices_(GRACE_NSPACEDIM * itree + 2UL) + l_coords[2] * tree_spacings_(GRACE_NSPACEDIM * itree + 2UL);
        )
        return ;
    };

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    get_logical_coordinates( int itree, double * p_coords, double * l_coords) const 
    {
        EXPR(
        l_coords[0] = 
            (p_coords[0] - tree_vertices_(GRACE_NSPACEDIM * itree + 0UL)) / tree_spacings_(GRACE_NSPACEDIM * itree + 0UL);,
        l_coords[1] = 
            (p_coords[1] - tree_vertices_(GRACE_NSPACEDIM * itree + 1UL)) / tree_spacings_(GRACE_NSPACEDIM * itree + 1UL);,
        l_coords[2] = 
            (p_coords[2] - tree_vertices_(GRACE_NSPACEDIM * itree + 2UL)) / tree_spacings_(GRACE_NSPACEDIM * itree + 2UL);

        )
        return ;
    };

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    transfer_coordinates( int tree_a, int tree_b, 
                          int face_a, int face_b,
                          double * l_coords_a, double * l_coords_b  )
    {
        get_physical_coordinates(tree_a, l_coords_a, l_coords_a) ; 
        get_logical_coordinates(tree_b, l_coords_a, l_coords_b ) ; 
    }
 private:
    Kokkos::View<double*,grace::default_space> tree_vertices_, tree_spacings_ ;

} ; 
//**************************************************************************************************
/**
 * @brief Implementation of coordinate system class for cartesian grids.
 * \ingroup grace_coordinates
 */
//**************************************************************************************************
class cartesian_coordinate_system_impl_t 
{
 public: 
    //**************************************************************************************************
    /**
     * @brief Get the physical coordinates of a point
     * 
     * @param itree Index of the tree containing the point
     * @param logical_coordinates Logical coordinates of the point within the tree
     * @return std::array<double, GRACE_NSPACEDIM> An array containing the point's 
     *                                               physical coordinates.
     */
    std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
    get_physical_coordinates( 
          int const itree
        , std::array<double, GRACE_NSPACEDIM> const& logical_coordinates ) const ;  
    //**************************************************************************************************
    /**
     * @brief Get the physical coordinates of a point
     * 
     * @param ijk Indices of the cell containing the point.
     * @param q Local quadrant index
     * @param cell_coordinates Coordinates of point within the cell (should be in \f$[0,1]^N_d\f$)
     * @param use_ghostzones Set to true if the indices are zero-offset, false if they are 
     *                       ngz-offset
     * @return std::array<double, GRACE_NSPACEDIM> An array containing the point's 
     *                                               physical coordinates.
     */
    std::array<double, GRACE_NSPACEDIM>
    GRACE_HOST get_physical_coordinates(
           std::array<size_t, GRACE_NSPACEDIM> const& ijk
        , int64_t q 
        , std::array<double, GRACE_NSPACEDIM> const& cell_coordinates
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
     * @return std::array<double, GRACE_NSPACEDIM> An array containing the point's 
     *                                               physical coordinates.
     */
    std::array<double, GRACE_NSPACEDIM>
    GRACE_HOST get_physical_coordinates(
           std::array<size_t, GRACE_NSPACEDIM> const& ijk
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
     * @return std::array<double, GRACE_NSPACEDIM> An array containing the point's 
     *                                               logical coordinates.
     */
    std::array<double, GRACE_NSPACEDIM>
    GRACE_HOST get_logical_coordinates(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk
    , int64_t q 
    , std::array<double, GRACE_NSPACEDIM> const& cell_coordinates
    , bool use_ghostzones) const ;
    //**************************************************************************************************
    /**
     * @brief Get the logical coordinates of a point
     * 
     * @param itree Index of tree containing the point
     * @param physical_coordinates Physical coordinates of requested point
     * @return std::array<double, GRACE_NSPACEDIM> An array containing the point's 
     *                                               logical coordinates.
     */
    std::array<double,GRACE_NSPACEDIM> 
    GRACE_HOST get_logical_coordinates(
          int itree
        , std::array<double,GRACE_NSPACEDIM> const& physical_coordinates
    ) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the logical coordinates of a point
     * 
     * @param physical_coordinates Physical coordinates of requested point
     * @return std::array<double, GRACE_NSPACEDIM> An array containing the point's 
     *                                               logical coordinates.
     */
    std::array<double,GRACE_NSPACEDIM> 
    GRACE_HOST get_logical_coordinates(
        std::array<double,GRACE_NSPACEDIM> const& physical_coordinates
    ) const ;
    //**************************************************************************************************
    /**
     * @brief Get the inverse cell spacing within a quadrant
     * 
     * @param q Local index of the quadrant
     */
    double GRACE_HOST get_inv_spacing(size_t const& q) const ; 
    //**************************************************************************************************
    /**
     * @brief Get the cell spacing within a quadrant
     * 
     * @param q Local index of the quadrant
     */
    double GRACE_HOST get_spacing(size_t const& q) const ; 
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
    GRACE_HOST get_jacobian(
          std::array<size_t, GRACE_NSPACEDIM> const& ijk 
        , int64_t q 
        , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
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
    GRACE_HOST get_jacobian(
          std::array<size_t, GRACE_NSPACEDIM> const& ijk 
        , int64_t q 
        , int itree
        , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
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
    GRACE_HOST get_jacobian(
          int itree
        , std::array<double,GRACE_NSPACEDIM> const& lcoords 
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
    GRACE_HOST get_inverse_jacobian(
          std::array<size_t, GRACE_NSPACEDIM> const& ijk 
        , int64_t q 
        , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
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
    GRACE_HOST get_inverse_jacobian(
          std::array<size_t, GRACE_NSPACEDIM> const& ijk 
        , int64_t q 
        , int itree
        , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
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
    GRACE_HOST get_inverse_jacobian(
          int itree
        , std::array<double,GRACE_NSPACEDIM> const& lcoords 
    ) const;
    //**************************************************************************************************
    /**
     * @brief Get the Jacobian matrix of the coordinate transformation at a given point
     * 
     * @param ijk Indices of cell containing the point 
     * @param q   Local index of quadrant containing the point
     * @param cell_coordinates Coordinates of point within the cell 
     * @param use_ghostzones True if indices are zero-offset, false if ngz-offset
     * @return std::array<double, GRACE_NSPACEDIM*GRACE_NSPACEDIM> The Jacobian matrix.
     */
    std::array<double, 9>
    GRACE_HOST get_jacobian_matrix(
          std::array<size_t, GRACE_NSPACEDIM> const& ijk 
        , int64_t q 
        , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
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
     * @return std::array<double, 9> The Jacobian matrix.
     */
    std::array<double, 9>
    GRACE_HOST get_jacobian_matrix(
          std::array<size_t, GRACE_NSPACEDIM> const& ijk 
        , int64_t q 
        , int itree
        , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
        , bool use_ghostzones 
    ) const; 
    //**************************************************************************************************
    /**
     * @brief Get the Jacobian matrix of the coordinate transformation at a given point
     * 
     * @param itree Index of tree containing the point
     * @param lcoords Logical coordinates of point
     * @return std::array<double, GRACE_NSPACEDIM*GRACE_NSPACEDIM> The Jacobian matrix.
     * NB: This function checks for tree boundaries.
     */
    std::array<double, 9>
    GRACE_HOST get_jacobian_matrix(
          int itree
        , std::array<double,GRACE_NSPACEDIM> const& lcoords 
    ) const;
    //**************************************************************************************************
    /**
     * @brief Get the Jacobian matrix of the inverse coordinate transformation at a given point
     * 
     * @param ijk Indices of cell containing the point 
     * @param q   Local index of quadrant containing the point
     * @param cell_coordinates Coordinates of point within the cell 
     * @param use_ghostzones True if indices are zero-offset, false if ngz-offset
     * @return std::array<double, GRACE_NSPACEDIM*GRACE_NSPACEDIM> The inverse Jacobian matrix.
     */
    std::array<double, 9>
    GRACE_HOST get_inverse_jacobian_matrix(
          std::array<size_t, GRACE_NSPACEDIM> const& ijk 
        , int64_t q 
        , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
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
     * @return std::array<double, GRACE_NSPACEDIM*GRACE_NSPACEDIM> The inverse Jacobian matrix.
     */
    std::array<double, 9>
    GRACE_HOST get_inverse_jacobian_matrix(
          std::array<size_t, GRACE_NSPACEDIM> const& ijk 
        , int64_t q 
        , int itree
        , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
        , bool use_ghostzones 
    ) const; 
    //**************************************************************************************************
    /**
     * @brief Get the Jacobian matrix of the inverse coordinate transformation at a given point
     * 
     * @param itree Index of tree containing the point
     * @param lcoords Logical coordinates of point
     * @return std::array<double, GRACE_NSPACEDIM*GRACE_NSPACEDIM> The inverse Jacobian matrix.
     * NB: This function checks for tree boundaries.
     */
    std::array<double, 9>
    GRACE_HOST get_inverse_jacobian_matrix(
          int itree
        , std::array<double,GRACE_NSPACEDIM> const& lcoords 
    ) const; 
    //************************************************************************************************** 
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
    GRACE_HOST get_cell_volume(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
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
    GRACE_HOST get_cell_volume(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const ;
    //**************************************************************************************************
    /**
     * @brief Get the volume of a cell
     * 
     * @param lcoords Logical coordinates of cell left bottom corner.
     * @param itree Index of the tree containing the cell.
     * @param dxl Cell spacing in logical coordinates
     * @param use_ghostzones Set to true if the indices are zero-offset, false if they are 
     *                       ngz-offset
     * @return double The volume of the requested cell.
     */
    double
    GRACE_HOST get_cell_volume(
      std::array<double, GRACE_NSPACEDIM> const& lcoords 
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const ;
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
    GRACE_HOST 
    get_cell_face_surface(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
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
    GRACE_HOST 
    get_cell_face_surface(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q
    , int8_t face 
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
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
    GRACE_HOST 
    get_cell_face_surface(
      std::array<double, GRACE_NSPACEDIM> const& lcoords 
    , int8_t face 
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const ;
    //**************************************************************************************************
    /**
     * @brief Get the length of a cell edge.
     * 
     * @param ijk     Cell indices.
     * @param q       Local quadrant index.
     * @param edge    Cell edge index (between 0 and <code>GRACE_NSPACEDIM</code>).
     * @param dxl     (Logical) Cell coordinate spacing.
     * @param use_ghostzones Set to true if indices are 0-offset, false if they are ngz-offset
     * @return double The length of the cell edge. 
     * NB: By convention, cell edge indices are staggered backwards, meaning that given an index \f$i_e\f$
     * of an edge, this routine returns the length of the edge whose center is located at index:
     * \f[
     *   (I,J,K) = (i-\frac{1-\delta_{i,i_f}}{2}, j-\frac{1-\delta_{j,i_f}}{2}, k-\frac{1-\delta_{k,i_f}}{2})
     * \f]
     */
    double GRACE_HOST 
    get_cell_edge_length(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk
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
     * @param edge    Cell edge index (between 0 and <code>GRACE_NSPACEDIM</code>).
     * @param dxl     (Logical) Cell coordinate spacing.
     * @param use_ghostzones Set to true if indices are 0-offset, false if they are ngz-offset
     * @return double The length of the cell edge. 
     * NB: By convention, cell edge indices are staggered backwards, meaning that given an index \f$i_e\f$
     * of an edge, this routine returns the length of the edge whose center is located at index:
     * \f[
     *   (I,J,K) = (i-\frac{1-\delta_{i,i_f}}{2}, j-\frac{1-\delta_{j,i_f}}{2}, k-\frac{1-\delta_{k,i_f}}{2})
     * \f]
     */
    double GRACE_HOST 
    get_cell_edge_length(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk
    , int64_t q 
    , int8_t edge
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const ;
    //**************************************************************************************************
    /**
     * @brief Get the length of a cell edge.
     * 
     * @param lcoords Logical coordinates of cell's lowest corner (z-ordering) in 
     *                tree <code>itree</code>'s coordinate system.
     * @param itree   Source tree id.
     * @param edge    Cell edge index (between 0 and <code>GRACE_NSPACEDIM</code>).
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
    GRACE_HOST get_cell_edge_length(
      std::array<double, GRACE_NSPACEDIM> const& lcoords
    , int8_t edge
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const ;
    //**************************************************************************************************
    /**
     * @brief Get the device coord system object
     * 
     * @return cartesian_device_coordinate_system_impl_t A lightweight coordinate system object
     *                                                   whose methods are accessible from device.
     */
    cartesian_device_coordinate_system_impl_t GRACE_ALWAYS_INLINE 
    get_device_coord_system() const {
        return cartesian_device_coordinate_system_impl_t{tree_vertices_,tree_spacings_} ;
    }

 private:
    //**************************************************************************************************
    //! Tree vertices and spacings        
    Kokkos::View<double*,grace::default_space> tree_vertices_, tree_spacings_ ;
    //**************************************************************************************************
    /**
     * @brief Construct a new cartesian coordinate system.
     */
    cartesian_coordinate_system_impl_t() ; 
    //**************************************************************************************************
    /**
     * @brief Destroy the cartesian coordinate system
     */
    ~cartesian_coordinate_system_impl_t() = default ; 
    //**************************************************************************************************
    static constexpr size_t longevity = GRACE_COORDINATE_SYSTEM ; //!< Singleton longevity
    //**************************************************************************************************
    friend class utils::singleton_holder<cartesian_coordinate_system_impl_t, memory::default_create> ;         //!< Give access
    friend class memory::new_delete_creator<cartesian_coordinate_system_impl_t,memory::new_delete_allocator> ; //!< Give access
    //**************************************************************************************************
} ; 

} /* namespace grace */

#endif /* GRACE_AMR_COORDINATES_SYSTEMS_HH */