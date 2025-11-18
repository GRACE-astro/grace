/**
 * @file coordinates.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-06-14
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference 
 * methods to simulate relativistic astrophysical systems and plasma
 * dynamics.
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
#ifndef F25FCED7_32FD_48EF_A294_4D29ABC78524
#define F25FCED7_32FD_48EF_A294_4D29ABC78524

#include <grace_config.h>

#include <grace/data_structures/variable_properties.hh>

#include <array> 

namespace grace { 
/*****************************************************************************************/
/**
 * @brief Fill cell coordinates array.
 * \ingroup coordinates 
 */
void fill_cell_coordinates( scalar_array_t<GRACE_NSPACEDIM>&
                          , scalar_array_t<GRACE_NSPACEDIM>&
                          , scalar_array_t<GRACE_NSPACEDIM>&
                          , cell_vol_array_t<GRACE_NSPACEDIM>&
                          , staggered_coordinate_arrays_t& ) ; 
/*****************************************************************************************/
/**
 * @brief Fill a device view with physical coordinates on the grid. 
 * \ingroup coordinates
 * @param pcoords           The view to be filled with coordinates.
 * @param cell_coordinates  Coordinates within the cell where physical coordinates
 *                          should be computed (defaults to 0.5 in all directions, i.e. 
 *                          the center of the cell).
 * @param spherical         If true, returned coordinates are spherical-like, else cartesian-like.
 * @param stag              Staggering (increases pcoords array dimension by 1 in each staggered direction).
 */
void fill_physical_coordinates( coord_array_t<GRACE_NSPACEDIM>& pcoords 
                              , std::array<double, GRACE_NSPACEDIM> const& cell_coordinates = {VEC(0.5,0.5,0.5)} 
                              , grace::var_staggering_t stag = grace::STAG_CENTER
                              , bool spherical = false ) ;
/*****************************************************************************************/
/**
 * @brief Fill a device view with physical coordinates on the grid. 
 * \ingroup coordinates
 * @param pcoords           The view to be filled with coordinates.
 * @param stag              Staggering (increases pcoords array dimension by 1 in each staggered direction).
 * @param spherical         If true, returned coordinates are spherical-like, else cartesian-like.
 */
void fill_physical_coordinates( coord_array_t<GRACE_NSPACEDIM>& pcoords 
                              , grace::var_staggering_t stag
                              , bool spherical = false ) ;
/*****************************************************************************************/
/**
 * @brief Fill a device view with jacobians of the logical-to-physical 
 *        coordinate transformation and its inverse. 
 * \ingroup coordinates
 * @param jac               The view to be filled with the jacobian matrix.
 * @param inv_jac           The view to be filled with the inverse jacobian matrix.
 * @param cell_coordinates  Coordinates within the cell where jacobian matrices
 *                          should be computed (defaults to 0.5 in all directions, i.e. 
 *                          the center of the cell).
 */
void fill_jacobian_matrices( jacobian_array_t& jac 
                           , jacobian_array_t& inv_jac 
                           , std::array<double, GRACE_NSPACEDIM> const& cell_coordinates = {VEC(0.5,0.5,0.5)} 
                           ) ; 
/*****************************************************************************************/
namespace detail{
template< typename ViewT >
static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE std::array<double,3> 
apply_jacobian_vec(std::array<double,3> const& v, ViewT& J)
{
    std::array<double,3> w{0,0,0} ; 
    #pragma unroll 9
    for( int ii=0; ii<GRACE_NSPACEDIM; ++ii){
        for(int jj=0; jj<GRACE_NSPACEDIM; ++jj){
            w[ii] += J(ii,jj) * v[jj] ;  
        }
    }
    return std::move(w) ; 
}

template< typename ViewT >
static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE std::array<double,6> 
apply_jacobian_symtens(std::array<double,6> const& v, ViewT& J)
{
    int idx[3][3] = {
          {0,1,2}
        , {1,3,4}
        , {2,4,5}
    } ; 

    std::array<double,6> w{0,0,0,0,0,0} ; 
    #pragma unroll 
    for( int ii=0; ii<GRACE_NSPACEDIM; ++ii){
        for(int jj=0; jj<GRACE_NSPACEDIM; ++jj){
            for( int kk=0; kk<GRACE_NSPACEDIM; ++kk){
                for( int ll=0; ll<GRACE_NSPACEDIM; ++ll){
                    w[idx[ii][jj]] += J(ii,ll) * J(jj,kk) * v[idx[ll][kk]] ; 
                }
            }
        }
    }
    return std::move(w) ; 
}

} /* namespace detail*/

} /* namespace grace */ 

#endif /* F25FCED7_32FD_48EF_A294_4D29ABC78524 */
