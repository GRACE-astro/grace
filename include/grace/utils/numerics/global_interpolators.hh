/**
 * @file global_interpolators.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-11-25
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
#ifndef GRACE_UTILS_NUMERICS_GLOBAL_INTERPOLATORS_HH
#define GRACE_UTILS_NUMERICS_GLOBAL_INTERPOLATORS_HH

#include <grace_config.h>

#include <grace/utils/inline.h>

#include <grace/utils/numerics/grid_transfer.hh>

#include <Kokkos_Core.hpp>

namespace grace {
/**
 * @brief Interpolate a grid variable from cell corners to cell centers ignoring the ghostzones.
 * 
 * @tparam order Order of interpolation
 * @tparam view_t Type of input/output Views
 * @param [in]  in_view Input View
 * @param [out] out_view Output View
 * The input/output format for this routine is the same as obtained when taking a subview
 * of a var_array_t where a single variable is selected.
 * NB: For higher-than-second order accuracy, the in_view must be defined in the ghostzones.
 */
template< size_t order 
        , typename in_view_t 
        , typename out_view_t >
void interp_corner_to_center(
    in_view_t in_view,
    out_view_t out_view 
) 
{
    DECLARE_GRID_EXTENTS ; 

    using namespace Kokkos ; 
    using namespace grace  ; 
    static constexpr const size_t rank = in_view_t::Rank ; 

    static_assert(rank==GRACE_NSPACEDIM+1, 
                 "In global interpolate in and out views must have the same rank as a "
                 "var array or one less.") ; 
    using interp_t = corner_to_center<order> ; 
    MDRangePolicy<Rank<rank>,default_execution_space> policy{
        {VEC(0,0,0),0}, {nx,ny,nz,nq}
    } ; 
    parallel_for(
            GRACE_EXECUTION_TAG("utils", "global_interpolation")
        , policy 
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int q)
    {
        auto const sview_in  = subview(in_view , VEC(ALL(),ALL(),ALL()), q) ; 
        out_view(VEC(i,j,k),q) = interp_t::interpolate(sview_in,VEC(i,j,k)) ; 
    }
    ) ; 
    
}
/**
 * @brief Interpolate a certain amount of grid variables
 *        from cell corners to cell centers ignoring the ghostzones.
 * 
 * @tparam order Order of interpolation
 * @tparam view_t Type of input/output Views
 * @param [in]  in_view Input View
 * @param [out] out_view Output View
 * @param [in] var_in_idx Indices of interpolated variables in <code>in_view</code>
 * This function can take a view which represents multiple variables.
 * The format of the in/out views must be the same as for a <code>grace::var_array_t<GRACE_NSPACEDIM></code>
 * in that case.
 * NB: For higher-than-second order accuracy, the in_view must be defined in the ghostzones.
 */
template< size_t order 
        , typename in_view_t 
        , typename out_view_t >
void interp_corner_to_center(
    in_view_t in_view,
    out_view_t out_view,
    Kokkos::View<int*> var_in_idx 
) 
{
    DECLARE_GRID_EXTENTS ; 

    using namespace Kokkos ; 
    using namespace grace  ; 
    static constexpr const size_t rank = in_view_t::Rank ; 

    static_assert( rank==GRACE_NSPACEDIM+2, 
                 "In global interpolate in and out views must have the same rank as a "
                 "var array or one less.") ; 
    using interp_t = corner_to_center<order> ; 
    
    auto nvars = out_view.extent(GRACE_NSPACEDIM) ; 
    MDRangePolicy<Rank<rank>,default_execution_space> policy{
        {VEC(0,0,0),0,0}, {nx,ny,nz,nvars, nq}
    } ; 
    parallel_for(
            GRACE_EXECUTION_TAG("utils", "global_interpolation")
        , policy 
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
    {
        auto const sview_in  = subview(in_view , VEC(ALL(),ALL(),ALL()), var_in_idx(ivar), q) ; 
        out_view(VEC(i,j,k),ivar,q) = interp_t::interpolate(sview_in,VEC(i,j,k)) ; 
    }
    ) ;
    
}
/**
 * @brief Interpolate a certain amount of grid variables
 *        from cell corners to cell centers ignoring the ghostzones.
 * 
 * @tparam order Order of interpolation
 * @tparam view_t Type of input/output Views
 * @param [in]  in_view Input View
 * @param [out] out_view Output View
 * @param [in] var_in_idx Indices of interpolated variables in <code>in_view</code>
 * This function can take a view which represents multiple variables.
 * The format of the in/out views must be the same as for a <code>grace::var_array_t<GRACE_NSPACEDIM></code>
 * in that case.
 * NB: For higher-than-second order accuracy, the in_view must be defined in the ghostzones.
 */
template< size_t order 
        , typename in_view_t 
        , typename out_view_t >
void interp_corner_to_center_scatter_out(
    in_view_t in_view,
    out_view_t out_view,
    Kokkos::View<int*> var_out_idx 
) 
{
    DECLARE_GRID_EXTENTS ; 

    using namespace Kokkos ; 
    using namespace grace  ; 
    static constexpr const size_t rank = in_view_t::Rank ; 

    static_assert( rank==GRACE_NSPACEDIM+2, 
                 "In global interpolate in and out views must have the same rank as a "
                 "var array or one less.") ; 
    using interp_t = corner_to_center<order> ; 
    
    auto nvars = in_view.extent(GRACE_NSPACEDIM) ; 
    MDRangePolicy<Rank<rank>,default_execution_space> policy{
        {VEC(0,0,0),0,0}, {nx,ny,nz,nvars, nq}
    } ; 
    parallel_for(
            GRACE_EXECUTION_TAG("utils", "global_interpolation")
        , policy 
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
    {
        auto const sview_in  = subview(in_view , VEC(ALL(),ALL(),ALL()), ivar, q) ; 
        out_view(VEC(i+ngz,j+ngz,k+ngz),var_out_idx(ivar),q) = interp_t::interpolate(sview_in,VEC(i+ngz,j+ngz,k+ngz)) ; 
    }
    ) ;
    
}


} /* namespace grace */


#endif /* GRACE_UTILS_NUMERICS_GLOBAL_INTERPOLATORS_HH */