/**
 * @file gridloop.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-07-18
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

#ifndef GRACE_UTILS_GRIDLOOPS_HH
#define GRACE_UTILS_GRIDLOOPS_HH

#include <grace_config.h>
#include <grace/utils/inline.h>
#include <grace/utils/device.h>
#include <grace/amr/amr_functions.hh>
#include <grace/system/print.hh>

#include <utility>
#include <tuple>
#include <array> 
#include <omp.h>

#include <iostream> 

namespace grace {

namespace detail {

/**
 * @brief Util to check whether a template  
 *        pack is a series of pairs of size_t .
 * \ingroup utils
 * \cond grace_detail 
 * @tparam Args Parameter pack to be checked 
 */
template<typename... Args>
struct are_pairs_of_size_t {
    static constexpr bool value = std::conjunction<std::is_same<Args, std::pair<std::size_t, std::size_t>>...>::value;
};
/**
 * @brief Value of are_pairs_of_size_t util.
 * \ingroup utils
 * \cond grace_detail 
 * @tparam Args Parameter pack to be checked 
 */
template<typename... Args>
constexpr bool are_pairs_of_size_t_v = are_pairs_of_size_t<Args...>::value ;
/**
 * @brief Unpack a 1D index into N indices given ranges.
 * \ingroup utils
 * \cond grace_detail
 * @tparam Tuple The tuple containing index ranges
 * @tparam Is    Index sequence
 * @param index  The 1D collapsed index.
 * @param ranges Ranges for individual indices.
 * @return auto Tuple containing unpacked indices.
 */
template< typename Tuple 
        , std::size_t ... Is >
auto GRACE_ALWAYS_INLINE GRACE_HOST 
index_unpacker(std::size_t index, Tuple const& ranges, std::index_sequence<Is...> ) {
    std::tuple<decltype(Is)...> result;

    auto unpack = [&](auto&... indices) {
        size_t sizes[] = { (std::get<Is>(ranges).second - std::get<Is>(ranges).first)... };
        (..., (indices = (index % sizes[Is] + std::get<Is>(ranges).first), index /= sizes[Is]));
    };

    std::apply(unpack, result);
    return result;
}
/**
 * @brief Get the total number of iteration for collapsed ND loop.
 * \ingroup utils 
 * \cond grace_detail
 * @tparam Tuple Type of tuple containing index ranges
 * @tparam Is    Index sequence
 * @param ranges Ranges of individual loop indices.
 * @return std::size_t The extent of the collapsed ND loop.
 */
template<typename Tuple, std::size_t... Is>
std::size_t get_cumulative_range(const Tuple& ranges, std::index_sequence<Is...>) {
    return (... * (std::get<Is>(ranges).second - std::get<Is>(ranges).first));
}
/**
 * @brief Implementation of host loop with closure as body
 * \ingroup utils 
 * \cond grace_detail
 * @tparam omp_parallel Whether the loop should be OpenMP parallelized.
 */
template< bool omp_parallel >
struct host_loop_impl_t {
    template< typename Ft
            , typename ... Idxt > 
    void GRACE_HOST GRACE_ALWAYS_INLINE 
    loop(Ft&& _func, Idxt&& ... ranges ) ; 
} ; 
/**
 * @brief Specialization of <code>host_loop_impl_t</code>
 *        without OpenMP parallelization.
 * \ingroup utils 
 * \cond grace_detail 
 */
template<>
struct host_loop_impl_t<false> {

    template< typename Ft
            , typename ... Idxt >
    static void GRACE_HOST GRACE_ALWAYS_INLINE 
    loop(Ft&& _func, Idxt && ... args ) {
        static_assert( are_pairs_of_size_t_v<Idxt...>
                     , "Loop index ranges must be provided as "
                       "a parameter pack that can be interpreted "
                       "as a list of std::pair<std::size_t,std::size_t>.") ; 
        static constexpr const std::size_t ndim = sizeof...(Idxt) ; 
        auto const ranges = std::make_tuple(args...) ; 
        const size_t ncumulative = get_cumulative_range(ranges,std::index_sequence_for<Idxt...>{}) ;
        for (std::size_t i = 0UL; i < ncumulative; i+=1UL) {
            auto indices = index_unpacker(i, ranges, std::index_sequence_for<Idxt...>{}) ; 
            std::apply(_func, indices) ; 
        }
    }

} ; 
/**
 * @brief Specialization of <code>host_loop_impl_t</code>
 *        with OpenMP parallelization.
 * \ingroup utils 
 * \cond grace_detail 
 */
template<>
struct host_loop_impl_t<true> {

    template< typename Ft
            , typename ... Idxt >
    static void GRACE_HOST GRACE_ALWAYS_INLINE 
    loop(Ft&& _func, Idxt && ... args ) {
        static_assert( are_pairs_of_size_t_v<Idxt...>
                     , "Loop index ranges must be provided as "
                       "a parameter pack that can be interpreted "
                       "as a list of std::pair<std::size_t,std::size_t>.") ; 
        static constexpr const std::size_t ndim = sizeof...(Idxt) ; 
        auto const ranges = std::make_tuple(args...) ; 
        const size_t ncumulative = get_cumulative_range(ranges,std::index_sequence_for<Idxt...>{}) ;
        #pragma omp parallel for 
        for (std::size_t i = 0UL; i < ncumulative; i+=1UL) {
            auto indices = index_unpacker(i, ranges, std::index_sequence_for<Idxt...>{}) ; 
            std::apply(_func, indices) ; 
        }
    }

} ;

} // namespace detail 
/**
 * @brief Perform a potentially OMP parallelized 
 *        ND loop on host
 * \ingroup utils
 * @tparam omp_parallel Whether OMP parallelization should be used.
 * @tparam Ft           Type of loop body lambda.
 * @tparam Idxt         Type index ranges (should be a series of <code>std::pair<std::size_t,std::size_t></code>)
 * @param _func         Loop body as a lambda.
 * @param args          Loop index ranges (as <code>std::pair<std::size_t,std::size_t></code>)
 * 
 * Note that on modern CPUs launching an OMP parallel region takes approximately 
 * 40k clock cycles. Be aware of that when deciding whether it's worth for your 
 * loop to be parallelized.
 */
template< bool omp_parallel
        , typename Ft 
        , typename ... Idxt > 
void GRACE_ALWAYS_INLINE GRACE_HOST 
host_ndloop(Ft&& _func, Idxt&& ... args ) {
    using namespace detail ; 
    host_loop_impl_t<omp_parallel>::template loop<Ft>(
          std::forward<Ft>(_func)
        , std::forward<Idxt>(args)...
    ) ; 
}
/**
 * @brief Perform a potentially OMP parallelized 
 *        host loop over the computational domain
 * \ingroup utils
 * @tparam omp_parallel  Whether OMP parallelization should be used.
 * @tparam Ft            Type of loop body lambda.
 * @param _func          Loop body as a lambda.
 * @param stagger        Whether the loop is staggered in each direction.
 * @param include_ghosts Whether the loop should run in the ghostzones.
 * 
 * This could be utilized in very similar fashion to Kokkos::parallel_loop
 * except for the fact that this loop \b always runs on host. This is useful for a lot of 
 * book-keeping tasks that can only run on host such as the computation of various coordinate
 * related quantities and amr operations. Initial data often also relies on Host for at least 
 * part of its computations.
 * Example
 * \code 
 * DECLARE_GRID_PROPERTIES ; 
 * 
 * grace::coord_array_t<GRACE_NSPACEDIM> phys_coords{
 *        "physical_coordinates"
 *      , VEC( nx + 2*ngz, ny + 2*ngz, nz + 2*ngz)
 *      , GRACE_NSPACEDIM
 *      , nq 
 * } ; 
 * 
 * auto h_phys_coords = Kokkos::create_mirror_view(phys_coords) ; 
 * 
 * auto coord_system = grace::coordinate_system::get() ; 
 * 
 * grace::host_grid_loop(
 *    [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
 *          auto const itree = grace::amr::get_quadrant_owner(q) ; 
 *          auto pcoords = coord_system.get_physical_coordinates(
 *              {VEC(i,j,k)}, q, itree, true 
 *          ) ;
 *          EXPR(
 *          h_phys_coords(VEC(i,j,k),0,q) = pcoords[0];,
 *          h_phys_coords(VEC(i,j,k),1,q) = pcoords[1];,
 *          h_phys_coords(VEC(i,j,k),2,q) = pcoords[2];
 *          )
 * }, true ) ; 
 * 
 * Kokkos::deep_copy(phys_coords,h_phys_coords) ; 
 * \endcode
 * 
 * Note that on modern CPUs launching an OMP parallel region takes approximately 
 * 40k clock cycles. Be aware of that when deciding whether it's worth for your 
 * loop to be parallelized.
 */
template< bool omp_parallel
        , typename Ft > 
void GRACE_ALWAYS_INLINE GRACE_HOST 
host_grid_loop(Ft&& _func, std::array<bool,GRACE_NSPACEDIM> stagger={VEC(false,false,false)}, bool include_ghosts=false ) {

    using namespace grace ; 
    using namespace detail ; 

    auto const nq = amr::get_local_num_quadrants() ; 
    size_t nx, ny, nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto const ngz = amr::get_n_ghosts() ; 
    
    std::size_t const nx_st = nx + static_cast<int>(stagger[0]) ; 
    std::pair<std::size_t, std::size_t> range_x{ 
          include_ghosts ? 0 : ngz 
        , include_ghosts ? nx_st + 2*ngz : nx_st + ngz 
    } ;
    std::size_t const ny_st = ny + static_cast<int>(stagger[1]) ; 
    std::pair<std::size_t, std::size_t> range_y{ 
          include_ghosts ? 0 : ngz 
        , include_ghosts ? ny_st + 2*ngz : ny_st + ngz 
    } ;  
    #ifdef GRACE_3D
    std::size_t const nz_st = nz + static_cast<int>(stagger[2]) ; 
    std::pair<std::size_t, std::size_t> range_z{ 
          include_ghosts ? 0 : ngz 
        , include_ghosts ? nz_st + 2*ngz : nz_st + ngz 
    } ;
    #endif 
    
    std::pair<std::size_t, std::size_t> range_q{
        0, nq 
    } ; 

    host_ndloop<omp_parallel,Ft>( std::forward<Ft>(_func)
                                , VEC( std::move(range_x) 
                                     , std::move(range_y) 
                                     , std::move(range_z) )
                                , std::move(range_q) ) ; 
}


} // namespace grace 

#endif 