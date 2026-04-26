/**
 * @file nan_check.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Implementation of grace::scan_nans() and check_nans_and_act_if_due().
 *
 * @copyright This file is part of GRACE. See LICENSE.
 */
#include <grace_config.h>

#include <grace/system/nan_check.hh>

#include <grace/system/runtime_functions.hh>
#include <grace/system/print.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/execution_tag.hh>
#include <grace/errors/error.hh>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <string>

namespace grace {

namespace {

/**
 * @brief Count NaN values in a Kokkos::View using rank-aware iteration via
 *        MDRangePolicy. Routes the access through the view's own indexing
 *        operator, so it is correct regardless of padding, layout, or
 *        whether the view is a subview with non-contiguous strides.
 *
 * Supports rank-4 (2D builds) and rank-5 (3D builds) var_array_t shapes;
 * static_asserts on any other rank.
 */
template <typename ViewT>
std::size_t count_nans_view(ViewT const& v, char const* tag)
{
    if (v.size() == 0) return 0 ;
    using exec = typename ViewT::execution_space ;
    std::int64_t count = 0 ;
    if constexpr (ViewT::rank == 5) {
        Kokkos::parallel_reduce(
            GRACE_EXECUTION_TAG("NAN_CHECK", tag),
            Kokkos::MDRangePolicy<Kokkos::Rank<5>, exec>(
                {0,0,0,0,0},
                { static_cast<std::int64_t>(v.extent(0))
                , static_cast<std::int64_t>(v.extent(1))
                , static_cast<std::int64_t>(v.extent(2))
                , static_cast<std::int64_t>(v.extent(3))
                , static_cast<std::int64_t>(v.extent(4)) }
            ),
            KOKKOS_LAMBDA(int i, int j, int k, int q, int r, std::int64_t& acc) {
                if (std::isnan(v(i,j,k,q,r))) acc++ ;
            },
            count
        ) ;
    } else if constexpr (ViewT::rank == 4) {
        Kokkos::parallel_reduce(
            GRACE_EXECUTION_TAG("NAN_CHECK", tag),
            Kokkos::MDRangePolicy<Kokkos::Rank<4>, exec>(
                {0,0,0,0},
                { static_cast<std::int64_t>(v.extent(0))
                , static_cast<std::int64_t>(v.extent(1))
                , static_cast<std::int64_t>(v.extent(2))
                , static_cast<std::int64_t>(v.extent(3)) }
            ),
            KOKKOS_LAMBDA(int i, int j, int k, int q, std::int64_t& acc) {
                if (std::isnan(v(i,j,k,q))) acc++ ;
            },
            count
        ) ;
    } else {
        static_assert(ViewT::rank == 4 || ViewT::rank == 5,
                      "count_nans_view: only rank-4 and rank-5 views are supported") ;
    }
    return static_cast<std::size_t>(count) ;
}

}

std::size_t scan_nans()
{
    auto& vlist = grace::variable_list::get() ;

    std::size_t local_total = 0 ;

    local_total += count_nans_view(vlist.getstate(), "state") ;
    local_total += count_nans_view(vlist.getaux(),   "aux") ;

    auto& sstate = vlist.getstaggeredstate() ;
    local_total += count_nans_view(sstate.face_staggered_fields_x, "stag_face_x") ;
    local_total += count_nans_view(sstate.face_staggered_fields_y, "stag_face_y") ;
    local_total += count_nans_view(sstate.face_staggered_fields_z, "stag_face_z") ;
    #ifdef GRACE_3D
    local_total += count_nans_view(sstate.edge_staggered_fields_xy, "stag_edge_xy") ;
    local_total += count_nans_view(sstate.edge_staggered_fields_xz, "stag_edge_xz") ;
    local_total += count_nans_view(sstate.edge_staggered_fields_yz, "stag_edge_yz") ;
    #endif
    local_total += count_nans_view(sstate.corner_staggered_fields, "stag_corner") ;

    /* MPI sum across ranks. mpi_allreduce<T> requires a typed buffer; size_t
     * mapping isn't guaranteed in mpi_type_utils, so go via uint64_t. */
    std::uint64_t local_u64  = static_cast<std::uint64_t>(local_total) ;
    std::uint64_t global_u64 = 0 ;
    parallel::mpi_allreduce(&local_u64, &global_u64, 1, sc_MPI_SUM) ;

    return static_cast<std::size_t>(global_u64) ;
}

void check_nans_and_act_if_due(bool is_initial)
{
    if (!grace::get_param<bool>("nan_check","enabled")) return ;

    if (is_initial) {
        if (!grace::get_param<bool>("nan_check","check_before_first_step")) return ;
    } else {
        auto const every = grace::get_param<int>("nan_check","check_every") ;
        if (every <= 0) return ;
        auto const iter = grace::get_iteration() ;
        if (iter == 0) return ; // initial pass handles iter 0
        if (static_cast<int>(iter % static_cast<std::size_t>(every)) != 0) return ;
    }

    auto const total = scan_nans() ;
    if (total == 0) {
        GRACE_VERBOSE("NaN check: clean (iteration {}).", grace::get_iteration()) ;
        return ;
    }

    auto const action = grace::get_param<std::string>("nan_check","action") ;
    auto const iter   = grace::get_iteration() ;
    auto const stage  = is_initial ? "before first step" : "in-loop" ;

    if (action == "warn") {
        GRACE_WARN("NaN check ({}, iteration {}): found {} NaN values in evolved arrays. Continuing.",
                   stage, iter, total) ;
    } else if (action == "terminate") {
        GRACE_WARN("NaN check ({}, iteration {}): found {} NaN values in evolved arrays. "
                   "Requesting clean termination.", stage, iter, total) ;
        grace::request_termination() ;
    } else if (action == "abort") {
        ERROR("NaN check (" << stage << ", iteration " << iter << "): found " << total
              << " NaN values in evolved arrays. Aborting.") ;
    } else {
        ERROR("Unknown nan_check.action: '" << action
              << "'. Expected one of: warn, terminate, abort.") ;
    }
}

}
