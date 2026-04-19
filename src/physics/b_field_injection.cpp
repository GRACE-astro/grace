/**
 * @file b_field_injection.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Mid-run poloidal B-field injection at co_tracker locations.
 * @date 2026-04-18
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
 */

#include <grace_config.h>

#include <grace/physics/b_field_injection.hh>

#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/config/config_parser.hh>
#include <grace/IO/diagnostics/co_tracker.hh>
#include <grace/physics/grmhd_B_from_A.hh>
#include <grace/physics/grmhd.hh>

#include <Kokkos_Core.hpp>

#include <array>

namespace grace {

namespace {
    bool s_has_fired = false ;
}

bool b_field_injection_has_fired() { return s_has_fired ; }
void b_field_injection_set_fired(bool v) { s_has_fired = v ; }

void maybe_inject_b_field()
{
    bool const enable = get_param<bool>("b_field_injection", "enable") ;
    if ( !enable ) return ;
    if ( s_has_fired ) return ;

    double const t_trigger = get_param<double>("b_field_injection", "trigger_time") ;
    double const t_now     = grace::runtime::get().time() ;
    if ( t_now < t_trigger ) return ;

    auto& tracker = grace::co_tracker::get() ;
    ASSERT(tracker.is_active(),
        "b_field_injection enabled but co_tracker is not active.") ;

    int const n_cos = tracker.get_n_cos() ;
    ASSERT(n_cos >= 1 && n_cos <= 2,
        "b_field_injection expects 1 or 2 compact objects, got " << n_cos) ;

    bool const is_binary = get_param<bool>("grmhd", "Avec_ID", "is_binary") ;
    double const B_target = get_param<double>("grmhd", "Avec_ID", "Bmax_target") ;
    double const r1 = get_param<double>("grmhd", "Avec_ID", "radius_1") ;

    GRACE_INFO("Injecting poloidal B field at t = {} (trigger {}), n_cos = {}",
               t_now, t_trigger, n_cos) ;

    {
        auto const& co0 = tracker.get(0) ;
        auto const c1 = co0->get_loc() ;
        GRACE_INFO("  CO 0 ({}): center = ({}, {}, {}), radius = {}",
                   co0->get_name(), c1[0], c1[1], c1[2], r1) ;
        setup_confined_poloidal_B_field_single(c1, r1, B_target) ;
    }

    if ( is_binary && n_cos >= 2 ) {
        double const r2 = get_param<double>("grmhd", "Avec_ID", "radius_2") ;
        auto const& co1 = tracker.get(1) ;
        auto const c2 = co1->get_loc() ;
        GRACE_INFO("  CO 1 ({}): center = ({}, {}, {}), radius = {}",
                   co1->get_name(), c2[0], c2[1], c2[2], r2) ;
        setup_confined_poloidal_B_field_single(c2, r2, B_target) ;
    }

    set_conservs_from_prims() ;
    Kokkos::fence() ;

    s_has_fired = true ;
    GRACE_INFO("B field injection complete.") ;
}

}
