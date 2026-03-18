/**
 * @file read_eos_table.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
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

#ifndef GRACE_PHYS_EOS_UNIT_SYSTEM_HH
#define GRACE_PHYS_EOS_UNIT_SYSTEM_HH


#include <grace_config.h>

#include <grace/physics/eos/physical_constants.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

namespace grace {

struct unit_system {
    double length, time, mass;
    double velocity, acceleration, force;
    double pressure, energy;
    double energy_density, mass_density;

private:
    constexpr unit_system(double _length, double _time, double _mass)
    : length          (_length)
    , time            (_time)
    , mass            (_mass)
    , velocity        (_length/_time)
    , acceleration    (_length/_time/_time)
    , force           (_mass*_length/_time/_time)
    , pressure        (_mass/_time/_time/_length)
    , energy          (_mass*_length*_length/_time/_time)
    , energy_density  (_mass/_time/_time/_length/_length)
    , mass_density    (_mass/_length/_length/_length)
    {}

public:
    unit_system operator/ (unit_system const& other) const 
    {
        return unit_system(length/other.length, time/other.time, mass/other.mass) ; 
    }

    static constexpr unit_system make_constexpr(double length, double time, double mass) {
        return unit_system(length, time, mass);
    }

    static unit_system make(double length, double time, double mass) {
        return unit_system(length, time, mass);
    }
};


constexpr auto SI_units = unit_system::make_constexpr(1.0,1.0,1.0) ; 
constexpr auto CGS_units = unit_system::make_constexpr(0.01, 1.0, 0.001);
constexpr auto GEOM_units = unit_system::make_constexpr(
    physical_constants::G_si * physical_constants::Msun_si /physical_constants::clight_si/physical_constants::clight_si, 
    physical_constants::G_si * physical_constants::Msun_si /physical_constants::clight_si/physical_constants::clight_si/physical_constants::clight_si, 
    physical_constants::Msun_si
);

}

#endif /*GRACE_PHYS_EOS_UNIT_SYSTEM_HH*/