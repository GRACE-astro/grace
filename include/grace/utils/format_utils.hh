/**
 * @file format_utils.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Some utilities to pipe common containers into a <code>std::ostream</code>.
 * @date 2024-06-21
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

#ifndef GRACE_UTILS_FORMAT_UTILS_HH
#define GRACE_UTILS_FORMAT_UTILS_HH

#include <Kokkos_Core.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <iomanip>
#include <type_traits>

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::ostream&>::type
operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << std::fixed << std::setprecision(15) << vec[i]; 
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template <typename T>
typename std::enable_if<!std::is_arithmetic<T>::value, std::ostream&>::type
operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i]; 
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, std::ostream&>::type
operator<<(std::ostream& os, const std::array<T,N>& arr) {
    os << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        os << std::fixed << std::setprecision(15) << arr[i]; 
        if (i != arr.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template <typename T, size_t N>
typename std::enable_if<!std::is_arithmetic<T>::value, std::ostream&>::type
operator<<(std::ostream& os, const std::array<T,N>& arr) {
    os << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        os << arr[i]; 
        if (i != arr.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::ostream&>::type
operator<<(std::ostream& os, const Kokkos::View<T*,Kokkos::HostSpace>& view) {
    os << "(";
    for (size_t i = 0; i < view.extent(0); ++i) {
        os << std::fixed << std::setprecision(15) << view(i); 
        if (i != view.extent(0) - 1) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::ostream&>::type
operator<<(std::ostream& os, const Kokkos::View<T**,Kokkos::HostSpace>& view) {
    os << "[";
    for (size_t i = 0; i < view.extent(1); ++i) {
        os << "[";
        for( size_t j=0; j < view.extent(0); ++j) {
            os << std::fixed << std::setprecision(15) << view(j,i); 
            if (i != view.extent(0) - 1) {
                os << ", ";
            }
        }
        os << "],\n" ; 
    }
    os << "]";
    return os;
}

#endif 