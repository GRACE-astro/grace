/**
 * @file bitset.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-24
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

#ifndef GRACE_UTILS_BITSET_HH
#define GRACE_UTILS_BITSET_HH

#include <cstdint>
#include <cstddef>

#include <Kokkos_Core.hpp>

namespace grace {


template <size_t N>
struct bitset_t {
    static constexpr size_t kWords = (N + 63) / 64;
    uint64_t words[kWords] = {};

    // Set bit i
    KOKKOS_INLINE_FUNCTION void set(size_t i) {
        words[i >> 6] |= (uint64_t(1) << (i & 63));
    }

    // Clear bit i
    KOKKOS_INLINE_FUNCTION void unset(size_t i) {
        words[i >> 6] &= ~(uint64_t(1) << (i & 63));
    }

    // Test bit i
    KOKKOS_INLINE_FUNCTION bool test(size_t i) const {
        return (words[i >> 6] >> (i & 63)) & 1;
    }

    // Clear all bits
    KOKKOS_INLINE_FUNCTION void reset() {
        for (size_t w = 0; w < kWords; ++w) words[w] = 0;
    }

    // Set all bits 
    KOKKOS_INLINE_FUNCTION void set_all() {
        for (size_t w = 0; w < kWords; ++w) words[w] = ~uint64_t(0);
    }
};

}

#endif /* GRACE_UTILS_BITSET_HH */