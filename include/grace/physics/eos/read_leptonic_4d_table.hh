/**
 * @file read_leptonic_4d_table.hh
 * @author Marie Cassing (mcassing@itp.uni-frankfurt.de)
 * @author Keneth Miler (miler@itp.uni-frankfurt.de)
 * @brief  Host-side reader for the Margherita-style 4D leptonic EOS.
 *         Loads the baryon side via the existing read_eos_table()
 *         pipeline and the lepton side from the leptonic HDF5 file.
 * @date   2026-05-15
 *
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale (GRACE).
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas.
 * Copyright (C) 2026 Marie Cassing, Keneth Miler
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef GRACE_PHYS_EOS_READ_LEPTONIC_4D_TABLE_HH
#define GRACE_PHYS_EOS_READ_LEPTONIC_4D_TABLE_HH

#include <grace_config.h>
#include <grace/physics/eos/leptonic_eos_4d.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/utils/grace_utils.hh>
#include <Kokkos_Core.hpp>

namespace grace {

/**
 * @brief Construct a leptonic_eos_4d_t from:
 *          - the baryon EOS table referenced by [eos.tabulated_eos.*]
 *            (read via the existing read_eos_table() pipeline);
 *          - the leptonic HDF5 file referenced by [eos.leptonic.*].
 */
grace::leptonic_eos_4d_t read_leptonic_4d_table() ;

/**
 * @brief Generate a cold-slice table for leptonic_eos_4d_t by solving
 *        muon-inclusive beta equilibrium at T = T_cold over the full
 *        rho axis of the baryon table.
 *
 * @param eos             Fully initialised leptonic EOS (baryon + lepton tables
 *                        must be valid; cold_table is not used).
 * @param T_cold          Temperature [MeV] at which to evaluate beta equilibrium.
 * @param d_data          Output: [nrho, N_CTAB_VARS] table data on device.
 * @param d_rho           Output: [nrho] log(rho) axis on device.
 * @param output_filename If non-empty, dump the generated table to this path
 *                        in the same plain-text .grace format the reader expects.
 */
void generate_leptonic_cold_table(
    const leptonic_eos_4d_t& eos,
    double T_cold,
    Kokkos::View<double**, grace::default_execution_space>& d_data,
    Kokkos::View<double*,  grace::default_execution_space>& d_rho,
    const std::string& output_filename = "") ;

} /* namespace grace */

#endif /* GRACE_PHYS_EOS_READ_LEPTONIC_4D_TABLE_HH */
