/**
 * @file read_leptonic_4d_table.cpp
 * @author Marie Cassing (mcassing@itp.uni-frankfurt.de)
 * @author Keneth Miler (miler@itp.uni-frankfurt.de)
 * @brief  Reader for the Margherita-style leptonic EOS HDF5, combined
 *         with a GRACE-native baryon table.
 *
 *         The baryon side is loaded via the existing read_eos_table()
 *         pipeline (so it honours [eos.tabulated_eos.table_format] =
 *         "stellarcollapse" or "compose").  The leptonic side is read
 *         here: the electronic and muonic 9-variable tables are
 *         layered on top of the same (rho, T, Y_e) grid as the baryon
 *         table.
 *
 *         Lepton-table conventions:
 *           - Pressures, epsilons, entropies, chemical potentials are
 *             stored in their final units (geometric pressure / eps,
 *             k_B/baryon for s, MeV for mu).  No unit conversion is
 *             applied here.
 *           - The (rho, T) axes from the leptonic file are not used:
 *             we reuse the baryon table's natural-log (rho, T) axes.
 *           - yle_table stores linear Y_le values; ymu_table stores
 *             log(Ymu) values (Margherita convention).
 *           - eos_ylemin/max should equal eos_yemin/max of the baryon
 *             table (Margherita convention; a warning is issued if not).
 *
 *         File layout (h5ls -r):
 *             /nrho /ntemp /nyle /nymu             int
 *             /logrho_table /logtemp_table          double  -- unused
 *             /yle_table  /ymu_table                double[nyle], double[nymu]
 *             /eos_ylemin /eos_ylemax /eos_ymumin /eos_ymumax  double
 *             /electronic_eos_tables   double[9, nyle, nT, nrho]
 *             /muonic_eos_tables       double[9, nymu, nT, nrho]
 *
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

#include <grace/physics/eos/leptonic_eos_4d.hh>
#include <grace/physics/eos/read_leptonic_4d_table.hh>
#include <grace/physics/eos/read_eos_table.hh>
#include <grace/physics/eos/tabulated_eos.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/physics/eos/unit_system.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/error.hh>

#include <hdf5.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace grace {

// ============================================================
//  Small HDF5 helper.
// ============================================================
namespace {

template <typename T>
inline void h5_read(hid_t file, const char* name, T* out, hid_t type)
{
    hid_t ds = H5Dopen(file, name, H5P_DEFAULT) ;
    ASSERT(ds >= 0, "Could not open HDF5 dataset '" << name << "'") ;
    herr_t e = H5Dread(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) ;
    ASSERT(e >= 0, "Failed to read HDF5 dataset '" << name << "'") ;
    H5Dclose(ds) ;
}

} // anonymous

// ============================================================
//  Constants
// ============================================================
namespace {
    // Muon rest mass [MeV] — same value used in leptonic_eos_4d.hh
    constexpr double MU_MASS_MEV = 105.6583755 ;
} // anonymous namespace


// ============================================================
//  npe beta equilibrium (Y_mu fixed)
//
//  Solves  mu_e(Y_e) + mu_p(Y_e + Y_mu) - mu_n(Y_e + Y_mu) = 0
//  for Y_e, with Y_mu held constant.  Used both as the pure-npe
//  fallback and as the inner solve inside solve_muon_beta_eq.
// ============================================================
static double
solve_npe_beta_eq(
    const leptonic_eos_4d_t& eos,
    double lrho, double ltemp,
    double ymu,          ///< fixed muon fraction for this solve
    double ye_guess,
    double tol = 1e-14)
{
    using E = leptonic_eos_4d_t ;

    double const ye_lo = eos.ele_table._ye(0) ;
    double const ye_hi = eos.ele_table._ye(eos.nye - 1) ;

    auto residual = [&](double ye) -> double {
        double const yp  = std::min(eos.get_c2p_ye_max(),
                           std::max(eos.get_c2p_ye_min(), ye + ymu)) ;
        double const mue = eos.ele_table   .interp(lrho, ltemp, ye, E::TABMUELE) ;
        double const mup = eos.baryon_table.interp(lrho, ltemp, yp, E::TABMUP)   ;
        double const mun = eos.baryon_table.interp(lrho, ltemp, yp, E::TABMUN)   ;
        return mue + mup - mun ;
    } ;

    // Try a narrow bracket around the guess first; widen to full range
    // if there is no sign change.
    double a = std::max(ye_lo, ye_guess - 0.05) ;
    double b = std::min(ye_hi, ye_guess + 0.05) ;
    if (residual(a) * residual(b) > 0.) {
        a = ye_lo ;
        b = ye_hi ;
    }

    auto f_npe = [&](double ye){ return residual(ye) ; } ;
    return utils::brent(f_npe, a, b, tol) ;
}


// ============================================================
//  Full npe-mu beta equilibrium
//
//  Two coupled conditions:
//      (1)  mu_e(Y_e) = mu_mu(Y_mu)        => Y_e = f(Y_mu)
//      (2)  mu_n - mu_p - mu_mu = 0        => outer Brent on log(Y_mu)
//
//  Algorithm (mirrors Margherita / leptonic_eos_impl.hh):
//    Outer Brent over log(Y_mu) in [log(ymu_min), log(ymu_max)]:
//      a. Evaluate mu_mu from the muon table at the trial log(Y_mu).
//      b. Find Y_e such that mu_e(Y_e) = mu_mu  [inner Brent on Y_e].
//      c. yp = clamp(Y_e + Y_mu, yemin, yemax)  (charge neutrality).
//      d. Residual = mu_n(yp) - mu_p(yp) - mu_mu.
//
//  If mu_mu < m_mu even at ymu_min (muons energetically forbidden),
//  fall back to a pure npe solve with Y_mu = ymu_min.
//
//  Returns {Y_e, Y_mu, muons_present}.
// ============================================================
static std::tuple<double,double,bool>
solve_muon_beta_eq(
    const leptonic_eos_4d_t& eos,
    double lrho, double ltemp,
    double ye_guess, double /*ymu_guess*/,
    double tol = 1e-14)
{
    using E = leptonic_eos_4d_t ;

    double const lymu_lo = std::log(eos.get_c2p_ymu_min()) ;
    double const lymu_hi = std::log(eos.get_c2p_ymu_max()) ;
    double const ye_lo   = eos.ele_table._ye(0) ;
    double const ye_hi   = eos.ele_table._ye(eos.nye - 1) ;

    // ---- Inner solve: Y_e such that mu_e(Y_e) = mu_mu_target ----
    auto find_ye_for_mue = [&](double mu_mu_target) -> double {
        double const mue_lo = eos.ele_table.interp(lrho, ltemp, ye_lo, E::TABMUELE) ;
        double const mue_hi = eos.ele_table.interp(lrho, ltemp, ye_hi, E::TABMUELE) ;
        // Clamp if the target lies outside the table's mu_e range.
        if (mu_mu_target <= std::min(mue_lo, mue_hi)) return ye_lo ;
        if (mu_mu_target >= std::max(mue_lo, mue_hi)) return ye_hi ;
        auto f_inner = [&](double ye){
            return eos.ele_table.interp(lrho, ltemp, ye, E::TABMUELE)
                   - mu_mu_target ;
        } ;
        return utils::brent(f_inner, ye_lo, ye_hi, tol) ;
    } ;

    // ---- Check muon onset ----
    // If mu_mu at the minimum Y_mu is below the rest mass, there are
    // no physical muons at this (rho, T): fall back to npe.
    double const mu_mu_at_min = eos.muon_table.interp(
        lrho, ltemp, lymu_lo, E::TABMUMU) ;
    GRACE_TRACE("Cold table solve: lrho={:.3f} (rho={:.3e})  "
                "mu_mu_excess(Ymu_min)={:.3f} MeV  (onset at > 0)",
                lrho, std::exp(lrho), mu_mu_at_min) ;
    if (mu_mu_at_min < 0.0) {
        double const ye = solve_npe_beta_eq(
            eos, lrho, ltemp, eos.get_c2p_ymu_min(), ye_guess, tol) ;
        return { ye, eos.get_c2p_ymu_min(), false } ;
    }

    // ---- Outer Brent over log(Y_mu) ----
    // ye_running is updated as a side-effect so the converged Y_e is
    // available without an extra table call after the root is found.
    double ye_running = ye_guess ;

    auto outer_residual = [&](double lymu) -> double {
        double const ymu_loc = std::exp(lymu) ;
        double const mu_mu   = eos.muon_table.interp(
            lrho, ltemp, lymu, E::TABMUMU) ;
        ye_running           = find_ye_for_mue(mu_mu) ;
        double const yp      = std::min(eos.get_c2p_ye_max(),
                               std::max(eos.get_c2p_ye_min(),
                                        ye_running + ymu_loc)) ;
        double const mu_p    = eos.baryon_table.interp(lrho, ltemp, yp, E::TABMUP) ;
        double const mu_n    = eos.baryon_table.interp(lrho, ltemp, yp, E::TABMUN) ;
        return mu_n - mu_p - mu_mu ;   // = 0 at equilibrium
    } ;

    // If no sign change across the full Y_mu range, muons are
    // suppressed at this thermodynamic point: fall back to npe.
    double const f_lo = outer_residual(lymu_lo) ;
    double const f_hi = outer_residual(lymu_hi) ;
    if (f_lo * f_hi > 0.) {
        GRACE_TRACE("Cold table: no sign change in outer Brent at "
                    "lrho={:.4f} (rho={:.3e})  f_lo={:.4e}  f_hi={:.4e} "
                    "-> npe fallback",
                    lrho, std::exp(lrho), f_lo, f_hi) ;
        double const ye = solve_npe_beta_eq(
            eos, lrho, ltemp, eos.get_c2p_ymu_min(), ye_running, tol) ;
        return { ye, eos.get_c2p_ymu_min(), false } ;
    }

    auto f_outer = [&](double lymu){ return outer_residual(lymu) ; } ;
    double const lymu_eq = utils::brent(f_outer, lymu_lo, lymu_hi, tol) ;

    // One final evaluation at the converged log(Y_mu) to get the
    // consistent Y_e (outer_residual updates ye_running each call,
    // but the last internal call in brent may not be at the root).
    double const ymu_eq = std::exp(lymu_eq) ;
    double const mu_mu  = eos.muon_table.interp(
        lrho, ltemp, lymu_eq, E::TABMUMU) ;
    double const ye_eq  = find_ye_for_mue(mu_mu) ;

    // If Ye is pinned to the ceiling, the solution is unphysical
    // — fall back to npe
    if (ye_eq >= ye_hi * 0.9999) {
        double const ye = solve_npe_beta_eq(
            eos, lrho, ltemp, eos.get_c2p_ymu_min(), ye_running, tol) ;
        return { ye, eos.get_c2p_ymu_min(), false } ;
    }

    return { ye_eq, ymu_eq, true } ;
}


// ============================================================
//  .grace file writer
//
//  Format (mirrors read_leptonic_cold_table):
//    Line 1 : description comment
//    Line 2 : nrows
//    Lines 3+: log(rho)  col0  col1 ... col(NCOLS-1)
// ============================================================
static void
write_leptonic_cold_table(
    const std::string& filename,
    const Kokkos::View<double**, Kokkos::HostSpace>& h_data,
    const Kokkos::View<double*,  Kokkos::HostSpace>& h_rho,
    double T_cold)
{
    int const nrho  = static_cast<int>(h_rho .extent(0)) ;
    int const ncols = static_cast<int>(h_data.extent(1)) ;

    std::ofstream out(filename) ;
    if (!out.is_open()) {
        GRACE_WARN("Cold table writer: could not open '{}' for writing — skipping.",
                   filename) ;
        return ;
    }

    // Line 1: description
    out << "# GRACE leptonic cold table  T=" << std::fixed << std::setprecision(4)
        << T_cold << " MeV  nrho=" << nrho
        << "  ncols=" << ncols + 1   // +1 because reader expects rho prepended
        << "  col0=log(rho)  col1=log(T)  col2=Ye  col3=Ymu"
           "  col4=log(P)  col5=log(eps+shift)  col6=cs2  col7=entropy\n" ;

    // Line 2: number of rows
    out << nrho << "\n" ;

    // Data rows: log(rho) then the NCOLS columns
    out << std::scientific << std::setprecision(15) ;
    for (int i = 0; i < nrho; ++i) {
        out << h_rho(i) ;
        for (int j = 0; j < ncols; ++j) {
            out << "  " << h_data(i, j) ;
        }
        out << "\n" ;
    }

    GRACE_INFO("Cold table written to '{}'  ({} rows, {} data cols).",
               filename, nrho, ncols) ;
}


// ============================================================
//  Public entry point
// ============================================================
void generate_leptonic_cold_table(
    const leptonic_eos_4d_t& eos,
    double T_cold,
    Kokkos::View<double**, grace::default_execution_space>& d_data,
    Kokkos::View<double*,  grace::default_execution_space>& d_rho,
    const std::string& output_filename)
{
    constexpr int NCOLS = leptonic_eos_4d_t::COLD_VIDX::N_CTAB_VARS ;

    auto h_logrho = Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(), eos.baryon_table._logrho) ;
    int const nrho = static_cast<int>(h_logrho.extent(0)) ;
    double const lT = std::log(T_cold) ;

    Kokkos::View<double**, Kokkos::HostSpace> h_data("cold_data_h", nrho, NCOLS) ;
    Kokkos::View<double*,  Kokkos::HostSpace> h_rho ("cold_rho_h",  nrho)        ;

    GRACE_INFO("Generating leptonic cold table at T = {:.4f} MeV "
               "over {} rho points.  "
               "rho range [{:.3e}, {:.3e}]  "
               "Ymu range [{:.3e}, {:.3e}]  "
               "Ye  range [{:.3f}, {:.3f}]",
               T_cold, nrho,
               std::exp(h_logrho(0)), std::exp(h_logrho(nrho-1)),
               eos.get_c2p_ymu_min(), eos.get_c2p_ymu_max(),
               eos.get_c2p_ye_min(),  eos.get_c2p_ye_max()) ;

    double ye_guess  = 0.10 ;
    double ymu_guess = eos.get_c2p_ymu_min() ;

    // Diagnostic counters / accumulators
    int    n_muon_rows  = 0 ;
    int    n_npe_rows   = 0 ;
    int    n_eos_errors = 0 ;
    double ye_min_seen  =  1e99, ye_max_seen  = -1e99 ;
    double ymu_min_seen =  1e99, ymu_max_seen = -1e99 ;
    int    muon_onset_row = -1 ;

    // Print a progress line every ~5% of the table
    int const dbg_stride = std::max(1, nrho / 20) ;

    for (int i = 0; i < nrho; ++i) {
        double const lrho = h_logrho(i) ;

        // ---- Beta equilibrium ----
        auto [ye_eq, ymu_eq, muons_present] = solve_muon_beta_eq(
            eos, lrho, lT, ye_guess, ymu_guess) ;

        ye_guess  = ye_eq  ;    // warm start for next density
        ymu_guess = ymu_eq ;

        if (muons_present) {
            ++n_muon_rows ;
            if (muon_onset_row < 0) {
                muon_onset_row = i ;
                GRACE_INFO("Cold table: muon onset at row {}/{}  "
                           "rho={:.3e}  Ye={:.4f}  Ymu={:.4e}",
                           i, nrho-1, std::exp(lrho), ye_eq, ymu_eq) ;
            }
        } else {
            ++n_npe_rows ;
        }

        if (i % dbg_stride == 0) {
            GRACE_TRACE("Cold table row {:4d}/{}: "
                        "rho={:.3e}  Ye={:.4f}  Ymu={:.4e}  muons={}",
                        i, nrho-1, std::exp(lrho), ye_eq, ymu_eq,
                        muons_present ? "yes" : "no") ;
        }

        ye_min_seen  = std::min(ye_min_seen,  ye_eq)  ;
        ye_max_seen  = std::max(ye_max_seen,  ye_eq)  ;
        ymu_min_seen = std::min(ymu_min_seen, ymu_eq) ;
        ymu_max_seen = std::max(ymu_max_seen, ymu_eq) ;

        // ---- EOS evaluation ----
        double rho_q  = std::exp(lrho) ;
        double temp_q = T_cold ;
        double ye_q   = ye_eq  ;
        double ymu_q  = ymu_eq ;
        eos_err_t err{} ;

        double eps_q{}, csnd2_q{}, entropy_q{} ;
        double const press_q =
            eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                eps_q, csnd2_q, entropy_q,
                temp_q, rho_q, ye_q, ymu_q, err) ;

        if (   err.test(EOS_RHO_TOO_LOW)        || err.test(EOS_RHO_TOO_HIGH)
            || err.test(EOS_EPS_TOO_LOW)        || err.test(EOS_EPS_TOO_HIGH)
            || err.test(EOS_YE_TOO_LOW)         || err.test(EOS_YE_TOO_HIGH)
            #ifdef M1_NU_FIVESPECIES
            || err.test(EOS_YMU_TOO_LOW)        || err.test(EOS_YMU_TOO_HIGH)
            #endif
            || err.test(EOS_TEMPERATURE_TOO_LOW)|| err.test(EOS_TEMPERATURE_TOO_HIGH)
            || err.test(EOS_ENTROPY_TOO_LOW)    || err.test(EOS_ENTROPY_TOO_HIGH)) {
            ++n_eos_errors ;
            GRACE_WARN("Cold table row {}/{}: EOS clamping at "
                       "lrho={:.4f} (rho={:.3e})  Ye={:.4f}  Ymu={:.4e}  "
                       "press={:.3e}  eps={:.3e}",
                       i, nrho-1, lrho, rho_q, ye_eq, ymu_eq,
                       press_q, eps_q) ;
        }

        // ---- Store (COLD_VIDX order) ----
        h_rho(i) = lrho ;
        h_data(i, leptonic_eos_4d_t::CTABTEMP)    = lT ;
        h_data(i, leptonic_eos_4d_t::CTABYE)       = ye_q ;
        h_data(i, leptonic_eos_4d_t::CTABYMU)      = ymu_q ;         // plain fraction
        h_data(i, leptonic_eos_4d_t::CTABPRESS)    = std::log(press_q) ;
        h_data(i, leptonic_eos_4d_t::CTABEPS)      = std::log(eps_q + eos.energy_shift) ;
        h_data(i, leptonic_eos_4d_t::CTABCSND2)    = csnd2_q ;
        h_data(i, leptonic_eos_4d_t::CTABENTROPY)  = entropy_q ;
    }

    GRACE_INFO("Leptonic cold table done: {} rows x {} cols  "
               "({} muon rows, {} npe-only rows, {} EOS errors)  "
               "Ye [{:.4f}, {:.4f}]  Ymu [{:.4e}, {:.4e}]  "
               "muon onset row: {}",
               nrho, NCOLS,
               n_muon_rows, n_npe_rows, n_eos_errors,
               ye_min_seen,  ye_max_seen,
               ymu_min_seen, ymu_max_seen,
               muon_onset_row >= 0 ? std::to_string(muon_onset_row) : "none") ;

    Kokkos::realloc(d_data, nrho, NCOLS) ;
    Kokkos::realloc(d_rho,  nrho)        ;
    Kokkos::deep_copy(d_data, h_data)    ;
    Kokkos::deep_copy(d_rho,  h_rho)     ;

    // ---- Optional .grace dump ----
    if (!output_filename.empty() && parallel::mpi_comm_rank() == 0) {
        write_leptonic_cold_table(output_filename, h_data, h_rho, T_cold) ;
    }
}

// ============================================================
//  Cold-table reader (8 cols).
// ============================================================
static void
read_leptonic_cold_table(
    const std::string& filename,
    Kokkos::View<double**, grace::default_execution_space>& d_data,
    Kokkos::View<double* , grace::default_execution_space>& d_rho,
    int expected_cols = -1)
{

    std::ifstream file(filename);

    ASSERT(file.is_open(),"Can't open leptonic cold table file") ;

    std::string line;

    // ---- 1. Skip description line
    std::getline(file, line);

    // ---- 2. Read number of rows
    std::getline(file, line);
    std::istringstream iss_n(line);

    size_t nrows;
    iss_n >> nrows;
    ASSERT(iss_n, "Failed to read number of rows in leptonic cold eos table") ;


    // ---- 3. Peek first data line to determine columns
    std::streampos data_start = file.tellg();

    if (!std::getline(file, line)) {
        ERROR("Unexpected EOF when reading leptonic cold eos table");
    }

    std::istringstream iss_first(line);
    std::vector<double> first_row;
    double val;

    while (iss_first >> val) {
        first_row.push_back(val);
    }
    ASSERT(!first_row.empty(), "Invalid leptonic cold eos table format, first line is empty.") ;

    size_t ncols = first_row.size();

    if (expected_cols > 0 && ncols != static_cast<size_t>(expected_cols)) {
        ERROR("Column count mismatch in leptonic cold eos table");
    }

    // ---- 4. Allocate view
    Kokkos::realloc(d_data, nrows, ncols-1) ;
    Kokkos::realloc(d_rho, nrows) ;
    // ----- Create mirrors
    auto data = Kokkos::create_mirror_view(d_data) ;
    auto rho  = Kokkos::create_mirror_view(d_rho)  ;

    // ---- 5. Fill first row
    rho(0) = first_row[0] ;
    for (size_t j = 1; j < ncols; ++j) {
        data(0, j-1) = first_row[j];
    }

    // ---- 6. Read remaining rows
    size_t i = 1;
    std::vector<double> row ;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        row.clear() ;

        std::istringstream iss(line);
        while (iss >> val) {
            row.push_back(val);
        }
        ASSERT(row.size() == ncols, "Malformed eos file at line " << i + 2 ) ;

        rho(i) = row[0] ;
        for( int j=1; j<ncols; ++j) {
            data(i,j-1) = row[j] ;
        }

        ++i;
    }

    if (i != nrows) {
        ERROR("Row count mismatch: expected " +
                std::to_string(nrows) + ", got " +
                std::to_string(i));
    }


    Kokkos::deep_copy(d_data, data) ;
    Kokkos::deep_copy(d_rho, rho)   ;
}

// ============================================================
//  Reshape a flat (NVAR, ny, nT, nrho) HDF5 buffer into a
//  Kokkos View [irho, iT, iy, ivar] (tabeos_linterp_t layout).
//
//  HDF5 is row-major, slowest = variable, fastest = rho:
//      flat = iv*ny*nT*nrho + iy*nT*nrho + it*nrho + ir
// ============================================================
template <typename HostView4d>
static void deal_species_into_view(
    const std::vector<double>& buf,
    int nrho, int ntemp, int ny, int nvar,
    HostView4d& hv)
{
    for (int iv=0; iv<nvar; ++iv)
    for (int iy=0; iy<ny;   ++iy)
    for (int it=0; it<ntemp;++it)
    for (int ir=0; ir<nrho; ++ir) {
        size_t flat = static_cast<size_t>(iv)*ny*ntemp*nrho
                    + static_cast<size_t>(iy)*ntemp*nrho
                    + static_cast<size_t>(it)*nrho
                    + static_cast<size_t>(ir) ;
        hv(ir, it, iy, iv) = buf[flat] ;
    }
}

// ============================================================
//  Main reader
// ============================================================
grace::leptonic_eos_4d_t read_leptonic_4d_table()
{
    // -------------------------------------------------------
    //  1) Load the baryon EOS via the existing GRACE pipeline.
    //     Honours [eos.tabulated_eos.*] settings in the parfile.
    // -------------------------------------------------------
    GRACE_INFO("Reading baryon EOS for leptonic_4d setup...") ;
    tabulated_eos_t baryon_eos = read_eos_table() ;
    auto const& bt = baryon_eos.tables ;

    int const nrho_b  = baryon_eos.nrho ;
    int const ntemp_b = baryon_eos.nT   ;

    // -------------------------------------------------------
    //  2) Open the leptonic HDF5 file.
    // -------------------------------------------------------
    auto const fname      = grace::get_param<std::string>("eos","leptonic","table_filename") ;
    auto const cold_fname = grace::get_param<std::string>("eos","leptonic","cold_table_filename") ;

    GRACE_INFO("Reading 4D leptonic EOS table: {}", fname) ;
    GRACE_INFO("Leptonic cold table:           {}", cold_fname) ;

    hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT) ;
    ASSERT(file >= 0, "Could not open leptonic HDF5: " << fname) ;

    // Grid sizes
    int nrho, ntemp, nyle, nymu ;
    h5_read(file, "nrho",  &nrho,  H5T_NATIVE_INT) ;
    h5_read(file, "ntemp", &ntemp, H5T_NATIVE_INT) ;
    h5_read(file, "nyle",  &nyle,  H5T_NATIVE_INT) ;
    h5_read(file, "nymu",  &nymu,  H5T_NATIVE_INT) ;

    GRACE_INFO("Leptonic file: nrho={} nT={} nyle={} nymu={}",
               nrho, ntemp, nyle, nymu) ;

    // The lepton tables must live on exactly the same (rho, T) grid as
    // the baryon table (the additive sum is pointwise).  The Y_le axis
    // of the electronic table is independent of the baryon Y_e axis --
    // only the physical bounds need to agree, not the grid size.
    ASSERT(nrho  == nrho_b,
           "Leptonic and baryon tables disagree on nrho ("
           << nrho << " vs " << nrho_b << ")") ;
    ASSERT(ntemp == ntemp_b,
           "Leptonic and baryon tables disagree on ntemp ("
           << ntemp << " vs " << ntemp_b << ")") ;

    // Axes.  yle and ymu are read in as the linear-valued axes used
    // by the lepton interpolators.  logrho_table / logtemp_table are
    // read in only to verify that the leptonic file lives on the
    // same physical (rho, T) grid as the baryon table -- they're
    // *not* used as axes downstream (the lepton interpolators reuse
    // the baryon table's already-unit-converted natural-log axes).
    std::vector<double> yle(nyle), ymu(nymu) ;
    std::vector<double> logrho10(nrho), logtemp10(ntemp) ;
    double ylemin_f, ylemax_f, ymumin_f, ymumax_f ;
    h5_read(file, "logrho_table",  logrho10 .data(), H5T_NATIVE_DOUBLE) ;
    h5_read(file, "logtemp_table", logtemp10.data(), H5T_NATIVE_DOUBLE) ;
    h5_read(file, "yle_table",     yle      .data(), H5T_NATIVE_DOUBLE) ;
    h5_read(file, "ymu_table",     ymu      .data(), H5T_NATIVE_DOUBLE) ;
    h5_read(file, "eos_ylemin",    &ylemin_f,        H5T_NATIVE_DOUBLE) ;
    h5_read(file, "eos_ylemax",    &ylemax_f,        H5T_NATIVE_DOUBLE) ;
    h5_read(file, "eos_ymumin",    &ymumin_f,        H5T_NATIVE_DOUBLE) ;
    h5_read(file, "eos_ymumax",    &ymumax_f,        H5T_NATIVE_DOUBLE) ;

    // -------------------------------------------------------
    //  Verify the leptonic file's (rho, T) grid matches the baryon
    //  table's pointwise.  The leptonic file stores log10 of CGS
    //  values; the baryon axes were already converted to natural log
    //  in geometric units by read_eos_table().  The mapping is:
    //      bt._logrho[i] = log(10) * logrho10[i]  + log(RHOGF)
    //      bt._logT  [i] = log(10) * logtemp10[i]
    //
    //  A mismatch here means the two HDF5 files were generated on
    //  different grids and the additive sum is unphysical.
    // -------------------------------------------------------
    {
        auto const uconv = CGS_units / GEOM_units ;  // for log(RHOGF) == log(uconv.mass_density)
        double const lnRHOGF = std::log(uconv.mass_density) ;
        double const ln10    = std::log(10.) ;

        auto h_bar_lr = Kokkos::create_mirror_view_and_copy(
                            Kokkos::HostSpace(), bt._logrho) ;
        auto h_bar_lT = Kokkos::create_mirror_view_and_copy(
                            Kokkos::HostSpace(), bt._logT) ;

        constexpr double axis_tol = 1e-8 ;  // log-axis spacing is typically ~1e-2
        double max_drho = 0., max_dT = 0. ;
        for (int i=0; i<nrho;  ++i) {
            double const expected = ln10 * logrho10[i] + lnRHOGF ;
            double const observed = h_bar_lr(i) ;
            double const d = std::abs(expected - observed) ;
            if (d > max_drho) max_drho = d ;
        }
        for (int i=0; i<ntemp; ++i) {
            double const expected = ln10 * logtemp10[i] ;
            double const observed = h_bar_lT(i) ;
            double const d = std::abs(expected - observed) ;
            if (d > max_dT)   max_dT   = d ;
        }
        if (max_drho > axis_tol || max_dT > axis_tol) {
            GRACE_WARN("Leptonic and baryon (rho, T) grids differ pointwise:"
                       " max |dlogrho| = {:.3e}, max |dlogT| = {:.3e}."
                       " The lepton tables will be evaluated on the baryon"
                       " grid regardless; results may be inconsistent if"
                       " the two HDF5 files were generated on different grids.",
                       max_drho, max_dT) ;
        } else {
            GRACE_INFO("Leptonic and baryon (rho, T) grids agree pointwise"
                       " (max |dlogrho| = {:.3e}, max |dlogT| = {:.3e}).",
                       max_drho, max_dT) ;
        }
    }

    // Sanity-check Y_le bounds against the baryon Y_e bounds.
    {
        double const baryon_yemax = baryon_eos.get_c2p_ye_max() ;
        double const baryon_yemin = baryon_eos.get_c2p_ye_min() ;
        if (std::abs(ylemax_f - baryon_yemax) > 1e-10 ||
            std::abs(ylemin_f - baryon_yemin) > 1e-10) {
            GRACE_WARN("Leptonic ylemin/max = [{}, {}] differs from baryon "
                       "yemin/max = [{}, {}]; this is required to be equal "
                       "by Margherita.  Continuing but expect inconsistencies.",
                       ylemin_f, ylemax_f, baryon_yemin, baryon_yemax) ;
        }
    }

    // -------------------------------------------------------
    //  3) Read both 9-variable lepton datasets.
    // -------------------------------------------------------
    constexpr int NVAR_ELE  = leptonic_eos_4d_t::ELE_VIDX::N_TAB_VARS_ELE  ;
    constexpr int NVAR_MUON = leptonic_eos_4d_t::MUON_VIDX::N_TAB_VARS_MUON ;

    std::vector<double> buf_ele (static_cast<size_t>(NVAR_ELE)*nyle*ntemp*nrho) ;
    std::vector<double> buf_muon(static_cast<size_t>(NVAR_MUON)*nymu*ntemp*nrho) ;

    h5_read(file, "electronic_eos_tables", buf_ele .data(), H5T_NATIVE_DOUBLE) ;
    h5_read(file, "muonic_eos_tables",     buf_muon.data(), H5T_NATIVE_DOUBLE) ;

    H5Fclose(file) ;

    using view4d = Kokkos::View<double****, grace::default_space> ;
    view4d v_ele ("eos4d_ele",  nrho, ntemp, nyle, NVAR_ELE)  ;
    view4d v_muon("eos4d_muon", nrho, ntemp, nymu, NVAR_MUON) ;
    auto h_ele  = Kokkos::create_mirror_view(v_ele)  ;
    auto h_muon = Kokkos::create_mirror_view(v_muon) ;

    deal_species_into_view(buf_ele,  nrho, ntemp, nyle, NVAR_ELE,  h_ele)  ;
    deal_species_into_view(buf_muon, nrho, ntemp, nymu, NVAR_MUON, h_muon) ;

    Kokkos::deep_copy(v_ele,  h_ele)  ;
    Kokkos::deep_copy(v_muon, h_muon) ;

    // -------------------------------------------------------
    //  4) Axis Kokkos views.  rho and T axes are shared with the
    //     baryon table (we pass bt._logrho / bt._logT below).
    //     yle stores linear Y_le values (Margherita convention).
    //     ymu stores log(Y_mu) values — the HDF5 ymu_table is
    //     log-spaced; queries must pass log(ymu) to muon_table.
    // -------------------------------------------------------
    using view1d = Kokkos::View<double*, grace::default_space> ;
    view1d v_yle("eos4d_yle", nyle) ;
    view1d v_ymu("eos4d_ymu", nymu) ;
    auto h_yle = Kokkos::create_mirror_view(v_yle) ;
    auto h_ymu = Kokkos::create_mirror_view(v_ymu) ;
    for (int i=0; i<nyle; ++i) h_yle(i) = yle[i] ;
    for (int i=0; i<nymu; ++i) h_ymu(i) = ymu[i] ;
    Kokkos::deep_copy(v_yle, h_yle) ;
    Kokkos::deep_copy(v_ymu, h_ymu) ;

    // -------------------------------------------------------
    //  5) Cold slice.
    // -------------------------------------------------------
    Kokkos::View<double**, grace::default_execution_space> cold_tabs ;
    Kokkos::View<double*,  grace::default_execution_space> cold_lrho ;
    read_leptonic_cold_table(cold_fname, cold_tabs, cold_lrho,leptonic_eos_4d_t::COLD_VIDX::N_CTAB_VARS+1) ;

    // -------------------------------------------------------
    //  6) Limits.  rho/T/Y_e/eps/h/atmosphere come from the
    //     baryon table; Y_mu from the leptonic file.
    // -------------------------------------------------------
    double const rhomax  = baryon_eos.density_maximum() ;
    double const rhomin  = baryon_eos.density_minimum() ;
    double const tempmax = baryon_eos.temperature_maximum() ;
    double const tempmin = baryon_eos.temperature_minimum() ;
    double const yemax   = baryon_eos.get_c2p_ye_max() ;
    double const yemin   = baryon_eos.get_c2p_ye_min() ;

    // ymu_table stores log(Ymu) (Margherita convention: "Ymu is in log
    // scale just like rho, temp").  eos_ymumin/max are the linear Ymu
    // bounds.  Convert axis endpoints before comparing.
    double const ymumin = std::max(ymumin_f, std::exp(ymu.front())) ;
    double const ymumax = std::min(ymumax_f, std::exp(ymu.back()))  ;

    double const epsmin = baryon_eos.get_c2p_eps_min() ;
    double const epsmax = baryon_eos.get_c2p_eps_max() ;
    double const hmin   = baryon_eos.enthalpy_minimum() ;
    double const hmax   = baryon_eos.get_c2p_h_max() ;

    double const temp_atm = baryon_eos.temp_atmosphere() ;
    double const ye_atm   = baryon_eos.ye_atmosphere() ;
    double const ymu_atm  = ymumin ;

    bool const atm_beta_eq =
        grace::get_param<bool>("grmhd","atmosphere","atmosphere_is_beta_eq") ;

    // add_ele_contribution: set true only if the baryon table was
    // generated *without* electrons (uncommon).  For standard SFHo /
    // DD2 / BHBlp tables the baryon table already includes electrons,
    // so this must stay false to avoid double-counting.
    bool const add_ele = grace::get_param<bool>("eos","leptonic","add_ele_contribution") ;

    GRACE_INFO("4D leptonic EOS rho [{:.4e}, {:.4e}]  T [{:.4e}, {:.4e}]  "
               "Ye [{:.3f}, {:.3f}]  Ymu [{:.3e}, {:.3e}]  add_ele={}",
               rhomin, rhomax, tempmin, tempmax, yemin, yemax, ymumin, ymumax, add_ele) ;

    // Copy the baryon rho/T/ye axes into fresh managed views so the
    // leptonic_eos_4d_t constructor receives View<double*> (not the
    // readonly_view_t that tabeos_linterp_t stores internally).
    view1d bar_logrho("eos4d_bar_logrho", bt._logrho.extent(0)) ;
    view1d bar_logT  ("eos4d_bar_logT",   bt._logT  .extent(0)) ;
    view1d bar_ye    ("eos4d_bar_ye",     bt._ye    .extent(0)) ;
    Kokkos::deep_copy(bar_logrho, bt._logrho) ;
    Kokkos::deep_copy(bar_logT,   bt._logT)   ;
    Kokkos::deep_copy(bar_ye,     bt._ye)     ;

    //// -------------------------------------------------------
    ////  7) Construct.
    //// -------------------------------------------------------
    //return leptonic_eos_4d_t(
    //    bt._tables,
    //    bar_logrho, bar_logT, bar_ye,
    //    v_ele,  v_yle,
    //    v_muon, v_ymu,
    //    cold_tabs, cold_lrho,
    //    rhomax, rhomin,
    //    tempmax, tempmin,
    //    yemax,  yemin,
    //    ymumax, ymumin,
    //    baryon_eos.get_baryon_mass(),
    //    baryon_eos.energy_shift,
    //    epsmin, epsmax,
    //    hmin,   hmax,
    //    temp_atm,
    //    ye_atm, ymu_atm,
    //    atm_beta_eq,
    //    add_ele
    //) ;
    // -------------------------------------------------------
    //  7) Construct with the file-read cold table as a placeholder,
    //     then overwrite it with the generated beta-eq slice.
    // -------------------------------------------------------
    leptonic_eos_4d_t eos(
        bt._tables,
        bar_logrho, bar_logT, bar_ye,
        v_ele,  v_yle,
        v_muon, v_ymu,
        cold_tabs, cold_lrho,
        rhomax, rhomin,
        tempmax, tempmin,
        yemax,  yemin,
        ymumax, ymumin,
        baryon_eos.get_baryon_mass(),
        baryon_eos.energy_shift,
        epsmin, epsmax,
        hmin,   hmax,
        temp_atm,
        ye_atm, ymu_atm,
        atm_beta_eq,
        add_ele
    ) ;

    // -------------------------------------------------------
    //  8) Generate cold slice via beta equilibrium and patch in.
    // -------------------------------------------------------
    double T_cold = grace::get_param<double>("eos","leptonic","cold_table_temperature") ;
    if ( T_cold < 0 ) {
        T_cold = std::exp(eos.ltempmin) ;
    }

    std::string cold_out_fname = "" ;
    try {
        cold_out_fname = grace::get_param<std::string>(
            "eos","leptonic","cold_table_output_filename") ;
    } catch (...) { /* optional param — silence if absent */ }

    Kokkos::View<double**, grace::default_execution_space> gen_cold_tabs ;
    Kokkos::View<double*,  grace::default_execution_space> gen_cold_lrho ;
    generate_leptonic_cold_table(eos, T_cold, gen_cold_tabs, gen_cold_lrho,
                                 cold_out_fname) ;

    eos.cold_table = cold_eos_linterp_t(gen_cold_tabs, gen_cold_lrho) ;

    return eos ;
}

} /* namespace grace */
