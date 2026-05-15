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
//  Cold-table reader (8 cols).
// ============================================================
static void read_leptonic_cold_table_0(
    const std::string& filename,
    Kokkos::View<double**, grace::default_space>& d_data,
    Kokkos::View<double*,  grace::default_space>& d_rho)
{
    std::ifstream f(filename) ;
    ASSERT(f.is_open(), "Cannot open leptonic cold table: " << filename) ;

    std::string line ;
    std::getline(f, line) ;  // description
    std::getline(f, line) ;  // nrows
    std::istringstream iss_n(line) ;
    size_t nrows ;
    iss_n >> nrows ;
    ASSERT(iss_n, "Failed to read nrows in leptonic cold table") ;

    constexpr int NCOLS = leptonic_eos_4d_t::COLD_VIDX::N_CTAB_VARS ;

    Kokkos::realloc(d_data, nrows, NCOLS) ;
    Kokkos::realloc(d_rho,  nrows) ;
    auto h_data = Kokkos::create_mirror_view(d_data) ;
    auto h_rho  = Kokkos::create_mirror_view(d_rho)  ;

    for (size_t i = 0; i < nrows; ++i) {
        ASSERT(std::getline(f,line),
               "Unexpected EOF in leptonic cold table at row " << i) ;
        std::istringstream iss(line) ;
        double v ;
        iss >> v ; h_rho(i) = v ;
        for (int j = 0; j < NCOLS; ++j) { iss >> v ; h_data(i,j) = v ; }
    }
    Kokkos::deep_copy(d_data, h_data) ;
    Kokkos::deep_copy(d_rho,  h_rho)  ;
}

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
    Kokkos::View<double**, grace::default_space> cold_tabs ;
    Kokkos::View<double*,  grace::default_space> cold_lrho ;
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

    // -------------------------------------------------------
    //  7) Construct.
    // -------------------------------------------------------
    return leptonic_eos_4d_t(
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
}

} /* namespace grace */
