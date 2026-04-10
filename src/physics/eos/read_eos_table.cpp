/**
 * @file read_eos_table.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de) & Khalil Pierre (pierre@itp.uni-frankfurt.de)
 * @brief 
 * @date 2026-03-28
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

#include <hdf5.h>

#include <string>
#include <filesystem>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <fstream>

#include <grace/errors/error.hh>
#include <grace/system/print.hh>

#include <grace/config/config_parser.hh>

#include <grace_config.h>

#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/tabulated_eos.hh>
#include <grace/physics/eos/physical_constants.hh>
#include <grace/physics/eos/unit_system.hh>

#define HDF5_CALL(result,cmd) \
    do {  \
        (result) = (cmd) ; \
        if((result)<0) { \
            ERROR("HDF5 API call failed with error code " << result ) ; \
        } \
    } while(false)

#define READ_ATTR_HDF5_COMPOSE(GROUP,NAME, VAR, TYPE)                     \
  do {                                                                  \
    hid_t dataset;                                                      \
    HDF5_CALL(dataset,H5Aopen(GROUP, NAME, H5P_DEFAULT));            \
    HDF5_CALL(h5err,H5Aread(dataset, TYPE, VAR)); \
    HDF5_CALL(h5err,H5Aclose(dataset));                                      \
  } while (0)

#define READ_EOS_HDF5_COMPOSE(GROUP,NAME, VAR, TYPE, MEM)                     \
  do {                                                                  \
    hid_t dataset;                                                      \
    HDF5_CALL(dataset,H5Dopen2(GROUP, NAME, H5P_DEFAULT));            \
    HDF5_CALL(h5err,H5Dread(dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR)); \
    HDF5_CALL(h5err,H5Dclose(dataset));                                      \
  } while (0)

// Use these two defines to easily read in a lot of variables in the same way
// The first reads in one variable of a given type completely
#define READ_SCOLLAPSE_EOS_HDF5(NAME, VAR, TYPE, MEM)                             \
  do {                                                                  \
    hid_t dataset;                                                      \
    HDF5_CALL(dataset,H5Dopen(file, NAME, H5P_DEFAULT));                          \
    HDF5_CALL(h5err,H5Dread(dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR)); \
    HDF5_CALL(h5err,H5Dclose(dataset));                                      \
  } while (0)
// The second reads a given variable into a hyperslab of the alltables_temp
// array
#define READ_SCOLLAPSE_EOSTABLE_HDF5(NAME, OFF)                                    \
  do {                                                                   \
    hsize_t offset[2] = {OFF, 0};                                        \
    H5Sselect_hyperslab(mem3, H5S_SELECT_SET, offset, NULL, var3, NULL); \
    READ_SCOLLAPSE_EOS_HDF5(NAME, alltables_temp, H5T_NATIVE_DOUBLE, mem3);        \
  } while (0)

namespace grace{


static void 
read_cold_table(
    const std::string& filename, 
    Kokkos::View<double**, grace::default_execution_space>& d_data,
    Kokkos::View<double* , grace::default_execution_space>& d_rho,
    int expected_cols = -1)
{

    std::ifstream file(filename);

    ASSERT(file.is_open(),"Can't open cold table file") ; 

    std::string line;

    // ---- 1. Skip description line
    std::getline(file, line);

    // ---- 2. Read number of rows
    std::getline(file, line);
    std::istringstream iss_n(line);

    size_t nrows;
    iss_n >> nrows;
    ASSERT(iss_n, "Failed to read number of rows in cold eos table") ; 


    // ---- 3. Peek first data line to determine columns
    std::streampos data_start = file.tellg();

    if (!std::getline(file, line)) {
        ERROR("Unexpected EOF when reading cold eos table");
    }

    std::istringstream iss_first(line);
    std::vector<double> first_row;
    double val;

    while (iss_first >> val) {
        first_row.push_back(val);
    }
    ASSERT(!first_row.empty(), "Invalid cold eos table format, first line is empty.") ; 

    size_t ncols = first_row.size();

    if (expected_cols > 0 && ncols != static_cast<size_t>(expected_cols)) {
        ERROR("Column count mismatch in cold eos table");
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

grace::tabulated_eos_t read_scollapse_table(std::string const& fname, std::string const& cold_tab_fname) 
{
    using namespace grace ; 
    using namespace grace::physical_constants ; 

    constexpr size_t NTABLES = tabulated_eos_t::TEOS_VIDX::N_TAB_VARS;

    auto const uconv = CGS_units / GEOM_units; 

    GRACE_INFO("Reading stellarcollapse table {}", fname) ;

    herr_t h5err ; 

    hid_t file ; 
    HDF5_CALL(file,H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT) ) ;

    int nrho, ntemp, nye;

    // Read size of tables
    READ_SCOLLAPSE_EOS_HDF5("pointsrho", &nrho, H5T_NATIVE_INT, H5S_ALL);
    READ_SCOLLAPSE_EOS_HDF5("pointstemp", &ntemp, H5T_NATIVE_INT, H5S_ALL);
    READ_SCOLLAPSE_EOS_HDF5("pointsye", &nye, H5T_NATIVE_INT, H5S_ALL);

    std::vector<double> logrho(nrho), logtemp(ntemp), ye(nye) ; 

    double *alltables_temp;
    if (!(alltables_temp=(double*)malloc(nrho*ntemp*nye*NTABLES*sizeof(double)))) {
        ERROR("Could not allocate memory for tabulated eos") ; 
    }

    // Prepare HDF5 to read hyperslabs into alltables_temp
    hsize_t table_dims[2] = {NTABLES, (hsize_t)nrho * ntemp * nye};
    hsize_t var3[2] = {1, (hsize_t)nrho * ntemp * nye};
    hid_t mem3 = H5Screate_simple(2, table_dims, NULL);

    // Read alltables_temp
    READ_SCOLLAPSE_EOSTABLE_HDF5("logpress", tabulated_eos_t::TEOS_VIDX::TABPRESS);
    READ_SCOLLAPSE_EOSTABLE_HDF5("logenergy", tabulated_eos_t::TEOS_VIDX::TABEPS);
    READ_SCOLLAPSE_EOSTABLE_HDF5("entropy", tabulated_eos_t::TEOS_VIDX::TABENTROPY);
    READ_SCOLLAPSE_EOSTABLE_HDF5("cs2", tabulated_eos_t::TEOS_VIDX::TABCSND2);

    READ_SCOLLAPSE_EOSTABLE_HDF5("mu_e", tabulated_eos_t::TEOS_VIDX::TABMUE);
    READ_SCOLLAPSE_EOSTABLE_HDF5("mu_p", tabulated_eos_t::TEOS_VIDX::TABMUP);
    READ_SCOLLAPSE_EOSTABLE_HDF5("mu_n", tabulated_eos_t::TEOS_VIDX::TABMUN);

    READ_SCOLLAPSE_EOSTABLE_HDF5("Xa", tabulated_eos_t::TEOS_VIDX::TABXA);
    READ_SCOLLAPSE_EOSTABLE_HDF5("Xh", tabulated_eos_t::TEOS_VIDX::TABXH);
    READ_SCOLLAPSE_EOSTABLE_HDF5("Xn", tabulated_eos_t::TEOS_VIDX::TABXN);
    READ_SCOLLAPSE_EOSTABLE_HDF5("Xp", tabulated_eos_t::TEOS_VIDX::TABXP);

    READ_SCOLLAPSE_EOSTABLE_HDF5("Abar", tabulated_eos_t::TEOS_VIDX::TABABAR);
    READ_SCOLLAPSE_EOSTABLE_HDF5("Zbar", tabulated_eos_t::TEOS_VIDX::TABZBAR);

    // axes and energy shift 
    double energy_shift ; 
    READ_SCOLLAPSE_EOS_HDF5("logrho", logrho.data(), H5T_NATIVE_DOUBLE, H5S_ALL);
    READ_SCOLLAPSE_EOS_HDF5("logtemp", logtemp.data(), H5T_NATIVE_DOUBLE, H5S_ALL);
    READ_SCOLLAPSE_EOS_HDF5("ye", ye.data(), H5T_NATIVE_DOUBLE, H5S_ALL);
    READ_SCOLLAPSE_EOS_HDF5("energy_shift", &energy_shift, H5T_NATIVE_DOUBLE, H5S_ALL);
    energy_shift *= SQR(uconv.velocity) ; 
    // get baryon mass if available 
    // Read in baryon mass if contained in the table
    hid_t mb_data;
    auto status = H5Lexists(file, "/mass_factor", H5P_DEFAULT);

    double baryon_mass ; 
    if (status) {
        HDF5_CALL(mb_data, H5Dopen(file, "mass_factor", H5P_DEFAULT));
        H5Dread(mb_data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                &baryon_mass);
        GRACE_INFO("Read baryon mass from file {} g", baryon_mass) ; 
    } else {
        baryon_mass = mn_MeV * MeV_to_g * uconv.mass ;
        GRACE_INFO("Using default baryon mass {} g", baryon_mass);
    }

    auto have_rel_cs2 = H5Lexists(file, "/have_rel_cs2", H5P_DEFAULT);
    if (have_rel_cs2) GRACE_INFO("Sound speed from table is already relativistic!");

    HDF5_CALL(h5err,H5Sclose(mem3));
    HDF5_CALL(h5err,H5Fclose(file));

    // copy into view 
    Kokkos::View<double ****, grace::default_space> _tables("eos_table", nrho, ntemp, nye, tabulated_eos_t::TEOS_VIDX::N_TAB_VARS) ; 
    auto alltables = Kokkos::create_mirror_view(_tables) ; 

    for (int iv = 0; iv < NTABLES; iv++)
    for (int k = 0; k < nye; k++)
    for (int j = 0; j < ntemp; j++)
    for (int i = 0; i < nrho; i++) {
        int indold = i + nrho * (j + ntemp * (k + nye * iv));
        alltables(i,j,k,iv) = alltables_temp[indold] ; 
    }

    free(alltables_temp);

    // convert units and log10 to loge 
    for( int i=0; i<nrho; ++i) {
        logrho[i] = logrho[i] * log(10.) + log(uconv.mass_density) ; 
    }
    for( int i=0; i<ntemp; ++i) {
        logtemp[i] = logtemp[i] * log(10.)  ; 
    }

    double const rhomin{exp(logrho[0])}, rhomax{exp(logrho[nrho-1])} ; 
    double const tempmin{exp(logtemp[0])}, tempmax{exp(logtemp[ntemp-1])} ;  
    double const yemax{ye[nye-1]}, yemin{ye[0]}     ;

    double hmin{std::numeric_limits<double>::max()}, hmax{std::numeric_limits<double>::min()} ;
    double epsmin{std::numeric_limits<double>::max()}, epsmax{std::numeric_limits<double>::min()} ;

    for (int k = 0; k < nye; k++)
    for (int j = 0; j < ntemp; j++)
    for (int i = 0; i < nrho; i++) {
        double pressL, epsL, rhoL, eL ; 
        rhoL = exp(logrho[i]) ; 
        { // press 
            int idx = tabulated_eos_t::TEOS_VIDX::TABPRESS + NTABLES * i;

            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABPRESS) = alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABPRESS) * log(10.0) + log(uconv.pressure) ; 
            pressL = exp(alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABPRESS)) ; 
        }

        { // eps 
            int idx = tabulated_eos_t::TEOS_VIDX::TABEPS + NTABLES * i; 
            double epsT =  pow(10,alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABEPS))* SQR(uconv.velocity);
            epsL = ( epsT - energy_shift  )  ; 
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABEPS) = log(epsT) ; 
        }

        const double hL = 1. + epsL + pressL / rhoL;
        hmax = fmax(hmax, hL) ; 
        hmin = fmin(hmin, hL) ; 

        epsmax = fmax(epsmax, epsL) ; 
        epsmin = fmin(epsmin, epsL) ; 

        { // cs2 
            int idx = tabulated_eos_t::TEOS_VIDX::TABCSND2 + NTABLES * i;  
            double cs2L = alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABCSND2) * SQR(uconv.velocity) ;
            
            if (!have_rel_cs2) {
                cs2L /= hL ; 
            }
            cs2L = fmax(fmin(cs2L,1.-1.e-10),1e-6) ; 

            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABCSND2) = cs2L ; 
        }
    }

    Kokkos::deep_copy(_tables, alltables) ; 
    Kokkos::View<double*, grace::default_space> _lrho("tab_eos_logrho", nrho), _lt("tab_eos_logtemp", ntemp), _ye("tab_eos_ye", nye) ;  
    grace::deep_copy_vec_to_view(_lrho, logrho)  ; 
    grace::deep_copy_vec_to_view(_lt  , logtemp) ; 
    grace::deep_copy_vec_to_view(_ye  , ye)      ; 
    GRACE_INFO("Table shape: ({}, {}, {}, {})", _tables.extent(0), _tables.extent(1), _tables.extent(2), _tables.extent(3)) ; 
    GRACE_INFO("Rest mass density max {}, min {}\n Temperature max {}, min {}\n ye max {}, min {}\n minimum enthalpy {}\n energy shift {}", rhomax, rhomin, tempmax, tempmin, yemax,yemin, hmin, energy_shift) ; 
    // figure out if atmo is beta equilibrated,
    // if so, find the beta equilibrium ye 
    double temp_floor = get_param<double>("grmhd", "atmosphere", "temp_fl") ; 
    double rho_floor = get_param<double>("grmhd", "atmosphere", "rho_fl") ; 

    double tmin_gracefact = std::exp(logtemp[1]) ; 
    if( temp_floor < tmin_gracefact ) {
        GRACE_WARN("Requested atmo temperature is below second point in the table {}, will be overridden.", tmin_gracefact) ; 
        temp_floor = tmin_gracefact; 
    }
    if (rho_floor < rhomin ) {
        ERROR("Requested atmo density is below table bound.") ; 
    }


    bool atm_beta_eq = grace::get_param<bool>("grmhd", "atmosphere", "atmosphere_is_beta_eq") ; 
    double ye_atmo = get_param<double>("grmhd", "atmosphere", "ye_fl") ; 
    if ( atm_beta_eq ) {
        // find beta equilibrium, we do this on device cause it's simpler that way 
        tabeos_linterp_t interpolator(_tables,_lrho,_lt,_ye) ;
        Kokkos::parallel_reduce("find_betaeq_atmo_ye", 1, 
            KOKKOS_LAMBDA (int dummy, double& acc) {
                auto const find_betaeq = [=] (double rho, double T) {
                    double logrhoL = log(rho) ; 
                    double logtempL = log(T) ; 
                    auto const dmu = [&] (double ye) {
                        double mup = interpolator.interp(logrhoL,logtempL,ye,tabulated_eos_t::TEOS_VIDX::TABMUP) ; 
                        double mue = interpolator.interp(logrhoL,logtempL,ye,tabulated_eos_t::TEOS_VIDX::TABMUE) ; 
                        double mun = interpolator.interp(logrhoL,logtempL,ye,tabulated_eos_t::TEOS_VIDX::TABMUN) ; 
                        return mue + mup - mun ; 
                    } ; 
                    return utils::brent(dmu, yemin, yemax, 1e-14) ; 
                } ; 
                acc = find_betaeq(rho_floor,temp_floor) ; 
            }, Kokkos::Max<double>(ye_atmo)
        ) ; 
    }

    auto usr_eps_max = grace::get_param<double>("eos", "eps_maximum");
    if ( usr_eps_max < epsmax ) {
        epsmax = usr_eps_max ; 
    }

    GRACE_INFO("Atmosphere settings: rho: {}, temperature: {}, ye: {}", rho_floor, temp_floor, ye_atmo) ; 

    // read in the cold table 
    Kokkos::View<double **, grace::default_execution_space> cold_tables("eos_cold_table") ; 
    Kokkos::View<double *, grace::default_execution_space> cold_table_rho("eos_cold_table_log_rho") ; 
    GRACE_INFO("Reading cold table {}", cold_tab_fname) ; 
    read_cold_table(
        cold_tab_fname,
        cold_tables, 
        cold_table_rho,
        tabulated_eos_t::COLD_TEOS_VIDX::N_CTAB_VARS+1
    ) ; 

    GRACE_INFO("Done reading cold table, size rho {} size table {} {}", cold_table_rho.extent(0), cold_tables.extent(0), cold_tables.extent(1)) ; 
    return tabulated_eos_t(
        _tables, 
        _lrho, _lt, _ye,
        cold_tables,
        cold_table_rho,
        rhomax, rhomin, 
        tempmax, tempmin, 
        yemax, yemin, 
        baryon_mass, 
        energy_shift,
        epsmin, epsmax, 
        hmin, hmax,
        temp_floor, ye_atmo, atm_beta_eq
    ) ;

}

grace::tabulated_eos_t read_compose_table(std::string const& fname, std::string const& cold_tab_fname) 
{
    using namespace grace ; 
    using namespace grace::physical_constants ; 

    auto const uconv = COMPOSE_units / GEOM_units; 

    GRACE_INFO("Reading compose table {}", fname) ; 

    herr_t h5err ; 

    hid_t file ; 
    HDF5_CALL(file,H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT) ) ; 

    hid_t parameters ; 
    HDF5_CALL(parameters, H5Gopen(file,"/Parameters",H5P_DEFAULT)) ; 

    int nrho, ntemp, nye ; 
    READ_ATTR_HDF5_COMPOSE(parameters,"pointsnb", &nrho, H5T_NATIVE_INT);
    READ_ATTR_HDF5_COMPOSE(parameters,"pointst", &ntemp, H5T_NATIVE_INT);
    READ_ATTR_HDF5_COMPOSE(parameters,"pointsyq", &nye, H5T_NATIVE_INT);

    std::vector<double> logrho(nrho), logtemp(ntemp), yes(nye) ; 


    auto num_points =
        std::array<size_t, 3>{size_t(nrho), size_t(ntemp), size_t(nye)};

    // Read additional tables and variables
    READ_EOS_HDF5_COMPOSE(parameters,"nb", logrho.data(), H5T_NATIVE_DOUBLE, H5S_ALL);
    READ_EOS_HDF5_COMPOSE(parameters,"t", logtemp.data(), H5T_NATIVE_DOUBLE, H5S_ALL);
    READ_EOS_HDF5_COMPOSE(parameters,"yq", yes.data(), H5T_NATIVE_DOUBLE, H5S_ALL);


    hid_t thermo_id;
    HDF5_CALL(thermo_id,H5Gopen(file, "/Thermo_qty",H5P_DEFAULT));
    int nthermo;
    READ_ATTR_HDF5_COMPOSE(thermo_id,"pointsqty", &nthermo, H5T_NATIVE_INT);

    // Read thermo index array
    int *thermo_index = new int[nthermo];
    READ_EOS_HDF5_COMPOSE(thermo_id,"index_thermo", thermo_index, H5T_NATIVE_INT, H5S_ALL);

    // Allocate memory and read table
    double *thermo_table = new double[nthermo * nrho * ntemp * nye];
    READ_EOS_HDF5_COMPOSE(thermo_id,"thermo", thermo_table, H5T_NATIVE_DOUBLE, H5S_ALL);


    int ncomp=0;
    hid_t comp_id;

    int status_e = H5Eset_auto(H5E_DEFAULT,NULL, NULL);
    int status_comp = H5Gget_objinfo(file,"/Composition_pairs",0,nullptr);
    if(status_comp ==0){
        HDF5_CALL(comp_id,H5Gopen(file, "/Composition_pairs",H5P_DEFAULT));
        READ_ATTR_HDF5_COMPOSE(comp_id, "pointspairs", &ncomp, H5T_NATIVE_INT);
    }

    int *index_yi = nullptr;
    double *yi_table = nullptr;

    if(ncomp > 0){

        // index identifying particle type
        index_yi = new int[ncomp];
        READ_EOS_HDF5_COMPOSE(comp_id,"index_yi", index_yi, H5T_NATIVE_INT, H5S_ALL);

        // Read composition
        yi_table = new double[ncomp * nrho * ntemp * nye];
        READ_EOS_HDF5_COMPOSE(comp_id,"yi", yi_table, H5T_NATIVE_DOUBLE, H5S_ALL);
    }

    // Read average charge and mass numbers
    int nav=0;
    double *zav_table = nullptr;
    double *yav_table = nullptr;
    double *aav_table = nullptr;

    int status_av = H5Gget_objinfo(file,"Composition_quadruples",0,nullptr);

    hid_t av_id;

    if(status_av ==0){
        HDF5_CALL(av_id, H5Gopen(file, "/Composition_quadruples", H5P_DEFAULT));
        READ_ATTR_HDF5_COMPOSE(av_id, "pointsav", &nav, H5T_NATIVE_INT);
    }

    if(nav >0){

        assert(nav == 1 &&
        "nav != 1 in this table, so there is none or more than "
        "one definition of an average nucleus."
        "Please check and generalize accordingly.");

        // Read average tables
        zav_table = new double[nrho * ntemp * nye];
        yav_table = new double[nrho * ntemp * nye];
        aav_table = new double[nrho * ntemp * nye];
        READ_EOS_HDF5_COMPOSE(av_id, "zav", zav_table, H5T_NATIVE_DOUBLE, H5S_ALL);
        READ_EOS_HDF5_COMPOSE(av_id, "yav", yav_table, H5T_NATIVE_DOUBLE, H5S_ALL);
        READ_EOS_HDF5_COMPOSE(av_id, "aav", aav_table, H5T_NATIVE_DOUBLE, H5S_ALL);
    }

    HDF5_CALL(h5err, H5Fclose(file));

    auto const find_index = [&](size_t const &index) {
        for (int i = 0; i < nthermo; ++i) {
        if (thermo_index[i] == index) return i;
        }
        assert(!"Could not find index of all required quantities. This should not "
                "happen.");
        return -1;
    };


    constexpr size_t PRESS_C = 1;
    constexpr size_t S_C = 2;
    constexpr size_t MUN_C = 3;
    constexpr size_t MUP_C = 4;
    constexpr size_t MUE_C = 5;
    // CHECK: is this really the same eps as in the stellar collapse tables?
    constexpr size_t EPS_C = 7;
    constexpr size_t CS2_C = 12;

    int thermo_index_conv[7]{find_index(PRESS_C), find_index(EPS_C),
                             find_index(CS2_C),   find_index(S_C),
                             find_index(MUE_C),   find_index(MUP_C),
                             find_index(MUN_C) };

    Kokkos::View<double ****> _tables("eos_table", nrho, ntemp, nye, tabulated_eos_t::TEOS_VIDX::N_TAB_VARS) ; 
    auto alltables = Kokkos::create_mirror_view(_tables) ; 

    for (int iv = 0; iv <= 6; iv++)
    for (int k = 0; k < nye; k++)
    for (int j = 0; j < ntemp; j++)
    for (int i = 0; i < nrho; i++) {
        auto const iv_thermo = thermo_index_conv[iv];
        int indold = i + nrho * (j + ntemp * (k + nye * iv_thermo));
        alltables(i,j,k,iv) = thermo_table[indold];
    }

    // find minimum un-shifted epsilon 
    double epsmin=std::numeric_limits<double>::max() ; 
    for (int k = 0; k < nye; k++)
    for (int j = 0; j < ntemp; j++)
    for (int i = 0; i < nrho; i++) {
        epsmin = fmin(epsmin, alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABEPS)) ; 
    }
    double energy_shift = epsmin < 0 ? (-2.*epsmin) : 0.0 ; 

    auto const find_index_yi = [&](size_t const &index) {
        for (int i = 0; i < ncomp; ++i) {
        if (index_yi[i] == index) return i;
        }
        assert(!"Could not find index of all required quantities. This should not "
                "happen.");
        return -1;
    };

    for (int k = 0; k < nye; k++)
    for (int j = 0; j < ntemp; j++)
    for (int i = 0; i < nrho; i++) {
        int indold = i + nrho * (j + ntemp * k);
        if(nav >0){
            // ABAR
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABABAR) = aav_table[indold];
            // ZBAR
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABZBAR) = zav_table[indold];
            // Xh
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABXH) = aav_table[indold] * yav_table[indold];
        }
        if(ncomp>0){
            // Xn
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABXN) =
                yi_table[indold + nrho * nye * ntemp * find_index_yi(10)];
            // Xp
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABXP) =
                yi_table[indold + nrho * nye * ntemp * find_index_yi(11)];
            // Xa
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABXA) =
                4. * yi_table[indold + nrho * nye * ntemp * find_index_yi(4002)];
        }
    }

    // Free all storage
    delete[] thermo_index;
    delete[] thermo_table;

    if(index_yi != nullptr) delete[] index_yi;
    if(yi_table != nullptr) delete[] yi_table;

    if(zav_table != nullptr) delete[] zav_table;
    if(yav_table != nullptr) delete[] yav_table;
    if(aav_table != nullptr) delete[] aav_table;

    double baryon_mass = mn_MeV * uconv.mass ;


    for (int i = 0; i < nrho; i++) logrho[i] = log(logrho[i] * mn_MeV * uconv.mass_density );
    
    for (int i = 0; i < ntemp; i++) logtemp[i] = log(logtemp[i]);


    double rhomax{exp(logrho[nrho-1])}, rhomin{exp(logrho[0])}   ; 
    double tempmax{exp(logtemp[ntemp-1])}, tempmin{exp(logtemp[0])} ; 
    double yemax{yes[nye-1]}, yemin{yes[0]}     ;
    double epsmax{std::numeric_limits<double>::min()}  ;
    double hmax{std::numeric_limits<double>::min()}, hmin{std::numeric_limits<double>::max()}     ;

    epsmin=std::numeric_limits<double>::max();   

    // convert units
    for (int i = 0; i < nrho; i++) for(int j=0; j<ntemp; ++j) for( int k=0; k<nye; ++k) {

        double pressL, epsL, rhoL;
        rhoL = exp(logrho[i]) ; 

        {  // pressure
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABPRESS) = log(alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABPRESS) * uconv.pressure );
            pressL = exp(alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABPRESS));
        }


        // shift epsilon to a positive range if necessary
        {
            epsL = alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABEPS) ;
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABEPS) = log(epsL + energy_shift) ;
        }

        {  // cs2
            double csnd2 = alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABCSND2) ; 
            if ( csnd2 < 0 ) {
                alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABCSND2) = 0.0 ; 
            }
            if ( csnd2 >= 1 ) {
                alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABCSND2) = 1-1e-10 ; 
            }
        }

        {  // chemical potentials
            auto const mu_q = alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABMUP);
            auto const mu_b = alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABMUN);
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABMUP) = mu_q + mu_b ; 
            alltables(i,j,k,tabulated_eos_t::TEOS_VIDX::TABMUE) -= mu_q  ; 
        }
        
        const double hL = 1. + epsL + pressL / rhoL;
        hmax = fmax(hmax, hL) ; 
        hmin = fmin(hmin, hL) ; 

        epsmax = fmax(epsmax, epsL) ; 
        epsmin = fmin(epsmin, epsL) ; 
    }

    // copy to device 
    Kokkos::deep_copy(_tables, alltables) ; 
    Kokkos::View<double*, grace::default_space> _lrho("tab_eos_logrho", nrho), _lt("tab_eos_logtemp", ntemp), _ye("tab_eos_ye", nye) ;  
    grace::deep_copy_vec_to_view(_lrho, logrho)  ; 
    grace::deep_copy_vec_to_view(_lt  , logtemp) ; 
    grace::deep_copy_vec_to_view(_ye  , yes)     ; 
    GRACE_INFO("Table shape: ({}, {}, {}, {})", _tables.extent(0), _tables.extent(1), _tables.extent(2), _tables.extent(3)) ; 
    GRACE_INFO("Rest mass density max {}, min {} Temperature max {}, min {}", rhomax, rhomin, tempmax, tempmin) ; 

    // figure out if atmo is beta equilibrated,
    // if so, find the beta equilibrium ye 
    double temp_floor = get_param<double>("grmhd", "atmosphere", "temp_fl") ; 
    double rho_floor = get_param<double>("grmhd", "atmosphere", "rho_fl") ; 

    double tmin_gracefact = std::exp(logtemp[1]) ; 
    if( temp_floor < tmin_gracefact ) {
        GRACE_WARN("Requested atmo temperature is below second point in the table {}, will be overridden.", tmin_gracefact) ; 
        temp_floor = tmin_gracefact; 
    }
    if (rho_floor < rhomin ) {
        ERROR("Requested atmo density is below table bound.") ; 
    }


    bool atm_beta_eq = grace::get_param<bool>("grmhd", "atmosphere", "atmosphere_is_beta_eq") ; 
    double ye_atmo = get_param<double>("grmhd", "atmosphere", "ye_fl") ; 
    if ( atm_beta_eq ) {
        // find beta equilibrium, we do this on device cause it's simpler that way 
        tabeos_linterp_t interpolator(_tables,_lrho,_lt,_ye) ;
        Kokkos::parallel_reduce("find_betaeq_atmo_ye", 1, 
            KOKKOS_LAMBDA (int dummy, double& acc) {
                auto const find_betaeq = [=] (double rho, double T) {
                    double logrhoL = log(rho) ; 
                    double logtempL = log(T) ; 
                    auto const dmu = [&] (double ye) {
                        double mup = interpolator.interp(logrhoL,logtempL,ye,tabulated_eos_t::TEOS_VIDX::TABMUP) ; 
                        double mue = interpolator.interp(logrhoL,logtempL,ye,tabulated_eos_t::TEOS_VIDX::TABMUE) ; 
                        double mun = interpolator.interp(logrhoL,logtempL,ye,tabulated_eos_t::TEOS_VIDX::TABMUN) ; 
                        return mue + mup - mun ; 
                    } ; 
                    return utils::brent(dmu, yemin, yemax, 1e-14) ; 
                } ; 
                acc = find_betaeq(rho_floor,temp_floor) ; 
            }, Kokkos::Max<double>(ye_atmo)
        ) ; 
    }

    auto usr_eps_max = grace::get_param<double>("eos", "eps_maximum");
    if ( usr_eps_max < epsmax ) {
        epsmax = usr_eps_max ; 
    }

    // read in the cold table 
    Kokkos::View<double **, grace::default_execution_space> cold_tables("eos_cold_table") ; 
    Kokkos::View<double *, grace::default_execution_space> cold_table_rho("eos_cold_table_log_rho") ; 
    GRACE_INFO("Reading cold table {}", cold_tab_fname) ; 
    read_cold_table(
        cold_tab_fname,
        cold_tables, 
        cold_table_rho,
        tabulated_eos_t::COLD_TEOS_VIDX::N_CTAB_VARS+1
    ) ; 

    GRACE_INFO("Done reading cold table, size rho {} size table {} {}", cold_table_rho.extent(0), cold_tables.extent(0), cold_tables.extent(1)) ; 
    return tabulated_eos_t(
        _tables, 
        _lrho, _lt, _ye,
        cold_tables,
        cold_table_rho,
        rhomax, rhomin, 
        tempmax, tempmin, 
        yemax, yemin, 
        baryon_mass, 
        energy_shift,
        epsmin, epsmax, 
        hmin, hmax,
        temp_floor, ye_atmo, atm_beta_eq
    ) ; 
    
}

grace::tabulated_eos_t read_eos_table() 
{
    auto const eos_tab_name = grace::get_param<std::string>("eos", "tabulated_eos", "table_filename") ; 
    auto const eos_cold_tab_name = grace::get_param<std::string>("eos", "tabulated_eos", "cold_table_filename") ;
    auto const eos_tab_kind =  grace::get_param<std::string>("eos", "tabulated_eos", "table_format") ;
    
    if ( eos_tab_kind == "compose" ) {
        return read_compose_table(eos_tab_name, eos_cold_tab_name) ; 
    } else {
        ASSERT(eos_tab_kind=="stellarcollapse", "Should have been caught at parcheck") ; 
        return read_scollapse_table(eos_tab_name, eos_cold_tab_name) ;  
    } 

}


}
