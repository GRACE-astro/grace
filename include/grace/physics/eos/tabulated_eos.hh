/**
 * @file piecewise_polytropic_eos.hh
 * @author Khalil Pierre (khalil3.14erre@gmail.com"
 * @brief 
 * @date 2025-02-03
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


#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/physics/eos/physical_constants.hh>
#include <hdf5.h>

#include <Kokkos_Core.hpp>
#include <iostream>

namespace grace { 
/**
 * @brief 
 *        
 * \ingroup eos
 * 
 */
class tabulated_eos_t
{
    using error_type = unsigned int ;
 public:
    
    enum EV {
    PRESS = 0,
    EPS,
    S,
    CS2,
    MUE,
    MUP,
    MUN,
    XA,
    XH,
    XN,
    XP,
    ABAR,
    ZBAR,
    NUM_VARS
    };

    tabulated_eos_t() = default;

    tabulated_eos_t(
      Kokkos::View<double****, grace::default_space> alltables,
      Kokkos::View<double*, grace::default_space> logrho,
      Kokkos::View<double*, grace::default_space> logtemp,
      Kokkos::View<double*, grace::default_space> yes,
      Kokkos::View<double***, grace::default_space> epstable,
      double c2p_eps_min,
      double c2p_h_min,
      double c2p_h_max,
      double c2p_press_max,
      double energy_shift)
    : _alltables(alltables), _logrho(logrho), _logtemp(logtemp), _yes(yes)
    , _epstable(epstable), _c2p_eps_min(c2p_eps_min), _c2p_h_min(c2p_h_min)
    , _c2p_h_max(c2p_h_max), _c2p_press_max(c2p_press_max), _energy_shift(energy_shift)
    {}

  private:

    Kokkos::View<double****, grace::default_space> _alltables ; 
    Kokkos::View<double***, grace::default_space> _epstable;
    Kokkos::View<double*, grace::default_space> _logrho, _logtemp, _yes ;
    double _c2p_eps_min, _c2p_h_min, _c2p_h_max, _c2p_press_max, _energy_shift;

} ;


// Catch HDF5 errors
#define HDF5_ERROR(fn_call)                                          \
  do {                                                               \
    int _error_code = fn_call;                                       \
    if (_error_code < 0) {                                           \
      printf(    "HDF5 call '%s' returned error code %d", #fn_call,  \
                  _error_code);                                      \
    }                                                                \
  } while (0)

//Check to see if file is readable
static inline int file_is_readable(const char *filename) {
  FILE *fp = NULL;
  fp = fopen(filename, "r");
  if (fp != NULL) {
    fclose(fp);
    return 1;
  }
  return 0;
}
                               

//Following two functions are used to read in a lot of variables in the same way
//The first reads the meta date of a group in our case we are interested in the number of points
static inline void READ_ATTR_HDF5_COMPOSE(hid_t GROUP, const char *NAME, void * VAR, hid_t TYPE) {
  hid_t dataset;
  HDF5_ERROR(dataset = H5Aopen(GROUP, NAME, H5P_DEFAULT));            
  HDF5_ERROR(H5Aread(dataset, TYPE, VAR));                            
  HDF5_ERROR(H5Aclose(dataset));
}

//The second function reads in values from the HDF5 data set
//A memory buffer is passed into var for the data to be read into
static inline void READ_EOS_HDF5_COMPOSE(hid_t GROUP, const char *NAME, void * VAR, hid_t TYPE, hid_t MEM) {
  hid_t dataset;                                                      
  HDF5_ERROR(dataset = H5Dopen2(GROUP, NAME, H5P_DEFAULT));         
  HDF5_ERROR(H5Dread(dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR)); 
  HDF5_ERROR(H5Dclose(dataset));    
}


static tabulated_eos_t setup_tabulated_eos_compose(const char *nuceos_table_name) {
//static int setup_tabulated_eos_compose(const char *nuceos_table_name) {
  
  using namespace physical_constants;

  constexpr size_t NTABLES = tabulated_eos_t::EV::NUM_VARS;

  GRACE_INFO("*******************************");
  GRACE_INFO("Reading COMPOSE nuc_eos table file:");
  GRACE_INFO("{}", nuceos_table_name);
  GRACE_INFO("*******************************");

  hid_t file;

  if (!file_is_readable(nuceos_table_name)){
      ERROR("Could not read nuceos_table_name " << nuceos_table_name);
  }

  //HDF5 file is opened
  HDF5_ERROR(file = H5Fopen(nuceos_table_name, H5F_ACC_RDONLY, H5P_DEFAULT));
  
  hid_t parameters;

  //Parameter group is opened
  //From the parameter group the dimensions of the tables can be read 
  HDF5_ERROR(parameters = H5Gopen(file, "/Parameters", H5P_DEFAULT));

  //Table dimensions will be stored in these variables
  int nrho, ntemp, nye;

  // Read size of tables
  READ_ATTR_HDF5_COMPOSE(parameters,"pointsnb", &nrho, H5T_NATIVE_INT);
  READ_ATTR_HDF5_COMPOSE(parameters,"pointst", &ntemp, H5T_NATIVE_INT);
  READ_ATTR_HDF5_COMPOSE(parameters,"pointsyq", &nye, H5T_NATIVE_INT);

  //Will be exported at end of function, is not used within function scope
  std::array<size_t, 3> num_points = {size_t(nrho), size_t(ntemp), size_t(nye)};    

  // Allocate memory for tables
  double *logrho = new double[nrho];
  double *logtemp = new double[ntemp];
  double *yes = new double[nye];

  // Read values of denisty, tempreature and electron fraction respectivley
  READ_EOS_HDF5_COMPOSE(parameters,"nb", logrho, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_EOS_HDF5_COMPOSE(parameters,"t", logtemp, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_EOS_HDF5_COMPOSE(parameters,"yq", yes, H5T_NATIVE_DOUBLE, H5S_ALL);

  //Density, temperatur and electron fraction make up the basis of the grid
  //Now we load in the data that correspond to the values at each table point
  //We start with the thermal tables

  hid_t thermo_id;
  HDF5_ERROR(thermo_id = H5Gopen(file, "/Thermo_qty", H5P_DEFAULT));
  
  //We need to find the number of thermal tables in the HDF5 file
  int nthermo;
  READ_ATTR_HDF5_COMPOSE(thermo_id,"pointsqty", &nthermo, H5T_NATIVE_INT);

  // Read thermo index array
  int *thermo_index = new int[nthermo];
  READ_EOS_HDF5_COMPOSE(thermo_id,"index_thermo", thermo_index, H5T_NATIVE_INT, H5S_ALL);

  // Allocate memory and read table
  double *thermo_table = new double[nthermo * nrho * ntemp * nye];
  READ_EOS_HDF5_COMPOSE(thermo_id,"thermo", thermo_table, H5T_NATIVE_DOUBLE, H5S_ALL);

  // Now read compositions!

  // number of available particle information
  int ncomp = 0;
  hid_t comp_id;

  //Turns off some HDF5 error messages
  int status_e = H5Eset_auto(H5E_DEFAULT, NULL, NULL);
  int status_comp = H5Gget_objinfo(file, "/Composition_pairs", 0, nullptr);

  //
  if(status_comp == 0){
    HDF5_ERROR(comp_id = H5Gopen(file, "/Composition_pairs", H5P_DEFAULT));
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

  int status_av = H5Gget_objinfo(file, "Composition_quadruples", 0, nullptr);

  hid_t av_id;

  if(status_av ==0){
    HDF5_ERROR(av_id = H5Gopen(file, "/Composition_quadruples", H5P_DEFAULT));
    READ_ATTR_HDF5_COMPOSE(av_id, "pointsav", &nav, H5T_NATIVE_INT);
  }

  if(nav >0){
    //If nav is not equal to 1 the code will terminate 
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

  HDF5_ERROR(H5Fclose(file));

  // Need to sort the thermo indices to match the tabulated_eos_t ordering

  //Compose associates table variables with specific numerical values
  constexpr size_t PRESS_C = 1;
  constexpr size_t S_C = 2;
  constexpr size_t MUN_C = 3;
  constexpr size_t MUP_C = 4;
  constexpr size_t MUE_C = 5;
  constexpr size_t EPS_C = 7;
  constexpr size_t CS2_C = 12;


  //Lambda function to go through the thermo_index array and 
  //finds array location of quiered index
  auto const find_index = [&](size_t const &index) {
    for (int i = 0; i < nthermo; ++i) {
      if (thermo_index[i] == index) return i;
    }
    assert(!"Could not find index of all required quantities. This should not "
            "happen.");
    return -1;
  };

  // IMPORTANT: The order here needs to match EV from tabulated_eos_t object!
  //Array here contains location of variables in the thermo_index array
  int thermo_index_conv[7]{find_index(PRESS_C), find_index(EPS_C),
                           find_index(S_C),     find_index(CS2_C),
                           find_index(MUE_C),   find_index(MUP_C),
                           find_index(MUN_C)};


  //Want to copy table data to the all table array with correct ordering 

  //Allocate memory for the all table array, good point to introduce kokkos views


  //Create Kokkos views to pass data too
  //TODO! What is the best odering for access patterns
  Kokkos::View<double****, grace::default_space> alltables("AllTables", nrho, ntemp, nye, NTABLES); 
  Kokkos::View<double *, grace::default_space> logrhoview("LogRhoView", nrho);
  Kokkos::View<double *, grace::default_space> logtempview("LogTempView", ntemp);
  Kokkos::View<double *, grace::default_space> yesview("yesView", nye);

  
  auto h_alltables = Kokkos::create_mirror_view(alltables); 
  auto h_logrhoview = Kokkos::create_mirror_view(logrhoview); 
  auto h_logtempview = Kokkos::create_mirror_view(logtempview); 
  auto h_yesview = Kokkos::create_mirror_view(yesview);


  //Allocate data to kokkos views and convert units/convert logs to natural log
  for (int i = 0; i < nrho; i++) h_logrhoview(i) = log(logrho[i] * baryon_mass * cm3_to_fm3 * densCGS_to_CU);
  for (int i = 0; i < ntemp; i++) h_logtempview(i) = log(logtemp[i]);
  for (int i = 0; i < nye; i++) h_yesview(i) = yes[i];

  //Every element of the thermal table is itterated through. The old
  //index is saved and the index required for the GRACE odering is calculated
  //Data is then transfered from the thermal tables to the all tables 
  for (int iv = tabulated_eos_t::EV::PRESS; iv <= tabulated_eos_t::EV::MUN; iv++)
    for (int k = 0; k < nye; k++)
      for (int j = 0; j < ntemp; j++)
        for (int i = 0; i < nrho; i++) {
          auto const iv_thermo = thermo_index_conv[iv];
          int indold = i + nrho * (j + ntemp * (k + nye * iv_thermo));
          h_alltables(i, j, k, iv) = thermo_table[indold];
        }

  //Lambda function to work out index_yi location of table identifier ID
  auto const find_index_yi = [&](size_t const &index) {
    for (int i = 0; i < ncomp; ++i) {
      if (index_yi[i] == index) return i;
    }
    assert(!"Could not find index of all required quantities. This should not "
            "happen.");
    return -1;
  };


  //A similar method as above is used to fix average compositions!
  for (int k = 0; k < nye; k++)
    for (int j = 0; j < ntemp; j++)
      for (int i = 0; i < nrho; i++) {
        int indold = i + nrho * (j + ntemp * k);
        int indnew = NTABLES * (i + nrho * (j + ntemp * k));

	      if(nav >0){
	        // ABAR
          h_alltables(i, j, k, tabulated_eos_t::EV::ABAR) = aav_table[indold];
	        // ZBAR
          h_alltables(i, j, k, tabulated_eos_t::EV::ZBAR) = zav_table[indold];
	        // Xh
          h_alltables(i, j, k, tabulated_eos_t::EV::XH) = aav_table[indold] * yav_table[indold];
	      }
	
        //Here the identifier ID is hard coded 
        if(ncomp>0){
	        // Xn
          h_alltables(i, j, k, tabulated_eos_t::EV::XN) =
            yi_table[indold + nrho * nye * ntemp * find_index_yi(10)];
	        // Xp
          h_alltables(i, j, k, tabulated_eos_t::EV::XP) =
            yi_table[indold + nrho * nye * ntemp * find_index_yi(11)];
	        // Xa
          h_alltables(i, j, k, tabulated_eos_t::EV::XA) = 
            4. * yi_table[indold + nrho * nye * ntemp * find_index_yi(4002)];
	      }
  }

  //Free memory
  delete[] thermo_index;
  delete[] thermo_table;
  delete[] logrho;
  delete[] logtemp;
  delete[] yes;

  if(index_yi != nullptr) delete[] index_yi;
  if(yi_table != nullptr) delete[] yi_table;

  if(zav_table != nullptr) delete[] zav_table;
  if(yav_table != nullptr) delete[] yav_table;
  if(aav_table != nullptr) delete[] aav_table;

  //Allocate memory for linear energy density table
  //linear scale is used to extrapolate negative energy densities
  //double *epstable;
  Kokkos::View<double *** , grace::default_space> epstable("linear_energy_table", nrho, ntemp, nye);
  auto h_epstable = Kokkos::create_mirror_view(epstable); 
  

  double c2p_eps_min = 1.e99;
  double c2p_h_min = 1.e99;
  double c2p_h_max = 0.;
  double c2p_press_max = 0.;

  double energy_shift = 0;

  //Get eps_min and convert units
  for (int k = 0; k < nye; k++)
    for (int j = 0; j < ntemp; j++)
      for (int i = 0; i < nrho; i++) {
        double pressL, epsL, rhoL;

        c2p_eps_min = math::min(c2p_eps_min, h_alltables(i, j, k, tabulated_eos_t::EV::EPS));

        
        { //pressure
        h_alltables(i, j, k, tabulated_eos_t::EV::PRESS) = 
          log(h_alltables(i, j, k, tabulated_eos_t::EV::PRESS) * MeV_to_erg * cm3_to_fm3 * pressCGS_to_CU);

        pressL = exp(h_alltables(i, j, k, tabulated_eos_t::EV::PRESS));
        c2p_press_max = math::max(c2p_press_max, pressL);
        }

        { //eps
        if (c2p_eps_min < 0) {
          energy_shift = -2.0 * c2p_eps_min;
          h_alltables(i, j, k, tabulated_eos_t::EV::EPS) += energy_shift;
        }
        
        h_epstable(i, j, k) = h_alltables(i, j, k, tabulated_eos_t::EV::EPS);
        h_alltables(i, j, k, tabulated_eos_t::EV::EPS) = log(h_alltables(i, j, k, tabulated_eos_t::EV::EPS));
        epsL = h_epstable(i, j, k) - energy_shift;
        }

        { //cs2
        if (h_alltables(i, j, k, tabulated_eos_t::EV::CS2) < 0) h_alltables(i, j, k, tabulated_eos_t::EV::CS2) = 0;
        h_alltables(i, j, k, tabulated_eos_t::EV::CS2) = 
          math::min(0.9999999, h_alltables(i, j, k, tabulated_eos_t::EV::CS2));
        }

        { //chemical potential
        auto const mu_q = h_alltables(i, j, k, tabulated_eos_t::EV::MUP);
        auto const mu_b = h_alltables(i, j, k, tabulated_eos_t::EV::MUN);
        
        h_alltables(i, j, k, tabulated_eos_t::EV::MUP) += mu_b;
        h_alltables(i, j, k, tabulated_eos_t::EV::MUE) -= mu_q;

        }

        rhoL = exp(h_logrhoview(i));
        const double hL = 1. + epsL + pressL / rhoL;
        c2p_h_min = math::min(c2p_h_min, hL);
        c2p_h_max = math::max(c2p_h_max, hL);

      }

  //Copy data from host to device
  Kokkos::deep_copy(alltables, h_alltables);
  Kokkos::deep_copy(logrhoview, h_logrhoview);
  Kokkos::deep_copy(logtempview, h_logtempview); 
  Kokkos::deep_copy(yesview, h_yesview);
  Kokkos::deep_copy(epstable, h_epstable);


  return tabulated_eos_t{
      alltables 
    , logrhoview
    , logtempview
    , yesview
    , epstable
    , c2p_eps_min
    , c2p_h_min
    , c2p_h_max
    , c2p_press_max
    , energy_shift};
    

} 

} 