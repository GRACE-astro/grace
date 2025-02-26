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


//!TODO replace int return with tabulated_eos_t when finished
//static tabulated_eos_t setup_tabulated_eos_compose(const char *nuceos_table_name) {
static int setup_tabulated_eos_compose(const char *nuceos_table_name) {
    
  constexpr size_t NTABLES = tabulated_eos_t::EV::NUM_VARS;

  //!TODO ask Carlo how to get the info to work with the test case
  // GRACE_INFO("*******************************");
  // GRACE_INFO("Reading COMPOSE nuc_eos table file:");
  // GRACE_INFO("{}", nuceos_table_name);
  // GRACE_INFO("*******************************");

  hid_t file;

  //!TODO need to work out how to call ERROR calls
  // if (!file_is_readable(nuceos_table_name)){
  //     ERROR("Could not read nuceos_table_name " << nuceos_table_name);
  // }

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


  //!TODO take this out when done
  std::cout << "The dimensions of the table are (" << nrho << ", " << ntemp << ", " << nye << ")" << std::endl; 

  std::cout << "The data range for the density is (" << logrho[0] << ", " << logrho[nrho - 1] << ")" << std::endl;
  std::cout << "The data range for the tempreature is (" << logtemp[0] << ", " << logtemp[ntemp - 1] << ")" << std::endl;
  std::cout << "The data range for the electron fraction is (" << yes[0] << ", " << yes[nye - 1] << ")" << std::endl;

  //Density, temperatur and electron fraction make up the basis of the grid
  //Now we load in the data that correspond to the values at each table point
  //We start with the thermal tables

  hid_t thermo_id;
  HDF5_ERROR(thermo_id = H5Gopen(file, "/Thermo_qty", H5P_DEFAULT));
  
  //We need to find the number of thermal tables in the HDF5 file
  int nthermo;
  READ_ATTR_HDF5_COMPOSE(thermo_id,"pointsqty", &nthermo, H5T_NATIVE_INT);

  std::cout << "The number of thermal tables is " << nthermo << std::endl;

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

  std::cout << "The number of composite tables is " << nthermo << std::endl;

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

  // IMPORTANT: The order here needs to match EV in tabulated.hh !
  //Array here contains location of variables in the thermo_index array
  int thermo_index_conv[7]{find_index(PRESS_C), find_index(EPS_C),
                           find_index(S_C),     find_index(CS2_C),
                           find_index(MUE_C),   find_index(MUP_C),
                           find_index(MUN_C)};


  //Want to copy table data to the all table array with correct ordering 

  //!TODO Talk with Carlo to see if the flattened array is how data should be stroed in GRACE


  //Allocate memory for the flattened all table array, allocated using 
  //smart pointer so that the array can be exported out of the scope of function 
  auto alltables =
      std::unique_ptr<double[]>(new double[nrho * ntemp * nye * NTABLES]);

  //Every element of the thermal table is itterated through. The old
  //index is saved and the index required for the GRACE odering is calculated
  //Data is then transfered from the thermal tables to the all tables 
  for (int iv = tabulated_eos_t::EV::PRESS; iv <= tabulated_eos_t::EV::MUN; iv++)
    for (int k = 0; k < nye; k++)
      for (int j = 0; j < ntemp; j++)
        for (int i = 0; i < nrho; i++) {
          auto const iv_thermo = thermo_index_conv[iv];
          int indold = i + nrho * (j + ntemp * (k + nye * iv_thermo));
          int indnew = iv + NTABLES * (i + nrho * (j + ntemp * k));
          alltables[indnew] = thermo_table[indold];
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
	        alltables[tabulated_eos_t::EV::ABAR + indnew] = aav_table[indold];
	        // ZBAR
	        alltables[tabulated_eos_t::EV::ZBAR + indnew] = zav_table[indold];
	        // Xh
	        alltables[tabulated_eos_t::EV::XH + indnew] = aav_table[indold] * yav_table[indold];
	      }
	
        //Here the identifier ID is hard coded 
        if(ncomp>0){
	        // Xn
	        alltables[tabulated_eos_t::EV::XN + indnew] =
	          yi_table[indold + nrho * nye * ntemp * find_index_yi(10)];
	        // Xp
	        alltables[tabulated_eos_t::EV::XP + indnew] =
	        yi_table[indold + nrho * nye * ntemp * find_index_yi(11)];
	        // Xa
	        alltables[tabulated_eos_t::EV::XA + indnew] =
	        4. * yi_table[indold + nrho * nye * ntemp * find_index_yi(4002)];
	      }
      
  }

  //Free memory
  delete[] thermo_index;
  delete[] thermo_table;

  if(index_yi != nullptr) delete[] index_yi;
  if(yi_table != nullptr) delete[] yi_table;

  if(zav_table != nullptr) delete[] zav_table;
  if(yav_table != nullptr) delete[] yav_table;
  if(aav_table != nullptr) delete[] aav_table;

  //Convert units and convert logs to natural log
  // The latter is great, because exp() is way faster than pow()
  // pressure
  //TODO! Is there a nice way to write this
  for (int i = 0; i < nrho; i++) {
    logrho[i] = log(logrho[i] * physical_constants::baryon_mass * physical_constants::cm3_to_fm3 * physical_constants::densCGS_to_CU);
  }

  for (int i = 0; i < ntemp; i++) {
    logtemp[i] = log(logtemp[i]);
  }

  //Allocate memory for linear energy density table
  //linear scale is used to extrapolate negative energy densities
  double *epstable;


  //if statement is used to handel error
  //TODO! Talk to Carlo about how to handel error within GRACE framework
  //This is done on the CPU may be better to run this check on GPU
  if (!(epstable = (double *)malloc(nrho * ntemp * nye * sizeof(double)))) {
    std::cout << "Cannot allocate memory for EOS table" << std::endl;
  }

  
  //TODO! These variables are used elsewhere in GRACE what is the best way to utilise
  double c2p_eps_min = 1.e99;
  double c2p_h_min = 1.e99;
  double c2p_h_max = 0.;
  double c2p_press_max = 0.;

  double energy_shift = 0;

  //Get eps_min
  for (int i = 0; i < nrho * ntemp * nye; i++) {
    int idx = tabulated_eos_t::EV::EPS + NTABLES * i;
    c2p_eps_min = math::min(c2p_eps_min, alltables[idx]);
  };

  //convert units
  for (int i = 0; i < nrho * ntemp * nye; i++) {
    double pressL, epsL, rhoL;
    
    { // pressure
      int idx = tabulated_eos_t::EV::PRESS + NTABLES * i;
      alltables[idx] = log(alltables[idx] * physical_constants::MeV_to_erg * physical_constants::cm3_to_fm3 * physical_constants::pressCGS_to_CU);
      pressL = exp(alltables[idx]);
      c2p_press_max = math::max(c2p_press_max, pressL);
    }

    { //eps
      int idx = tabulated_eos_t::EV::EPS + NTABLES * i;
      //shift eps to a postive range if necessary
      if (c2p_eps_min < 0) {
        energy_shift = -2.0 * c2p_eps_min;
        alltables[idx] += energy_shift;
      }

      epstable[i] = alltables[idx];
      alltables[idx] = log(alltables[idx]);
      epsL = epstable[i] - energy_shift; 
    }

    { // cs2
      int idx = tabulated_eos_t::EV::CS2 + NTABLES * i;
      if (alltables[idx] < 0) alltables[idx] = 0;
      alltables[idx] = math::min(0.9999999, alltables[idx]);
    }

    { // chemical potentials

      int idx_p = tabulated_eos_t::EV::MUP + NTABLES * i;
      int idx_n = tabulated_eos_t::EV::MUN + NTABLES * i;
      int idx_e = tabulated_eos_t::EV::MUE + NTABLES * i;

      auto const mu_q = alltables[idx_p];
      
      //Note that this does not include the rest mass contribution of the
      // neutron!
      auto const mu_b = alltables[idx_n];

      //TODO! Ask Carlo about this and the following comment
      // mu_p = mu_b + mu_q = mu_n + mu_q
      // Important: To be consistent we should actually subtract the mass
      // difference between proton and neutron here, but this makes beta eq.
      // complicated. Hence we leave it this way, but fix it in the Leakage!

      alltables[idx_p] += mu_b;
      // mu_e = mu_le - mu_q
      // CHECK: we have mu_l = effective lepton chemical potential (4.7) here
      //        after (3.23) it says, mu_e = mu_le - mu_q
      //        charge neutrality says n_l = n_q = n_le + n_lmu
      //        but how do we get mu_e then?
      //   ERM: We make the explicit assumption that we have no muons....
      //        I know we have to fix this later, but for most EOS this is ok
      //        And if we had muons, I'm pretty sure the Leakage would become
      //        inconsistent...
      // Page 8: Assumptions on the relation between
      //         the electron and muon chemical potentials are discussed
      //         in the description of each model separately.
      // Page 10: In this case, the balance between the
      //          electron and muon densities depends on the assumed relation of
      //          the electron and muon chemical potentials.
      alltables[idx_e] -= mu_q;
    }

    const int irhoL = i % nrho;
    rhoL = exp(logrho[irhoL]);
    const double hL = 1. + epsL + pressL / rhoL;
    c2p_h_min = math::min(c2p_h_min, hL);
    c2p_h_max = math::max(c2p_h_max, hL);
  }


  }






  delete[] logrho;
  delete[] logtemp;
  delete[] yes;

  

  return 0;
    


    

    

} 

} 