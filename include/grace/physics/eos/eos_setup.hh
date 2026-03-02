/**
 * @file eos_setup.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de), Khalil Pierre (khalil3.14erre@gmail.com"
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

#include <grace/config/config_parser.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/rootfinding.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/physics/eos/physical_constants.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/eos/tabulated_eos.hh>
#include <hdf5.h>

#include <Kokkos_Core.hpp>


#include <grace/utils/format_utils.hh>
// #include <grace/data_structures/grace_data_structures.hh>
#include <grace/system/grace_system.hh>

namespace grace {

//----------------------------------piecewise_polytropic----------------------------------//

static piecewise_polytropic_eos_t setup_cold_politrope() 
{
    auto& params = grace::config_parser::get() ;

    unsigned int _pwpoly_n_pieces = 
                params["eos"]["piecewise_polytrope"]["n_pieces"].as<unsigned int>() ; 
            
    double _pwpoly_kappas_0 =
        params["eos"]["piecewise_polytrope"]["kappa_0"].as<double>() ; 
    std::vector<double> _pwpoly_gammas_vec  =
        params["eos"]["piecewise_polytrope"]["gammas"].as<std::vector<double>>() ; 
    std::vector<double> _pwpoly_rhos_vec  =
        params["eos"]["piecewise_polytrope"]["rhos"].as<std::vector<double>>() ; 
    
    
    ASSERT( _pwpoly_gammas_vec.size() == _pwpoly_n_pieces
          , "Number of gammas does not coincide with n_pieces.") ;
    ASSERT( _pwpoly_rhos_vec.size() == _pwpoly_n_pieces - 1 
          , "The piecewise polytrope densities must be n_pieces-1.") ;
    /* Add 0 as first density */
    _pwpoly_rhos_vec.insert(_pwpoly_rhos_vec.begin(), 0.);
    for( int i=0; i<_pwpoly_rhos_vec.size()-1; ++i)
        ASSERT( _pwpoly_rhos_vec[i+1] > _pwpoly_rhos_vec[i]
              , "Piecewise polytrope densities must be in ascending order.") ; 
    
    /* Fill kappas eps and press */
    std::vector<double> _pwpoly_kappas_vec(_pwpoly_n_pieces)
                      , _pwpoly_press_vec(_pwpoly_n_pieces)
                      , _pwpoly_eps_vec(_pwpoly_n_pieces) ; 
    
    _pwpoly_kappas_vec[0] = _pwpoly_kappas_0 ; 
    _pwpoly_press_vec[0]  = 0 ; 
    _pwpoly_eps_vec[0]    = 0 ; 

    for( int i=1; i < _pwpoly_n_pieces; ++i ) {
        _pwpoly_kappas_vec[i] = 
            _pwpoly_kappas_vec[i-1] * 
            pow( _pwpoly_rhos_vec[i], _pwpoly_gammas_vec[i-1]-_pwpoly_gammas_vec[i]) ; 
        _pwpoly_eps_vec[i]  = 
            _pwpoly_eps_vec[i-1] +
            _pwpoly_kappas_vec[i-1] *
                pow(_pwpoly_rhos_vec[i], _pwpoly_gammas_vec[i-1]-1.) 
                / ( _pwpoly_gammas_vec[i-1]-1. ) -
            _pwpoly_kappas_vec[i] *
                pow(_pwpoly_rhos_vec[i], _pwpoly_gammas_vec[i]-1.) 
                / ( _pwpoly_gammas_vec[i]-1. ) ; 
        _pwpoly_press_vec[i] =
            _pwpoly_kappas_vec[i] *
            pow(_pwpoly_rhos_vec[i], _pwpoly_gammas_vec[i]) ; 
    }


    Kokkos::View<double [piecewise_polytropic_eos_t::max_n_pieces], grace::default_space> 
        _pwpoly_kappas("Piecewise polytropic indices") ; 
    Kokkos::View<double [piecewise_polytropic_eos_t::max_n_pieces], grace::default_space> 
        _pwpoly_gammas("Piecewise polytropic adiabatic compressibilities") ; 
    Kokkos::View<double [piecewise_polytropic_eos_t::max_n_pieces], grace::default_space> 
        _pwpoly_rhos("Piecewise polytropic densities") ;
    Kokkos::View<double [piecewise_polytropic_eos_t::max_n_pieces], grace::default_space> 
        _pwpoly_press("Piecewise polytropic pressures") ;
    Kokkos::View<double [piecewise_polytropic_eos_t::max_n_pieces], grace::default_space> 
        _pwpoly_eps("Piecewise polytropic specific internal energies") ;
    
    #define DEEP_COPY_VEC_TO_VIEW(vec,view) \
            do { \
                auto host_view = Kokkos::create_mirror_view(view) ; \
                for( int i=0; i < vec.size(); ++i){                 \
                    host_view(i) = vec[i] ;                         \
                }                                                   \
                Kokkos::deep_copy(view,host_view) ;                 \
            } while(0)
    DEEP_COPY_VEC_TO_VIEW(_pwpoly_gammas_vec,_pwpoly_gammas) ; 
    DEEP_COPY_VEC_TO_VIEW(_pwpoly_kappas_vec,_pwpoly_kappas) ;
    DEEP_COPY_VEC_TO_VIEW(_pwpoly_rhos_vec,_pwpoly_rhos) ;
    DEEP_COPY_VEC_TO_VIEW(_pwpoly_press_vec,_pwpoly_press) ;
    DEEP_COPY_VEC_TO_VIEW(_pwpoly_eps_vec,_pwpoly_eps) ; 

    GRACE_INFO("Polytropic EOS initialized.") ; 

    std::ostringstream _pwpoly_gammas_str, _pwpoly_rhos_str
                     , _pwpoly_kappas_str, _pwpoly_press_str
                     , _pwpoly_eps_str;
    _pwpoly_gammas_str << _pwpoly_gammas_vec ; 
    _pwpoly_rhos_str << _pwpoly_rhos_vec ; 
    _pwpoly_kappas_str << _pwpoly_kappas_vec ; 
    _pwpoly_press_str << _pwpoly_press_vec ; 
    _pwpoly_eps_str << _pwpoly_eps_vec ; 

    GRACE_INFO("Polytropic has {} segments.", _pwpoly_n_pieces) ;
    GRACE_INFO("Polytropic Gammas: {}.", _pwpoly_gammas_str.str()) ;
    GRACE_INFO("Polytropic rhos: {}.", _pwpoly_rhos_str.str()) ;
    GRACE_INFO("Polytropic K: {}.", _pwpoly_kappas_str.str()) ;
    GRACE_INFO("Polytropic press: {}.", _pwpoly_press_str.str()) ;
    GRACE_INFO("Polytropic eps: {}.", _pwpoly_eps_str.str()) ;

    return std::move(piecewise_polytropic_eos_t{
          _pwpoly_kappas
        , _pwpoly_gammas
        , _pwpoly_rhos
        , _pwpoly_eps
        , _pwpoly_press
        , _pwpoly_n_pieces
        , 1e+10
        , 1e-20
    }) ; 

}

//----------------------------------TABULATED EOS----------------------------------//

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


static tabulated_eos_t setup_tabulated_eos_compose(const char *nuceos_table_name, bool test = false) {
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
  for (int i = 0; i < nrho; i++) h_logrhoview(i) = log(logrho[i] * baryon_mass_tabulated * cm3_to_fm3 * densCGS_to_CU);
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


  //Table bounds
  double eos_rhomax = exp(h_logrhoview(nrho - 1));
  double eos_rhomin = exp(h_logrhoview(0));

  double eos_tempmax = exp(h_logtempview[ntemp - 1]);
  double eos_tempmin = exp(h_logtempview[0]);

  double eos_yemax = h_yesview[nye - 1];
  double eos_yemin = h_yesview[0];

  //Calculate coordinate spacing

  Kokkos::View<double [tabulated_eos_t::dim::num_dim], grace::default_space> coord_spacing("CoordSpacing");
  Kokkos::View<double [tabulated_eos_t::dim::num_dim], grace::default_space> inverse_coord_spacing("InverseCoordSpacing");

  auto h_coord_spacing = Kokkos::create_mirror_view(coord_spacing); 
  auto h_inverse_coord_spacing = Kokkos::create_mirror_view(inverse_coord_spacing);

  h_coord_spacing(tabulated_eos_t::dim::rho) = h_logrhoview(1) - h_logrhoview(0);
  h_coord_spacing(tabulated_eos_t::dim::temp) = h_logtempview(1) - h_logtempview(0);
  h_coord_spacing(tabulated_eos_t::dim::yes) = h_yesview(1) - h_yesview(0);

  h_inverse_coord_spacing(tabulated_eos_t::dim::rho) = 1. / h_coord_spacing(tabulated_eos_t::dim::rho);
  h_inverse_coord_spacing(tabulated_eos_t::dim::temp) = 1. / h_coord_spacing(tabulated_eos_t::dim::temp);
  h_inverse_coord_spacing(tabulated_eos_t::dim::yes) = 1. / h_coord_spacing(tabulated_eos_t::dim::yes);
  
  //Need to read in some paramters from config file
  auto& params = grace::config_parser::get() ;

  double c2p_ye_atm   = params["eos"]["ye_atmosphere"].as<double>(); 
  double c2p_rho_atm  = params["eos"]["rho_atmosphere"].as<double>(); 
  double c2p_temp_atm = params["eos"]["temp_atmosphere"].as<double>(); 
  double c2p_eps_max  = params["eos"]["eps_maximum"].as<double>();

  bool atm_is_beta_eq = params["eos"]["atm_is_beta_eq"].as<bool>();

  bool extend_table_high = params["eos"]["extend_table_high"].as<bool>(); 

  //---------------------------------------Cold Slices---------------------------------------//
  // Compute cold beta-equilibrium slice at T = T_min
  // Code is inspired from pizzatools:
  // for each rho, find Ye s.t. mu_n = mu_p + mu_e at T_min,
  // then read off P_cold and eps_cold.

  Kokkos::View<double*, grace::default_space> cold_logpress("ColdLogPress", nrho);
  Kokkos::View<double*, grace::default_space> cold_eps     ("ColdEps",      nrho);
  Kokkos::View<double*, grace::default_space> cold_ye      ("ColdYe",       nrho);

  auto h_cold_logpress = Kokkos::create_mirror_view(cold_logpress);
  auto h_cold_eps      = Kokkos::create_mirror_view(cold_eps);
  auto h_cold_ye       = Kokkos::create_mirror_view(cold_ye);

  {
      // Ye grid spacing (assumed uniform, consistent with how coord_spacing
      // is computed for the full 3D table above)
      const double dye_spacing = h_yesview(1) - h_yesview(0);

      for (int i = 0; i < nrho; ++i) {

          // ----------------------------------------------------------------
          // Step 1: Find beta-equilibrium Ye at (rho_i, T_min)
          // Condition: mu_hat(Ye) = mu_e + mu_p - mu_n = 0
          // Note: chemical potentials in h_alltables have already had the
          // COMPOSE sign convention fixed above (MUP += MUN, MUE -= MUQ)
          // ----------------------------------------------------------------
          int    iye_lo    = -1;
          double mu_hat_lo = 0.;
          double mu_hat_hi = 0.;

          for (int k = 0; k < nye - 1; ++k) {
              const double f0 = h_alltables(i, 0, k,   tabulated_eos_t::EV::MUE)
                              + h_alltables(i, 0, k,   tabulated_eos_t::EV::MUP)
                              - h_alltables(i, 0, k,   tabulated_eos_t::EV::MUN);
              const double f1 = h_alltables(i, 0, k+1, tabulated_eos_t::EV::MUE)
                              + h_alltables(i, 0, k+1, tabulated_eos_t::EV::MUP)
                              - h_alltables(i, 0, k+1, tabulated_eos_t::EV::MUN);

              if (f0 * f1 <= 0.0) {
                  iye_lo    = k;
                  mu_hat_lo = f0;
                  mu_hat_hi = f1;
                  break;
              }
          }

          double ye_eq;

          if (iye_lo < 0) {
              // No sign change found: beta-eq is outside the table range.
              // Check sign at the first point to decide which edge to clamp to.
              // If mu_hat > 0 everywhere, matter wants to be more neutron-rich
              // (lower Ye) than the table allows -> clamp to Ye_min.
              // If mu_hat < 0 everywhere, matter wants higher Ye -> clamp to Ye_max.
              // This matches the warning behaviour in find_beta_eq() in betaeq.py.
              const double f_first = h_alltables(i, 0, 0, tabulated_eos_t::EV::MUE)
                                    + h_alltables(i, 0, 0, tabulated_eos_t::EV::MUP)
                                    - h_alltables(i, 0, 0, tabulated_eos_t::EV::MUN);
              ye_eq = (f_first > 0.0) ? h_yesview(0) : h_yesview(nye - 1);

              GRACE_INFO("Cold beta-eq slice: no beta equilibrium found in table "
                          "at rho index {}, clamping Ye to {}", i, ye_eq);
          } else {
              // Bracket found in [iye_lo, iye_lo+1].
              // Use Brent's method within this single cell for accuracy.
              // Since the bracket is already tight (one Ye cell width),
              // convergence is very fast regardless.
              auto const mu_hat_func = [&](double ye_try) {
                  // Linear interpolation in Ye at fixed (rho=i, temp=0)
                  int ik = std::max(0, std::min(nye - 2,
                            (int)((ye_try - h_yesview(0)) / dye_spacing)));
                  const double t = (ye_try      - h_yesview(ik))
                                  / dye_spacing ;

                  const double mue = h_alltables(i, 0, ik,   tabulated_eos_t::EV::MUE) * (1. - t)
                                    + h_alltables(i, 0, ik+1, tabulated_eos_t::EV::MUE) * t;
                  const double mup = h_alltables(i, 0, ik,   tabulated_eos_t::EV::MUP) * (1. - t)
                                    + h_alltables(i, 0, ik+1, tabulated_eos_t::EV::MUP) * t;
                  const double mun = h_alltables(i, 0, ik,   tabulated_eos_t::EV::MUN) * (1. - t)
                                    + h_alltables(i, 0, ik+1, tabulated_eos_t::EV::MUN) * t;

                  return mue + mup - mun;
              };

              ye_eq = utils::brent(mu_hat_func,
                                    h_yesview(iye_lo),
                                    h_yesview(iye_lo + 1),
                                    1.e-14);
          }

          // Clamp to table range as a safety measure
          ye_eq = std::max(h_yesview(0), std::min(h_yesview(nye - 1), ye_eq));

          // ----------------------------------------------------------------
          // Step 2: Interpolate P_cold and eps_cold at (rho_i, T_min, Ye_eq)
          // ----------------------------------------------------------------

          // Find Ye index for interpolation
          int iye = static_cast<int>((ye_eq - h_yesview(0)) / dye_spacing);
          iye     = std::max(0, std::min(nye - 2, iye));

          const double t_ye = (ye_eq        - h_yesview(iye))
                              / (h_yesview(iye + 1) - h_yesview(iye));

          // Pressure is stored as log(P) after the unit conversion loop above
          const double lp0          = h_alltables(i, 0, iye,     tabulated_eos_t::EV::PRESS);
          const double lp1          = h_alltables(i, 0, iye + 1, tabulated_eos_t::EV::PRESS);
          h_cold_logpress(i)        = lp0 + t_ye * (lp1 - lp0);

          // Eps is stored as log(eps + energy_shift) after the unit conversion loop.
          // We want linear eps_cold (matching eps_cold__rho_ye_impl: exp(vars) - energy_shift)
          const double le0          = h_alltables(i, 0, iye,     tabulated_eos_t::EV::EPS);
          const double le1          = h_alltables(i, 0, iye + 1, tabulated_eos_t::EV::EPS);
          h_cold_eps(i)             = exp(le0 + t_ye * (le1 - le0)) - energy_shift;

          h_cold_ye(i)              = ye_eq;
      }
  }

  // Copy cold slices to device
  Kokkos::deep_copy(cold_logpress, h_cold_logpress);
  Kokkos::deep_copy(cold_eps,      h_cold_eps);
  Kokkos::deep_copy(cold_ye,       h_cold_ye);

  GRACE_INFO("Cold beta-eq slice computed over {} density points", nrho);
  GRACE_INFO("  Ye   range: [{:.4f}, {:.4f}]", h_cold_ye(0),       h_cold_ye(nrho - 1));
  GRACE_INFO("  logP range: [{:.4f}, {:.4f}]", h_cold_logpress(0), h_cold_logpress(nrho - 1));
  GRACE_INFO("  eps  range: [{:.4e}, {:.4e}]", h_cold_eps(0),      h_cold_eps(nrho - 1));
  
  //---------------------------------------For testing---------------------------------------//

  //For unit testing I want to overwrite the pressure table with a linear function
  
  if (test == true) {

    const double x0 = h_logrhoview(0);
    const double y0 = h_logtempview(0);
    const double z0 = h_yesview(0);

    auto const z = [&] ( double x, double y, double z ) {
      return 2.5*x + 4.2*y - 5.1*z + 3.7 ;
    };

    for (int k = 0; k < nye; k++)
      for (int j = 0; j < ntemp; j++)
        for (int i = 0; i < nrho; i++) {
    
          h_alltables(i, j, k, tabulated_eos_t::EV::PRESS) = z( x0 + i * h_coord_spacing(tabulated_eos_t::dim::rho)
                                                              , y0 + j * h_coord_spacing(tabulated_eos_t::dim::temp)
                                                              , z0 + k * h_coord_spacing(tabulated_eos_t::dim::yes)) ;
      
        }


  }

  //-----------------------------------------------------------------------------------------//

  //TODO! Should this be brought more inline with hybrid eos i.e a template for tables


  GRACE_INFO("*******************************");
  GRACE_INFO("Tabulated data read, transfering to GPU");
  GRACE_INFO("*******************************");

  //Copy data from host to device
  Kokkos::deep_copy(alltables, h_alltables);
  Kokkos::deep_copy(logrhoview, h_logrhoview);
  Kokkos::deep_copy(logtempview, h_logtempview); 
  Kokkos::deep_copy(yesview, h_yesview);
  Kokkos::deep_copy(epstable, h_epstable);
  Kokkos::deep_copy(coord_spacing, h_coord_spacing);
  Kokkos::deep_copy(inverse_coord_spacing, h_inverse_coord_spacing);

  GRACE_INFO("Spacing of logrho {}", h_coord_spacing(0)) ;
  GRACE_INFO("Spacing of logtemp {}", h_coord_spacing(1)) ;
  GRACE_INFO("Spacing of yes {}", h_coord_spacing(2)) ;


  GRACE_INFO("Inverse spacing of logrho {}", h_inverse_coord_spacing(0)) ;
  GRACE_INFO("Inverse spacing of logtemp {}", h_inverse_coord_spacing(1)) ;
  GRACE_INFO("Inverse spacing of yes {}", h_inverse_coord_spacing(2)) ;

  GRACE_INFO("logrho range {} to {}", h_logrhoview(0), h_logrhoview(h_logrhoview.extent(0) - 1)) ;
  GRACE_INFO("logtemp range {} to {}", h_logtempview(0), h_logtempview(h_logtempview.extent(0) - 1)) ;
  GRACE_INFO("yes range {} to {}", h_yesview(0), h_yesview(h_yesview.extent(0) - 1)) ;




  return tabulated_eos_t{
      alltables 
    , logrhoview
    , logtempview
    , yesview
    , epstable
    , cold_logpress   
    , cold_eps        
    , cold_ye
    , coord_spacing
    , inverse_coord_spacing
    , c2p_ye_atm
    , c2p_rho_atm
    , c2p_temp_atm
    , 0 //TODO! Need to work out how to implement c2p_eps_atm best. Can use the eps__temp_rho_ye_impl routine to calculate.
    , c2p_eps_min
    , c2p_eps_max
    , c2p_h_min
    , c2p_h_max
    , c2p_press_max
    , eos_rhomax
    , eos_rhomin
    , eos_tempmax
    , eos_tempmin
    , eos_yemax
    , eos_yemin
    , baryon_mass_tabulated
    , energy_shift
    , atm_is_beta_eq
    , extend_table_high
   } ;

} 

} 



