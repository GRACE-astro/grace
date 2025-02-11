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

// //Macro to catch HDF5 errors
// #define HDF5_ERROR(fn_call)                                          
//   do {                                                               
//     int _error_code = fn_call;                                       
//     if (_error_code < 0) {                                           
//       printf(    "HDF5 call '%s' returned error code %d", #fn_call, 
//                   _error_code);                                      
//     }                                                                
//   } while (0)

// //Check to see if file is readable
// static inline int file_is_readable(const char *filename) {
//   FILE *fp = NULL;
//   fp = fopen(filename, "r");
//   if (fp != NULL) {
//     fclose(fp);
//     return 1;
//   }
//   return 0;
// }
// #endif

// // Use these two defines to easily read in a lot of variables in the same way
// // The first reads in one variable of a given type completely
// #define READ_EOS_HDF5_COMPOSE(GROUP, NAME, VAR, TYPE, MEM)                     
//   do {                                                                  
//     hid_t dataset;                                                      
//     HDF5_ERROR(dataset = H5Dopen2(GROUP, NAME, H5P_DEFAULT));            
//     HDF5_ERROR(H5Dread(dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR)); 
//     HDF5_ERROR(H5Dclose(dataset));                                      
//   } while (0)

// #define READ_ATTR_HDF5_COMPOSE(GROUP, NAME, VAR, TYPE)                     
//   do {                                                                  
//     hid_t dataset;                                                      
//     HDF5_ERROR(dataset = H5Aopen(GROUP, NAME, H5P_DEFAULT));            
//     HDF5_ERROR(H5Aread(dataset, TYPE, VAR)); 
//     HDF5_ERROR(H5Aclose(dataset));                                      
//   } while (0)


// static tabulated_eos_t setup_tabulated_eos_compose(const char *nuceos_table_name) {
    
//     constexpr size_t NTABLES = tabulated_eos_t::EV::NUM_VARS;

//     std::cout << NTABLES << std::endl;

// } 

}