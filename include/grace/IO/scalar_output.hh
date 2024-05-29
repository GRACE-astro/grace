/**
 * @file scalar_output.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-22
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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

#ifndef GRACE_IO_SCALAR_OUTPUT_HH
#define GRACE_IO_SCALAR_OUTPUT_HH 

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/config/config_parser.hh>
#include <grace/amr/amr_functions.hh>

#include <map>
#include <string>

namespace grace { namespace IO {

template< typename T> 
struct minmax_res_t {
    T min_val; T max_val ; 
} ; 

namespace detail {
extern std::map<std::string,minmax_res_t<double>> _minmax_reduction_vars_results ;
extern std::map<std::string,minmax_res_t<double>> _minmax_reduction_aux_results  ;
extern std::map<std::string,double> _norm2_reduction_vars_results    ;
extern std::map<std::string,double> _norm2_reduction_aux_results     ;
extern std::map<std::string,double> _integral_reduction_vars_results ;
extern std::map<std::string,double> _integral_reduction_aux_results  ;

}

void compute_reductions() ; 

void write_scalar_output() ; 

void initialize_output_files() ; 

void info_output() ; 



}}

#endif 