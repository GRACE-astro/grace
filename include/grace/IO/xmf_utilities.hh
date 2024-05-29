/**
 * @file xmf_utilities.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-24
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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
#ifndef GRACE_IO_XMF_UTILITIES_HH
#define GRACE_IO_XMF_UTILITIES_HH

#include <string> 

namespace grace { namespace IO{

void write_xmf_file( const std::string &filename 
                   , std::vector<int64_t> const & iters
                   , std::vector<int64_t> const& ncells 
                   , std::vector<double> const& times 
                   , std::vector<std::string> const& filenames ) ; 

}}

#endif /* GRACE_IO_XMF_UTILITIES_HH */