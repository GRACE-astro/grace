/**
 * @file thunder/utils/type_name.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief print the name of a type in a pretty way
 * @version 0.1
 * @date 2023-03-10
 * 
 * Reference:
 *      https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c/56766138#56766138
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference 
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
#ifndef THUNDER_UTILS_TYPE_NAME_HH
#define THUNDER_UTILS_TYPE_NAME_HH

#include <string_view>

namespace utils 
{
/**
 * @brief Format a type-name nicely.
 * \ingroup utils
 * @tparam T type-name to be formatted.
 * @return constexpr auto <code>std::string_view</code> containing 
 *         formatted type-name.
 */
template <typename T>
constexpr auto type_name() {
  std::string_view name, prefix, suffix;
#ifdef __clang__
  name = __PRETTY_FUNCTION__;
  prefix = "auto utils::type_name() [T = ";
  suffix = "]";
#elif defined(__GNUC__)
  name = __PRETTY_FUNCTION__;
  prefix = "constexpr auto utils::type_name() [with T = ";
  suffix = "]";
#elif defined(_MSC_VER)
  name = __FUNCSIG__;
  prefix = "auto __cdecl utils::type_name<";
  suffix = ">(void)";
#else
  name = __PRETTY_FUNCTION__;
  prefix = "constexpr auto utils::type_name() [with T = ";
  suffix = "]";
#endif
  name.remove_prefix(prefix.size());
  name.remove_suffix(suffix.size());
  return name;
}

}

#endif