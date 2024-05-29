/**
 * @file grace/utils/make_string.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-03-10
 * 
 * Adapted from analogous function in SpeCTRE.
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

 #ifndef GRACE_UTILS_MAKE_STRING_HH
 #define GRACE_UTILS_MAKE_STRING_HH

 #include <sstream>

namespace utils {
/**
 * @brief make a string out of a stream
 * \ingroup utils
 * Usage: 
 * \code 
 * std::string str = make_string{} << "your personal string stream\n" ; 
 * \endcode 
 */
class make_string {
 public:
  make_string() = default;
  make_string(const make_string&) = delete;
  make_string& operator=(const make_string&) = delete;
  make_string(make_string&&) = default;
  make_string& operator=(make_string&&) = delete;
  ~make_string() = default;

  operator std::string() const { return stream_.str(); }

  template <class T>
  friend make_string operator<<(make_string&& ms, const T& t) {
    ms.stream_ << t;  
    return std::move(ms);
  }

  template <class T>
  friend make_string& operator<<(make_string& ms, const T& t) {
    ms.stream_ << t;  
    return ms;
  }

 private:
  std::stringstream stream_{};
};

inline std::ostream& operator<<(std::ostream& os, const make_string& t) {
  return os << std::string{t};
}

} 

 #endif 