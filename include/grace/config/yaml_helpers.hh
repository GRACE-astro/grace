/**
 * @file yaml_helpers.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief this header contains the definition of grace's configuration file parser.
 * @version 1.0
 * @date 2026.02.20
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

#ifndef GRACE_SYSTEM_YAML_HELPERS_HH
#define GRACE_SYSTEM_YAML_HELPERS_HH

#include <grace_config.h> 

#include <yaml-cpp/yaml.h>
#include <grace/utils/type_name.hh>

#include <fstream>
#include <string>

#include <array>
#include <limits>
#include <sstream>

#include <vector>
#include <iostream> 

// CM: NB we can't use the normal error
// and print macros because when the 
// parfile is loaded the loggers are 
// not yet initialized! 
#define GRACE_PRINT_ERROR_AND_EXIT(m)        \
do {                                  \
    std::cerr << m << std::endl;       \
    std::abort();                     \
} while(false)

namespace grace {

// helper 
using node_t = YAML::Node ; 

/**
 * @brief Path from root of yaml file to parameter
 */
struct param_path {
    std::vector<std::string> identifiers;

    std::string to_string() const {
        if (identifiers.empty())
            return {};

        std::ostringstream oss;

        oss << identifiers[0];

        for (std::size_t i = 1; i < identifiers.size(); ++i) {
            oss << "->" << identifiers[i];
        }

        return oss.str();
    }

    void append(std::string const& name) { identifiers.push_back(name) ; } 
};

template <typename T>
static T get_param_safe(YAML::Node const& n, param_path const& path)
{
    auto name = path.identifiers.back() ; 
    if (!n[name]) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter missing at " << path.to_string());
    }
    T v; 
    try {
        v = n[name].as<T>();
    }
    catch (const YAML::BadConversion&) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter cannot be converted to "
              << utils::type_name<T>());
    }
    return v ; 
}

/**
 * @brief Add to a path 
 */
static param_path operator+(param_path path, std::string const& name) {
    path.append(name);
    return path;
}

/**
 * @brief Helper to represent numeric parameter ranges 
 */
template<typename T>
struct numeric_range {
    std::array<bool, 2> inclusive;
    std::array<T, 2> range;

    numeric_range(const std::string& desc) {
        if (desc.size() < 5)
            GRACE_PRINT_ERROR_AND_EXIT("Invalid range format");

        // ---- Left bracket ----
        if (desc.front() == '[')
            inclusive[0] = true;
        else if (desc.front() == '(')
            inclusive[0] = false;
        else
            GRACE_PRINT_ERROR_AND_EXIT("Invalid left bracket");

        // ---- Right bracket ----
        if (desc.back() == ']')
            inclusive[1] = true;
        else if (desc.back() == ')')
            inclusive[1] = false;
        else
            GRACE_PRINT_ERROR_AND_EXIT("Invalid right bracket");

        // ---- Remove brackets ----
        std::string inner = desc.substr(1, desc.size() - 2);

        // ---- Split at comma ----
        auto comma = inner.find(',');
        if (comma == std::string::npos)
            GRACE_PRINT_ERROR_AND_EXIT("Missing comma");

        std::string left  = inner.substr(0, comma);
        std::string right = inner.substr(comma + 1);

        // ---- Parse left bound ----
        if (left == "*") {
            if constexpr (std::numeric_limits<T>::has_infinity) {
                range[0] = -std::numeric_limits<T>::infinity();
            } else {
                range[0] = std::numeric_limits<T>::lowest();
            }

        } else {
            std::stringstream ss(left);
            ss >> range[0];

            if (!ss)
                GRACE_PRINT_ERROR_AND_EXIT("Invalid left bound");
        }

        // ---- Parse right bound ----
        if (right == "*") {
            if constexpr (std::numeric_limits<T>::has_infinity) {
                range[1] = std::numeric_limits<T>::infinity();
            } else {
                range[1] = std::numeric_limits<T>::max();
            }
        } else {
            std::stringstream ss(right);
            ss >> range[1];

            if (!ss)
                GRACE_PRINT_ERROR_AND_EXIT("Invalid right bound");
        }

        // ---- Sanity check ----
        if (range[0] > range[1])
            GRACE_PRINT_ERROR_AND_EXIT("Invalid range ordering");
    }

    bool check(const T& val) const {
        bool in_range = true;

        in_range &= inclusive[0] ? val >= range[0] : val > range[0];
        in_range &= inclusive[1] ? val <= range[1] : val < range[1];

        return in_range;
    }
};

/**
 * @brief Helper to check range and type of numeric param
 */
template< typename T >
static void check_type_and_range_impl_numeric(
    param_path const& path,
    node_t param,
    node_t schema
)
{
    if (!param.IsScalar()) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter " << path.to_string()
              << " is not scalar.");
    }

    T val;

    try {
        val = param.as<T>();
    }
    catch (const YAML::BadConversion&) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter " << path.to_string()
              << " cannot be converted to " << utils::type_name<T>() );
    }

    auto range_node = schema["range"];

    if (!range_node) {
        GRACE_PRINT_ERROR_AND_EXIT("Missing range for parameter "
              << path.to_string());
    }

    numeric_range<T> range(
        range_node.as<std::string>());

    if (!range.check(val)) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter " << path.to_string()
              << " not in allowed range.");
    }
}

void check_type_and_range_impl_bool(
    param_path const& path,
    node_t param,
    node_t schema
) ; 


/**
 * @brief Check string parameter
 */
void check_type_and_range_impl_string(
    param_path const& path,
    node_t param,
    node_t schema
) ; 

/**
 * @brief Check keyword parameter
 */
void check_type_and_range_impl_kw(
    param_path const& path,
    node_t param,
    node_t schema
) ; 

/**
 * @brief Check type and range of any parameter
 */
void check_type_and_range(
    param_path const& path,
    node_t param,
    node_t schema 
) ; 
/**
 * @brief Parse a scalar parameter of any kind 
 */
void parse_scalar_parameter(
    param_path const& path,
    node_t schema,
    node_t list 
); 

/**
 * @brief Parse a list of numbers
 */
template< typename T >
static void parse_list_numeric_impl(
    param_path const& path,
    node_t schema,
    node_t list 
)
{
    if (!list.IsSequence()) {
        GRACE_PRINT_ERROR_AND_EXIT("Expected list at " << path.to_string() );
    }

    for (std::size_t i = 0; i < list.size(); ++i) {
        YAML::Node elem = list[i];
        check_type_and_range_impl_numeric<T>(path+("[" + std::to_string(i) + "]"),elem,schema) ; 
    }
}

/**
 * @brief Parse a list of bools 
 */
void parse_list_impl_bool(
    param_path const& path,
    node_t schema,
    node_t list 
) ;

/**
 * @brief Parse a list of strings
 */
void parse_list_impl_string(
    param_path const& path,
    node_t schema,
    node_t list 
) ;

/**
 * @brief Parse a list of custom types
 */
void parse_node_list_impl(
    param_path const& path,
    node_t schema,
    node_t list 
) ;

/**
 * @brief Parse a list 
 */
void parse_list_parameter(
    param_path const& path,
    node_t schema,
    node_t list 
) ; 


// forward declare 
void check_unknown_parameters(
    param_path const& path,
    node_t schema,
    node_t params
); 

void check_unknown_parameter_list(
    param_path const& path,
    node_t schema,
    node_t params
) ; 

void check_unknown_parameters(
    param_path const& path,
    node_t schema,
    node_t params
) ;

/**
 * @brief Traverse a section of a yaml file 
 */
void traverse_section(
    param_path const& path,
    node_t schema,
    node_t params
) ; 

} /* namespace grace */

#endif 