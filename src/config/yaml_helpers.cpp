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

#include <grace_config.h> 

#include <yaml-cpp/yaml.h>
#include <grace/config/yaml_helpers.hh>

#include <grace/system/print.hh> 
#include <grace/errors/error.hh>

namespace grace {

void check_type_and_range_impl_bool(
    param_path const& path,
    node_t param,
    node_t schema
)
{
    if (!param.IsScalar()) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter " << path.to_string()
              << " is not scalar.");
    }

    try {
        (void)param.as<bool>();
    }
    catch (const YAML::BadConversion&) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter " << path.to_string()
              << " cannot be converted to bool" );
    }
}

void check_type_and_range_impl_string(
    param_path const& path,
    node_t param,
    node_t schema
)
{
    if (!param.IsScalar()) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter " << path.to_string()
              << " is not scalar.");
    }

    std::string val;

    try {
        val = param.as<std::string>();
    }
    catch (const YAML::BadConversion&) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter " << path.to_string()
              << " cannot be converted to string" );
    }
}

void check_type_and_range(
    param_path const& path,
    node_t param,
    node_t schema 
)
{
    const auto type_str = schema["type"].as<std::string>();

    if (type_str == "double") {
        check_type_and_range_impl_numeric<double>(path, param, schema);
    }
    else if (type_str == "unsigned int") {
        check_type_and_range_impl_numeric<unsigned int>(path, param, schema);
    }
    else if (type_str == "int") {
        check_type_and_range_impl_numeric<int>(path, param, schema);
    }
    else if (type_str == "bool") {
        check_type_and_range_impl_bool(path, param, schema);
    }
    else if (type_str == "keyword") {
        check_type_and_range_impl_kw(path, param, schema);
    }
    else if (type_str == "string") {
        check_type_and_range_impl_string(path, param, schema);
    }
    else {
        GRACE_PRINT_ERROR_AND_EXIT("Unknown type '" << type_str
              << "' for parameter " << path.to_string());
    }
}

void check_type_and_range_impl_kw(
    param_path const& path,
    node_t param,
    node_t schema
)
{
    // ---- Must be scalar ----
    if (!param.IsScalar()) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter " << path.to_string()
              << " is not scalar.");
    }

    // ---- Read value ----
    std::string val;

    try {
        val = param.as<std::string>();
    }
    catch (const YAML::BadConversion&) {
        GRACE_PRINT_ERROR_AND_EXIT("Parameter " << path.to_string()
              << " cannot be converted to string");
    }

    // ---- Read allowed list ----
    auto allowed_node = schema["allowed"];

    if (!allowed_node || !allowed_node.IsSequence()) {
        GRACE_PRINT_ERROR_AND_EXIT("Schema for parameter " << path.to_string()
              << " does not define a valid 'allowed' list");
    }

    std::vector<std::string> allowed;

    try {
        allowed = allowed_node.as<std::vector<std::string>>();
    }
    catch (const YAML::BadConversion&) {
        GRACE_PRINT_ERROR_AND_EXIT("Schema for parameter " << path.to_string()
              << " has invalid 'allowed' values");
    }

    // ---- Check membership ----
    bool ok = false;

    for (const auto& s : allowed) {
        if (s == val) {
            ok = true;
            break;
        }
    }

    if (!ok) {
        GRACE_PRINT_ERROR_AND_EXIT("Invalid value '" << val
              << "' for parameter " << path.to_string() );
    }
}

void parse_scalar_parameter(
    param_path const& path,
    node_t schema,
    node_t list 
)
{
    auto default_val = schema["default"] ;
    if (!default_val) {
        GRACE_PRINT_ERROR_AND_EXIT("Missing default for parameter " << path.to_string() ) ; 
    }
    auto key = path.identifiers.back() ; 
    if (!list[key]) {
        list[key] = default_val ;
    }

    check_type_and_range(path, list[key], schema) ; 
}

void parse_list_impl_bool(
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
        check_type_and_range_impl_bool(path+("[" + std::to_string(i) + "]"),elem,schema) ; 
    }
}

void parse_list_impl_string(
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
        check_type_and_range_impl_string(path+("[" + std::to_string(i) + "]"),elem,schema) ; 
    }
}

void parse_node_list_impl(
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
        traverse_section(path+("[" + std::to_string(i) + "]"), schema["item_schema"], elem) ; 
    }
}

void parse_list_parameter(
    param_path const& path,
    node_t schema,
    node_t list 
)
{
    auto key = path.identifiers.back() ; 
    if (!list[key]) {
        if (schema["default"]) {
            list[key] = schema["default"] ; 
        } else {
            list[key] = YAML::Node(YAML::NodeType::Sequence);
        }
    }
    
    auto item_type_str = get_param_safe<std::string>(schema,path+"item_type") ; 
    if (item_type_str == "node") {
        parse_node_list_impl(path,schema,list[key]) ; 
    } else if (item_type_str == "double" ) {
        parse_list_numeric_impl<double>(path,schema,list[key]) ; 
    } else if (item_type_str == "unsigned int") {
        parse_list_numeric_impl<unsigned int>(path,schema,list[key]) ; 
    } else if (item_type_str == "bool") {
        parse_list_impl_bool(path,schema,list[key]) ; 
    } else if (item_type_str == "int") {
        parse_list_numeric_impl<int>(path,schema,list[key]) ; 
    } else if (item_type_str == "string") {
        parse_list_impl_string(path,schema,list[key]) ; 
    } else {
        GRACE_PRINT_ERROR_AND_EXIT("Unrecognized item_type " << item_type_str << " for list at " << path.to_string()) ; 
    }
}


void traverse_section(
    param_path const& path,
    node_t schema,
    node_t params
) {

    for (auto it = schema.begin(); it != schema.end(); ++it) {

        const std::string name =
            it->first.as<std::string>();

        // NB: We are assuming here that if a 
        // node is scalar it is NOT a parameter. 
        // This is because of the nested nature of 
        // the parameter file, where sections sometimes
        // have a description or even a type == "schema"
        // which should not be misunderstood as parameters
        if ( it->second.IsScalar() ) continue ; 

        const std::string type_str =
            it->second["type"].as<std::string>();
        
        if (type_str == "list") {
            parse_list_parameter(path + name,
                                 it->second,
                                 params);
        }
        else if ( type_str == "schema" ) {
            if (!params[name] || !params[name].IsMap()) {
                params[name] = YAML::Node(YAML::NodeType::Map);
            }
            traverse_section(
                path+name,
                it->second,
                params[name]
            ) ; 
        } else {
            parse_scalar_parameter(path + name,
                                   it->second,
                                   params);
        }
    }

} 

void check_unknown_parameter_list(
    param_path const& path,
    node_t schema,
    node_t params
)
{
    if (!params.IsSequence()) return ; 

    for (std::size_t i = 0; i < params.size(); ++i) {
        YAML::Node elem = params[i];
        for (auto it = elem.begin(); it != elem.end(); ++it) {
            const std::string name =
                it->first.as<std::string>();
            check_unknown_parameters(
                path + "["+ std::to_string(i) + "]" + name , 
                schema["item_schema"],
                it->second
            ) ; 
        }
    }
}

void check_unknown_parameters(
    param_path const& path,
    node_t schema,
    node_t params
) 
{
    if (!params || !params.IsMap()) return;
    if (!schema || !schema.IsMap()) return;

    for (auto it = params.begin(); it != params.end(); ++it) {

        const std::string name =
            it->first.as<std::string>();

        // Ignore keys that exist in schema
        if (!schema[name]) {
            std::cerr << "Unknown parameter at path " <<  (path+name).to_string() << std::endl; 
            continue;
        }

        // If schema says "schema", recurse
        if (schema[name]["type"]
            && schema[name]["type"].as<std::string>() == "schema")
        {
            check_unknown_parameters(
                path + name,
                schema[name],
                it->second
            );
        }

        // If schema says "schema", recurse
        if (schema[name]["type"]
            && schema[name]["type"].as<std::string>() == "list"
            && schema[name]["item_type"].as<std::string>() == "node" )
        {
            check_unknown_parameter_list(
                path + name,
                schema[name],
                it->second
            );
        }
    }
}

}