/**
 * @file config_parser.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief this header contains the definition of grace's configuration file parser.
 * @version 0.1
 * @date 2023-03-13
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

#ifndef GRACE_SYSTEM_CONFIG_PARSER_HH
#define GRACE_SYSTEM_CONFIG_PARSER_HH

#include <grace_config.h>
#include <grace/utils/singleton_holder.hh>
#include <code_modules.h> 
#include <grace/utils/inline.h>
#include <grace/utils/type_name.hh>
#include <yaml-cpp/yaml.h>

#include "yaml_helpers.hh"

#include <fstream>
#include <string>

namespace grace { namespace detail {

//**************************************************************************************************
/**
 * @brief configuration file parser implementation.
 * \ingroup system
 * This class will be wrapped by a singleton_holder since only one instance of the 
 * parameters can exist at any given time. 
 */
class config_parser_impl_t 
{
    //**************************************************************************************************
    public:
    //**************************************************************************************************
    /**
     * @brief get instance.
     * 
     * @return YAML::Node the configuration file.
     */
    inline YAML::Node get() const { return config ; }
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief return const entry or section in config file
     * 
     * @param key identifier of queried entry or section
     * @return const YAML::Node the requested entry or section, may be null if invalid key was requested.
     */
    template<typename key_t>
    inline const YAML::Node operator[] (const key_t & key) const { return config[key]; }
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief return entry or section in config file
     * 
     * @param key identifier of queried entry or section
     * @return const YAML::Node the requested entry or section, may be null if invalid key was requested.
     */
    template<typename key_t>
    inline YAML::Node operator[] (const key_t & key) { return config[key]; }
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Write config file to disk.
     * 
     * @param filename where to write. 
     */
    void to_file(std::string const& filename) const 
    {
        std::ofstream os(filename) ; 
        os << config ; 
    }; 
    //**************************************************************************************************

    //**************************************************************************************************
    protected:
    //**************************************************************************************************
    static constexpr unsigned long longevity = SYSTEM_UTILITIES ; 
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Construct a new config parser object
     * 
     * This method simply calls the yaml-cpp LoadFile method.
     * Note that this function will never be called by user code,
     * the global config file is a singleton and will be initialized 
     * by singleton_handler.
     * @param filename yaml file to read
     */
    config_parser_impl_t(std::string const& filename) 
        : config( YAML::LoadFile(filename) ) {
            set_parameters_to_default_value() ; 
    }; 
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Destroy the config parser object
     */
    ~config_parser_impl_t() = default; 
    //**************************************************************************************************

    //**************************************************************************************************
    YAML::Node config ; //!< instance 
    //**************************************************************************************************
    friend class utils::singleton_holder<config_parser_impl_t> ;
    friend class memory::new_delete_creator<config_parser_impl_t, memory::new_delete_allocator> ;  
    //**************************************************************************************************
 private:
    /**
    * @brief Set parameters which are not specified in the config file 
    *        to their default value.
    * This function checks all the <module>_defaults.yaml for all known modules
    * (registered in code_modules.h.in). It then  goes through the parsed parameter 
    * file and sets default values for all parameters which are not overridden. This 
    * ensures that caller code can access all parameters without exceptions.
    */
    void set_parameters_to_default_value() { 
        for( int i=0 ; i < code_modules.size(); ++i ) { 
            auto mod = code_modules[i] ; 
            auto defaults = YAML::LoadFile(code_modules_default_configs[i]) ; 
            if (!config[mod] || !config[mod].IsMap()) {
                config[mod] = YAML::Node(YAML::NodeType::Map);
        } 
            
            param_path path ; 
            traverse_section(path + mod, defaults[mod], config[mod]) ; 
            
        }
    }; 
} ; 


} // namespace detail 
//*****************************************************************************************************
/**
 * @brief configuration parser singleton.
 * \ingroup system
 */
using config_parser = utils::singleton_holder<detail::config_parser_impl_t> ;
//*****************************************************************************************************

namespace detail {
template <typename... Keys>
std::string join_path(const std::string& first, const Keys&... rest) {
    std::ostringstream oss;
    oss << first;
    ((oss << " -> " << rest), ...);  // fold expression, no string literal on LHS
    return oss.str();
}

template <typename Key, typename... Keys>
YAML::Node get_node(const YAML::Node& prev, const Key& first, const Keys&... rest) {
    // if prev is null at this stage, the key path was already invalid
    if (!prev) {
        ERROR("Missing parameter at path " << detail::join_path(first, rest...));
    }

    YAML::Node next = prev[first];

    if (!next) {
        ERROR("Missing parameter at path " << detail::join_path(first, rest...));
    }

    if constexpr (sizeof...(rest) == 0) {
        // leaf reached
        return next;
    } else {
        // keep going down
        return get_node(next, rest...);
    }
}


}
/**
 * @brief Get a parameter from the config file.
 * 
 * @tparam T Type to which the parameter should be cast (always explicit)
 * @param module Code module where to fetch the param.
 * @param name   Name of the parameter to be fetched.
 * @return T The parameter interpreted as a <code>T</code>.
 */
template <typename T, typename... Keys>
static GRACE_ALWAYS_INLINE T
get_param(const std::string& first, const Keys&... rest) {
    auto& parfile = grace::config_parser::get();

    try {
        YAML::Node entry = parfile[first];

        YAML::Node node = detail::get_node(entry,rest...) ; 
        if (!node) {
            ERROR("Missing parameter at path: " << detail::join_path(first, rest...));
        }
        return node.as<T>();
    } catch (const std::exception& e) {
        ERROR("Parameter " << detail::join_path(first, rest...)
              << " could not be retrieved as "
              << utils::type_name<T>() << ".\nException raised:\n"
              << e.what());
    }
}
//*****************************************************************************************************
} // namespace grace 

#endif 