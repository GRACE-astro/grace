/**
 * @file variables.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-07
 * 
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

#ifndef THUNDER_DATA_STRUCTURES_VARIABLES_HH
#define THUNDER_DATA_STRUCTURES_VARIABLES_HH

#include<code_modules.h>
#include<thunder/data_structures/data_vector.hh>

#include<thunder/utils/inline.hh>

namespace thunder { 

template< size_t ndim > 
struct variable_properties_t 
{ } ; 

template<> 
struct variable_properties_t<2>
{
    using view_t = Kokkos::View<double ****, DefaultSpace> ; 
    bool stagger_x, stagger_y ; 
    unsigned int ngz ; 
    unsigned int ntl ; 

    variable_type_t type ;
    std::string name ; 

} ; 

template<> 
struct variable_properties_t<3>
{
    using view_t = Kokkos::View<double *****, DefaultSpace> ; 
    bool stagger_x, stagger_y, stagger_z ; 
    unsigned int ngz ; 
    unsigned int ntl ; 

    variable_type_t type ;
    std::string name ;
} ; 

template< size_t ndim > 
using var_array_t = variable_properties_t::view_t ; 

/**
 * @brief Register a variable within Thunder.
 * 
 * @tparam ndim Number of spatial dimensions
 * @param name Name of the variable.
 * @param staggered Staggering of variable in each direction.
 * @param need_reconstruction Whether the variable needs to be reconstructed.
 * @param is_evolved Whether the variable is evolved.
 * @param need_fluxes Whether the variables needs fluxes. 
 * @return size_t Index of the variable in respective state array.
 */
template<size_t ndim=THUNDER_NSPACEDIM> 
static size_t register_variable(  std::string const& name
                                , std::array<ndim, bool> staggered  
                                , bool need_reconstruction 
                                , bool is_evolved 
                                , bool need_fluxes ) ; 



enum variable_type_t
{
    EVOLVED_VARIABLE=0,
    AUXILIARY_VARIABLE,
    FLUX_VARIABLE,
    NUM_VARIABLE_TYPES 
} ; 






template< size_t ndim = THUNDER_NSPACEDIM > 
class variable_list_impl_t
{
 private:
    using gf_type       = variable_properties_t<ndim>::view_t ; 
    using vec_type      = Kokkos::vector<gf_type, Device>;

public: 
    
    vec_type THUNDER_FORCE_INLINE 
    getaux() { return _aux ; }

    vec_type THUNDER_FORCE_INLINE 
    getstate() { return _state ; }

    vec_type& THUNDER_FORCE_INLINE 
    getstate(int tl) {
        ASSERT_DBG( n_active_timelevels > tl+1,
                    "Requested inactive timelevel, request memory allocation first.") ; 
        vec_type ret = _state ;
        if ( tl == 1 ) { 
            ret = _state
        }
        return 
    }

    void THUNDER_FORCE_INLINE 
    alloc_state() {
        ASSERT_DBG( n_active_timelevels < 4, 
        "Maximum number of active timelevels is 4.") ; 
        if( n_active_timelevels == 2 ) { 
            alloc_state_impl(_state_p_p) ; 
        } else {
            alloc_state_impl(_state_p_p_p) ; 
        }
    }

private: 

    variable_list_impl_t() ; 

    ~variable_list_impl_t() ; 
    int n_active_timelevels ; 
    gf_type  _coords  ;                                  //!< Gridpoint coordinates    
    vec_type _state   ;                                  //!< State variables 
    vec_type _state_p ;                                  //!< Second timelevel, allocated at all times 
    vec_type _state_p_p ;                                //!< Third timelevel, needs to be explicitly allocated / deallocated 
    vec_type _state_p_p_p ;                              //!< Fourth timelevel, needs to be explicitly allocated / deallocated 
    vec_type _aux     ;                                  //!< Auxiliary variables 
    std::vector<variable_properties_t<ndim>> _varprops ; //!< Host only, used to keep track of all variables.

}

template<size_t ndim>
variable_list_impl_t<ndim>::variable_list_impl_t() 
{
    using namespace thunder; 

    auto& params = config_parser::get() ; 
    auto& forest = amr::forest::get()   ; 
    /* Read parameters from config file */
    /* Grid quadrant (octant) dimensions */
    size_t nx {params["amr"]["npoints_block_x"].as<size_t>()} ; 
    size_t ny {params["amr"]["npoints_block_y"].as<size_t>()} ; 
    size_t nz {params["amr"]["npoints_block_z"].as<size_t>()} ; 
    /* number of ghostzones for evolved vars */
    size_t ngz { params["amr"]["n_ghostzones"].as<size_t>() } ; 
    /* number of timelevels for evolved vars */
    size_t ntl { params["system"]["n_timelevels"].as<size_t>() } ; 
    /* Read active physics modules */
    std::vector<std::string> physical_modules ; 

    auto& node = params["system"]["active_modules"] ; 
    for(int i=0; i<node.size(); ++i) {
        physical_modules.push_back( node[i].as<std::string>() ) ; 
    }
    /* 
    *  for each module read variable properties file 
    *  and initialize grid functions accordingly 
    */
    for( auto const& module: physical_modules ) 
    {
       auto file = YAML::LoadFile( physical_modules_variable_lists[module] ) ;
       for(int ivar=0; ivar<file.size(); ++ivar)
       {
        // concatenate module name :: varname to enforce uniqueness of varnames
        varnames.push_back( 
                detail::concat_module_varname(module, file[ivar]["name"].as<std::string>() ) 
                ) ; 

       }

    }
}


} /* thunder */

#endif /* THUNDER_DATA_STRUCTURES_VARIABLES_HH */ 