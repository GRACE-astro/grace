/**
 * @file variable_indices.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-12
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Volume
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

#include <code_modules.h>
#include <thunder_config.h>
#include <thunder/utils/device.h>
#include <thunder/utils/execution_tag.hh>

#include <thunder/errors/assert.hh>
#include <thunder/data_structures/variable_indices.hh>
#include <thunder/data_structures/macros.hh>
#include <thunder/errors/error.hh>


#include <Kokkos_Core.hpp>

#include <string> 

#undef DECLARE_VAR_INDEX_IMPL
#define DECLARE_VAR_INDEX_IMPL(var) int var = -1 ;
DECLARE_VARIABLE_INDICES;
#undef DECLARE_VAR_INDEX_IMPL

#define DECLARE_VAR_INDEX_IMPL(var) THUNDER_DEVICE int var##_ ;
DECLARE_VARIABLE_INDICES;
#undef DECLARE_VAR_INDEX_IMPL
 


namespace thunder { namespace variables { 

namespace detail {

int num_vars      = 0 ; 
int last_evolved  = -1 ; 
int num_fluxes    = 0 ;
int last_flux     = -1 ; 
int first_flux    = -1 ; 

/****************************************************/
/*                Variable arrays sizes             */
/****************************************************/
int num_evolved   = 0 ;
int num_auxiliary = 0 ;

int num_face_staggered_vars = 0 ;
int num_face_staggered_aux = 0 ;

int num_edge_staggered_vars = 0 ;
int num_edge_staggered_aux = 0 ;

int num_corner_staggered_vars = 0 ;
int num_corner_staggered_aux  = 0 ;

int num_vector_vars = 0 ;
int num_tensor_vars = 0 ; 
/****************************************************/
/****************************************************/

/****************************************************/
/*                Variable name arrays              */
/****************************************************/
std::vector<std::string> _varnames ; 
std::vector<std::string> _auxnames ; 

std::vector<std::string> _face_staggered_varnames ;
std::vector<std::string> _face_staggered_auxnames ;

std::vector<std::string> _edge_staggered_varnames ; 
std::vector<std::string> _edge_staggered_auxnames ; 

std::vector<std::string> _corner_staggered_varnames ; 
std::vector<std::string> _corner_staggered_auxnames ;
/****************************************************/
/****************************************************/

/****************************************************/
/*             Boundary condition arrays            */
/****************************************************/
std::vector<std::string> _var_bc_types ;
std::vector<std::string> _aux_bc_types ;

std::vector<std::string> _face_vars_bc_types ;
std::vector<std::string> _face_aux_bc_types ;

std::vector<std::string> _edge_vars_bc_types ;
std::vector<std::string> _edge_aux_bc_types ;

std::vector<std::string> _corner_vars_bc_types ;
std::vector<std::string> _corner_aux_bc_types ;
/****************************************************/
/****************************************************/

/****************************************************/
/*              Handling of vector/tensor           */
/*                    components                    */
/****************************************************/
std::vector<int> _vector_var_indices ; 
std::vector<int> _tensor_var_indices ; 
/****************************************************/
/* NB: here we assume that all face/edge staggered  */
/*     variables are vector components.             */
/****************************************************/

/****************************************************/
/*              Reconstructed variables             */
/*                    indices                       */
/****************************************************/
std::vector<int> _recon_var_indices ; 
std::vector<int> _recon_aux_indices ; 

std::vector<int> _face_staggered_recon_var_indices ; 
std::vector<int> _face_staggered_recon_aux_indices ; 

std::vector<int> _edge_staggered_recon_var_indices ; 
std::vector<int> _edge_staggered_recon_aux_indices ; 

std::vector<int> _corner_staggered_recon_var_indices ;
std::vector<int> _corner_staggered_recon_aux_indices ;
/****************************************************/
/****************************************************/
std::unordered_map<std::string, variable_properties_t<THUNDER_NSPACEDIM>> 
    _varprops; 
std::unordered_map<std::string, variable_properties_t<THUNDER_NSPACEDIM>> 
    _auxprops; 


std::vector<int> flux_var_indices         ; 
std::vector<int> prolongation_var_indices ;

std::vector<int> face_staggered_flux_var_indices         ;
std::vector<int> face_staggered_prolongation_var_indices ;

std::vector<int> edge_staggered_flux_var_indices         ;
std::vector<int> edge_staggered_prolongation_var_indices ;

} /* namespace thunder::variables::detail */

void register_variables() {
    #ifdef THUNDER_ENABLE_BURGERS 
    U = register_variable("U", {VEC(false,false,false)}
                                , true 
                                , true 
                                , true
                                , "outgoing"
                                , false ) ; 
    X_SOURCE = register_variable("source[0]", {VEC(false,false,false)}
                                , true 
                                , false 
                                , false
                                , "none"
                                , false
                                , false
                                , 0
                                , "source" ) ; 
    Y_SOURCE = register_variable("source[1]", {VEC(false,false,false)}
                                , true 
                                , false 
                                , false
                                , "none"
                                , false
                                , false
                                , 1
                                , "source" ) ;
    Z_SOURCE = register_variable("source[2]", {VEC(false,false,false)}
                                , true 
                                , false 
                                , false
                                , "none"
                                , false
                                , false
                                , 2
                                , "source" ) ;
    #endif 
    #ifdef THUNDER_ENABLE_SCALAR_ADV
    U = register_variable("U", {VEC(false,false,false)}
                                , true 
                                , true 
                                , true
                                , "outgoing"
                                , false ) ; 
    ERR = register_variable("err", {VEC(false,false,false)}
                                , true 
                                , false 
                                , false
                                , "none"
                                , false ) ; 
    #endif 
    #ifdef THUNDER_ENABLE_GRMHD 
    /* Valencia hydrodynamics */
    DENS = register_variable( "dens"
                                , {VEC(false,false,false)}
                                , true 
                                , true 
                                , true
                                , "outgoing"
                                , false ) ; 
    SX = register_variable( "S[0]"
                                , {VEC(false,false,false)}
                                , true 
                                , true 
                                , true
                                , "outgoing"
                                , true
                                , false
                                , 0
                                , "S" ) ;
    SY = register_variable( "S[1]"
                                , {VEC(false,false,false)}
                                , true 
                                , true 
                                , true
                                , "outgoing"
                                , true
                                , false 
                                , 1 
                                , "S") ;
    SZ = register_variable( "S[2]"
                                , {VEC(false,false,false)}
                                , true 
                                , true 
                                , true
                                , "outgoing"
                                , true
                                , false 
                                , 2
                                , "S") ;
    TAU = register_variable( "tau"
                                , {VEC(false,false,false)}
                                , true 
                                , true 
                                , true
                                , "outgoing"
                                , false ) ;
    /* registration of metric variables */
    GXX = register_variable( "gxx"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false
                            , "outgoing" 
                            , false
                            , true
                            , 0
                            , "gamma" 
                             ) ; 

    GXY = register_variable( "gxy"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false 
                            , "outgoing"
                            , false
                            , true
                            , 1
                            , "gamma" 
                            ) ;

    GXZ = register_variable( "gxz"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false 
                            , "outgoing"
                            , false
                            , true
                            , 2
                            , "gamma") ;

    GYY = register_variable( "gyy"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false 
                            , "outgoing"
                            , false
                            , true
                            , 3
                            , "gamma" 
                            ) ;

    GYZ = register_variable( "gyz"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false
                            , "outgoing"
                            , false
                            , true
                            , 4
                            , "gamma"  ) ;

    GZZ = register_variable( "gzz"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false 
                            , "outgoing"
                            , false
                            , true
                            , 5
                            , "gamma" ) ;

    ALP = register_variable( "alp"
                            , {VEC(false,false,false)} 
                            , true 
                            , true
                            , false
                            , "outgoing"
                            , false 
                            ) ;

    BETAX = register_variable( "beta[0]"
                                , {VEC(false,false,false)} 
                                , true 
                                , true
                                , false 
                                , "outgoing"
                                , true
                                , false 
                                , 0
                                , "beta"
                                ) ;


    BETAY = register_variable( "beta[1]"
                                , {VEC(false,false,false)} 
                                , true 
                                , true
                                , false 
                                , "outgoing"
                                , true
                                , false 
                                , 1
                                , "beta"
                                ) ;


    BETAZ = register_variable( "beta[2]"
                                , {VEC(false,false,false)} 
                                , true 
                                , true
                                , false 
                                , "outgoing"
                                , true
                                , false 
                                , 2
                                , "beta" ) ;
    #endif 
    ASSERT_DBG( detail::_var_bc_types.size() == detail::num_evolved, 
                detail::num_evolved << " evolved variables but "
                "only " << detail::_var_bc_types.size() << " have BCs.\n") ; 
    #define DECLARE_VAR_INDEX_IMPL(var) int const var##0 = var;
    DECLARE_VARIABLE_INDICES ;
    #undef DECLARE_VAR_INDEX_IMPL
    #define DECLARE_VAR_INDEX_IMPL(var) var##_ = var##0;
    Kokkos::parallel_for(THUNDER_EXECUTION_TAG("SYSTEM","init_var_indices"), 1, 
                        KOKKOS_LAMBDA (int const i) {
                            DECLARE_VARIABLE_INDICES ; 
                        } ) ; 
    #undef DECLARE_VAR_INDEX_IMPL
}
namespace detail {
static int register_scalar( std::string const& name
                          , bool is_evolved 
                          , bool need_fluxes
                          , std::string const & bc_type )
{
    using namespace detail ; 
    if( need_fluxes ) {
        if( first_flux == -1 ){
            ASSERT(num_vars == 0,
                   "The first registered evolved variable"
                   " must be a flux variable." ) ; 
            first_flux = num_vars ;
        } else {
            ASSERT( last_flux == num_vars - 1,
                    "Flux variables need to be a contiguous"
                    " block at the start of the evolved array" ) ; 
        } 
        last_flux = num_vars ; 
        num_fluxes ++ ; 
    }
    num_vars++ ; 
    if( is_evolved ){
        last_evolved = num_vars; 
        num_evolved ++ ; 
        _varnames.push_back(name) ; 
        _var_bc_types.push_back(bc_type) ; 
    } else {
        num_auxiliary ++ ; 
        _auxnames.push_back(name) ; 
        _aux_bc_types.push_back(bc_type) ;
    }

    return  is_evolved ? num_evolved-1 : num_auxiliary-1 ; 
}

static int register_staggered_variable( std::string const& name
                                      , bool is_evolved 
                                      , bool need_fluxes
                                      , std::string const & bc_type 
                                      , std::array<bool,THUNDER_NSPACEDIM> const& staggering )
{
    using namespace detail ; 
    num_vars++;
    int nstagger = 0 ; 
    for( int idim=0; idim<THUNDER_NSPACEDIM; ++idim) nstagger += int(staggering[idim]) ;
    if( nstagger == 1 ) {
        if( is_evolved ) {
            _face_staggered_varnames.push_back(name) ; 
            _face_vars_bc_types.push_back(bc_type) ;
            return (num_face_staggered_vars ++) - 1 ; 
        } else {
            _face_staggered_auxnames.push_back(name) ; 
            _face_aux_bc_types.push_back(bc_type) ; 
            return (num_face_staggered_aux ++) - 1 ; 
        }
    } else if (nstagger == 2) {
        if( is_evolved ) {
            _edge_staggered_varnames.push_back(name) ; 
            _edge_vars_bc_types.push_back(bc_type) ; 
            return (num_edge_staggered_vars ++) - 1 ; 
        } else {
            _face_staggered_auxnames.push_back(name) ; 
            _edge_aux_bc_types.push_back(bc_type) ; 
            return (num_edge_staggered_aux ++) - 1 ; 
        }
    } else if (nstagger == 3) {
        if( is_evolved ) {
            _corner_staggered_varnames.push_back(name) ; 
            _corner_vars_bc_types.push_back(bc_type) ; 
            return (num_corner_staggered_vars ++) - 1 ; 
        } else {
            _corner_staggered_auxnames.push_back(name) ; 
            _corner_aux_bc_types.push_back(bc_type) ; 
            return (num_corner_staggered_aux ++) - 1 ; 
        }
    } else {
        ERROR("Something wrong!") ; 
    }
}

static int register_vector( std::string const& name
                          , bool is_evolved 
                          , bool need_fluxes
                          , int num_comp
                          , std::string const & bc_type )
{
    using namespace detail ; 


    if( need_fluxes ) {
        if( first_flux == -1 ){
            ASSERT(num_vars == 0,
                   "The first registered evolved variable"
                   " must be a flux variable." ) ; 
            first_flux = num_vars ;
        } else {
            ASSERT( last_flux == num_vars - 1,
                    "Flux variables need to be a contiguous"
                    " block at the start of the evolved array" ) ; 
        } 
        last_flux = num_vars ; 
        num_fluxes ++ ; 
    }
    num_vars++ ; 
    if( is_evolved ){
        _varnames.push_back(name) ;
        if( num_comp == 0 ) {
            _vector_var_indices.push_back(num_evolved) ; 
        }
        last_evolved = num_vars; 
        num_evolved ++ ; 
        _var_bc_types.push_back(bc_type) ; 
    } else {
        _auxnames.push_back(name) ; 
        num_auxiliary ++ ; 
        _aux_bc_types.push_back(bc_type) ;
    }

    return  is_evolved ? num_evolved-1 : num_auxiliary-1 ; 
}

static int register_tensor( std::string const& name
                          , bool is_evolved 
                          , bool need_fluxes
                          , int num_comp
                          , std::string const & bc_type )
{
    using namespace detail ; 


    if( need_fluxes ) {
        if( first_flux == -1 ){
            ASSERT(num_vars == 0,
                   "The first registered evolved variable"
                   " must be a flux variable." ) ; 
            first_flux = num_vars ;
        } else {
            ASSERT( last_flux == num_vars - 1,
                    "Flux variables need to be a contiguous"
                    " block at the start of the evolved array" ) ; 
        } 
        last_flux = num_vars ; 
        num_fluxes ++ ; 

    }
    num_vars++ ; 
    if( is_evolved ){
        _varnames.push_back(name) ; 
        if( num_comp == 0 ) {
            _tensor_var_indices.push_back(num_evolved) ; 
        }
        last_evolved = num_vars; 
        num_evolved ++ ; 
        _var_bc_types.push_back(bc_type) ; 
    } else {
        _auxnames.push_back(name) ; 
        num_auxiliary ++ ; 
        _aux_bc_types.push_back(bc_type) ;
    }

    return  is_evolved ? num_evolved-1 : num_auxiliary-1 ; 
}
}


/**
 * @brief Register a variable within Thunder.
 * 
 * @param name Name of the variable.
 * @param staggered Staggering of variable in each direction.
 * @param need_prolongation Whether the variable needs to be prlongated/restricted.
 * @param is_evolved Whether the variable is evolved.
 * @param need_fluxes Whether the variables needs fluxes. 
 * @return size_t Index of the variable in respective state array.
 */
static int register_variable(     std::string const& name
                                , std::array<bool, THUNDER_NSPACEDIM> staggering  
                                , bool need_prolongation
                                , bool is_evolved 
                                , bool need_fluxes
                                , std::string const & bc_type 
                                , bool is_vector
                                , bool is_tensor
                                , int comp_num
                                , std::string const& vec_name ) 
{
    using namespace detail ; 

    variable_properties_t<THUNDER_NSPACEDIM> props ;
    props.staggering = staggering ; 
    props.has_gz     = is_evolved ; 
    props.is_vector  = is_vector  ; 
    props.is_tensor  = is_tensor  ; 
    props.name   = (is_tensor || is_vector) ?  vec_name : name  ;
    if ( is_evolved ) {
        detail::_varprops[name] = props ; 
    } else {
        detail::_auxprops[name] = props ; 
    }
    num_vector_vars += static_cast<int>(is_vector) ; 
    num_tensor_vars += static_cast<int>(is_tensor) ; 
    bool is_staggered = false ; 
    for( auto const & s: staggering ) is_staggered |= s ; 
    if( is_staggered ) {
        return register_staggered_variable(name,is_evolved,need_fluxes,bc_type,staggering) ; 
    } else {
        if ( is_vector ) {
            return register_vector(name,is_evolved,need_fluxes,comp_num,bc_type) ; 
        } else if ( is_tensor ) {
            return register_tensor(name,is_evolved,need_fluxes,comp_num,bc_type) ; 
        } else {
            return register_scalar(name,is_evolved,need_fluxes,bc_type) ; 
        }   
    }
}
} } /* namespace thunder::variables */