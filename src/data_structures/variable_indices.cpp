/**
 * @file variable_indices.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Macros galore.
 * @date 2024-03-12
 * 
 * @copyright This file is part of GRACE.
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

#include <code_modules.h>
#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/execution_tag.hh>

#include <grace/errors/assert.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variable_registration_helpers.hh>
#include <grace/data_structures/macros.hh>
#include <grace/errors/error.hh>


#include <Kokkos_Core.hpp>

#include <string> 

#undef DECLARE_VAR_INDEX_IMPL
#define DECLARE_VAR_INDEX_IMPL(var) int var = -1 ;
DECLARE_VARIABLE_INDICES;
#undef DECLARE_VAR_INDEX_IMPL

#define DECLARE_VAR_INDEX_IMPL(var) GRACE_DEVICE int var##_ ;
DECLARE_VARIABLE_INDICES;
#undef DECLARE_VAR_INDEX_IMPL
 


namespace grace { namespace variables { 

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
std::unordered_map<std::string, variable_properties_t<GRACE_NSPACEDIM>> 
    _varprops; 
std::unordered_map<std::string, variable_properties_t<GRACE_NSPACEDIM>> 
    _auxprops; 


std::vector<int> flux_var_indices         ; 
std::vector<int> prolongation_var_indices ;

std::vector<int> face_staggered_flux_var_indices         ;
std::vector<int> face_staggered_prolongation_var_indices ;

std::vector<int> edge_staggered_flux_var_indices         ;
std::vector<int> edge_staggered_prolongation_var_indices ;

} /* namespace grace::variables::detail */

void register_variables() {
    #ifdef GRACE_ENABLE_BURGERS 
    U = register_variable("U", {VEC(false,false,false)}
                                , true 
                                , true 
                                , true
                                , "outgoing"
                                , false ) ; 
    #endif 
    #ifdef GRACE_ENABLE_SCALAR_ADV
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
    #ifdef GRACE_ENABLE_GRMHD 
    /********************************************************************************/
    /* Valencia hydrodynamics */
    /* Conserved variables    */
    REGISTER_EVOLVED_SCALAR(DENS,"dens","outgoing") ; 
    REGISTER_EVOLVED_VECTOR(SX,SY,SZ,"stilde","outgoing") ;
    REGISTER_EVOLVED_SCALAR(TAU,"tau","outgoing") ; 
    REGISTER_EVOLVED_SCALAR(YESTAR,"ye_star","outgoing") ; 
    REGISTER_EVOLVED_SCALAR(ENTROPYSTAR,"s_star", "outgoing") ;
    /* GRMHD primitives */
    REGISTER_AUX_SCALAR(RHO,"rho","none") ; 
    REGISTER_AUX_VECTOR(VELX,VELY,VELZ,"vel","none") ; 
    REGISTER_AUX_SCALAR(YE,"ye","none") ; 
    REGISTER_AUX_SCALAR(TEMP,"temperature", "none") ;
    REGISTER_AUX_SCALAR(ENTROPY,"entropy","none") ; 
    REGISTER_AUX_SCALAR(EPS,"eps","none") ; 
    REGISTER_AUX_SCALAR(PRESS,"press","none") ; 
    /* registration of metric variables */
    REGISTER_AUX_TENSOR(GXX,GXY,GXZ,GYY,GYZ,GZZ,"gamma","none") ; 
    REGISTER_AUX_SCALAR(ALP,"alp","none") ; 
    REGISTER_AUX_VECTOR(BETA,"beta","none");
    /********************************************************************************/
    /********************************************************************************/
    /*                           COPY INDICES TO GPU                                */
    /********************************************************************************/
    /********************************************************************************/
    #endif 
    ASSERT_DBG( detail::_var_bc_types.size() == detail::num_evolved, 
                detail::num_evolved << " evolved variables but "
                "only " << detail::_var_bc_types.size() << " have BCs.\n") ; 
    #define DECLARE_VAR_INDEX_IMPL(var) int const var##0 = var;
    DECLARE_VARIABLE_INDICES ;
    #undef DECLARE_VAR_INDEX_IMPL
    #define DECLARE_VAR_INDEX_IMPL(var) var##_ = var##0;
    Kokkos::parallel_for(GRACE_EXECUTION_TAG("SYSTEM","init_var_indices"), 1, 
                        KOKKOS_LAMBDA (int const i) {
                            DECLARE_VARIABLE_INDICES ; 
                        } ) ; 
    #undef DECLARE_VAR_INDEX_IMPL
    /********************************************************************************/
    /********************************************************************************/
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
                                      , std::array<bool,GRACE_NSPACEDIM> const& staggering )
{
    using namespace detail ; 
    num_vars++;
    int nstagger = 0 ; 
    for( int idim=0; idim<GRACE_NSPACEDIM; ++idim) nstagger += int(staggering[idim]) ;
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
 * @brief Register a variable within GRACE.
 * 
 * @param name Name of the variable.
 * @param staggered Staggering of variable in each direction.
 * @param need_prolongation Whether the variable needs to be prlongated/restricted.
 * @param is_evolved Whether the variable is evolved.
 * @param need_fluxes Whether the variables needs fluxes. 
 * @return size_t Index of the variable in respective state array.
 */
static int register_variable(     std::string const& name
                                , std::array<bool, GRACE_NSPACEDIM> staggering  
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

    variable_properties_t<GRACE_NSPACEDIM> props ;
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
} } /* namespace grace::variables */