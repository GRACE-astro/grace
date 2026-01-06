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

#include <grace/system/grace_system.hh>
#include <grace/errors/assert.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variable_registration_helpers.hh>
#include <grace/data_structures/macros.hh>
#include <grace/errors/error.hh>


#include <Kokkos_Core.hpp>

#include <string> 
#include <sstream>


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
/****************************************************/
/* These mappings go idx -> name                    */
/****************************************************/
std::vector<std::string> _varnames ; 
std::vector<std::string> _auxnames ; 

std::vector<std::string> _facex_staggered_varnames ;
std::vector<std::string> _facex_staggered_auxnames ;

std::vector<std::string> _facey_staggered_varnames ;
std::vector<std::string> _facey_staggered_auxnames ;

std::vector<std::string> _facez_staggered_varnames ;
std::vector<std::string> _facez_staggered_auxnames ;

std::vector<std::string> _edgexy_staggered_varnames ; 
std::vector<std::string> _edgexy_staggered_auxnames ;

std::vector<std::string> _edgexz_staggered_varnames ; 
std::vector<std::string> _edgexz_staggered_auxnames ;

std::vector<std::string> _edgeyz_staggered_varnames ; 
std::vector<std::string> _edgeyz_staggered_auxnames ;

std::vector<std::string> _corner_staggered_varnames ; 
std::vector<std::string> _corner_staggered_auxnames ;
/****************************************************/
/****************************************************/

/****************************************************/
/*             Boundary condition arrays            */
/****************************************************/
std::vector<bc_t> _var_bc_types ;

std::vector<bc_t> _facex_vars_bc_types ;

std::vector<bc_t> _facey_vars_bc_types ;

std::vector<bc_t> _facez_vars_bc_types ;

std::vector<bc_t> _edgexy_vars_bc_types ;

std::vector<bc_t> _edgexz_vars_bc_types ;

std::vector<bc_t> _edgeyz_vars_bc_types ;

std::vector<bc_t> _corner_vars_bc_types ;
/****************************************************/
/*      Prolong/Restrict operator arrays            */
/****************************************************/
std::vector<grace::var_amr_interp_t> _var_interp_types ;

std::vector<grace::var_amr_interp_t> _facex_vars_interp_types ;

std::vector<grace::var_amr_interp_t> _facey_vars_interp_types ;

std::vector<grace::var_amr_interp_t> _facez_vars_interp_types ;

std::vector<grace::var_amr_interp_t> _edgexy_vars_interp_types ;

std::vector<grace::var_amr_interp_t> _edgexz_vars_interp_types ;

std::vector<grace::var_amr_interp_t> _edgeyz_vars_interp_types ;

std::vector<grace::var_amr_interp_t> _corner_vars_interp_types ;
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
/****************************************************/
std::unordered_map<std::string, variable_properties_t<GRACE_NSPACEDIM>> 
    _varprops; 
std::unordered_map<std::string, variable_properties_t<GRACE_NSPACEDIM>> 
    _auxprops; 

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
    REGISTER_EVOLVED_SCALAR(DENS,"dens","outgoing","second_order",true) ; 
    REGISTER_EVOLVED_VECTOR(SX,SY,SZ,"stilde","outgoing","second_order",true) ;
    REGISTER_EVOLVED_SCALAR(TAU,"tau","outgoing","second_order",true) ; 
    REGISTER_EVOLVED_SCALAR(YESTAR,"ye_star","outgoing","second_order",true) ; 
    REGISTER_EVOLVED_SCALAR(ENTROPYSTAR,"s_star", "outgoing","second_order",true) ;
    #ifdef GRACE_ENABLE_M1
    REGISTER_EVOLVED_SCALAR(ERAD,"Erad","outgoing","second_order",true) ;
    REGISTER_EVOLVED_SCALAR(NRAD,"Nrad","outgoing","second_order",true) ;
    REGISTER_EVOLVED_VECTOR(FRADX,FRADY,FRADZ,"Frad","outgoing","second_order",true) ;
    #endif 
    REGISTER_EVOLVED_FACE_STAGGERED_VECTOR(BSX,BSY,BSZ,"B_face", "outgoing"/*FIXME?*/, "div_preserving", false);
    /* GRMHD primitives */
    REGISTER_AUX_SCALAR(RHO,"rho","none") ; 
    REGISTER_AUX_VECTOR(VELX,VELY,VELZ,"vel","none") ; 
    REGISTER_AUX_VECTOR(ZVECX,ZVECY,ZVECZ,"zvec","none") ;
    REGISTER_AUX_VECTOR(BX,BY,BZ,"Bvec","none") ; 
    REGISTER_AUX_SCALAR(YE,"ye","none") ; 
    REGISTER_AUX_SCALAR(TEMP,"temperature", "none") ;
    REGISTER_AUX_SCALAR(ENTROPY,"entropy","none") ; 
    REGISTER_AUX_SCALAR(EPS,"eps","none") ; 
    REGISTER_AUX_SCALAR(PRESS,"press","none") ;
    REGISTER_AUX_SCALAR(BDIV, "Bdiv","none") ; 
    REGISTER_AUX_SCALAR(SMALLB2, "b2","none") ;
    REGISTER_AUX_SCALAR(C2P_ERR, "c2p_err","none") ;
    #ifdef GRACE_ENABLE_M1
    REGISTER_AUX_SCALAR(KAPPAA,"kappa_a","none"); 
    REGISTER_AUX_SCALAR(KAPPAS,"kappa_s","none"); 
    REGISTER_AUX_SCALAR(ETA,"eta","none"); 
    REGISTER_AUX_SCALAR(ETAN,"eta_n","none"); 
    REGISTER_AUX_SCALAR(KAPPAAN,"kappa_n","none"); 
    #endif
    /* registration of metric variables */
    #ifdef GRACE_ENABLE_COWLING_METRIC
    REGISTER_EVOLVED_TENSOR(GXX,GXY,GXZ,GYY,GYZ,GZZ,"gamma","outgoing","second_order",false) ; 
    REGISTER_EVOLVED_SCALAR(ALP,"alp","outgoing","second_order",false) ; 
    REGISTER_EVOLVED_VECTOR(BETAX,BETAY,BETAZ,"beta","outgoing","second_order",false);
    REGISTER_EVOLVED_TENSOR(KXX,KXY,KXZ,KYY,KYZ,KZZ,"ext_curv","outgoing","second_order",false) ; 
    #elif defined(GRACE_ENABLE_Z4C_METRIC)
    REGISTER_EVOLVED_TENSOR(GTXX,GTXY,GTXZ,GTYY,GTYZ,GTZZ,"gamma_tilde","sommerfeld","fourth_order",false) ; 
    REGISTER_EVOLVED_SCALAR(CHI,"conf_fact","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_SCALAR(THETA,"z4c_theta","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_VECTOR(GAMMATX,GAMMATY,GAMMATZ,"z4c_Gamma","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_TENSOR(ATXX,ATXY,ATXZ,ATYY,ATYZ,ATZZ,"A_tilde","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_SCALAR(KHAT,"z4c_Khat","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_SCALAR(ALP,"alp","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_VECTOR(BETAX,BETAY,BETAZ,"beta","sommerfeld","fourth_order",false);
    REGISTER_AUX_SCALAR(HAM,"z4c_H","none");
    REGISTER_AUX_VECTOR(MOMX,MOMY,MOMZ,"z4c_M","none");
    #elif defined(GRACE_ENABLE_BSSN_METRIC)
    REGISTER_EVOLVED_TENSOR(GTXX,GTXY,GTXZ,GTYY,GTYZ,GTZZ,"gamma_tilde","sommerfeld","fourth_order",false) ; 
    REGISTER_EVOLVED_SCALAR(CHI,"conf_fact","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_VECTOR(GAMMATX,GAMMATY,GAMMATZ,"z4c_Gamma","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_TENSOR(ATXX,ATXY,ATXZ,ATYY,ATYZ,ATZZ,"A_tilde","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_SCALAR(KTR,"bssn_K","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_SCALAR(ALP,"alp","sommerfeld","fourth_order",false);
    REGISTER_EVOLVED_VECTOR(BETAX,BETAY,BETAZ,"beta","sommerfeld","fourth_order",false);
    REGISTER_AUX_SCALAR(HAM,"bssn_H","none");
    REGISTER_AUX_VECTOR(MOMX,MOMY,MOMZ,"bssn_M","none");
    #endif
    /********************************************************************************/
    /********************************************************************************/
    /*                           COPY INDICES TO GPU                                */
    /********************************************************************************/
    /********************************************************************************/
    /*                           DON'T TOUCH THIS CODE!!                            */
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
    GRACE_INFO("{} total variables registered, of which {} evolved, {} auxiliary. {} variables require fluxes.", 
                detail::num_vars, detail::num_evolved, detail::num_auxiliary, detail::num_fluxes ) ; 
    std::ostringstream ss ; 
    ss << "Evolved variables:\n" ; 
    for( int ii=0; ii<detail::num_evolved; ++ii) {
        auto const vname = detail::_varnames[ii] ; 
        ss << vname << " needs fluxes: " << (ii<detail::num_fluxes ? "yes" : "no"); 
        if ( ii < detail::num_evolved - 1 ){ 
            ss << ",\n" ;  
        } else {
            ss << "\n" ; 
        }
    }
    GRACE_VERBOSE(ss.str()) ; 
    ss.str("");          // Clear the buffer
    ss.clear();          // Clear the error flags
    ss << "Auxiliary variables:\n" ; 
    for( int ii=0; ii<detail::num_auxiliary; ++ii) {
        auto const vname = detail::_auxnames[ii] ; 
        ss << vname ; 
        if ( ii < detail::num_evolved - 1 ){ 
            ss << ",\n" ;  
        } else {
            ss << "\n" ; 
        }
    }
    GRACE_VERBOSE(ss.str()) ; 
}
namespace detail {

static bc_t get_bc_type(std::string const& bc_string)
{
    if ( bc_string == "outgoing" ) {
        return bc_t::BC_OUTFLOW ;
    } else if ( bc_string == "third_order_lagrange") {
        return bc_t::BC_LAGRANGE_EXTRAP ; 
    } else if ( bc_string == "sommerfeld") { 
        return bc_t::BC_SOMMERFELD ; 
    } else if ( bc_string == "none" ) {
        return bc_t::BC_NONE ; 
    } else{
        ERROR("Invalid bc_type string " << bc_string) ; 
    }
}

static var_amr_interp_t get_interp_type(std::string const& interp_string)
{
    if ( interp_string == "second_order" ) {
        return var_amr_interp_t::INTERP_SECOND_ORDER ;
    } else if ( interp_string == "fourth_order") {
        return var_amr_interp_t::INTERP_FOURTH_ORDER ; 
    } else if ( interp_string == "div_preserving") { 
        return var_amr_interp_t::INTERP_DIV_PRESERVING ; 
    } else if ( interp_string == "none" ) {
        return var_amr_interp_t::INTERP_NONE ; 
    } else{
        ERROR("Invalid prolongation/restriction string " << interp_string) ; 
    }
}

static var_staggering_t get_staggering(std::array<bool,3> s) {
    return static_cast<var_staggering_t>(static_cast<uint8_t>(s[0]) + (static_cast<uint8_t>(s[1])<<1) + (static_cast<uint8_t>(s[2])<<2)) ; 
}

static int register_scalar( std::string const& name
                          , bool is_evolved 
                          , bool need_fluxes
                          , bc_t const & bc_type 
                          , var_amr_interp_t const & interp_type )
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
        _var_interp_types.push_back(interp_type) ;  
    } else {
        num_auxiliary ++ ; 
        _auxnames.push_back(name) ; 
    }

    return  is_evolved ? num_evolved-1 : num_auxiliary-1 ; 
}

static int register_staggered_variable( std::string const& name
                                      , bool is_evolved 
                                      , bool need_fluxes
                                      , bc_t const & bc_type 
                                      , var_amr_interp_t const & interp_type
                                      , grace::var_staggering_t const& staggering 
                                      , bool is_vector 
                                      , bool is_tensor
                                      , int num_comp )
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
    num_vars++;
    if( staggering == var_staggering_t::STAG_FACEX ) {
        ASSERT(is_vector, "Face staggered variables must be vectors.") ;
        ASSERT(num_comp == 0, "Staggering and vector indices don't match") ; 
        if ( is_evolved ) {
            num_face_staggered_vars ++ ; 
            _facex_staggered_varnames.push_back(name) ; 
            _facex_vars_bc_types.push_back(bc_type)  ;
            _facex_vars_interp_types.push_back(interp_type);
            return num_face_staggered_vars - 1 ; 
        } else {
            num_face_staggered_aux ++ ;
            _facex_staggered_auxnames.push_back(name) ; 
            return num_face_staggered_aux - 1 ;
        }
        
    } else if (staggering == var_staggering_t::STAG_FACEY) {
        ASSERT(is_vector, "Face staggered variables must be vectors.") ;
        ASSERT(num_comp == 1, "Staggering and vector indices don't match") ;
        if ( is_evolved ) {
            _facey_staggered_varnames.push_back(name) ; 
            _facey_vars_bc_types.push_back(bc_type)  ;
            _facey_vars_interp_types.push_back(interp_type);
            return num_face_staggered_vars - 1 ; 
        } else {
            _facey_staggered_auxnames.push_back(name) ; 
            return num_face_staggered_aux - 1 ;
        }
    } else if  (staggering == var_staggering_t::STAG_FACEZ) {
        ASSERT(is_vector, "Face staggered variables must be vectors.") ;
        ASSERT(num_comp == 2, "Staggering and vector indices don't match") ;
        if ( is_evolved ) {
            _facez_staggered_varnames.push_back(name) ; 
            _facez_vars_bc_types.push_back(bc_type)  ;
            _facez_vars_interp_types.push_back(interp_type);
            return num_face_staggered_vars - 1 ; 
        } else {
            _facez_staggered_auxnames.push_back(name) ; 
            return num_face_staggered_aux - 1 ;
        }
    } else if (staggering == var_staggering_t::STAG_EDGEYZ) {
        ASSERT(is_vector, "Edge staggered variables must be vectors.") ;
        ASSERT(num_comp == 0, "Staggering and vector indices don't match") ; 
        if( is_evolved ) {
            num_edge_staggered_vars ++ ; 
            _edgeyz_staggered_varnames.push_back(name) ; 
            _edgeyz_vars_bc_types.push_back(bc_type) ;
            _edgeyz_vars_interp_types.push_back(interp_type);
            return (num_edge_staggered_vars) - 1 ; 
        } else {
            num_edge_staggered_aux ++ ; 
            _edgeyz_staggered_auxnames.push_back(name) ; 
            return (num_edge_staggered_aux) - 1 ; 
        }
    } else if (staggering == var_staggering_t::STAG_EDGEXZ) {
        ASSERT(is_vector, "Edge staggered variables must be vectors.") ;
        ASSERT(num_comp == 1, "Staggering and vector indices don't match") ; 
        if( is_evolved ) {
            _edgexz_staggered_varnames.push_back(name) ; 
            _edgexz_vars_bc_types.push_back(bc_type) ; 
            _edgexz_vars_interp_types.push_back(interp_type);
            return (num_edge_staggered_vars) - 1 ; 
        } else {
            _edgexz_staggered_auxnames.push_back(name) ; 
            return (num_edge_staggered_aux) - 1 ; 
        }
    } else if (staggering == var_staggering_t::STAG_EDGEXY) {
        ASSERT(is_vector, "Edge staggered variables must be vectors.") ;
        ASSERT(num_comp == 2, "Staggering and vector indices don't match") ; 
        if( is_evolved ) {
            _edgexy_staggered_varnames.push_back(name) ; 
            _edgexy_vars_bc_types.push_back(bc_type) ; 
            _edgexy_vars_interp_types.push_back(interp_type);
            return (num_edge_staggered_vars) - 1 ; 
        } else {
            _edgexy_staggered_auxnames.push_back(name) ; 
            return (num_edge_staggered_aux) - 1 ; 
        }
    } else {
        if( is_evolved ) {
            _corner_staggered_varnames.push_back(name) ; 
            _corner_vars_bc_types.push_back(bc_type) ; 
            _corner_vars_interp_types.push_back(interp_type);
            return (++num_corner_staggered_vars) - 1 ; 
        } else {
            _corner_staggered_auxnames.push_back(name) ; 
            return (++num_corner_staggered_aux) - 1 ; 
        }
    }
}

static int register_vector( std::string const& name
                          , bool is_evolved 
                          , bool need_fluxes
                          , int num_comp
                          , bc_t const & bc_type 
                          , var_amr_interp_t const& interp_type )
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
        _var_interp_types.push_back(interp_type) ; 
    } else {
        _auxnames.push_back(name) ; 
        num_auxiliary ++ ; 
    }

    return  is_evolved ? num_evolved-1 : num_auxiliary-1 ; 
}

static int register_tensor( std::string const& name
                          , bool is_evolved 
                          , bool need_fluxes
                          , int num_comp
                          , bc_t const & bc_type
                          , var_amr_interp_t const& interp_type )
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
        _var_interp_types.push_back(interp_type) ; 
    } else {
        _auxnames.push_back(name) ; 
        num_auxiliary ++ ; 
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
                                , bool is_evolved 
                                , bool need_fluxes
                                , std::string const & bc_type 
                                , std::string const & var_interp_type
                                , bool is_vector
                                , bool is_tensor
                                , int comp_num
                                , std::string const& vec_name ) 
{
    using namespace detail ; 

    if ( need_fluxes ) {
        ASSERT(is_evolved, "Not evolved variable can't need fluxes.") ; 
    }

    int var_staggering = 0 ; 
    for( auto const & s: staggering ) var_staggering += static_cast<int>(s) ; 

    auto bc = get_bc_type(bc_type) ; 
    auto int_kind = get_interp_type(var_interp_type) ; 

    variable_properties_t<GRACE_NSPACEDIM> props ;
    props.staggering = get_staggering(staggering) ; 
    props.is_evolved = is_evolved ; 
    props.is_vector  = is_vector  ; 
    props.is_tensor  = is_tensor  ; 
    props.name       = (is_tensor || is_vector) ?  vec_name : name  ;
    props.comp_num   = (is_tensor || is_vector) ?  comp_num : -1    ;
    props.bc_type    = bc; 
    props.interp_op_kind = int_kind;

    num_vector_vars += static_cast<int>(is_vector) ; 
    num_tensor_vars += static_cast<int>(is_tensor) ;
    size_t varidx ; 
    if( ( var_staggering != 0 ) ) {
        varidx = register_staggered_variable(name,is_evolved,need_fluxes,bc,int_kind,get_staggering(staggering), is_vector, is_tensor, comp_num) ; 
    } else {
        if ( is_vector ) {
            varidx = register_vector(name,is_evolved,need_fluxes,comp_num,bc,int_kind) ; 
        } else if ( is_tensor ) {
            varidx = register_tensor(name,is_evolved,need_fluxes,comp_num,bc,int_kind) ; 
        } else {
            varidx = register_scalar(name,is_evolved,need_fluxes,bc,int_kind) ; 
        }   
    }
    props.index = varidx ;
    if ( is_evolved ) {
        detail::_varprops[name] = props ;
    } else {
        detail::_auxprops[name] = props ;
    } 
    return varidx ;
}
} } /* namespace grace::variables */