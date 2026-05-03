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
#include <grace/errors/error.hh>


#include <Kokkos_Core.hpp>

#include <string> 
#include <sstream>

namespace grace { namespace variables { 

namespace detail {

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

std::vector<std::string> _facey_staggered_varnames ;

std::vector<std::string> _facez_staggered_varnames ;

std::vector<std::string> _edgexy_staggered_varnames ; 

std::vector<std::string> _edgexz_staggered_varnames ; 

std::vector<std::string> _edgeyz_staggered_varnames ; 

std::vector<std::string> _corner_staggered_varnames ; 
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
/* NB: here we assume that all face/edge staggered  */
/*     variables are vector components.             */
/****************************************************/

/****************************************************/
/****************************************************/
std::unordered_map<std::string, variable_properties_t<GRACE_NSPACEDIM>> 
    _varprops; 
std::unordered_map<std::string, variable_properties_t<GRACE_NSPACEDIM>> 
    _auxprops; 

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

} /* namespace grace::variables::detail */

static void register_evolved_scalar(
    int const idx, 
    std::string const& name, 
    bc_t bc, 
    std::string const& interp) 
{
    auto itp = detail::get_interp_type(interp) ; 
    detail::_varnames[idx]         = name   ; 
    detail::_var_bc_types[idx]     = bc     ; 
    detail::_var_interp_types[idx] = itp    ;
    variable_properties_t<GRACE_NSPACEDIM> props ; 
    props.name = name ;
    props.bc_type = bc ; 
    props.interp_op_kind = itp ; 
    props.is_evolved = true ; 
    props.is_vector = props.is_tensor = false ; 
    props.comp_num = -1 ; 
    props.index = idx ;
    props.staggering = var_staggering_t::STAG_CENTER;

    detail::_varprops[name] = props ; 
} ; 

static void register_evolved_vector(
    std::array<int,3> const& idxs,
    std::string const& name, 
    bc_t bc, 
    std::string const& interp) 
{
    auto itp = detail::get_interp_type(interp) ; 
    for ( int ic=0; ic<3; ++ic) {
        auto idx = idxs[ic] ; 
        std::string const cname = name  + "[" + std::to_string(ic) + "]"; 
        detail::_varnames[idx]         = cname  ;
        detail::_var_bc_types[idx]     = bc     ; 
        detail::_var_interp_types[idx] = itp    ;

        variable_properties_t<GRACE_NSPACEDIM> props ; 
        props.name = name ;
        props.bc_type = bc ; 
        props.interp_op_kind = itp ; 
        props.is_evolved = true ; 
        props.is_vector = true ; 
        props.is_tensor = false ; 
        props.comp_num = ic ; 
        props.index = idx ;
        props.staggering = var_staggering_t::STAG_CENTER;

        detail::_varprops[cname] = props ; 
    }        
} ;

static void register_evolved_tensor(
    std::array<int,6> const& idxs,
    std::string const& name, 
    bc_t bc, 
    std::string const& interp) 
{
    auto itp = detail::get_interp_type(interp) ; 
    int cmp=0;
    for ( int ic=0; ic<3; ++ic) {
        for( int jc=ic; jc<3; ++jc) {
            auto idx = idxs[cmp] ; 
            std::string const cname = name  + "[" + std::to_string(ic) + "," + std::to_string(jc) + "]"; 
            detail::_varnames[idx]         = cname  ;
            detail::_var_bc_types[idx]     = bc     ; 
            detail::_var_interp_types[idx] = itp    ;
            variable_properties_t<GRACE_NSPACEDIM> props ; 
            props.name = name ;
            props.bc_type = bc ; 
            props.interp_op_kind = itp ; 
            props.is_evolved = true ; 
            props.is_vector = false ;
            props.is_tensor = true ; 
            props.comp_num = cmp ; 
            props.index = idx ;
            props.staggering = var_staggering_t::STAG_CENTER;

            detail::_varprops[cname] = props ; 

            cmp++ ; 
        }   
    }        
} ;


static void register_aux_scalar(
    int const idx, 
    std::string const& name) 
{
    detail::_auxnames[idx]         = name   ; 
    variable_properties_t<GRACE_NSPACEDIM> props ; 
    props.name = name ;
    props.is_evolved = false ; 
    props.is_vector = props.is_tensor = false ; 
    props.comp_num = -1 ; 
    props.index = idx ;
    props.staggering = var_staggering_t::STAG_CENTER;

    detail::_auxprops[name] = props ; 
} ; 

static void register_aux_vector(
    std::array<int,3> const& idxs,
    std::string const& name) 
{
    for ( int ic=0; ic<3; ++ic) {
        auto idx = idxs[ic] ; 
        std::string const cname = name  + "[" + std::to_string(ic) + "]"; 
        detail::_auxnames[idx]         = cname  ;

        variable_properties_t<GRACE_NSPACEDIM> props ; 
        props.name = name ;
        props.is_evolved = false ; 
        props.is_vector = true ; 
        props.is_tensor = false ; 
        props.comp_num = ic ; 
        props.index = idx ;
        props.staggering = var_staggering_t::STAG_CENTER;

        detail::_auxprops[cname] = props ; 
    }        
} ;

static void register_evolved_vector_fc(
    std::array<int,3> const& idxs,
    std::string const& name, 
    bc_t bc, 
    std::string const& interp) 
{
    auto itp = detail::get_interp_type(interp) ; 
    {
        auto idx = idxs[0] ; 
        std::string const cname = name  + "[0]"; 
        detail::_facex_staggered_varnames[idx] = cname  ;
        detail::_facex_vars_bc_types[idx]      = bc     ; 
        detail::_facex_vars_interp_types[idx]  = itp    ;

        variable_properties_t<GRACE_NSPACEDIM> props ; 
        props.name = name ;
        props.bc_type = bc ; 
        props.interp_op_kind = itp ; 
        props.is_evolved = true ; 
        props.is_vector = true ; 
        props.is_tensor = false ; 
        props.comp_num = 0 ; 
        props.index = idx ;
        props.staggering = var_staggering_t::STAG_FACEX;

        detail::_varprops[cname] = props ; 
    }
    {
        auto idx = idxs[1] ; 
        std::string const cname = name  + "[1]"; 
        detail::_facey_staggered_varnames[idx] = cname  ;
        detail::_facey_vars_bc_types[idx]      = bc     ; 
        detail::_facey_vars_interp_types[idx]  = itp    ;

        variable_properties_t<GRACE_NSPACEDIM> props ; 
        props.name = name ;
        props.bc_type = bc ; 
        props.interp_op_kind = itp ; 
        props.is_evolved = true ; 
        props.is_vector = true ; 
        props.is_tensor = false ; 
        props.comp_num = 1 ; 
        props.index = idx ;
        props.staggering = var_staggering_t::STAG_FACEY;

        detail::_varprops[cname] = props ; 
    }
    {
        auto idx = idxs[2] ; 
        std::string const cname = name  + "[2]"; 
        detail::_facez_staggered_varnames[idx] = cname  ;
        detail::_facez_vars_bc_types[idx]      = bc     ; 
        detail::_facez_vars_interp_types[idx]  = itp    ;

        variable_properties_t<GRACE_NSPACEDIM> props ; 
        props.name = name ;
        props.bc_type = bc ; 
        props.interp_op_kind = itp ; 
        props.is_evolved = true ; 
        props.is_vector = true ; 
        props.is_tensor = false ; 
        props.comp_num = 2 ; 
        props.index = idx ;
        props.staggering = var_staggering_t::STAG_FACEZ;

        detail::_varprops[cname] = props ; 
    }
       
} ;



void register_variables() {
    // purpose of this function is to 
    // register variable properties 
    detail::_varnames.resize(N_EVOL_VARS) ; 
    detail::_auxnames.resize(N_AUX_VARS) ; 
    detail::_facex_staggered_varnames.resize(N_FC_X) ; 
    detail::_facey_staggered_varnames.resize(N_FC_Y) ; 
    detail::_facez_staggered_varnames.resize(N_FC_Z) ; 
    detail::_edgeyz_staggered_varnames.resize(N_EC_YZ) ; 
    detail::_edgexz_staggered_varnames.resize(N_EC_XZ) ; 
    detail::_edgexy_staggered_varnames.resize(N_EC_XY) ;
    detail::_corner_staggered_varnames.resize(N_VC) ; 

    detail::_var_bc_types.resize(N_EVOL_VARS) ; 
    detail::_facex_vars_bc_types.resize(N_FC_X) ; 
    detail::_facey_vars_bc_types.resize(N_FC_Y) ; 
    detail::_facez_vars_bc_types.resize(N_FC_Z) ; 
    detail::_edgeyz_vars_bc_types.resize(N_EC_YZ) ; 
    detail::_edgexz_vars_bc_types.resize(N_EC_XZ) ; 
    detail::_edgexy_vars_bc_types.resize(N_EC_XY) ; 

    detail::_var_interp_types.resize(N_EVOL_VARS) ; 
    detail::_facex_vars_interp_types.resize(N_FC_X) ; 
    detail::_facey_vars_interp_types.resize(N_FC_Y) ; 
    detail::_facez_vars_interp_types.resize(N_FC_Z) ; 
    detail::_edgeyz_vars_interp_types.resize(N_EC_YZ) ; 
    detail::_edgexz_vars_interp_types.resize(N_EC_XZ) ; 
    detail::_edgexy_vars_interp_types.resize(N_EC_XY) ;
    detail::_corner_vars_interp_types.resize(N_VC) ; 

    // hydro 
    
    auto hydro_bc = detail::get_bc_type(get_param<std::string>("grmhd","bc_kind")) ; 
    // evolved 
    register_evolved_scalar(DENS_,"dens",hydro_bc,"second_order") ;
    register_evolved_vector({SX_,SY_,SZ_},"stilde",hydro_bc,"second_order")  ;
    register_evolved_scalar(TAU_,"tau",hydro_bc,"second_order") ;
    register_evolved_scalar(YESTAR_,"ye_star",hydro_bc,"second_order") ;
    register_evolved_scalar(ENTROPYSTAR_,"s_star",hydro_bc,"second_order") ;
    // stag 
    register_evolved_vector_fc({BSX_,BSY_,BSZ_}, "B_face", hydro_bc, "div_preserving") ;  
    // aux 
    register_aux_scalar(RHO_, "rho") ;
    register_aux_vector({ZVECX_,ZVECY_,ZVECZ_}, "zvec") ; 
    register_aux_vector({BX_,BY_,BZ_}, "Bvec") ;
    register_aux_scalar(YE_, "ye") ; 
    register_aux_scalar(TEMP_,"temperature") ; 
    register_aux_scalar(ENTROPY_,"entropy") ; 
    register_aux_scalar(EPS_,"eps") ; 
    register_aux_scalar(PRESS_,"press") ;
    register_aux_scalar(BDIV_, "Bdiv") ; 
    register_aux_scalar(C2P_DENS_ERR_,"c2p_dens_corr") ; 
    register_aux_scalar(C2P_ERR_,"c2p_err") ; 

    #ifdef GRACE_ENABLE_M1
    // m1 
    auto m1_bc = detail::get_bc_type(get_param<std::string>("m1","bc_kind")) ; 
    // evolved 
    register_evolved_scalar(ERAD1_,"Erad1",m1_bc,"second_order") ; 
    register_evolved_scalar(NRAD1_,"Nrad1",m1_bc,"second_order") ; 
    register_evolved_vector({FRADX1_,FRADY1_,FRADZ1_},"Frad1",m1_bc,"second_order") ; 
    #ifdef M1_NU_THREESPECIES
    register_evolved_scalar(ERAD2_,"Erad2",m1_bc,"second_order") ; 
    register_evolved_scalar(NRAD2_,"Nrad2",m1_bc,"second_order") ; 
    register_evolved_vector({FRADX2_,FRADY2_,FRADZ2_},"Frad2",m1_bc,"second_order") ; 
    register_evolved_scalar(ERAD3_,"Erad3",m1_bc,"second_order") ; 
    register_evolved_scalar(NRAD3_,"Nrad3",m1_bc,"second_order") ; 
    register_evolved_vector({FRADX3_,FRADY3_,FRADZ3_},"Frad3",m1_bc,"second_order") ; 
    #endif 
    #ifdef M1_NU_FIVESPECIES
    register_evolved_scalar(ERAD4_,"Erad4",m1_bc,"second_order") ; 
    register_evolved_scalar(NRAD4_,"Nrad4",m1_bc,"second_order") ; 
    register_evolved_vector({FRADX4_,FRADY4_,FRADZ4_},"Frad4",m1_bc,"second_order") ; 
    register_evolved_scalar(ERAD5_,"Erad5",m1_bc,"second_order") ; 
    register_evolved_scalar(NRAD5_,"Nrad5",m1_bc,"second_order") ; 
    register_evolved_vector({FRADX5_,FRADY5_,FRADZ5_},"Frad5",m1_bc,"second_order") ; 
    #endif 
    // aux 
    register_aux_scalar(KAPPAA1_,"kappa_a1") ; 
    register_aux_scalar(KAPPAS1_,"kappa_s1") ; 
    register_aux_scalar(ETA1_,"eta1") ; 
    register_aux_scalar(KAPPAAN1_,"kappa_n1") ; 
    register_aux_scalar(ETAN1_,"eta_n1") ; 
    #ifdef M1_NU_THREESPECIES
    register_aux_scalar(KAPPAA2_,"kappa_a2") ; 
    register_aux_scalar(KAPPAS2_,"kappa_s2") ; 
    register_aux_scalar(ETA2_,"eta2") ; 
    register_aux_scalar(KAPPAAN2_,"kappa_n2") ; 
    register_aux_scalar(ETAN2_,"eta_n2") ;
    register_aux_scalar(KAPPAA3_,"kappa_a3") ; 
    register_aux_scalar(KAPPAS3_,"kappa_s3") ; 
    register_aux_scalar(ETA3_,"eta3") ; 
    register_aux_scalar(KAPPAAN3_,"kappa_n3") ; 
    register_aux_scalar(ETAN3_,"eta_n3") ;
    #endif
    #ifdef M1_NU_FIVESPECIES
    register_aux_scalar(KAPPAA4_,"kappa_a4") ; 
    register_aux_scalar(KAPPAS4_,"kappa_s4") ; 
    register_aux_scalar(ETA4_,"eta4") ; 
    register_aux_scalar(KAPPAAN4_,"kappa_n4") ; 
    register_aux_scalar(ETAN4_,"eta_n4") ;
    register_aux_scalar(KAPPAA5_,"kappa_a5") ; 
    register_aux_scalar(KAPPAS5_,"kappa_s5") ; 
    register_aux_scalar(ETA5_,"eta5") ; 
    register_aux_scalar(KAPPAAN5_,"kappa_n5") ; 
    register_aux_scalar(ETAN5_,"eta_n5") ;
    register_aux_scalar(YMU_, "ymu") ; 

    #endif
    #endif  

    #ifdef GRACE_ENABLE_COWLING_METRIC
    auto metric_bc = detail::get_bc_type("none") ; 
    register_evolved_tensor({GXX_,GXY_,GXZ_,GYY_,GYZ_,GZZ_}, "gamma", metric_bc, "fourth_order") ; 
    register_evolved_scalar(ALP_,"alp",metric_bc,"fourth_order") ; 
    register_evolved_vector({BETAX_,BETAY_,BETAZ_}, "beta", metric_bc, "fourth_order") ; 
    register_evolved_tensor({KXX_,KXY_,KXZ_,KYY_,KYZ_,KZZ_}, "ext_curv", metric_bc, "fourth_order") ; 
    #elif defined(GRACE_ENABLE_Z4C_METRIC)
    auto metric_bc = detail::get_bc_type(get_param<std::string>("z4c","bc_kind")) ; 
    register_evolved_tensor({GTXX_,GTXY_,GTXZ_,GTYY_,GTYZ_,GTZZ_}, "gamma_tilde", metric_bc, "fourth_order") ; 
    register_evolved_scalar(CHI_,"conf_fact",metric_bc,"fourth_order") ; 
    register_evolved_scalar(THETA_,"z4c_theta", metric_bc, "fourth_order") ; 
    register_evolved_vector({GAMMATX_,GAMMATY_,GAMMATZ_}, "z4c_Gamma", metric_bc, "fourth_order") ; 
    register_evolved_tensor({ATXX_,ATXY_,ATXZ_,ATYY_,ATYZ_,ATZZ_}, "A_tilde", metric_bc, "fourth_order") ; 
    register_evolved_scalar(KHAT_,"z4c_Khat",metric_bc, "fourth_order") ; 
    register_evolved_scalar(ALP_,"alp",metric_bc,"fourth_order") ; 
    register_evolved_vector({BETAX_,BETAY_,BETAZ_}, "beta", metric_bc, "fourth_order") ; 
    register_evolved_vector({BDRIVERX_,BDRIVERY_,BDRIVERZ_}, "z4c_Bdriver", metric_bc, "fourth_order") ; 

    // aux 
    register_aux_scalar(PSI4RE_,"Psi4Re") ; 
    register_aux_scalar(PSI4IM_,"Psi4Im") ; 
    register_aux_scalar(HAM_,"z4c_H") ; 
    register_aux_vector({MOMX_,MOMY_,MOMZ_},"z4c_M") ; 
    #endif 
}

}  } /* namespace grace::variables */
