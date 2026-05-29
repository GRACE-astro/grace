/**
 * @file grmhd_helpers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Shared GRMHD helpers: atmosphere/c2p/excision parameter structs, pair-symmetric metric face interpolators, and per-cell math primitives.
 * @date 2024-06-17
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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

#ifndef GRACE_PHYSICS_GRMHD_HELPERS_HH
#define GRACE_PHYSICS_GRMHD_HELPERS_HH

#include <grace_config.h> 
#include <grace/utils/metric_utils.hh>
#include <grace/config/config_parser.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <array>

/**
 * @brief Atmosphere treatment parameters
 * 
 */
struct atmo_params_t {
    double ye_fl ;    //!< Atmo ye
    double rho_fl ;   //!< Atmo rho
    double temp_fl ;  //!< Atmo T 
    double rho_fl_scaling  ; //!< Radial scaling of atmo rho
    double temp_fl_scaling ; //!< Radial scaling of atmo T
    double atmo_tol        ; //!< Tolerance in setting points to atmosphere
} ;
/**
 * @brief Parameters controlling C2P behaviour 
 */
struct c2p_params_t {
  double tol           ; //!< C2P tolerance 
  double max_w         ; //!< Maximum Lorentz factor
  double max_sigma     ; //!< Maximum magnetization b^2/rho
  double beta_fallback ; //!< beta < fallback we use ent
  bool use_ent_backup  ; //!< Use backup c2p?
  double alp_bh_thresh ; //!< alp theshold for BH horizon
} ; 
/**
 * @brief FOFC parameters
 *
 * Discrete-maximum-principle (DMP) settings for flag_fofc_cells. Mirrors
 * AthenaK's dyn_grmhd <mhd>/enforce_maximum + dmp_M (Felker & Stone 2018).
 */
struct fofc_params_t {
  bool   dmp_enable ;  //!< Apply DMP check on tentative D / tau
  double dmp_M      ;  //!< Slack multiplier for the DMP threshold
} ;
/**
 * @brief Riemann-solver parameters
 *
 * Only the Rusanov / LLF flux gets a wavespeed floor: in atmosphere / near-
 * vacuum the local fast-magnetosonic speed collapses to ~c_s ~ 1e-3 and the
 * Rusanov dissipation budget along with it, letting sub-grid noise grow into
 * blastwaves at high resolution. Flooring cmin = cmax >= rusanov_wavespeed_floor
 * restores LLF-like artificial viscosity exactly where the physical wavespeed
 * is too small to provide it, without touching HLLE (where cmin, cmax carry
 * upwind direction info).
 */
struct riemann_params_t {
  double rusanov_wavespeed_floor ;        //!< Floor value on cmin=cmax in the Rusanov flux
  double rusanov_floor_rho_threshold ;    //!< Only apply the floor when min(rhoL,rhoR) < this
} ;
/**
 * @brief Excision parameters
 *
 */
struct excision_params_t {
    double rho_ex ;         //!< Excision rho
    double temp_ex ;        //!< Excision temp 
    double ye_ex   ;        //!< Excision ye
    double r_ex ;           //!< Excision radius
    double alp_ex ;         //!< Excision alpha
    bool excise_by_radius ; //!< Whether excision is radius based (CKS) or alpha based.
} ; 

//**************************************************************************************************/
/* Auxiliaries */
//**************************************************************************************************/
/**
 * @brief Helper indices for prim arrays
 * \ingroup physics
 */
enum GRMHD_PRIMS_LOC_INDICES {
    RHOL = 0,
    PRESSL,
    ZXL,
    ZYL,
    ZZL,
    YEL,
    TEMPL,
    EPSL,
    ENTL,
    BXL,
    BYL,
    BZL,
    CS2L, /*only filled in flux comp*/
    NUM_PRIMS_LOC
} ; 
enum GRMHD_FLUX_LOC_INDICES : int {
  DENSF=0,
  STXF,
  STYF,
  STZF,
  TAUF,
  YESTARF,
  ENTROPYSTARF,
  BXF,
  BYF,
  BZF
} ; 
/**
 * @brief Helper indices for cons array.
 * \ingroup physics
 */
enum GRMHD_CONS_LOC_INDICES {
    DENSL=0,
    STXL,
    STYL,
    STZL,
    TAUL,
    YESL,
    ENTSL,
    BSXL,
    BSYL,
    BSZL,
    NUM_CONS_LOC
} ; 
namespace grace {
/**
 * @brief Array of GRMHD primitives.
 * \ingroup physics
 */
using grmhd_prims_array_t = std::array<double,NUM_PRIMS_LOC> ; 
/**
 * @brief Array of GRMHD conservatives.
 * \ingroup physics
 */
using grmhd_cons_array_t  = std::array<double,NUM_CONS_LOC>  ;
} /* namespace grace */

/** @brief Get atmosphere settings
 */
GRACE_ALWAYS_INLINE
atmo_params_t get_atmo_params()
{
  atmo_params_t atmo_params ; 
    
  atmo_params.rho_fl = grace::get_param<double>("grmhd","atmosphere","rho_fl") ; 

  atmo_params.rho_fl_scaling = grace::get_param<double>("grmhd","atmosphere","rho_scaling") ; 
  atmo_params.temp_fl_scaling = grace::get_param<double>("grmhd","atmosphere","temp_scaling") ;

  auto eos_type = grace::get_param<std::string>("eos", "eos_type") ; 
  if( eos_type == "hybrid" ) {
    auto const cold_eos_type = 
        grace::get_param<std::string>("eos","hybrid_eos","cold_eos_type") ;  
    if( cold_eos_type == "piecewise_polytrope" ) {
      atmo_params.ye_fl =
        grace::eos::get().get_eos<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>().ye_atmosphere() ;
      atmo_params.temp_fl =
        grace::eos::get().get_eos<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>().temp_atmosphere() ;
    } else if ( cold_eos_type == "tabulated" ) {
      atmo_params.ye_fl =
        grace::eos::get().get_eos<grace::hybrid_eos_t<grace::tabulated_cold_eos_t>>().ye_atmosphere() ;
      atmo_params.temp_fl =
        grace::eos::get().get_eos<grace::hybrid_eos_t<grace::tabulated_cold_eos_t>>().temp_atmosphere() ;
    } else {
      ERROR("Unsupported cold_eos_type: " << cold_eos_type) ;
    }
  } else if ( eos_type == "tabulated" ) {
    atmo_params.ye_fl =   
        grace::eos::get().get_eos<grace::tabulated_eos_t>().ye_atmosphere() ;
    atmo_params.temp_fl =   
        grace::eos::get().get_eos<grace::tabulated_eos_t>().temp_atmosphere() ;
  } else if ( eos_type == "ideal_gas" ) {
    atmo_params.ye_fl =   
        grace::eos::get().get_eos<grace::ideal_gas_eos_t>().ye_atmosphere() ;
    atmo_params.temp_fl =   
        grace::eos::get().get_eos<grace::ideal_gas_eos_t>().temp_atmosphere() ;
  }
  
  atmo_params.atmo_tol = grace::get_param<double>("grmhd","atmosphere","atmo_tol") ; 

  return atmo_params ; 
}

GRACE_ALWAYS_INLINE 
c2p_params_t get_c2p_params() 
{
  c2p_params_t c2p_params ; 

  c2p_params.tol = grace::get_param<double>("grmhd","c2p","tolerance") ; 
  c2p_params.max_w = grace::get_param<double>("grmhd","c2p","max_lorentz") ; 
  c2p_params.max_sigma = grace::get_param<double>("grmhd","c2p","max_sigma") ; 
  c2p_params.beta_fallback = grace::get_param<double>("grmhd","c2p","beta_fallback") ; 
  c2p_params.use_ent_backup = grace::get_param<bool>("grmhd","c2p","use_c2p_entropy_backup") ;
  c2p_params.alp_bh_thresh = grace::get_param<double>("grmhd","c2p","bh_alp_thresh") ;
  return c2p_params ; 
}

/** @brief Get FOFC DMP parameters from the parameter store.
 */
GRACE_ALWAYS_INLINE
fofc_params_t get_fofc_params()
{
  fofc_params_t fofc_params ;
  fofc_params.dmp_enable = grace::get_param<bool>  ("grmhd","fofc","dmp_enable") ;
  fofc_params.dmp_M      = grace::get_param<double>("grmhd","fofc","dmp_M") ;
  return fofc_params ;
}

/** @brief Get Riemann-solver parameters from the parameter store.
 */
GRACE_ALWAYS_INLINE
riemann_params_t get_riemann_params()
{
  riemann_params_t riemann_params ;
  riemann_params.rusanov_wavespeed_floor =
    grace::get_param<double>("grmhd","riemann","rusanov_wavespeed_floor") ;
  riemann_params.rusanov_floor_rho_threshold =
    grace::get_param<double>("grmhd","riemann","rusanov_floor_rho_threshold") ;
  return riemann_params ;
}

/** @brief Get excision settings
 */
GRACE_ALWAYS_INLINE
excision_params_t get_excision_params()
{
  excision_params_t excision_params ; 
    auto excision_kind = grace::get_param<std::string>("grmhd","excision","excision_criterion"); 
    //excision_pars["excision_criterion"].as<std::string>() ;
    if ( excision_kind == "radius" ) {
        excision_params.excise_by_radius = true ;
    } else if ( excision_kind == "lapse") {
        excision_params.excise_by_radius = false ;
    } else {
        ERROR("Unrecognized excision criterion") ; 
    }
    excision_params.r_ex = grace::get_param<double>("grmhd","excision","excision_radius") ;
    excision_params.alp_ex = grace::get_param<double>("grmhd","excision","excision_lapse") ;

    excision_params.rho_ex  =  grace::get_param<double>("grmhd","atmosphere","rho_fl") ;


    // get excision temperature and ye 
    auto eos_type = grace::get_param<std::string>("eos", "eos_type") ; 
    if( eos_type == "hybrid" ) {
      auto const cold_eos_type = 
          grace::get_param<std::string>("eos","hybrid_eos","cold_eos_type") ;  
      if( cold_eos_type == "piecewise_polytrope" ) {
        excision_params.ye_ex =
          grace::eos::get().get_eos<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>().ye_atmosphere() ;
        excision_params.temp_ex =
          grace::eos::get().get_eos<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>().temp_atmosphere() ;
      } else if ( cold_eos_type == "tabulated" ) {
        excision_params.ye_ex =
          grace::eos::get().get_eos<grace::hybrid_eos_t<grace::tabulated_cold_eos_t>>().ye_atmosphere() ;
        excision_params.temp_ex =
          grace::eos::get().get_eos<grace::hybrid_eos_t<grace::tabulated_cold_eos_t>>().temp_atmosphere() ;
      } else {
        ERROR("Unsupported cold_eos_type: " << cold_eos_type) ;
      }
    } else if ( eos_type == "tabulated" ) {
      excision_params.ye_ex =   
          grace::eos::get().get_eos<grace::tabulated_eos_t>().ye_atmosphere() ;
      excision_params.temp_ex =   
          grace::eos::get().get_eos<grace::tabulated_eos_t>().temp_atmosphere() ;
    }
    
    return excision_params ; 
}

#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
#define FILL_METRIC_ARRAY(g, view, q, ...)                    \
g = grace::metric_array_t{  { view(__VA_ARGS__,GXX_,q)   \
                          , view(__VA_ARGS__,GXY_,q)     \
                          , view(__VA_ARGS__,GXZ_,q)     \
                          , view(__VA_ARGS__,GYY_,q)     \
                          , view(__VA_ARGS__,GYZ_,q)     \
                          , view(__VA_ARGS__,GZZ_,q) }   \
                          , { view(__VA_ARGS__,BETAX_,q) \
                          , view(__VA_ARGS__,BETAY_,q)   \
                          , view(__VA_ARGS__,BETAZ_,q) } \
                          , view(__VA_ARGS__,ALP_,q) } 
#else 
#define FILL_METRIC_ARRAY(g, view, q, ...)                    \
g = grace::metric_array_t{  { view(__VA_ARGS__,GTXX_,q)   \
                          , view(__VA_ARGS__,GTXY_,q)     \
                          , view(__VA_ARGS__,GTXZ_,q)     \
                          , view(__VA_ARGS__,GTYY_,q)     \
                          , view(__VA_ARGS__,GTYZ_,q)     \
                          , view(__VA_ARGS__,GTZZ_,q) }   \
                          , view(__VA_ARGS__,CHI_,q)     \
                          , { view(__VA_ARGS__,BETAX_,q) \
                          , view(__VA_ARGS__,BETAY_,q)   \
                          , view(__VA_ARGS__,BETAZ_,q) } \
                          , view(__VA_ARGS__,ALP_,q) } 
#endif 

#define FILL_PRIMS_ARRAY_ZVEC(primsarr,vview,q,...)        \
primsarr[RHOL] = vview(__VA_ARGS__,RHO_,q);      \
primsarr[PRESSL] = vview(__VA_ARGS__,PRESS_,q) ; \
primsarr[ZXL] = vview(__VA_ARGS__,ZVECX_,q) ;     \
primsarr[ZYL] = vview(__VA_ARGS__,ZVECY_,q) ;     \
primsarr[ZZL] = vview(__VA_ARGS__,ZVECZ_,q) ;     \
primsarr[YEL] = vview(__VA_ARGS__,YE_,q) ;       \
primsarr[TEMPL] = vview(__VA_ARGS__,TEMP_,q) ;   \
primsarr[EPSL] = vview(__VA_ARGS__,EPS_,q) ;     \
primsarr[ENTL] = vview(__VA_ARGS__,ENTROPY_,q);  \
primsarr[BXL] = vview(__VA_ARGS__,BX_,q);        \
primsarr[BYL] = vview(__VA_ARGS__,BY_,q);        \
primsarr[BZL] = vview(__VA_ARGS__,BZ_,q)       

#define FILL_CONS_ARRAY(consarr, vview,q,...)      \
consarr[DENSL] = vview(__VA_ARGS__,DENS_,q);       \
consarr[TAUL] = vview(__VA_ARGS__,TAU_,q);         \
consarr[STXL] = vview(__VA_ARGS__,SX_,q);          \
consarr[STYL] = vview(__VA_ARGS__,SY_,q);          \
consarr[STZL] = vview(__VA_ARGS__,SZ_,q);          \
consarr[YESL] = vview(__VA_ARGS__,YESTAR_,q);      \
consarr[ENTSL] = vview(__VA_ARGS__,ENTROPYSTAR_,q) 

namespace grace {

// Fourth-order Lagrange interpolation from cell centers to a face,
// restricted to the 1D stencil along axis `idir`.  Coefficients:
//   c_inner =  9/16 for the two cells adjacent to the face,
//   c_outer = -1/16 for the two cells at distance 3/2 dx from the face.
//
// Pair-symmetric summation (inner pair, outer pair, then add) so the
// result is bit-equivariant under the L<->R swap induced by the mirror
// symmetries: at a mirror face the same expression sees operand pairs
// that are either invariant (scalar component) or exact negations
// (vector component) of their counterparts at the original face, and
// the two-term inner sums commute exactly under IEEE.  Left-to-right
// summation of all four terms would be non-associative across the
// swap and seed a ~1 ulp drift per face that propagates through the
// Riemann solver into the boundary flux.
template <typename View>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
face_interp_4(View const& view,
              VEC(int const i, int const j, int const k),
              int const ivar, int const q, int const idir)
{
    constexpr double c_inner =  0.5625;   //  9/16
    constexpr double c_outer = -0.0625;   // -1/16
    int const di = utils::delta(0, idir);
    int const dj = utils::delta(1, idir);
#ifdef GRACE_3D
    int const dk = utils::delta(2, idir);
#endif
    double const s_inner =
          view(VEC(i -   di, j -   dj, k -   dk), ivar, q)
        + view(VEC(i,        j,        k       ), ivar, q);
    double const s_outer =
          view(VEC(i - 2*di, j - 2*dj, k - 2*dk), ivar, q)
        + view(VEC(i +   di, j +   dj, k +   dk), ivar, q);
    return c_inner * s_inner + c_outer * s_outer;
}

// Build the face-centered metric by face_interp_4'ing each component of
// the cell-centered metric stored in `state`.  Replaces the legacy
// COMPUTE_FCVAL macro.
template <typename View>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE grace::metric_array_t
compute_face_metric(View const& state,
                    VEC(int const i, int const j, int const k),
                    int const q, int const idir)
{
#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
    return grace::metric_array_t{
        { face_interp_4(state, VEC(i, j, k), GXX_, q, idir),
          face_interp_4(state, VEC(i, j, k), GXY_, q, idir),
          face_interp_4(state, VEC(i, j, k), GXZ_, q, idir),
          face_interp_4(state, VEC(i, j, k), GYY_, q, idir),
          face_interp_4(state, VEC(i, j, k), GYZ_, q, idir),
          face_interp_4(state, VEC(i, j, k), GZZ_, q, idir) },
        { face_interp_4(state, VEC(i, j, k), BETAX_, q, idir),
          face_interp_4(state, VEC(i, j, k), BETAY_, q, idir),
          face_interp_4(state, VEC(i, j, k), BETAZ_, q, idir) },
        face_interp_4(state, VEC(i, j, k), ALP_, q, idir)
    };
#else
    return grace::metric_array_t{
        { face_interp_4(state, VEC(i, j, k), GTXX_, q, idir),
          face_interp_4(state, VEC(i, j, k), GTXY_, q, idir),
          face_interp_4(state, VEC(i, j, k), GTXZ_, q, idir),
          face_interp_4(state, VEC(i, j, k), GTYY_, q, idir),
          face_interp_4(state, VEC(i, j, k), GTYZ_, q, idir),
          face_interp_4(state, VEC(i, j, k), GTZZ_, q, idir) },
        face_interp_4(state, VEC(i, j, k), CHI_, q, idir),
        { face_interp_4(state, VEC(i, j, k), BETAX_, q, idir),
          face_interp_4(state, VEC(i, j, k), BETAY_, q, idir),
          face_interp_4(state, VEC(i, j, k), BETAZ_, q, idir) },
        face_interp_4(state, VEC(i, j, k), ALP_, q, idir)
    };
#endif
}

} // namespace grace

//*****************************************************************************************************
// Constrained-Transport / GS-EMF inline helpers.
//
// These factor out two pieces of arithmetic that previously appeared in three
// separate kernels each (compute_emfs / apply_fofc_correction / flag_fofc_cells
// for the CT B-face update; compute_emfs / apply_fofc_correction for the GS
// edge-EMF assembly).  Keeping a single implementation removes the risk of
// one copy drifting from the others as the discretization evolves.
//
//   ct_update_B{x,y,z}_face : given a face-staggered B at (i,j,k), an EMF
//        array, inverse spacings and dt*dtfact, return the post-CT face value.
//
//   gs_edge_emf_{x,y,z}     : given Eface (per-face EMF contributions),
//        Ecenter (cell-centered E), and fluxes (used only for the DENS sign
//        upwinding), assemble the Gardiner-Stone edge EMF at corner (i,j,k).
//*****************************************************************************************************
namespace grace {

template <typename BFaceXView, typename EmfView, typename IdxView>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
ct_update_Bx_face(BFaceXView const& Bx, EmfView const& emf, IdxView const& idx,
                  VEC(int const i, int const j, int const k), int const q,
                  double const dt_eff)
{
    return Bx(VEC(i,j,k), BSX_, q) + dt_eff * (
        (emf(VEC(i,j,k+1), 1, q) - emf(VEC(i,j,  k), 1, q)) * idx(2, q)
      + (emf(VEC(i,j,k),   2, q) - emf(VEC(i,j+1,k), 2, q)) * idx(1, q)
    );
}

template <typename BFaceYView, typename EmfView, typename IdxView>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
ct_update_By_face(BFaceYView const& By, EmfView const& emf, IdxView const& idx,
                  VEC(int const i, int const j, int const k), int const q,
                  double const dt_eff)
{
    return By(VEC(i,j,k), BSY_, q) + dt_eff * (
        (emf(VEC(i+1,j,k), 2, q) - emf(VEC(i,j,  k), 2, q)) * idx(0, q)
      + (emf(VEC(i,j,k),   0, q) - emf(VEC(i,j,k+1), 0, q)) * idx(2, q)
    );
}

template <typename BFaceZView, typename EmfView, typename IdxView>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
ct_update_Bz_face(BFaceZView const& Bz, EmfView const& emf, IdxView const& idx,
                  VEC(int const i, int const j, int const k), int const q,
                  double const dt_eff)
{
    return Bz(VEC(i,j,k), BSZ_, q) + dt_eff * (
        (emf(VEC(i,j+1,k), 0, q) - emf(VEC(i,j,k), 0, q)) * idx(1, q)
      + (emf(VEC(i,j,k),   1, q) - emf(VEC(i+1,j,k), 1, q)) * idx(0, q)
    );
}

#if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS
//-----------------------------------------------------------------------------
// Gardiner-Stone edge EMF assembly at corner (i,j,k).
//
// The three flavors differ only in which pair of face-direction indices feed
// the dE/dperp upwinding and which Ecenter axis is read.  Kept as three
// distinct functions (rather than one templated on idir) because the index
// shuffles are not a simple permutation — staying explicit avoids subtle
// off-by-ones.
//
// E^x edge at (i, j-1/2, k-1/2):
//   N/S split runs in z (y-face contributions), E/W split runs in y
//   (z-face contributions).  Sign of v^y selects the z-direction upwind,
//   sign of v^z selects the y-direction upwind.
//-----------------------------------------------------------------------------
template <typename EfaceView, typename EcenterView, typename FluxView>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
gs_edge_emf_x(EfaceView const& Eface, EcenterView const& Ecenter,
              FluxView const& fluxes,
              VEC(int const i, int const j, int const k), int const q)
{
    // z-face contributions (idir=2, axis=0)
    double const Exzf  = Eface(i, j,   k,   0, 2, q);
    double const Exzfm = Eface(i, j-1, k,   0, 2, q);
    // y-face contributions (idir=1, axis=0)
    double const Exyf  = Eface(i, j,   k,   0, 1, q);
    double const Exyfm = Eface(i, j,   k-1, 0, 1, q);
    // Cell-centered E^x at the 4 corners of the edge.
    double const ExNE = Ecenter(VEC(i, j,   k  ), 0, q);
    double const ExNW = Ecenter(VEC(i, j-1, k  ), 0, q);
    double const ExSE = Ecenter(VEC(i, j,   k-1), 0, q);
    double const ExSW = Ecenter(VEC(i, j-1, k-1), 0, q);

    double const Eavg = 0.25 * (Exzf + Exzfm + Exyf + Exyfm);

    // dE/dz upwinded by v^y sign (face y at index j, j-? doesn't matter; we
    // need *signs* on both sides of the edge in z).
    double const Sy  = Kokkos::copysign(1., fluxes(i, j, k,   DENS_, 1, q));
    double const Sym = Kokkos::copysign(1., fluxes(i, j, k-1, DENS_, 1, q));
    double const dEdzN = (1.-Sy ) * (ExNE - Exzf )
                       + (1.+Sy ) * (ExNW - Exzfm);
    double const dEdzS = (1.-Sym) * (Exzf - ExSE)
                       + (1.+Sym) * (Exzfm - ExSW);
    double const dEdz  = (1./8.) * (dEdzS - dEdzN);

    // dE/dy upwinded by v^z sign.
    double const Sz  = Kokkos::copysign(1., fluxes(i, j,   k, DENS_, 2, q));
    double const Szm = Kokkos::copysign(1., fluxes(i, j-1, k, DENS_, 2, q));
    double const dEdyE = (1.-Sz ) * (ExNE - Exyf )
                       + (1.+Sz ) * (ExSE - Exyfm);
    double const dEdyW = (1.-Szm) * (Exyf  - ExNW)
                       + (1.+Szm) * (Exyfm - ExSW);
    double const dEdy  = (1./8.) * (dEdyW - dEdyE);

    return Eavg + dEdz + dEdy;
}

//-----------------------------------------------------------------------------
// E^y edge at (i-1/2, j, k-1/2):
//   N/S split in z (x-face contributions), E/W split in x
//   (z-face contributions).  Sign of v^x selects z-upwind, sign of v^z
//   selects x-upwind.
//-----------------------------------------------------------------------------
template <typename EfaceView, typename EcenterView, typename FluxView>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
gs_edge_emf_y(EfaceView const& Eface, EcenterView const& Ecenter,
              FluxView const& fluxes,
              VEC(int const i, int const j, int const k), int const q)
{
    // z-face contributions (idir=2, axis=1)
    double const Eyzf  = Eface(i,   j, k,   1, 2, q);
    double const Eyzfm = Eface(i-1, j, k,   1, 2, q);
    // x-face contributions (idir=0, axis=0 of E^y... matches compute_emfs).
    // NB: compute_emfs reads Eface(...,0,0,q) for E^y E/W — the x-face
    // contribution to E^y is stored in axis-0 slot of the x-face, by
    // convention of getflux<0> writing vb_HLL[0]/[1] for the two orthogonal
    // E components.
    double const Eyxf  = Eface(i, j, k,   0, 0, q);
    double const Eyxfm = Eface(i, j, k-1, 0, 0, q);
    // Cell-centered E^y at the 4 corners.
    double const EyNE = Ecenter(VEC(i,   j, k  ), 1, q);
    double const EyNW = Ecenter(VEC(i-1, j, k  ), 1, q);
    double const EySE = Ecenter(VEC(i,   j, k-1), 1, q);
    double const EySW = Ecenter(VEC(i-1, j, k-1), 1, q);

    double const Eavg = 0.25 * (Eyzf + Eyzfm + Eyxf + Eyxfm);

    double const Sx  = Kokkos::copysign(1., fluxes(i, j, k,   DENS_, 0, q));
    double const Sxm = Kokkos::copysign(1., fluxes(i, j, k-1, DENS_, 0, q));
    double const dEdzN = (1.-Sx ) * (EyNE - Eyzf )
                       + (1.+Sx ) * (EyNW - Eyzfm);
    double const dEdzS = (1.-Sxm) * (Eyzf - EySE)
                       + (1.+Sxm) * (Eyzfm - EySW);
    double const dEdz  = (1./8.) * (dEdzS - dEdzN);

    double const Sz  = Kokkos::copysign(1., fluxes(i,   j, k, DENS_, 2, q));
    double const Szm = Kokkos::copysign(1., fluxes(i-1, j, k, DENS_, 2, q));
    double const dEdxE = (1.-Sz ) * (EyNE - Eyxf )
                       + (1.+Sz ) * (EySE - Eyxfm);
    double const dEdxW = (1.-Szm) * (Eyxf  - EyNW)
                       + (1.+Szm) * (Eyxfm - EySW);
    double const dEdx  = (1./8.) * (dEdxW - dEdxE);

    return Eavg + dEdz + dEdx;
}

//-----------------------------------------------------------------------------
// E^z edge at (i-1/2, j-1/2, k):
//   N/S split in y (x-face contributions), E/W split in x
//   (y-face contributions).  Sign of v^x selects y-upwind, sign of v^y
//   selects x-upwind.
//-----------------------------------------------------------------------------
template <typename EfaceView, typename EcenterView, typename FluxView>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
gs_edge_emf_z(EfaceView const& Eface, EcenterView const& Ecenter,
              FluxView const& fluxes,
              VEC(int const i, int const j, int const k), int const q)
{
    // y-face contributions (idir=1, axis=1)
    double const Ezyf  = Eface(i,   j, k, 1, 1, q);
    double const Ezyfm = Eface(i-1, j, k, 1, 1, q);
    // x-face contributions (idir=0, axis=1 of x-face for E^z component).
    // compute_emfs reads Eface(...,1,0,q); see note in gs_edge_emf_y.
    double const Ezxf  = Eface(i, j,   k, 1, 0, q);
    double const Ezxfm = Eface(i, j-1, k, 1, 0, q);
    // Cell-centered E^z at the 4 corners.
    double const EzNE = Ecenter(VEC(i,   j,   k), 2, q);
    double const EzNW = Ecenter(VEC(i-1, j,   k), 2, q);
    double const EzSE = Ecenter(VEC(i,   j-1, k), 2, q);
    double const EzSW = Ecenter(VEC(i-1, j-1, k), 2, q);

    double const Eavg = 0.25 * (Ezyf + Ezyfm + Ezxf + Ezxfm);

    double const Sx  = Kokkos::copysign(1., fluxes(i, j,   k, DENS_, 0, q));
    double const Sxm = Kokkos::copysign(1., fluxes(i, j-1, k, DENS_, 0, q));
    double const dEdyN = (1.-Sx ) * (EzNE - Ezyf )
                       + (1.+Sx ) * (EzNW - Ezyfm);
    double const dEdyS = (1.-Sxm) * (Ezyf - EzSE)
                       + (1.+Sxm) * (Ezyfm - EzSW);
    double const dEdy  = (1./8.) * (dEdyS - dEdyN);

    double const Sy  = Kokkos::copysign(1., fluxes(i,   j, k, DENS_, 1, q));
    double const Sym = Kokkos::copysign(1., fluxes(i-1, j, k, DENS_, 1, q));
    double const dEdxE = (1.-Sy ) * (EzNE - Ezxf )
                       + (1.+Sy ) * (EzSE - Ezxfm);
    double const dEdxW = (1.-Sym) * (Ezxf  - EzNW)
                       + (1.+Sym) * (Ezxfm - EzSW);
    double const dEdx  = (1./8.) * (dEdxW - dEdxE);

    return Eavg + dEdy + dEdx;
}
#endif // GS

} // namespace grace

struct grmhd_id_t {
  double rho;
  double press;
  double eps; 
  double temp;
  double entropy;
  double ye;
  double gxx,gxy,gxz,gyy,gyz,gzz; 
  double kxx,kxy,kxz,kyy,kyz,kzz;
  double alp;  
  double betax, betay, betaz ; 
  double vx, vy, vz;
  double bx, by, bz;
} ; 

#endif /* GRACE_PHYSICS_GRMHD_HELPERS_HH */