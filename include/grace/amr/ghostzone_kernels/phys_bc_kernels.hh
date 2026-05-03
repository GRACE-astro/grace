/**
 * @file phys_bc_kernels.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-09-05
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
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
#ifndef GRACE_AMR_PHYS_BC_KERNELS_HH
#define GRACE_AMR_PHYS_BC_KERNELS_HH 

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/physics/fd_subexpressions.hh>

#include <grace/coordinates/coordinate_systems.hh>

#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

struct reflect_bc_t 
{
    template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t is, int8_t js, int8_t ks), double f
      ) const
      {
        view(i,j,k) = f*view(is,js,ks) ; 
      }
} ;

template< size_t order >
struct extrap_bc_t 
{
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
      ) const ; 
} ; 

template<>
struct extrap_bc_t<0>
{
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
      ) const
      {
            view(VEC(i,j,k)) = view(VEC(i-dx,j-dy,k-dz)) ; 
      }; 
} ; 

template<>
struct extrap_bc_t<3>
{
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
        ) const
      {
            view(VEC(i,j,k)) =( 4*view(VEC(i-dx,j-dy,k-dz)) 
                              - 6*view(VEC(i-2*dx,j-2*dy,k-2*dz)) 
                              + 4*view(VEC(i-3*dx,j-3*dy,k-3*dz)) 
                              -   view(VEC(i-4*dx,j-4*dy,k-4*dz)) ); 
      }; 
} ;

struct sommerfeld_bc_t  
{
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view, view_t view_p,
            double r, double invh, double v, double f0, double s[3], double dt, double dtfact,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz),
            VEC( size_t nx, size_t ny, size_t nz), size_t ngz 
        ) const
      {
        double dudx,dudy,dudz; 
        if ( dx>0 ) {
            fd_der_2_x_l1(view_p,i,j,k,invh,&dudx) ; 
        } else if ( dx<0 ) {
            fd_der_2_x_r1(view_p,i,j,k,invh,&dudx) ; 
        } else {
            if ( i > 0 and i < nx + 2*ngz - 1 ) {
                fd_der_2_x(view_p,i,j,k,invh,&dudx) ; 
            } else if ( i == 0 ) {
                fd_der_2_x_r1(view_p,i,j,k,invh,&dudx) ; 
            } else {
                fd_der_2_x_l1(view_p,i,j,k,invh,&dudx) ; 
            }
        }
        if ( dy>0 ) {
            fd_der_2_y_l1(view_p,i,j,k,invh,&dudy) ; 
        } else if ( dy<0 ) {
            fd_der_2_y_r1(view_p,i,j,k,invh,&dudy) ; 
        } else {
            if ( j > 0 and j < ny + 2*ngz - 1 ) {
                fd_der_2_y(view_p,i,j,k,invh,&dudy) ;  
            } else if ( j ==  0 ) {
                fd_der_2_y_r1(view_p,i,j,k,invh,&dudy) ; 
            } else {
                fd_der_2_y_l1(view_p,i,j,k,invh,&dudy) ; 
            }
        }
        if ( dz>0 ) {
            fd_der_2_z_l1(view_p,i,j,k,invh,&dudz) ; 
        } else if ( dz<0 ) {
            fd_der_2_z_r1(view_p,i,j,k,invh,&dudz) ; 
        } else {
            if ( k > 0 and k < nz + 2*ngz - 1 ) {
                fd_der_2_z(view_p,i,j,k,invh,&dudz) ;  
            } else if ( k == 0 ) {
                fd_der_2_z_r1(view_p,i,j,k,invh,&dudz) ; 
            } else {
                fd_der_2_z_l1(view_p,i,j,k,invh,&dudz) ; 
            }
        }
        double dudt = -v*(s[0]*dudx + s[1]*dudy + s[2]*dudz) + (f0-view_p(i,j,k))*v/r;
        view(i,j,k) += dudt * dt * dtfact ;  // dt should be passed in
      }; 
} ; 

/**
 * @brief Apply outgoing boundary conditions.
 * \ingroup amr
 */
using outflow_bc_t = extrap_bc_t<0> ;

// Work tags for the precomputed-bounds, (face x var)-parallel FACE_EXT launch
// and the follow-up Z4c algebraic-constraint enforcement pass.
struct phys_bc_face_ext_md_tag    {} ;
struct phys_bc_z4c_constr_tag     {} ;

// Classical (non-fused) MDRange work tags.  One per `elem_kind`; `bc_kind`
// is a compile-time template param on `phys_bc_op` but does not change the
// shape of the iteration space — only the bounds, which are precomputed on
// the host.  Shapes:
//   phys_bc_md_face_tag      Rank<4>(i, j, iv, iq)  — 2 parallel, 1 serial ghost
//   phys_bc_md_edge_tag      Rank<3>(i, iv, iq)     — 1 parallel, 2 serial ghost
//   phys_bc_md_corner_tag    Rank<2>(iv, iq)        — 0 parallel, 3 serial ghost
// The per-element `iq` axis lives on the slow end of the policy so launch
// geometry can tile the (parallel, iv) axes freely.
struct phys_bc_md_face_tag        {} ;
struct phys_bc_md_edge_tag        {} ;
struct phys_bc_md_corner_tag      {} ;

// Z4c algebraic-constraint follow-up variants: same shape minus the `iv`
// axis.  Launched only when `stag == STAG_CENTER` under
// GRACE_ENABLE_Z4C_METRIC; for other staggerings we never touch them.
struct phys_bc_md_face_z4c_tag    {} ;
struct phys_bc_md_edge_z4c_tag    {} ;
struct phys_bc_md_corner_z4c_tag  {} ;

KOKKOS_INLINE_FUNCTION 
static void get_somm_props(int iv, double *v, double *f0) {
    #ifdef GRACE_ENABLE_Z4C_METRIC
    if ( iv == ALP_ or iv == KHAT_ ) {
        *v = sqrt(2) ; 
    } else {
        *v = 1.0 ; 
    }
    if ( iv == ALP_ or iv == GTXX_ or iv == GTYY_ or iv == GTZZ_ or iv == CHI_ ) {
        *f0 = 1.0;
    } else {
        *f0 = 0.0 ;
    }
    #else
    *f0=0.0 ; *v=1.;
    #endif 
} 

template< element_kind_t elem_kind, element_kind_t bc_kind, typename view_t, bool extended = false >
struct phys_bc_op {

    readonly_view_t<std::size_t> qid   ;
    readonly_view_t<uint8_t> eid       ;
    readonly_twod_view_t<int8_t,3> dir ;
    readonly_twod_view_t<double,3> var_refl_fact ;

    readonly_view_t<bc_t> var_bcs      ;

    readonly_twod_view_t<int,3> exloop       ;
    readonly_twod_view_t<int,3> offloop      ;

    // Guard mask for the fused FACE_EXT kernel.  Bit layout per quad:
    //   0..3 : adjacent type=FACE edges (f2e ordering: A-lo, A-hi, B-lo, B-hi)
    //   4..7 : adjacent type=FACE corners (f2c ordering: AloBlo, AhiBlo, AloBhi, AhiBhi)
    // Unused (empty view) when `extended == false`.
    readonly_view_t<uint8_t> guard_mask ;

    // Precomputed per-face geometry, populated by the task factory at setup
    // time. Used by `phys_bc_face_ext_md_tag` and `phys_bc_z4c_constr_tag`
    // operator() variants to avoid redundant per-thread `compute_bounds`
    // calls. Empty (extent(0)==0) for legacy TeamPolicy launches that still
    // call `compute_bounds` inline.
    readonly_twod_view_t<int,3> bnd_lmin    ;
    readonly_twod_view_t<int,3> bnd_idir    ;
    readonly_twod_view_t<int,3> bnd_extents ;
    readonly_twod_view_t<int,3> bnd_pdim    ;
    readonly_twod_view_t<int,3> bnd_npdim   ;

    view_t data, data_p ;

    outflow_bc_t outflow_kernel ;
    extrap_bc_t<3> extrap_kernel ; 
    sommerfeld_bc_t sommerfeld_kernel ; 
    reflect_bc_t reflect_kernel ; 

    bool is_cbuf ; //!< If the data is cbuf set_data_ptr **must** be no-op

    var_staggering_t stag ; 

    // only one view involved, if nx needs to be 
    // halved, just do it here 
    size_t nx, ny, nz, ngz, nv ; 



    bool rx,ry,rz ; 

    double dt, dtfact ; 

    device_coordinate_system coords; 
    scalar_array_t<GRACE_NSPACEDIM> dx ; 

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        // NB this wont work if 
        // the phys boundaries
        // sit at an AMR boundary.
        // I don't see a way around 
        // the issue. 
        if (!is_cbuf) {
            data   = alias.get<stag>() ;
            data_p = alias.get_p<stag>() ;
            dt = alias._dt; dtfact = alias._dtfact;
        }

    }

    phys_bc_op(
        view_t _data, view_t _data_p, scalar_array_t<GRACE_NSPACEDIM> _dx, device_coordinate_system _coords,
        Kokkos::View<size_t*> _qid,
        Kokkos::View<uint8_t*> _eid,
        Kokkos::View<int8_t*[3]> _dir,
        Kokkos::View<int*[3]> _ext,
        Kokkos::View<int*[3]> _off_l,
        Kokkos::View<double*[3]> _var_rfact,
        Kokkos::View<bc_t*> _var_bcs,
         VEC(size_t _nx, size_t _ny, size_t _nz), size_t _ngz, size_t _nv, bool _is_cbuf, var_staggering_t _stag, bool _rx, bool _ry, bool _rz,
        Kokkos::View<uint8_t*> _guard_mask = Kokkos::View<uint8_t*>{}
    ) : qid(_qid),  eid(_eid), dir(_dir), var_refl_fact(_var_rfact), var_bcs(_var_bcs), exloop(_ext), offloop(_off_l),
        guard_mask(_guard_mask), dx(_dx), coords(_coords),
        data(_data), data_p(_data_p),
        nx(_nx), ny(_ny), nz(_nz), ngz(_ngz), nv(_nv), is_cbuf(_is_cbuf), stag(_stag), rx(_rx), ry(_ry), rz(_rz)
    {
        outflow_kernel = outflow_bc_t{} ; extrap_kernel = extrap_bc_t<3>{} ; sommerfeld_kernel = sommerfeld_bc_t{} ; reflect_kernel = reflect_bc_t{} ;
    }

    #ifdef GRACE_ENABLE_Z4C_METRIC
    KOKKOS_INLINE_FUNCTION 
    void impose_algebraic_constraintz_z4c(const int ijk[3], size_t qid) const {
        auto sv = Kokkos::subview(
            data, 
            ijk[0],ijk[1],ijk[2],
            Kokkos::ALL(), qid
        );
        double gtxx = sv(GTXX_);
        double gtxy = sv(GTXY_);
        double gtxz = sv(GTXZ_);
        double gtyy = sv(GTYY_);
        double gtyz = sv(GTYZ_);
        double gtzz = sv(GTZZ_);

        double const detgt     = -(gtxz*gtxz*gtyy) + 2*gtxy*gtxz*gtyz - gtxx*(gtyz*gtyz) - gtxy*gtxy*gtzz + gtxx*gtyy*gtzz;
        double const cbrtdetgt = Kokkos::cbrt(detgt);

        gtxx/=cbrtdetgt;
        gtxy/=cbrtdetgt;
        gtxz/=cbrtdetgt;
        gtyy/=cbrtdetgt;
        gtyz/=cbrtdetgt;
        gtzz/=cbrtdetgt;

        double const gtXX=(-(gtyz*gtyz) + gtyy*gtzz) ;
        double const gtXY=(gtxz*gtyz - gtxy*gtzz)    ;
        double const gtXZ=(-(gtxz*gtyy) + gtxy*gtyz) ;
        double const gtYY=(-(gtxz*gtxz) + gtxx*gtzz) ;
        double const gtYZ=(gtxy*gtxz - gtxx*gtyz)    ;
        double const gtZZ=(-(gtxy*gtxy) + gtxx*gtyy) ; 

        double const Atxx = sv(ATXX_);
        double const Atxy = sv(ATXY_);
        double const Atxz = sv(ATXZ_);
        double const Atyy = sv(ATYY_);
        double const Atyz = sv(ATYZ_);
        double const Atzz = sv(ATZZ_);

        double const ATR = Atxx*gtXX + 2*Atxy*gtXY + 2*Atxz*gtXZ + Atyy*gtYY + 2*Atyz*gtYZ + Atzz*gtZZ ; 
        
        sv(ATXX_) -= 1./3. * gtxx * ATR ; 
        sv(ATXY_) -= 1./3. * gtxy * ATR ; 
        sv(ATXZ_) -= 1./3. * gtxz * ATR ; 
        sv(ATYY_) -= 1./3. * gtyy * ATR ; 
        sv(ATYZ_) -= 1./3. * gtyz * ATR ; 
        sv(ATZZ_) -= 1./3. * gtzz * ATR ; 

        sv(GTXX_) = gtxx ; 
        sv(GTXY_) = gtxy ; 
        sv(GTXZ_) = gtxz ; 
        sv(GTYY_) = gtyy ; 
        sv(GTYZ_) = gtyz ; 
        sv(GTZZ_) = gtzz ; 
    }
    #endif 

    KOKKOS_INLINE_FUNCTION 
    void apply_bc_impl(int iv, const int ijk[3], const int8_t _dir[3], size_t qid) const {
        auto sv = Kokkos::subview(
            data, 
            VEC(Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
            static_cast<size_t>(iv), qid 
        ) ;

        bool do_reflection = false ; 
        double fact = 1.0;
        int ijk_s[3] = {ijk[0],ijk[1],ijk[2]}; 
        // first we detect reflections
        if ( rx or ry or rz ) {
            double parities[3] = {1.,1.,1.} ; 
            if ( stag == STAG_CENTER ) {
                parities[0] = var_refl_fact(iv,0);
                parities[1] = var_refl_fact(iv,1);
                parities[2] = var_refl_fact(iv,2); 
            } else if ( stag == STAG_FACEX ) {
                // B is a pseudovector: B_d is even under d-reflection,
                // B_{j!=d} are odd
                parities[0] = 1 ;
                parities[1] = parities[2] = -1;
            } else if ( stag == STAG_FACEY ) {
                parities[1] = 1 ;
                parities[0] = parities[2] = -1;
            } else if ( stag == STAG_FACEZ ) {
                parities[2] = 1 ;
                parities[0] = parities[1] = -1;
            }

            if ( _dir[0] == -1 and rx ) {
                ijk_s[0] = ngz + (ngz-1-ijk[0]) ; 
                ijk_s[0] += stag==STAG_FACEX ? 1 : 0; 
                fact *= parities[0] ; 
                do_reflection = true ; 
            }

            if ( _dir[1] == -1 and ry ) {
                ijk_s[1] = ngz + (ngz-1-ijk[1]) ; 
                ijk_s[1] += stag==STAG_FACEY ? 1 : 0; 
                fact *= parities[1] ; 
                do_reflection = true ; 
            }

            if ( _dir[2] == -1 and rz ) {
                ijk_s[2] = ngz + (ngz-1-ijk[2]) ; 
                ijk_s[2] += stag==STAG_FACEZ ? 1 : 0; 
                fact *= parities[2] ; 
                do_reflection = true ; 
            }
        }
        if (do_reflection) {
            reflect_kernel.template apply<decltype(sv)>(sv,ijk[0],ijk[1],ijk[2],ijk_s[0],ijk_s[1],ijk_s[2],fact) ;
        } else {
            auto _bc_kind = var_bcs(iv) ;       
            switch (_bc_kind) {
                case BC_OUTFLOW:{
                    outflow_kernel.template apply<decltype(sv)>(
                        sv, VEC(ijk[0],ijk[1],ijk[2]), VEC(_dir[0], _dir[1], _dir[2]));
                    break;
                }
                case BC_LAGRANGE_EXTRAP: {
                    extrap_kernel.template apply<decltype(sv)>(
                        sv, VEC(ijk[0],ijk[1],ijk[2]), VEC(_dir[0], _dir[1], _dir[2]));
                    break;
                }
                case BC_SOMMERFELD: {
                    double vel,f0;
                    get_somm_props(iv, &vel, &f0) ; 
                    double h = dx(0,qid) ; 
                    double s[3] ; 
                    coords.get_physical_coordinates(
                        ijk[0],ijk[1],ijk[2],qid,s
                    ) ; 
                    auto sv_p = Kokkos::subview(
                        data_p, 
                        VEC(Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
                        iv, qid 
                    ) ;
                    double r = sqrt(SQR(s[0])+SQR(s[1])+SQR(s[2]));
                    s[0]/=r; s[1]/=r; s[2]/=r ; 
                    sommerfeld_kernel.template apply<decltype(sv)>(
                        sv, sv_p, r,h,vel,f0,s,dt,dtfact,VEC(ijk[0],ijk[1],ijk[2]), VEC(_dir[0], _dir[1], _dir[2]),nx,ny,nz,ngz);
                    break ; 
                }
                case BC_NONE:
                    break;
                default:
                    // fallback or assert
                    break;
            }
        }
    }
    // FACE_EXT-only: returns true if the (ijk[pdim[0]], ijk[pdim[1]]) cell
    // is in an adjacent edge/corner region that the guard mask says we should
    // skip. Returns false for interior-of-face cells (no skip) and for cells
    // whose corresponding guard bit is set (apply BC).
    KOKKOS_INLINE_FUNCTION
    bool guard_skip(size_t iq, int const ijk[3], int const pdim[3]) const
    {
        if constexpr (!extended || elem_kind != element_kind_t::FACE) {
            return false;
        } else {
            uint8_t const mask = guard_mask(iq);
            size_t const _ncells[3] = {nx, ny, nz};
            size_t const nA = _ncells[pdim[0]];
            size_t const nB = _ncells[pdim[1]];
            int const pA = ijk[pdim[0]];
            int const pB = ijk[pdim[1]];
            int const clsA = (pA < (int)ngz) ? 0 : ((pA >= (int)(nA + ngz)) ? 2 : 1);
            int const clsB = (pB < (int)ngz) ? 0 : ((pB >= (int)(nB + ngz)) ? 2 : 1);
            if (clsA == 1 && clsB == 1) return false;
            int bit;
            if (clsA == 1) {
                bit = (clsB == 0) ? 2 : 3;
            } else if (clsB == 1) {
                bit = (clsA == 0) ? 0 : 1;
            } else {
                int const aHi = (clsA == 2) ? 1 : 0;
                int const bHi = (clsB == 2) ? 1 : 0;
                bit = 4 + (bHi << 1) + aHi;
            }
            return !((mask >> bit) & 1);
        }
    }

    // FACE_EXT MDRange operator. Iteration space is (i, j, iv, iq) where
    // i,j are the two parallel in-face axes bounded by `max_ext_par` (max
    // across all faces — ragged faces predicate out-of-bounds). The swept
    // ghost-depth axis (`ngz`) is an inner serial loop, which avoids pinning
    // one wavefront per face and lets the driver tile (i, j, iv, iq) freely.
    //
    // Z4c algebraic constraint enforcement is *not* run here — it requires
    // all six metric components written first. Use phys_bc_z4c_constr_tag
    // as a follow-up launch when stag == STAG_CENTER.
    KOKKOS_INLINE_FUNCTION
    void operator() (
        phys_bc_face_ext_md_tag,
        int i, int j, int iv, int iq
    ) const
    {
        if constexpr (!(extended
                        && elem_kind == element_kind_t::FACE
                        && bc_kind   == element_kind_t::FACE)) {
            return ;
        } else {
            int const pdim0  = bnd_pdim(iq, 0) ;
            int const pdim1  = bnd_pdim(iq, 1) ;
            int const npdim0 = bnd_npdim(iq, 0) ;

            int const e_p0 = bnd_extents(iq, pdim0) ;
            int const e_p1 = bnd_extents(iq, pdim1) ;
            if (i >= e_p0 || j >= e_p1) return ;

            int ijk[3] ;
            ijk[pdim0] = bnd_lmin(iq, pdim0) + bnd_idir(iq, pdim0) * i ;
            ijk[pdim1] = bnd_lmin(iq, pdim1) + bnd_idir(iq, pdim1) * j ;

            int const pdim_l[3] = {pdim0, pdim1, npdim0} ;
            if (guard_skip(iq, ijk, pdim_l)) return ;

            int8_t const _dir[3] = {dir(iq,0), dir(iq,1), dir(iq,2)} ;
            size_t const _qid = qid(iq) ;

            int const lm_n0 = bnd_lmin(iq, npdim0) ;
            int const id_n0 = bnd_idir(iq, npdim0) ;
            int const e_n0  = bnd_extents(iq, npdim0) ;

            for (int k = 0; k < e_n0; ++k) {
                ijk[npdim0] = lm_n0 + id_n0 * k ;
                apply_bc_impl(iv, ijk, _dir, _qid) ;
            }
        }
    }

    // Z4c algebraic-constraint follow-up pass. Rank<3>(i, j, iq) with the
    // swept axis serial, calls `impose_algebraic_constraintz_z4c` once per
    // (i,j,k). Only meaningful for STAG_CENTER under GRACE_ENABLE_Z4C_METRIC;
    // the task factory only launches this in that case.
    KOKKOS_INLINE_FUNCTION
    void operator() (
        phys_bc_z4c_constr_tag,
        int i, int j, int iq
    ) const
    {
        #ifdef GRACE_ENABLE_Z4C_METRIC
        if constexpr (!(extended
                        && elem_kind == element_kind_t::FACE
                        && bc_kind   == element_kind_t::FACE)) {
            return ;
        } else {
            int const pdim0  = bnd_pdim(iq, 0) ;
            int const pdim1  = bnd_pdim(iq, 1) ;
            int const npdim0 = bnd_npdim(iq, 0) ;

            int const e_p0 = bnd_extents(iq, pdim0) ;
            int const e_p1 = bnd_extents(iq, pdim1) ;
            if (i >= e_p0 || j >= e_p1) return ;

            int ijk[3] ;
            ijk[pdim0] = bnd_lmin(iq, pdim0) + bnd_idir(iq, pdim0) * i ;
            ijk[pdim1] = bnd_lmin(iq, pdim1) + bnd_idir(iq, pdim1) * j ;

            int const pdim_l[3] = {pdim0, pdim1, npdim0} ;
            if (guard_skip(iq, ijk, pdim_l)) return ;

            size_t const _qid = qid(iq) ;

            int const lm_n0 = bnd_lmin(iq, npdim0) ;
            int const id_n0 = bnd_idir(iq, npdim0) ;
            int const e_n0  = bnd_extents(iq, npdim0) ;

            for (int k = 0; k < e_n0; ++k) {
                ijk[npdim0] = lm_n0 + id_n0 * k ;
                impose_algebraic_constraintz_z4c(ijk, _qid) ;
            }
        }
        #endif
    }

    // ------------------------------------------------------------------
    // Classical MDRange operators.  Bounds (lmin/idir/extents/pdim/npdim)
    // are precomputed on the host and stored in bnd_* Views — kernel just
    // reads them.  Each operator early-outs on elem_kind mismatch so the
    // tag chooses the shape unambiguously at dispatch time.
    //
    // Ragged extent handling: we pass an upper bound `max_ext_par` on the
    // policy end for each parallel axis; elements with shorter extent
    // predicate-out the tail.  The swept ghost-depth axis (or axes, for
    // EDGE / CORNER) is serial inside the kernel so tiles are never
    // fragmented across per-element boundaries.
    //
    // Z4c algebraic-constraint enforcement is *not* run in the BC
    // operators — it requires all six metric components written first.
    // The _z4c_tag variants do that pass as a follow-up launch over the
    // same (precomputed-bounds) indices.
    // ------------------------------------------------------------------

    // FACE classical: (i, j, iv, iq), 2 parallel axes + 1 serial ghost axis.
    // Dispatched on `bc_kind == FACE` (not elem_kind) — covers elem_kind
    // FACE, EDGE, and CORNER when combined with a bc_kind=FACE region.
    KOKKOS_INLINE_FUNCTION
    void operator() (
        phys_bc_md_face_tag,
        int i, int j, int iv, int iq
    ) const
    {
        if constexpr (extended || bc_kind != element_kind_t::FACE) {
            return ;
        } else {
            int const pdim0  = bnd_pdim(iq, 0) ;
            int const pdim1  = bnd_pdim(iq, 1) ;
            int const npdim0 = bnd_npdim(iq, 0) ;

            int const e_p0 = bnd_extents(iq, pdim0) ;
            int const e_p1 = bnd_extents(iq, pdim1) ;
            if (i >= e_p0 || j >= e_p1) return ;

            int ijk[3] ;
            ijk[pdim0] = bnd_lmin(iq, pdim0) + bnd_idir(iq, pdim0) * i ;
            ijk[pdim1] = bnd_lmin(iq, pdim1) + bnd_idir(iq, pdim1) * j ;

            int8_t const _dir[3] = {dir(iq,0), dir(iq,1), dir(iq,2)} ;
            size_t const _qid = qid(iq) ;

            int const lm_n0 = bnd_lmin(iq, npdim0) ;
            int const id_n0 = bnd_idir(iq, npdim0) ;
            int const e_n0  = bnd_extents(iq, npdim0) ;

            for (int k = 0; k < e_n0; ++k) {
                ijk[npdim0] = lm_n0 + id_n0 * k ;
                apply_bc_impl(iv, ijk, _dir, _qid) ;
            }
        }
    }

    // EDGE classical: (i, iv, iq), 1 parallel axis + 2 serial ghost axes.
    // Dispatched on `bc_kind == EDGE` — covers elem_kind EDGE and CORNER
    // when combined with a bc_kind=EDGE region.
    KOKKOS_INLINE_FUNCTION
    void operator() (
        phys_bc_md_edge_tag,
        int i, int iv, int iq
    ) const
    {
        if constexpr (extended || bc_kind != element_kind_t::EDGE) {
            return ;
        } else {
            int const pdim0  = bnd_pdim(iq, 0) ;
            int const npdim0 = bnd_npdim(iq, 0) ;
            int const npdim1 = bnd_npdim(iq, 1) ;

            int const e_p0 = bnd_extents(iq, pdim0) ;
            if (i >= e_p0) return ;

            int ijk[3] ;
            ijk[pdim0] = bnd_lmin(iq, pdim0) + bnd_idir(iq, pdim0) * i ;

            int8_t const _dir[3] = {dir(iq,0), dir(iq,1), dir(iq,2)} ;
            size_t const _qid = qid(iq) ;

            int const lm_n0 = bnd_lmin(iq, npdim0) ;
            int const id_n0 = bnd_idir(iq, npdim0) ;
            int const e_n0  = bnd_extents(iq, npdim0) ;
            int const lm_n1 = bnd_lmin(iq, npdim1) ;
            int const id_n1 = bnd_idir(iq, npdim1) ;
            int const e_n1  = bnd_extents(iq, npdim1) ;

            for (int jj = 0; jj < e_n0; ++jj) {
                ijk[npdim0] = lm_n0 + id_n0 * jj ;
                for (int kk = 0; kk < e_n1; ++kk) {
                    ijk[npdim1] = lm_n1 + id_n1 * kk ;
                    apply_bc_impl(iv, ijk, _dir, _qid) ;
                }
            }
        }
    }

    // CORNER classical: (iv, iq), 0 parallel + 3 serial ghost axes.
    // Dispatched on `bc_kind == CORNER` — only elem_kind=CORNER reaches here.
    KOKKOS_INLINE_FUNCTION
    void operator() (
        phys_bc_md_corner_tag,
        int iv, int iq
    ) const
    {
        if constexpr (extended || bc_kind != element_kind_t::CORNER) {
            return ;
        } else {
            int8_t const _dir[3] = {dir(iq,0), dir(iq,1), dir(iq,2)} ;
            size_t const _qid = qid(iq) ;

            int const lm0 = bnd_lmin(iq, 0), id0 = bnd_idir(iq, 0), e0 = bnd_extents(iq, 0) ;
            int const lm1 = bnd_lmin(iq, 1), id1 = bnd_idir(iq, 1), e1 = bnd_extents(iq, 1) ;
            int const lm2 = bnd_lmin(iq, 2), id2 = bnd_idir(iq, 2), e2 = bnd_extents(iq, 2) ;

            int ijk[3] ;
            for (int ci = 0; ci < e0; ++ci) {
                ijk[0] = lm0 + id0 * ci ;
                for (int cj = 0; cj < e1; ++cj) {
                    ijk[1] = lm1 + id1 * cj ;
                    for (int ck = 0; ck < e2; ++ck) {
                        ijk[2] = lm2 + id2 * ck ;
                        apply_bc_impl(iv, ijk, _dir, _qid) ;
                    }
                }
            }
        }
    }

    // Z4c follow-up variants — same geometry minus `iv`, call
    // `impose_algebraic_constraintz_z4c` once per (ijk).

    KOKKOS_INLINE_FUNCTION
    void operator() (
        phys_bc_md_face_z4c_tag,
        int i, int j, int iq
    ) const
    {
        #ifdef GRACE_ENABLE_Z4C_METRIC
        if constexpr (extended || bc_kind != element_kind_t::FACE) {
            return ;
        } else {
            int const pdim0  = bnd_pdim(iq, 0) ;
            int const pdim1  = bnd_pdim(iq, 1) ;
            int const npdim0 = bnd_npdim(iq, 0) ;

            int const e_p0 = bnd_extents(iq, pdim0) ;
            int const e_p1 = bnd_extents(iq, pdim1) ;
            if (i >= e_p0 || j >= e_p1) return ;

            int ijk[3] ;
            ijk[pdim0] = bnd_lmin(iq, pdim0) + bnd_idir(iq, pdim0) * i ;
            ijk[pdim1] = bnd_lmin(iq, pdim1) + bnd_idir(iq, pdim1) * j ;

            size_t const _qid = qid(iq) ;
            int const lm_n0 = bnd_lmin(iq, npdim0) ;
            int const id_n0 = bnd_idir(iq, npdim0) ;
            int const e_n0  = bnd_extents(iq, npdim0) ;

            for (int k = 0; k < e_n0; ++k) {
                ijk[npdim0] = lm_n0 + id_n0 * k ;
                impose_algebraic_constraintz_z4c(ijk, _qid) ;
            }
        }
        #endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (
        phys_bc_md_edge_z4c_tag,
        int i, int iq
    ) const
    {
        #ifdef GRACE_ENABLE_Z4C_METRIC
        if constexpr (extended || bc_kind != element_kind_t::EDGE) {
            return ;
        } else {
            int const pdim0  = bnd_pdim(iq, 0) ;
            int const npdim0 = bnd_npdim(iq, 0) ;
            int const npdim1 = bnd_npdim(iq, 1) ;

            int const e_p0 = bnd_extents(iq, pdim0) ;
            if (i >= e_p0) return ;

            int ijk[3] ;
            ijk[pdim0] = bnd_lmin(iq, pdim0) + bnd_idir(iq, pdim0) * i ;

            size_t const _qid = qid(iq) ;
            int const lm_n0 = bnd_lmin(iq, npdim0) ;
            int const id_n0 = bnd_idir(iq, npdim0) ;
            int const e_n0  = bnd_extents(iq, npdim0) ;
            int const lm_n1 = bnd_lmin(iq, npdim1) ;
            int const id_n1 = bnd_idir(iq, npdim1) ;
            int const e_n1  = bnd_extents(iq, npdim1) ;

            for (int jj = 0; jj < e_n0; ++jj) {
                ijk[npdim0] = lm_n0 + id_n0 * jj ;
                for (int kk = 0; kk < e_n1; ++kk) {
                    ijk[npdim1] = lm_n1 + id_n1 * kk ;
                    impose_algebraic_constraintz_z4c(ijk, _qid) ;
                }
            }
        }
        #endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (
        phys_bc_md_corner_z4c_tag,
        int iq
    ) const
    {
        #ifdef GRACE_ENABLE_Z4C_METRIC
        if constexpr (extended || bc_kind != element_kind_t::CORNER) {
            return ;
        } else {
            size_t const _qid = qid(iq) ;

            int const lm0 = bnd_lmin(iq, 0), id0 = bnd_idir(iq, 0), e0 = bnd_extents(iq, 0) ;
            int const lm1 = bnd_lmin(iq, 1), id1 = bnd_idir(iq, 1), e1 = bnd_extents(iq, 1) ;
            int const lm2 = bnd_lmin(iq, 2), id2 = bnd_idir(iq, 2), e2 = bnd_extents(iq, 2) ;

            int ijk[3] ;
            for (int ci = 0; ci < e0; ++ci) {
                ijk[0] = lm0 + id0 * ci ;
                for (int cj = 0; cj < e1; ++cj) {
                    ijk[1] = lm1 + id1 * cj ;
                    for (int ck = 0; ck < e2; ++ck) {
                        ijk[2] = lm2 + id2 * ck ;
                        impose_algebraic_constraintz_z4c(ijk, _qid) ;
                    }
                }
            }
        }
        #endif
    }

} ;

}} /* namespace grace::amr */
// supercalifragilistichespiralidoso! 
#endif /* GRACE_AMR_PHYS_BC_KERNELS_HH */