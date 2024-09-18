/**
 * @file bc_helpers.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-21
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

#ifndef GRACE_AMR_BC_HELPERS_TPP
#define GRACE_AMR_BC_HELPERS_TPP
#include <grace_config.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

#include <grace/amr/grace_amr.hh> 
#include <grace/utils/grace_utils.hh>
#include <grace/utils/device_vector.hh>
#include <grace/data_structures/macros.hh>

#include <grace/amr/bc_kernels.tpp>

namespace grace { namespace amr {
/**
 * @brief Apply physical boundary conditions on a single variable.
 * \ingroup amr
 * @tparam BCT Boundary condition kernel.
 * @tparam ViewT Type of variable view.
 * @param u View containing a single variable.
 * @param face_info Information on physical boundary faces (quadrant ids and face ids).
 */
template< typename BCT
        , typename ViewT >
void apply_phys_bc(
    ViewT& u,
    int nx,
    int ny,
    int nz,
    int ngz,
    grace::device_vector<grace::amr::grace_phys_bc_info_t>& face_info
  )
{     
  /******************************************************/
  /* This function applies physical boundary conditions */
  /* to a single variable. Note that the loop in the    */
  /* ghostzone direction(s) is (are) always serial.     */
  /* This is because in general there might be loop     */
  /* carry dependencies. Also note that we do first     */
  /* faces then edges and corners, this is also because */
  /* there might be dependencies.                       */
  /******************************************************/

  using namespace grace ;
  using namespace Kokkos  ; 

  /* Faces */
  int const n_faces = face_info.size() ; 
  auto& d_face_info = face_info.d_view ;

  /******************************************************/
  /* In this case we have a heftier 2D loop over        */
  /* the whole cell range so we do that in parallel     */
  /* below in the loop over edges everything is serial  */
  /* for convenience                                    */
  /******************************************************/
  MDRangePolicy< IndexType<grace_loop_index_t>
               , grace::default_execution_space 
               , Rank<GRACE_NSPACEDIM> > 
    policy( {VECD(0,0), 0}, {VECD(nx+2*ngz,ny+2*ngz), n_faces}) ;
  parallel_for(
    GRACE_EXECUTION_TAG("AMR","impose_face_physical_BC"),
    policy,
    KOKKOS_LAMBDA( VECD(int const i, int const j), int const iface ) 
    {
        EXPR(
        int8_t const dx = d_face_info(iface).dir_x ;, 
        int8_t const dy = d_face_info(iface).dir_y ;, 
        int8_t const dz = d_face_info(iface).dir_z ;
        )
        int64_t iq     = d_face_info(iface).qid    ;
        int8_t face    = d_face_info(iface).face   ;

        int const lmin = (face%2==0) * (ngz-1) + (face%2==1) * (nx+ngz) ; 
        int const lmax = (face%2==0) * (-1)    + (face%2==1) * (nx+2*ngz) ;
        int const idir = (face%2==0) * (-1)    + (face%2==1) * (+1) ;  

        int const faceb2{ face / 2 } ; 

        #pragma unroll  
        for( int ig=lmin; ig!=lmax; ig+=idir ) {
          int const I = (faceb2==0) * ig + (faceb2!=0) * j ; 
          int const J = (faceb2==1) * ig + (faceb2==0) * j + (faceb2==2) * k ; 
          int const K = (faceb2==2) * ig + (faceb2!=2) * k ;  
          BCT::apply(u,VEC(I,J,K), VEC(dx,dy,dz), iq) ;
        }
    }
  ); 

  /* Edges */
  int const n_edges = edge_info.size() ; 
  auto& d_edge_info = edge_info.d_view ;
  
  parallel_for(
    GRACE_EXECUTION_TAG("AMR","impose_edge_physical_BC"),
    n_edges,
    KOKKOS_LAMBDA( int const iedge ) 
    {
        int const lbnd[3] = {
          nx + ngz,
          ny + ngz,
          nz + ngz 
        } ;
        int const ubnd[3] = {
          nx + 2*ngz,
          ny + 2*ngz,
          nz + 2*ngz 
        } ; 

        int dir[3] = {
          d_edge_info(iedge).dir_x,
          d_edge_info(iedge).dir_y, 
          d_edge_info(iedge).dir_z 
        } ; 
        
        int lmin[3], lmax[3], idir[3] ;
        for( int dd=0; dd<3; ++dd) {
          if (dir[dd] < 0) {
            lmin[dd] = ngz-1 ; 
            lmax[dd] = -1    ; // the loop goes til ig != lmax 
            idir[dd] = -1  ; 
          } else if ( dir[dd] > 0 ) {
            lmin[dd] = lbnd[dd]   ;
            lmax[dd] = ubnd[dd]   ;
            idir[dd] = +1 ;
          } else {
            lmin[dd] = 0          ;
            lmax[dd] = ubnd[dd]   ;
            idir[dd] = +1 ;
          }
        }     
        int64_t iq     = d_edge_info(iedge).qid    ;
        #pragma unroll 
        for( int ig=lmin[0]; ig!=lmax[0]; ig+=idir[0] ) 
        for( int jg=lmin[1]; jg!=lmax[1]; jg+=idir[1] ) 
        for( int kg=lmin[2]; kg!=lmax[2]; kg+=idir[2] )
        { 
          BCT::apply(u,VEC(ig,jg,kg), VEC(dx,dy,dz), iq) ;
        }
    }
  ) ; 
  
  /* Corners */
  int const n_corner  = corner_info.size() ; 
  auto& d_corner_info = corner_info.d_view ;
;
  parallel_for(
    GRACE_EXECUTION_TAG("AMR","impose_corner_physical_BC"),
    n_corners,
    KOKKOS_LAMBDA( int const icorner ) 
    {

      int const lbnd[3] = {
        nx + ngz,
        ny + ngz,
        nz + ngz 
      } ;
      int const ubnd[3] = {
        nx + 2*ngz,
        ny + 2*ngz,
        nz + 2*ngz 
      } ; 

      int dir[3] = {
        d_corner_info(icorner).dir_x,
        d_corner_info(icorner).dir_y, 
        d_corner_info(icorner).dir_z 
      } ; 
      
      int lmin[GRACE_NSPACEDIM], lmax[GRACE_NSPACEDIM], idir[GRACE_NSPACEDIM] ;
      for( int dd=0; dd<GRACE_NSPACEDIM; ++dd) {
        if (dir[dd] < 0) {
          lmin[dd] = ngz-1 ; 
          lmax[dd] = -1    ; // the loop goes til ig != lmax 
          idir[dd] = -1  ; 
        } else if ( dir[dd] > 0 ) {
          lmin[dd] = lbnd[dd]   ;
          lmax[dd] = ubnd[dd]   ;
          idir[dd] = +1 ;
        } else {
          lmin[dd] = 0          ;
          lmax[dd] = ubnd[dd]   ;
          idir[dd] = +1 ;
        }
      } 
        
      int64_t iq     = d_corner_info(icorner).qid       ;

      #pragma unroll
      EXPR( for( int ig=lmin[0]; ig!=lmax[0]; ig+=idir[0]), 
            for( int jg=lmin[1]; jg!=lmax[1]; jg+=idir[1]), 
            for( int kg=lmin[2]; kg!=lmax[2]; kg+=idir[2])) 
      {
        BCT::apply(u,VEC(ig,jg,kg), VEC(dx,dy,dz), iq) ;
      }
    }
  ); 
}

}}

#endif /* GRACE_AMR_BC_HELPERS_TPP */