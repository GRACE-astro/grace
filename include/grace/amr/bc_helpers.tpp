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
#include <grace/utils/device/device_vector.hh>


#include <grace/amr/bc_kernels.tpp>

namespace grace { namespace amr {
/**
 * @brief Apply physical boundary conditions on a single variable.
 * \ingroup amr
 * @tparam BCT Boundary condition kernel.
 * @tparam ViewT Type of variable view.
 * @param u View containing a single variable.
 * @param face_info Information on physical boundary faces (quadrant ids and face ids).
 * @param corner_info Information on physical boundary corners.
 * @param edge_info Information on physical boundary edges.
 * NB: Here by physical boundary faces is meant the face of a quadrant that does not have 
 * any neighbor in the grid. 
 * In the context of this routine, physical boundary edges / corners refers to those grid 
 * edges / corners which belong to only \b one quadrant. This means that the edges (resp. corners)
 * of a physical boundary face are not always considered physical boundary edges (resp. corners) since they 
 * may touch  2 (resp. 4) quadrants. The filling of these ghostzones is handled by the "face" loop.
 * In other words, we label a ghostzone outside the grid as a "face" ghostzone when it is outside the 
 * grid in w.r.t one coordinate direction, "edge" if it is outside the grid w.r.t two, and corner if it 
 * is outside the grid w.r.t all directions.
 */
template< typename BCT
        , typename ViewT >
void apply_phys_bc(
    ViewT dst,
    ViewT src,
    int nx, int ny, int nz, int ngz,
    grace::device_vector<grace::amr::grace_phys_bc_info_t>& face_info,
    grace::device_vector<grace::amr::grace_phys_bc_info_t>& corner_info,
    #ifdef GRACE_3D
    grace::device_vector<grace::amr::grace_phys_bc_info_t>& edge_info,
    #endif
    BCT const bc_kernel
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
  parallel_for(
  GRACE_EXECUTION_TAG("AMR", "impose_face_physical_BC"),
  MDRangePolicy<IndexType<int>, grace::default_execution_space, Rank<GRACE_NSPACEDIM>>(
      {VECD(ngz, ngz), 0},
      {VECD(nx + ngz, ny + ngz), n_faces}),
    KOKKOS_LAMBDA(VECD(int const i, int const j), int const iface) {

    // Load face info once
    auto face_info = d_face_info(iface);

    int8_t dir[3] = { face_info.dir_x, face_info.dir_y, face_info.dir_z };
    int64_t iq = face_info.qid;
    int8_t face = face_info.face;
    int faceb2 = face / 2;  // face index divided by 2: 0 for x, 1 for y, 2 for z

    // Compute sweep range in normal direction
    int lmin, lmax, idir;
    auto compute_bounds = [](int face, int ngz, int n, int& lmin, int& lmax, int& idir) {
      if (face % 2 == 0) {  // negative side
        lmin = ngz - 1;
        lmax = -1;
        idir = -1;
      } else {              // positive side
        lmin = n + ngz;
        lmax = n + 2 * ngz;
        idir = +1;
      }
     //test:
    //  if (face % 2 == 0) {  // negative side
    //      lmin = ngz - 1;
    //      lmax = 0;    
    //      idir = -1;
    //  } else {  // positive side
    //      lmin = n + ngz;
    //      lmax = n + ngz + (ngz - 1);
    //      idir = +1;
    //  }
    //  if (face % 2 == 0) {
    //    lmin = ngz - 1;
    //    lmax = ngz - stencil;
    //    idir = -1;
    //  } else {
    //    lmin = n + ngz;
    //    lmax = n + ngz + stencil -1;
    //    idir = +1;
    //  }
    };

    compute_bounds(face, ngz, (faceb2 == 0 ? nx : (faceb2 == 1 ? ny : nz)), lmin, lmax, idir);

    for (int ig = lmin; ig != lmax; ig += idir) {
      // Reconstruct full 3D index depending on face orientation
      int I = (faceb2 == 0) ? ig : i;
      int J = (faceb2 == 1) ? ig : (faceb2 == 0 ? i : j);
      int K = (faceb2 == 2) ? ig : j;

      bool is_edge = false;
      if (faceb2 != 0 && (I < ngz || I >= nx + ngz)) is_edge = true;
      if (faceb2 != 1 && (J < ngz || J >= ny + ngz)) is_edge = true;
      if (faceb2 != 2 && (K < ngz || K >= nz + ngz)) is_edge = true;
   
      if (is_edge) continue;

      bc_kernel.template apply<decltype(dst)>(
        dst, src, VEC(I, J, K), VEC(dir[0], dir[1], dir[2]), iq);
    }
  });

  /* Edges */
  int const n_edges = edge_info.size() ; 
  auto& d_edge_info = edge_info.d_view ;
  
  parallel_for(
  GRACE_EXECUTION_TAG("AMR", "impose_edge_physical_BC"),
  n_edges,
  KOKKOS_LAMBDA(int const iedge) {

    // Load edge metadata once
    auto edge = d_edge_info(iedge);

    // Outward normal direction
    int8_t dir[3] = { edge.dir_x, edge.dir_y, edge.dir_z };
    int64_t iq = edge.qid;

   // int dir_sum = abs(dir[0]) + abs(dir[1]) + abs(dir[2]);
   // if (dir_sum != 2) return; // skip if not exactly edge

    // Loop bounds
    int lmin[3], lmax[3], idir[3];

    // Compute loop bounds per direction
    auto compute_bounds = [](int8_t dir, int ngz, int n, int& lmin, int& lmax, int& idir) {
      if (dir < 0) {
        lmin = ngz - 1;
        lmax = -1;
        idir = -1;
      } else if (dir > 0) {
        lmin = n + ngz;
        lmax = n + 2 * ngz;
        idir = +1;
      } else {
        lmin = ngz;
        lmax = n + ngz;
        idir = +1;
      }
    // test:
      //  if (dir < 0) {
      //    lmin = ngz - 1;
      //    lmax = ngz - ngz;  // = 0
      //    idir = -1;
      //  } else if (dir > 0) {
      //    lmin = n + ngz;
      //    lmax = n + ngz + (ngz - 1);
      //    idir = +1;
      //  } else {
      //    lmin = ngz;
      //    lmax = n + ngz;
      //    idir = +1;
      //  }
        // test:
    //  if (dir < 0) {
    //    lmin = ngz - 1;
    //    lmax = ngz - 2;
    //    idir = -1;
    //  } else if (d > 0) {
    //    lmin = n + ngz;
    //    lmax = n + ngz + 1;
    //    idir = +1;
    //  } else {
    //    lmin = ngz;
    //    lmax = n + ngz - 1;
    //    idir = +1;
    //  }
    };

    compute_bounds(dir[0], ngz, nx, lmin[0], lmax[0], idir[0]);
    compute_bounds(dir[1], ngz, ny, lmin[1], lmax[1], idir[1]);
    compute_bounds(dir[2], ngz, nz, lmin[2], lmax[2], idir[2]);

    // Apply BC across all ghost zones along the edge
    for (int ig = lmin[0]; ig != lmax[0]; ig += idir[0])
    for (int jg = lmin[1]; jg != lmax[1]; jg += idir[1])
    for (int kg = lmin[2]; kg != lmax[2]; kg += idir[2]) {

      //   // Exclude corners (i.e., all directions outside grid)
      bool i_at_edge = (dir[0] != 0) && (ig < ngz || ig >= nx + ngz);
      bool j_at_edge = (dir[1] != 0) && (jg < ngz || jg >= ny + ngz);
      bool k_at_edge = (dir[2] != 0) && (kg < ngz || kg >= nz + ngz);
           // Count how many directions lie on ghost boundaries
      bool is_corner = i_at_edge && j_at_edge && k_at_edge;
      if (is_corner) continue;
    
      bc_kernel.template apply<decltype(dst)>(
        dst, src, VEC(ig, jg, kg), VEC(dir[0], dir[1], dir[2]), iq);
    }
  });
  
  /* Corners */
  int const n_corner  = corner_info.size() ; 
  auto& d_corner_info = corner_info.d_view ;

  parallel_for(
  GRACE_EXECUTION_TAG("AMR", "impose_corner_physical_BC"),
  n_corner,
  KOKKOS_LAMBDA(int const icorner) {
    auto corner = d_corner_info(icorner);

   // int8_t dir[GRACE_NSPACEDIM] = { VEC(corner.dir_x, corner.dir_y, corner.dir_z) };
    int8_t dir[GRACE_NSPACEDIM] = { corner.dir_x, corner.dir_y, corner.dir_z };
    int64_t iq = corner.qid;

    //int dir_sum = abs(dir[0]) + abs(dir[1]) + abs(dir[2]);
    ////if (dir_sum == 3) {
    ////  GRACE_ASSERT(dir[0] != 0 && dir[1] != 0 && dir[2] != 0, "Invalid corner direction");
    //// }
    //if (dir_sum != 3) return; // skip if not full 3D corner


    int lmin[GRACE_NSPACEDIM], lmax[GRACE_NSPACEDIM], idir[GRACE_NSPACEDIM];

    auto compute_bounds = [](int8_t dir, int ngz, int n, int& lmin, int& lmax, int& idir) {
      if (dir < 0) {
        lmin = ngz - 1; lmax = -1; idir = -1;
      } else if (dir > 0) {
        lmin = n + ngz; lmax = n + 2 * ngz; idir = +1;
      } else {
        // This should not happen!
        //GRACE_ASSERT(dir != 0, "Invalid zero direction in corner ghostzone BC");
        printf("Problems! \n") ; 
      }
     // test
    //  if (dir < 0) {
    //     lmin = ngz - 1;
    //     lmax = ngz - ngz;  // = 0
    //     idir = -1;
    //   } else if (dir > 0) {
    //     lmin = n + ngz;
    //     lmax = n + ngz + (ngz - 1);
    //     idir = +1;
    //   } else {
    //     printf("Problems! \n") ; 
    //   }
       // test
   //   if (dir < 0) {
   //     lmin = ngz - 1;
   //     lmax = ngz - 1;
   //     idir = -1;
   //   } else if (d > 0) {
   //     lmin = n + ngz;
   //     lmax = n + ngz;
   //     idir = +1;
   //   } else {
  //          printf("Problems! \n") ; 
   //     lmin = -1; lmax = -1; idir = 0;
   //   }
   //test
     //  for(int d =0; d<3; d++){
     //      if (dir[d] == -1) {
     //          lmin[d] = ngz - stencil;
     //          lmax[d] = ngz;
     //      } else if (dir[d] == +1) {
     //          lmin[d] = n + ngz;
     //          lmax[d] = n + ngz + stencil;
     //      } else {
     //          printf("Problems! \n") ; 
     //      }
     //  }
    };

    compute_bounds(dir[0], ngz, nx, lmin[0], lmax[0], idir[0]);
    compute_bounds(dir[1], ngz, ny, lmin[1], lmax[1], idir[1]);
    #ifdef GRACE_3D
    compute_bounds(dir[2], ngz, nz, lmin[2], lmax[2], idir[2]);
    #endif
    
    //printf("[CORNER DEBUG] icorner=%d, dir=(%d,%d,%d), iq=%ld\n               x=[%d→%d:%d], y=[%d→%d:%d], z=[%d→%d:%d]\n",
    //        icorner, dir[0], dir[1], dir[2], iq, lmin[0], lmax[0], idir[0], lmin[1], lmax[1], idir[1], lmin[2], lmax[2], idir[2]);
    
    EXPR(for (int ig = lmin[0]; ig != lmax[0]; ig += idir[0]),
    for (int jg = lmin[1]; jg != lmax[1]; jg += idir[1]),
    for (int kg = lmin[2]; kg != lmax[2]; kg += idir[2]))
    {
      bc_kernel.template apply<decltype(dst)>(
        dst, src, VEC(ig, jg, kg), VEC(dir[0], dir[1], dir[2]), iq);
    }
  });
}

}}

#endif /* GRACE_AMR_BC_HELPERS_TPP */