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
    Kokkos::vector<int64_t>& face_info ) 
{     
  using namespace grace ;
  using namespace Kokkos  ; 

  size_t const nq = face_info.size() ; 
  auto& d_face_info = face_info.d_view ; 

  size_t nx,ny,nz ; 
  std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
  int64_t ngz = amr::get_n_ghosts() ;
  
  TeamPolicy<default_execution_space> 
        policy( nq, AUTO() ) ; 
  using member_t = decltype(policy)::member_type ;

  parallel_for(
      GRACE_EXECUTION_TAG("AMR","impose_physical_BC")
    , policy
    , KOKKOS_LAMBDA( const member_t& team )
    {
      int which_face = d_face_info(team.league_rank()) % P4EST_FACES ; 
      int64_t iq     = d_face_info(team.league_rank()) / P4EST_FACES ; 

      int n0 = (which_face/2==0) * nx + ((which_face/2==1) * ny) + ((which_face/2==2) * nz) ;
      int n1 = (which_face/2==0) * ny + ((which_face/2==1) * nx) + ((which_face/2==2) * nx) ;
      int n2 = (which_face/2==0) * nz + ((which_face/2==1) * nz) + ((which_face/2==2) * ny) ;
      #ifdef GRACE_3D 
      TeamThreadMDRange<Rank<2>,member_t>
                team_range( team, n1,n2) ; 
      #else 
      auto team_range = TeamThreadRange(team,0,n1); 
      #endif 
      parallel_for( team_range 
                  , KOKKOS_LAMBDA (VECD(int& j, int& k))
                  {
                    BCT::apply(u,ngz,n0,VECD(j,k),which_face,iq) ; 
                  }) ; 
    }
  ); 

}

}}

#endif /* GRACE_AMR_BC_HELPERS_TPP */