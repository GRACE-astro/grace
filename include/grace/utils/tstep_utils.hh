/**
 * @file tstep_utils.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-10-14
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

#ifndef GRACE_UTILS_TSTEP_UTILS_HH
#define GRACE_UTILS_TSTEP_UTILS_HH

#include <grace_config.h>

#include <grace/data_structures/variable_properties.hh>
#include <grace/amr/amr_functions.hh>

#include <Kokkos_Core.hpp>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

namespace grace {
/**
 * @brief Linear operator on arrays (staggered and not staggered) 
 * @param[out] A Output, overwritten 
 * @param[in] B Input
 * @param[out] As Output, overwritten 
 * @param[in] Bs Input 
 * @param[in] b Coefficient 
 * Computes 
 * \[
 *  A = b B
 * \] 
 */
static void linop_apply(
    var_array_t A, var_array_t B,
    staggered_variable_arrays_t AS, staggered_variable_arrays_t BS, 
    double b
) 
{
    DECLARE_GRID_EXTENTS ;
    using namespace grace ;
    using namespace Kokkos ;

    auto nvars_face  = AS.face_staggered_fields_x.extent(GRACE_NSPACEDIM) ;
    auto nvars_cc  = A.extent(GRACE_NSPACEDIM) ;

    auto update_policy =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
                {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nvars_cc, nq}
        ) ;
    auto staggered_update_policy_x =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
            {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz),nvars_face, nq}
        ) ;
    auto staggered_update_policy_y =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
            {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz),nvars_face, nq}
        ) ;
    auto staggered_update_policy_z =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
            {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz+1),nvars_face, nq}
        ) ;
    
    Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
            , update_policy
            , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
            {
                A(VEC(i,j,k), ivar, q)
                    = b * B(VEC(i,j,k), ivar, q);
            }
        ) ;

    parallel_for(
    GRACE_EXECUTION_TAG("EVOL","substep_apply")
        , staggered_update_policy_x
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
        {
            AS.face_staggered_fields_x(VEC(i,j,k), ivar, q)
                = b * BS.face_staggered_fields_x(VEC(i,j,k), ivar, q);
        }
    ) ;
    
    parallel_for(
    GRACE_EXECUTION_TAG("EVOL","substep_apply")
        , staggered_update_policy_y
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
        {
            AS.face_staggered_fields_y(VEC(i,j,k), ivar, q)
                = b * BS.face_staggered_fields_y(VEC(i,j,k), ivar, q);
        }
    ) ;
    parallel_for(
    GRACE_EXECUTION_TAG("EVOL","substep_apply")
        , staggered_update_policy_z
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
        {
            AS.face_staggered_fields_z(VEC(i,j,k), ivar, q)
                = b * BS.face_staggered_fields_z(VEC(i,j,k), ivar, q);
        }
    ) ;
}
/**
 * @brief Linear operator on arrays 
 * @param[out] A Output, overwritten 
 * @param[in] B Input 
 * @param[in] C Input 
 * @param[in] b Coefficient 
 * @param[in] c Coefficient 
 * Computes 
 * \[
 *  A = b B + c C 
 * \] 
 */
static void linop_apply(
    var_array_t A, var_array_t B, var_array_t C,
    staggered_variable_arrays_t AS, staggered_variable_arrays_t BS, staggered_variable_arrays_t CS, 
    double b, double c
) 
{
    DECLARE_GRID_EXTENTS ;
    using namespace grace ;
    using namespace Kokkos ;

    auto nvars_face  = AS.face_staggered_fields_x.extent(GRACE_NSPACEDIM) ;
    auto nvars_cc  = A.extent(GRACE_NSPACEDIM) ;

    auto update_policy =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
                {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nvars_cc, nq}
        ) ;
    auto staggered_update_policy_x =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
            {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz),nvars_face, nq}
        ) ;
    auto staggered_update_policy_y =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
            {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz),nvars_face, nq}
        ) ;
    auto staggered_update_policy_z =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
            {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz+1),nvars_face, nq}
        ) ;
    
    Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
            , update_policy
            , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
            {
                A(VEC(i,j,k), ivar, q)
                    = b * B(VEC(i,j,k), ivar, q)
                    + c * C(VEC(i,j,k), ivar, q) ; 
            }
        ) ;

    parallel_for(
    GRACE_EXECUTION_TAG("EVOL","substep_apply")
        , staggered_update_policy_x
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
        {
            AS.face_staggered_fields_x(VEC(i,j,k), ivar, q)
                = b * BS.face_staggered_fields_x(VEC(i,j,k), ivar, q)
                + c * CS.face_staggered_fields_x(VEC(i,j,k), ivar, q) ; 
        }
    ) ;
    
    parallel_for(
    GRACE_EXECUTION_TAG("EVOL","substep_apply")
        , staggered_update_policy_y
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
        {
            AS.face_staggered_fields_y(VEC(i,j,k), ivar, q)
                = b * BS.face_staggered_fields_y(VEC(i,j,k), ivar, q)
                + c * CS.face_staggered_fields_y(VEC(i,j,k), ivar, q) ; 
        }
    ) ;
    parallel_for(
    GRACE_EXECUTION_TAG("EVOL","substep_apply")
        , staggered_update_policy_z
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
        {
            AS.face_staggered_fields_z(VEC(i,j,k), ivar, q)
                = b * BS.face_staggered_fields_z(VEC(i,j,k), ivar, q)
                + c * CS.face_staggered_fields_z(VEC(i,j,k), ivar, q) ; 
        }
    ) ;
}

/**
 * @brief Linear operator on arrays 
 * @param[out] A Output, overwritten 
 * @param[in] B Input 
 * @param[in] C Input 
 * @param[in] D Input 
 * @param[in] b Coefficient 
 * @param[in] c Coefficient 
 * @param[in] d Coefficient 
 * Computes 
 * \[
 *  A = b B + c C + d D
 * \] 
 */
static void linop_apply(
    var_array_t A, var_array_t B, var_array_t C, var_array_t D,
    staggered_variable_arrays_t AS, staggered_variable_arrays_t BS, staggered_variable_arrays_t CS, staggered_variable_arrays_t DS, 
    double b, double c, double d
) 
{
    DECLARE_GRID_EXTENTS ;
    using namespace grace ;
    using namespace Kokkos ;

    auto nvars_face  = AS.face_staggered_fields_x.extent(GRACE_NSPACEDIM) ;
    auto nvars_cc  = A.extent(GRACE_NSPACEDIM) ;

    auto update_policy =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
                {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nvars_cc, nq}
        ) ;
    auto staggered_update_policy_x =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
            {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz),nvars_face, nq}
        ) ;
    auto staggered_update_policy_y =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
            {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz),nvars_face, nq}
        ) ;
    auto staggered_update_policy_z =
        MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> (
            {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz+1),nvars_face, nq}
        ) ;
    
    Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
            , update_policy
            , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
            {
                A(VEC(i,j,k), ivar, q)
                    = b * B(VEC(i,j,k), ivar, q)
                    + c * C(VEC(i,j,k), ivar, q) 
                    + d * D(VEC(i,j,k), ivar, q); 
            }
        ) ;

    parallel_for(
    GRACE_EXECUTION_TAG("EVOL","substep_apply")
        , staggered_update_policy_x
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
        {
            AS.face_staggered_fields_x(VEC(i,j,k), ivar, q)
                = b * BS.face_staggered_fields_x(VEC(i,j,k), ivar, q)
                + c * CS.face_staggered_fields_x(VEC(i,j,k), ivar, q) 
                + d * DS.face_staggered_fields_x(VEC(i,j,k), ivar, q) ; 
        }
    ) ;
    
    parallel_for(
    GRACE_EXECUTION_TAG("EVOL","substep_apply")
        , staggered_update_policy_y
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
        {
            AS.face_staggered_fields_y(VEC(i,j,k), ivar, q)
                = b * BS.face_staggered_fields_y(VEC(i,j,k), ivar, q)
                + c * CS.face_staggered_fields_y(VEC(i,j,k), ivar, q) 
                + d * DS.face_staggered_fields_y(VEC(i,j,k), ivar, q) ; 
        }
    ) ;
    parallel_for(
    GRACE_EXECUTION_TAG("EVOL","substep_apply")
        , staggered_update_policy_z
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
        {
            AS.face_staggered_fields_z(VEC(i,j,k), ivar, q)
                = b * BS.face_staggered_fields_z(VEC(i,j,k), ivar, q)
                + c * CS.face_staggered_fields_z(VEC(i,j,k), ivar, q) 
                + d * DS.face_staggered_fields_z(VEC(i,j,k), ivar, q) ; 
        }
    ) ;
}

}

#endif /*GRACE_UTILS_TSTEP_UTILS_HH*/