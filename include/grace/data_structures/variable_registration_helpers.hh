/**
 * @file variable_registration_helpers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief A few macros to help with variable registration. Kept here because they're ugly
 * @date 2024-06-06
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#include <grace_config.h>

#ifndef GRACE_DATA_STRUCTURES_VARIABLE_REGISTRATION_HELPERS_HH
#define GRACE_DATA_STRUCTURES_VARIABLE_REGISTRATION_HELPERS_HH

/********************************************************************************/
/* Utility macros                                                               */
/********************************************************************************/
/**
 * @brief Register (FV) evolved (cell-center) scalar variable.
 * @param idx  Variable index 
 * @param name Variable name 
 * @param bc   Variable BC type
 * @param is_hrsc Is it evolved with FV scheme?
*/
#define REGISTER_EVOLVED_SCALAR(idx,name,bc, is_hrsc) \
idx = register_variable( name \
                        , {VEC(false,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , false ) 

/**
 * @brief Register (FV) evolved (cell-vertex) scalar variable.
 * @param idx  Variable index 
 * @param name Variable name 
 * @param bc   Variable BC type
 * @param is_hrsc Is it evolved with FV scheme?
*/
#define REGISTER_EVOLVED_CORNER_SCALAR(idx,name,bc, is_hrsc) \
idx = register_variable( name \
                        , {VEC(true,true,true)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , false ) 
/**
 * @brief Register (FV) evolved (cell-centered) vector variable.
 * @param idx0  Variable index (X component)
 * @param idx1  Variable index (Y component)
 * @param idx2  Variable index (Z component)
 * @param name  Variable name 
 * @param bc    Variable BC type
 * @param is_hrsc Is it evolved with FV scheme?
 */
#define REGISTER_EVOLVED_VECTOR(idx0, idx1, idx2, name,bc, is_hrsc) \
idx0 = register_variable( name "[0]"\
                        , {VEC(false,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , true \
                        , false \
                        , 0 \
                        , name ) ; \
idx1 = register_variable( name "[1]"\
                        , {VEC(false,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , true \
                        , false \
                        , 1 \
                        , name ) ; \
idx2 = register_variable( name "[2]"\
                        , {VEC(false,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , true \
                        , false \
                        , 2 \
                        , name ) 

/**
 * @brief Register evolved symmetric tensor.
 * @param idx0  Variable index (XX component)
 * @param idx1  Variable index (XY component)
 * @param idx2  Variable index (XZ component)
 * @param idx3  Variable index (YY component)
 * @param idx4  Variable index (YZ component)
 * @param idx5  Variable index (ZZ component)
 * @param name  Variable name 
 * @param bc    Variable BC type
 * @param is_hrsc Is it evolved with FV scheme?
 */
#define REGISTER_EVOLVED_TENSOR(idx0, idx1, idx2, idx3, idx4, idx5, name,bc, is_hrsc) \
idx0 = register_variable( name "[0,0]"\
                        , {VEC(false,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , false \
                        , true \
                        , 0  \
                        , name ) ;\
idx1 = register_variable( name "[0,1]"\
                        , {VEC(false,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , false\
                        , true\
                        , 1 \
                        , name ) ; \
idx2 = register_variable( name "[0,2]"\
                        , {VEC(false,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , false \
                        , true \
                        , 2  \
                        , name )  ; \
idx3 = register_variable( name "[1,1]"\
                        , {VEC(false,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , false \
                        , true \
                        , 3  \
                        , name )  ; \
idx4 = register_variable( name "[1,2]"\
                        , {VEC(false,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , false \
                        , true \
                        , 4  \
                        , name )  ; \
idx5 = register_variable( name "[2,2]"\
                        , {VEC(false,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , false \
                        , true \
                        , 5 \
                        , name )
/**
 * @brief Register evolved vector with face staggering
 * @param idx0  Variable index (X component)
 * @param idx1  Variable index (Y component)
 * @param idx2  Variable index (Z component)
 * @param name  Variable name 
 * @param bc    Variable BC type
 */
#define REGISTER_EVOLVED_FACE_STAGGERED_VECTOR(idx0, idx1, idx2,name,bc,is_hrsc) \
idx0 = register_variable( name "[0]"\
                        , {VEC(true,false,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , true \
                        , false \
                        , 0 \
                        , name ) ; \
idx1 = register_variable( name "[1]"\
                        , {VEC(false,true,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , true \
                        , false \
                        , 1 \
                        , name ) ; \
idx2 = register_variable( name "[2]"\
                        , {VEC(false,false,true)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , true \
                        , false \
                        , 2 \
                        , name ) 
/**
 * @brief Register evolved vector with edge staggering
 * @param idx0  Variable index (X component)
 * @param idx1  Variable index (Y component)
 * @param idx2  Variable index (Z component)
 * @param name  Variable name 
 * @param bc    Variable BC type
 */
#define REGISTER_EVOLVED_EDGE_STAGGERED_VECTOR(idx0, idx1, idx2, idx3,name,bc,is_hrsc) \
idx0 = register_variable( name "[0]"\
                        , {VEC(false,true,true)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , true \
                        , false \
                        , 0 \
                        , name ) ; \
idx1 = register_variable( name "[1]"\
                        , {VEC(true,false,true)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , true \
                        , false \
                        , 1 \
                        , name ) ; \
idx2 = register_variable( name "[2]"\
                        , {VEC(true,true,false)} \
                        , true \
                        , is_hrsc \
                        , bc \
                        , true \
                        , false \
                        , 2 \
                        , name ) 
/**
 * @brief Register auxiliary scalar.
 * @param idx  Variable index 
 * @param name Variable name 
 * @param bc   Variable BC type
*/
#define REGISTER_AUX_SCALAR(idx,name,bc) \
idx = register_variable( name \
                        , {VEC(false,false,false)} \
                        , false \
                        , false \
                        , bc \
                        , false ) 
/**
 * @brief Register auxiliary vector.
 * @param idx0  Variable index (X component)
 * @param idx1  Variable index (Y component)
 * @param idx2  Variable index (Z component)
 * @param name  Variable name 
 * @param bc    Variable BC type
 */
#define REGISTER_AUX_VECTOR(idx0, idx1, idx2, name,bc) \
idx0 = register_variable( name "[0]"\
                        , {VEC(false,false,false)} \
                        , false \
                        , false \
                        , bc \
                        , true \
                        , false \
                        , 0 \
                        , name ) ; \
idx1 = register_variable( name "[1]"\
                        , {VEC(false,false,false)} \
                        , false \
                        , false \
                        , bc \
                        , true \
                        , false \
                        , 1 \
                        , name ) ; \
idx2 = register_variable( name "[2]"\
                        , {VEC(false,false,false)} \
                        , false \
                        , false \
                        , bc \
                        , true \
                        , false \
                        , 2 \
                        , name ) 
/**
 * @brief Register auxiliary symmetric tensor.
 * @param idx0  Variable index (XX component)
 * @param idx1  Variable index (XY component)
 * @param idx2  Variable index (XZ component)
 * @param idx3  Variable index (YY component)
 * @param idx4  Variable index (YZ component)
 * @param idx5  Variable index (ZZ component)
 * @param name  Variable name 
 * @param bc    Variable BC type
 */
#define REGISTER_AUX_TENSOR(idx0, idx1, idx2, idx3, idx4, idx5, name,bc) \
idx0 = register_variable( name "[0,0]"\
                        , {VEC(false,false,false)} \
                        , false \
                        , false \
                        , bc \
                        , false \
                        , true \
                        , 0  \
                        , name ) ;\
idx1 = register_variable( name "[0,1]"\
                        , {VEC(false,false,false)} \
                        , false \
                        , false \
                        , bc \
                        , false\
                        , true\
                        , 1 \
                        , name ) ; \
idx2 = register_variable( name "[0,2]"\
                        , {VEC(false,false,false)} \
                        , false \
                        , false \
                        , bc \
                        , false \
                        , true \
                        , 2  \
                        , name )  ; \
idx3 = register_variable( name "[1,1]"\
                        , {VEC(false,false,false)} \
                        , false \
                        , false \
                        , bc \
                        , false \
                        , true \
                        , 3  \
                        , name )  ; \
idx4 = register_variable( name "[1,2]"\
                        , {VEC(false,false,false)} \
                        , false \
                        , false \
                        , bc \
                        , false \
                        , true \
                        , 4  \
                        , name )  ; \
idx5 = register_variable( name "[2,2]"\
                        , {VEC(false,false,false)} \
                        , false \
                        , false \
                        , bc \
                        , false \
                        , true \
                        , 5 \
                        , name )  

#endif /* GRACE_DATA_STRUCTURES_VARIABLE_REGISTRATION_HELPERS_HH */