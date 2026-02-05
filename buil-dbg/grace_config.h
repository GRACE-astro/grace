/**
 * @file grace_config.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-03-13
 * 
 * @copyright This file is part of grace.
 * grace is an evolution framework that uses Finite Difference
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
#ifndef GRACE_CONFIG_HEADER_H_FB650CE9_1908_47F9_98EB_3C4E4087C9CC
#define GRACE_CONFIG_HEADER_H_FB650CE9_1908_47F9_98EB_3C4E4087C9CC

#include <stdlib.h>


namespace grace {

//*****************************************************************************************************
//*****************************************************************************************************

#define GRACE_NSPACEDIM 3 
#define GRACE_3D 

#define GRACE_ENABLE_CUDA 
/* #undef GRACE_ENABLE_HIP */
/* #undef GRACE_ENABLE_OMP */
/* #undef GRACE_ENABLE_SERIAL */

#define GRACE_CARTESIAN_COORDINATES
/* #undef GRACE_SPHERICAL_COORDINATES */

#define GRACE_ENABLE_GRMHD
/* #undef GRACE_ENABLE_BURGERS */
/* #undef GRACE_ENABLE_SCALAR_ADV */

/* #undef GRACE_ENABLE_PROFILING */

/* #undef GRACE_ENABLE_VTK */

#define GRACE_FREEZE_HYDRO

/* #undef GRACE_ENABLE_LORENE */

/* #undef GRACE_ENABLE_TWO_PUNCTURES */

#define GRACE_BANNER                                                          \
"                                                                           \n" \
"                                                                           \n" \
"                                                                           \n" \
"   _____ _____            _____ ______                                     \n" \
"  / ____|  __ \\     /\\   / ____|  ____|                                  \n" \
" | |  __| |__) |   /  \\ | |    | |__                                      \n" \
" | | |_ |  _  /   / /\\ \\| |    |  __|                                    \n" \
" | |__| | | \\ \\  / ____ \\ |____| |____                                  \n" \
"  \\_____|_|  \\_\\/_/    \\_\\_____|______|                               \n" \
"                                                                           \n" \
"                                                                           \n" \
"                                                                           \n" \
"                                                                           \n" \
"This is version 0.5\n" \
"of the General Relativistic Astrophysics Code for Exascale.                \n" \
"GRACE is an evolution framework that uses Finite Volume\n"                     \
"methods to simulate relativistic spacetimes and plasmas\n"                     \
"Copyright (C) 2023 Carlo Musolino\n"                                           \
"\n"                                                                            \
"This program is free software: you can redistribute it and/or modify\n"        \
"it under the terms of the GNU General Public License as published by\n"        \
"the Free Software Foundation, either version 3 of the License, or\n"           \
"any later version.\n"                                                          \
"\n"                                                                            \
"This program is distributed in the hope that it will be useful,\n"             \
"but WITHOUT ANY WARRANTY; without even the implied warranty of\n"              \
"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"               \
"GNU General Public License for more details.\n"                                \
"\n"                                                                            \
"You should have received a copy of the GNU General Public License\n"           \
"along with this program.  If not, see <https://www.gnu.org/licenses/>.\n\n"    \


#ifdef GRACE_3D 
#define GRACE_FACE_CHILDREN 4 
#else 
#define GRACE_FACE_CHILDREN 2 
#endif 

#ifdef GRACE_ENABLE_CUDA 
#define GRACE_BACKEND "CUDA"
#elif defined(GRACE_ENABLE_HIP)
#define GRACE_BACKEND "HIP"
#else 
#define GRACE_BACKEND "HOST"
#endif 

/**
 * @brief Macros designed to help with writing dimension independent code.
 * \ingroup utils 
 * 
 */
#ifdef GRACE_3D 
#define VEC(X,Y,Z) X,Y,Z
#define EXPR(X,Y,Z) X Y Z
#define PICK_D(X,Y) Y 
#define VECD(X,Y) X,Y
#define EXPRD(X,Y) X Y 
#else 
#define VEC(X,Y,Z) X,Y
#define EXPR(X,Y,Z) X Y
#define PICK_D(X,Y) X 
#define VECD(X,Y) X
#define EXPRD(X,Y) X
#endif 

/* #undef GRACE_ENABLE_COWLING_METRIC */
#define GRACE_ENABLE_Z4C_METRIC

/* #undef GRACE_ENABLE_M1 */

#define Z4C_DER_ORDER 4

// for spheres
#define LAGRANGE_INTERP_ORDER 4

#define SQR(a) ((a)*(a))

}
#endif /* FB650CE9_1908_47F9_98EB_3C4E4087C9CC */
