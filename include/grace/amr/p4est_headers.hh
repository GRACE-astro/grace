/**
 * @file p4est_headers.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Single dispatch header pulling in the correct p4est / p8est umbrella includes depending on GRACE_3D.
 *
 * @copyright This file is part of GRACE.
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
 */
#ifndef AEC986E5_311F_4FDF_9E46_3ABDF632C6FD
#define AEC986E5_311F_4FDF_9E46_3ABDF632C6FD

#include <sc.h>

#ifdef GRACE_3D
#include <p4est_to_p8est.h>
#include <p8est.h>
#include <p8est_extended.h>
#include <p8est_connectivity.h> 
#include <p8est_communication.h>
#include <p8est_bits.h>
#include <p8est_algorithms.h>
#include <p8est_balance.h>
#include <p8est_ghost.h>
#include <p8est_iterate.h>
#include <p8est_search.h>
#else
#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_connectivity.h> 
#include <p4est_communication.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>
#include <p4est_base.h>
#include <p4est_balance.h>
#include <p4est_ghost.h>
#include <p4est_iterate.h>
#include <p4est_search.h>
#endif 

#endif /* AEC986E5_311F_4FDF_9E46_3ABDF632C6FD */
