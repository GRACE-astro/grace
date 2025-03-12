/*
 * Copyright (C) 2022 Carlo Musolino
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; If not, see <http://www.gnu.org/licenses/>.
*/
#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Functions.h"
#include "cctk_Parameters.h"

extern "C" void healpix_paramcheck(CCTK_ARGUMENTS) {

  DECLARE_CCTK_ARGUMENTS ;
  DECLARE_CCTK_PARAMETERS ;

  if ( integrate_every <= 0 )
    CCTK_INFO("Integration is never scheduled to happend integrate_every <= 0") ;

  if ( output_every <= 0 )
    CCTK_INFO("Output is never scheduled to happend output_every <= 0") ;

  if ( ! num_slices )
    CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
	       "No slices requested!");

  return ;
  
}
