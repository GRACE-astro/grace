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

#pragma once

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

#include "healpix_sphere.hh"

#include <string>
#include <stdlib.h>

namespace healpix {
  namespace outpututils {

    void InitFile(CCTK_ARGUMENTS, healpix_sphere_t const& slice );
    
    void CreateIter(CCTK_ARGUMENTS, int det_id );
    
    void WriteField(CCTK_ARGUMENTS, CCTK_REAL * data_buffer, size_t gsh, int det_id, std::string const& vname ) ;
  
      } 
}
