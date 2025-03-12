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

namespace healpix {

  // TODO: this is currently a bit pathetic. Get rid of the asserts
  // and put meaningful codes here
  enum healpix_err_t { NO_ERROR=0, INTERP_ERROR, REDUCE_ERROR } ;

  struct symm {
    static bool ref_x ;
    static bool ref_y ;
    static bool ref_z ;
  };

  enum distrib_t { CONST=0, SINGLE, SPLIT } ;
  
}
