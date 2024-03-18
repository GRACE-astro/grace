/**
 * @file vtk_output_2D.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-18
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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
#ifndef THUNDER_IO_VTK_OUTPUT_2D_TPP
#define THUNDER_IO_VTK_OUTPUT_2D_TPP 

#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>

#include <string> 

namespace thunder { namespace IO {

template< typename ViewT > 
static vtkSmartPointer<vtkDoubleArray> vtk_create_cell_data(ViewT data_view, std::string const& name)
{
    constexpr bool is_vector  {ViewT::Rank == 4} ; 
    auto data = vtkSmartPointer<vtkDoubleArray>::New() ;
     
    data->SetNumberOfTuples( 
        data_view.extent(0) 
      * data_view.extent(1) 
      * data_view.extent(2) 
    ) ;

    if constexpr( !is_vector ) {
        data->SetNumberOfComponents(1) ;
        data->SetName(name.c_str()) ; 
        
    } else {
        data->SetNumberOfComponents(3) ;
        std::string comp_name = name + "[0]" ; 
        data->SetComponentName(0,comp_name.c_str()) ; 
        comp_name = name + "[1]" ;
        data->SetComponentName(1,comp_name.c_str()) ; 
        comp_name = name + "[2]" ;
        data->SetComponentName(2,comp_name.c_str()) ; 
    }   
    size_t const  nq{data_view.extent(2) }
                , ny{data_view.extent(1)}
                , nx{data_view.extent(0)} ; 

    for(size_t iq=0UL; iq<nq ; iq+=1UL) {
        for(size_t iy=0UL; iy<ny; iy+=1UL) {
            for(size_t ix=0UL; ix<nx; ix+=1UL) {
                size_t icell = ix + nx*(iy+ny*iq) ; 
                if constexpr ( is_vector ) 
                {
                for( int icomp=0; icomp<3; icomp++) 
                    data->SetComponent(icell,icomp,data_view(ix,iy,iq,icomp)) ;
                } else {
                    data->SetComponent(icell,0,data_view(ix,iy,iq)) ; 
                }
            }
        }
    }
    return data ;
}

}} /* namespace thunder::IO */ 

 #endif /* THUNDER_IO_VTK_OUTPUT_2D_TPP */ 