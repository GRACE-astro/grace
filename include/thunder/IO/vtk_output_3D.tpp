/**
 * @file vtk_output_3D.tpp
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
 #ifndef THUNDER_IO_VTK_OUTPUT_3D_TPP
 #define THUNDER_IO_VTK_OUTPUT_3D_TPP 

#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>

#include <string> 


namespace thunder { namespace IO {

template< typename ViewT > 
static vtkSmartPointer<vtkDoubleArray> vtk_create_cell_data(ViewT data_view, std::string const& name, bool is_vector=false)
{
    auto data = vtkSmartPointer<vtkDoubleArray>::New() ;
    data->SetNumberOfComponents( 1 + 2*is_vector ) ; 
    data->SetNumberOfTuples( 
        data_view.extent(0) 
      * data_view.extent(1) 
      * data_view.extent(2) 
      * data_view.extent(THUNDER_NSPACEDIM) 
    ) ;
    if( is_vector ) {
        std::string comp_name = name + "[0]" ; 
        data->SetComponentName(0,comp_name.c_str()) ; 
        comp_name = name + "[1]" ;
        data->SetComponentName(1,comp_name.c_str()) ; 
        comp_name = name + "[2]" ;
        data->SetComponentName(2,comp_name.c_str()) ; 
    } else {
        data->SetComponentName(0,name.c_str()) ; 
    }   
    size_t const  nq{data_view.extent(THUNDER_NSPACEDIM) }
                , nz{data_view.extent(2)}
                , ny{data_view.extent(1)}
                , nx{data_view.extent(0)} ; 

    for(size_t iq=0UL; iq<nq ; iq+=1UL) {
        for(size_t iz=0UL; iz<nz; iz+=1UL) {
            for(size_t iy=0UL; iy<ny; iy+=1UL) {
                for(size_t ix=0UL; ix<nx; ix+=1UL) {
                    size_t icell = ix + nx*(iy+ny*(iz+nz*iq)) ; 
                    for( int icomp=0; icomp<1+2*is_vector; icomp++) 
                        data->SetComponent(icell,icomp,data_view(ix,iy,iz,iq,icomp)) ;
                }
            }
        }
    }
    return data ;
}

}} /* namespace thunder::IO */ 

 #endif /* THUNDER_IO_VTK_OUTPUT_3D_TPP */ 