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
#ifndef THUNDER_IO_VTK_OUTPUT_TPP
#define THUNDER_IO_VTK_OUTPUT_TPP 

#include <thunder_config.h>

#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>

#include <string> 

namespace thunder { namespace IO {

template< typename ViewT > 
static vtkSmartPointer<vtkDoubleArray> 
vtk_create_cell_data( VEC(int nx, int ny, int nz), int nq
                    , ViewT data_view, std::string const& name )
{
    auto data = vtkSmartPointer<vtkDoubleArray>::New() ;
     
    data->SetNumberOfTuples( 
        EXPR(   nx 
            , * ny 
            , * nz ) * nq 
    ) ;

    data->SetNumberOfComponents(1) ;
    data->SetName(name.c_str()) ; 

    for(size_t iq=0UL; iq<nq ; iq+=1UL) {
        #ifdef THUNDER_3D 
        for(size_t iz=0UL; iz<nz; iz+=1UL) {
        #endif 
            for(size_t iy=0UL; iy<ny; iy+=1UL) {
                for(size_t ix=0UL; ix<nx; ix+=1UL) {
                    #ifdef THUNDER_3D 
                    size_t icell = ix + nx*(iy+ny*(iz+nz*iq)) ;
                    #else 
                    size_t icell = ix + nx*(iy+ny*iq) ; 
                    #endif 
                    data->SetComponent(icell,0,data_view(VEC(ix,iy,iz),iq)) ; 
                }
            }
        #ifdef THUNDER_3D
        }
        #endif 
    }
    return data ;
}

template< typename ViewT > 
static vtkSmartPointer<vtkDoubleArray> 
vtk_create_vector_cell_data( VEC(int nx, int ny, int nz), int nq
                           , ViewT data_view, std::string const& name )
{
    auto data = vtkSmartPointer<vtkDoubleArray>::New() ;
     
    data->SetNumberOfTuples( 
        EXPR(   nx 
            , * ny 
            , * nz ) 
        * nq * THUNDER_NSPACEDIM
    ) ;

    data->SetNumberOfComponents(3) ;
    data->SetName(name.c_str()) ; 

    for(size_t iq=0UL; iq<nq ; iq+=1UL) {
        #ifdef THUNDER_3D 
        for(size_t iz=0UL; iz<nz; iz+=1UL) {
        #endif 
            for(size_t iy=0UL; iy<ny; iy+=1UL) {
                for(size_t ix=0UL; ix<nx; ix+=1UL) {
                    #ifdef THUNDER_3D 
                    size_t icell = ix + nx*(iy+ny*(iz+nz*iq)) ;
                    #else 
                    size_t icell = ix + nx*(iy+ny*iq) ; 
                    #endif 
                    for( int icomp=0; icomp<THUNDER_NSPACEDIM; icomp++) 
                        data->SetComponent(icell,0,data_view(VEC(ix,iy,iz),icomp,iq)) ; 
                }
            }
        #ifdef THUNDER_3D
        }
        #endif 
    }
    return data ;
}

}} /* namespace thunder::IO */ 

 #endif /* THUNDER_IO_VTK_OUTPUT_2D_TPP */ 