/**
 * @file vtk_volume_output_2D.cpp
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


namespace thunder{ namespace IO { namespace detail {

 

vtkSmartPointer<vtkUnstructuredGrid> 
setup_vtk_grid_cartesian()
{
    vtkSmartPointer<vtkUnstructuredGrid> grid 
        = vtkSmartPointer<vtkUnstructuredGrid>::New() ;

    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    size_t ncells = nx*ny*nq ; // these are cells not vertices 
    size_t nvertex = 4 ; 
    using cell_type = vtkQuad ; 
    vtkNew<vtkPoints> points ; 
    points->SetNumberOfPoints( nvertex * ncells ) ; 

    auto const get_cell_coordinates = [&] ( size_t icell
                                          , double lx=0
                                          , double ly=0
                                          , double lz=0 )
    {
        /* unpack index assuming LayoutLeft */
        size_t const ix = icell%nx ; 
        size_t const iy = (icell/nx) % ny ;
        size_t const iq = (icell/nx/ny) ;
        /* Find quadrant index, level, coordinates, spacing */
        size_t itree              = thunder::amr::get_quadrant_owner(iq) ; 
        auto const tree_coords    = thunder::amr::get_tree_vertex(itree, 0UL) ; 
        auto const dx_tree        = thunder::amr::get_tree_spacing(itree)[0]  ;
        thunder::amr::quadrant_t const quadrant = thunder::amr::get_quadrant(itree, iq) ;  
        auto const dx_quad        = dx_tree / ( 1<<quadrant.level()) ; 
        auto const qcoords        = quadrant.qcoords() ; 
        auto const dx_cell        = dx_quad / nx ;
        auto const dy_cell        = dx_quad / ny ; 
        /* return physical coordinates of point within cell */ 
        return std::array<double,2> {
            tree_coords[0] + qcoords[0] * dx_quad + (ix + lx) * dx_cell, 
            tree_coords[1] + qcoords[1] * dx_quad + (iy + ly) * dy_cell
        } ; 
    } ; 

    for( size_t icell=0UL; icell<ncells; icell+=1UL )
    {
        vtkNew<cell_type> cell ; 
        auto lcoords = cell->GetParametricCoords() ;
        for( int iv=0; iv<nvertex; ++iv) { 
            size_t ipoint = iv + icell * nvertex ; 
            auto const coords = get_cell_coordinates( icell 
                                                    , lcoords[3*iv + 0]
                                                    , lcoords[3*iv + 1] ) ; 
            points->SetPoint(ipoint, coords[0], coords[1], 0.0) ; 
            cell->GetPointIds()->SetId(iv, ipoint) ; 
        }
        grid->InsertNextCell( cell->GetCellType(), cell->GetPointIds() ) ; 
    }

    grid->SetPoints(points); 
    return grid; 
}

vtkSmartPointer<vtkUnstructuredGrid> 
setup_vtk_grid_spherical()
{
    auto& params = thunder::config_parser::get()  ;  
    
    auto const L = params["amr"]["inner_region_side"].as<double>() ;
    auto const R = params["amr"]["outer_region_radius"].as<double>()   ;
    auto const Rl = params["amr"]["logarithmic_outer_radius"].as<double>()  ;

    vtkSmartPointer<vtkUnstructuredGrid> grid 
        = vtkSmartPointer<vtkUnstructuredGrid>::New() ;

    size_t nx,ny,dummy; 
    std::tie(nx,ny,dummy) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    size_t ncells = nx*ny*nq ; // these are cells not vertices 
    size_t nvertex = 6 ; 
    using cell_type = vtkQuadraticLinearQuad ; 

    vtkNew<vtkPoints> points ; 
    points->SetNumberOfPoints( nvertex * ncells ) ; 

    auto const get_cell_coordinates = [&] ( size_t icell
                                          , double lx=0
                                          , double ly=0 )
    {
        using namespace amr::detail ; 
        /* unpack index assuming LayoutLeft */
        size_t const ix = icell%nx ; 
        size_t const iy = (icell/nx) % ny ;
        size_t const iq = (icell/nx/ny) ;
        /* Find quadrant index, level, coordinates, spacing */
        size_t itree              = thunder::amr::get_quadrant_owner(iq) ; 
        thunder::amr::quadrant_t const quadrant = thunder::amr::get_quadrant(itree, iq) ;  
        auto const dx_quad        = 1.0 / ( 1<<quadrant.level()) ; 
        auto const qcoords        = quadrant.qcoords() ; 
        auto const dx_cell        = dx_quad / nx ;
        auto const dy_cell        = dx_quad / ny ; 
        /* return physical coordinates of point within cell */ 
        std::array<double,2> lcoords {
            { qcoords[0] * dx_quad + (ix + lx) * dx_cell, 
              qcoords[1] * dx_quad + (iy + ly) * dy_cell } 
        } ; 
        std::array<double,2> pcoords ;
        if( itree == CARTESIAN_TREE ){ 
            auto const tree_coords    = thunder::amr::get_tree_vertex(itree, 0UL) ; 
            auto const dx_tree        = thunder::amr::get_tree_spacing(itree)[0]  ;
            for(int idir=0; idir<2; ++idir) {
                pcoords[idir] = tree_coords[idir] + lcoords[idir] * dx_tree ;
            }
        } else {
            auto const H  = tan(M_PI/4. * (2.*lcoords[1]-1)) ; 
            auto const rho = sqrt( 1 + math::int_pow<2>(H) ) ; 
            auto const zeta     = ((1.-lcoords[0]) * L + lcoords[0]*R/rho)  ;
            auto const zeta_log = sqrt( std::pow(R, 2*(1-lcoords[0])) * std::pow(Rl, 2*lcoords[0]) ) / rho ;
            switch( itree )
            {
                case MX_TREE: 
                    pcoords[0] = -zeta     ; 
                    pcoords[1] = zeta * H  ;
                    break ; 
                case PX_TREE: 
                    pcoords[0] = zeta      ; 
                    pcoords[1] = zeta * H  ;
                    break; 
                case MY_TREE:
                    pcoords[0] =  zeta * H  ;
                    pcoords[1] = -zeta      ;
                    break;
                case PY_TREE:
                    pcoords[0] =  zeta * H   ;
                    pcoords[1] =  zeta       ;
                    break;
                case MXL_TREE: 
                    pcoords[0] = -zeta_log     ; 
                    pcoords[1] = zeta_log * H  ;
                    break ; 
                case PXL_TREE: 
                    pcoords[0] = zeta_log      ; 
                    pcoords[1] = zeta_log * H  ; 
                    break; 
                case MYL_TREE:
                    pcoords[0] =  zeta_log * H  ;
                    pcoords[1] = -zeta_log      ;
                    break;
                case PYL_TREE:
                    pcoords[0] =  zeta_log * H   ;
                    pcoords[1] =  zeta_log       ; 
                    break;
            }
        }
        return pcoords ;
    } ; 

    for( size_t icell=0UL; icell<ncells; icell+=1UL )
    {
        vtkNew<cell_type> cell ; 
        auto param_coords = cell->GetParametricCoords() ;
        for( int iv=0; iv<nvertex; ++iv) { 
            size_t ipoint = iv + icell * nvertex ; 
            auto const coords = get_cell_coordinates( icell 
                                                    , param_coords[3*iv + 0]
                                                    , param_coords[3*iv + 1] ) ; 
            points->SetPoint(ipoint, coords[0], coords[1], 0.0) ; 
            cell->GetPointIds()->SetId(iv, ipoint) ; 
        }
        grid->InsertNextCell( cell->GetCellType(), cell->GetPointIds() ) ; 
    }

    grid->SetPoints(points); 
    return grid; 
}


} } } /* namespace thunder::IO::detail */