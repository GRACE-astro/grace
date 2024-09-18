/**
 * @file xmf_utils.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-24
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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
#include <grace/IO/hdf5_output.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/xmf_utilities.hh>
#include <grace/system/grace_system.hh>

#include <string>
#include <vector>
#include <fstream>

namespace grace { namespace IO {

struct xmf_cell_centered_attr_node {
    int64_t ncells ; 
    std::string vname ; 
    std::string h5fname ;
    std::string vtype   ;
} ; 

struct xmf_grid_node {
    int64_t iteration ;
    int64_t ncells ;
    int64_t npoints ; 
    int nvertex ; 
    double time ; 
    std::string topology;
    std::string h5fname ;
    std::vector<xmf_cell_centered_attr_node> attrs ; 
} ; 

struct xmf_grid_collection_node {
    std::string name ; 
    std::vector<xmf_grid_node> grids ;
} ; 

struct xmf_domain_node {
    xmf_grid_collection_node coll ; 
} ; 

struct xmf_file_node {
    xmf_domain_node domain ; 
} ; 

std::ostream& operator<<(std::ostream& os, xmf_cell_centered_attr_node& attr ) {
    os << R"(<Attribute Center="Cell" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name=")" << attr.vname 
       << R"(" Type=")" << attr.vtype << R"(">)" << '\n'
       << R"(<DataItem DataType="Float" Dimensions=")" << attr.ncells << R"(" Format="HDF" Precision="8">)"
       << attr.h5fname << ":/" << attr.vname << R"(</DataItem>)"
       << R"(</Attribute>)" ; 
    return os ; 
}

std::ostream& operator<<(std::ostream& os, xmf_grid_node& grid ) {
    os << R"(<Grid Name="Grid_)" << grid.iteration << R"(">)" << '\n'
       << R"(<Time Value=")" << grid.time << R"("/>)" << '\n'
       << R"( <Geometry Origin="" Type="XYZ">)" << '\n'
       << R"(<DataItem DataType="Float" Dimensions=")"  << grid.npoints << " " << 3 << R"(" Format="HDF" Precision="8">)"
       << grid.h5fname << R"(:/Points</DataItem>)" << '\n'
       << R"(</Geometry>)" << '\n'
       << R"(<Topology Dimensions=")" << grid.ncells << R"(" Type=")" << grid.topology << R"(">)" << '\n'
       << R"(<DataItem DataType="UInt" Dimensions=")" << grid.ncells << " " << grid.nvertex << R"(" Format="HDF" Precision="4">)"
       << grid.h5fname << R"(:/Cells</DataItem>)" << '\n'
       << R"(</Topology>)" << '\n' ; 
    for(auto& attr: grid.attrs ) {
        os << attr << '\n'; 
    }
    os << R"(</Grid>)";
    return os ; 
}

std::ostream& operator<<(std::ostream& os, xmf_grid_collection_node& coll ) {
    os << R"(<Grid CollectionType="Temporal" GridType="Collection" Name=")" << coll.name << R"(">)" << '\n'; 
    for( auto& grid: coll.grids) {
        os << grid << '\n' ; 
    }
    os << R"(</Grid>)";
    return os ; 
}

std::ostream& operator<<(std::ostream& os, xmf_domain_node& domain ) {
    os << R"(<Domain>)" << '\n' ;
    os << domain.coll << '\n' ; 
    os << R"(</Domain>)" ;
    return os ;
}

std::ostream& operator<<(std::ostream& os, xmf_file_node& f ) {
    os << R"(<?xml version="1.0" encoding="utf-8"?>)" << '\n' 
       << R"(<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">)" << '\n' ;
    os << f.domain << '\n' ; 
    os << R"(</Xdmf>)" ;
    return os ;
}

void write_xmf_file( const std::string &filename 
                   , std::vector<int64_t> const & iters
                   , std::vector<int64_t> const& ncells_vec
                   , std::vector<double> const& times 
                   , std::vector<std::string> const& filenames ) 
{

    auto& runtime = grace::runtime::get() ;

    auto const scalars     = runtime.cell_volume_output_scalar_vars() ; 
    auto const vectors     = runtime.cell_volume_output_vector_vars() ; 

    std::ofstream outfile(filename);
    std::vector<xmf_grid_node> grids ;

    for(int i=0; i<iters.size(); ++i) {
        std::vector<xmf_cell_centered_attr_node> attributes ;
        double const time = times[i] ; 
        int64_t const it  = iters[i] ; 
        int64_t const ncells = ncells_vec[i]  ; 
        #ifdef GRACE_3D 
        int64_t const nvertex = 8 ;
        std::string const topology = "Hexahedron" ; 
        #else 
        int64_t const nvertex = 4 ;
        std::string const topology = "Quadrilateral" ;
        #endif 
        int64_t const npoints = nvertex * ncells ; 
        std::string  const h5fname = filenames[i] ;

        for( auto& vname: scalars ) {
            attributes.push_back( xmf_cell_centered_attr_node{ncells,vname,h5fname,"Scalar"} ) ; 
        } 
        for( auto& vname: vectors ) {
            attributes.push_back( xmf_cell_centered_attr_node{ncells,vname,h5fname,"Vector"} ) ; 
        }
        grids.push_back(xmf_grid_node{it,ncells,npoints,nvertex,time,topology,h5fname,attributes}) ; 
    }

    xmf_grid_collection_node collection{"Collection",grids} ; 
    xmf_domain_node domain{collection} ; 
    xmf_file_node file{domain} ;
    outfile << file ; 

}

}}