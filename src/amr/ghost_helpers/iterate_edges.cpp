/**
 * @file iterate_edges.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Index fiesta.
 * @date 2025-09-05
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

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/sc_wrappers.hh>

#include <grace/errors/assert.hh>
#include <grace/system/print.hh>

#include <grace/amr/p4est_headers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/forest.hh>

#include <grace/data_structures/memory_defaults.hh>

#include <grace/amr/amr_ghosts.hh>
#include <grace/amr/ghostzone_kernels/index_helpers.hh>


#include <vector>

namespace grace {

std::tuple<int8_t,int8_t,int8_t> get_dirs(
    sc_array_view_t<p8est_iter_edge_side_t> const& sides
)
{
    // depending on the number of sides, this
    // is either a grid edge or a face 

    auto const get_edge_dir = [&](int8_t edge) -> std::tuple<int8_t,int8_t,int8_t> {
        int8_t off[] = { 
            static_cast<int8_t>((edge>>0)&1),
            static_cast<int8_t>((edge>>1)&1) 
        };
        if ( edge < 4 ) {
            // along x 
            return {
                0, off[0] ? 1 : -1, off[1] ? 1 : -1 
            } ; 
        } else if (edge < 8) {
            return {
                off[0] ? 1 : -1, 0, off[1] ? 1 : -1 
            } ;
        } else {
            return {
                off[0] ? 1 : -1, off[1] ? 1 : -1, 0
            } ;
        }
    } ; 

    if ( sides.size() == 1 ) {
        // grid edge 
        return get_edge_dir(sides[0].edge) ; 
    } else {
        // grid face 
        auto const [d11,d12,d13] = get_edge_dir(sides[0].edge) ; 
        auto const [d21,d22,d23] = get_edge_dir(sides[1].edge) ; 
        // these two edges should agree in all directions except one,
        // where one will say -1 and the other +1. This direction 
        // lies in the face. 
        return {
            d11==d21 ? d11 : 0,
            d12==d22 ? d12 : 0,
            d13==d23 ? d13 : 0
        } ; 
    }
}


static void register_physical_boundary_edge(
    sc_array_view_t<p8est_iter_edge_side_t> const& sides, 
    std::vector<quad_neighbors_descriptor_t>& neighbors
)
{
    auto [dx,dy,dz] =  get_dirs(sides) ;  
    for( auto const& side: sides) {
        
        if ( side.is_hanging ) {
            for( int is=0; is<2; ++is) {
                if ( side.is.hanging.is_ghost[is]) continue ; 
                auto const offset = grace::amr::get_local_quadrants_offset(side.treeid); 
                // hanging local 
                auto qid = side.is.hanging.quadid[is] +  offset ; 
                auto& edge = neighbors[qid].edges[side.edge];
                edge.kind = interface_kind_t::PHYS ;
                edge.filled = true ;
                // grid normal 
                edge.data.phys.dir[0] = dx ;
                edge.data.phys.dir[1] = dy ;
                edge.data.phys.dir[2] = dz ;
                edge.data.phys.in_cbuf = false ;
                edge.data.phys.type = sides.size() == 1 ? amr::element_kind_t::EDGE : amr::element_kind_t::FACE ;
                neighbors[qid].n_registered_edges++ ; 
                edge.data.phys.task_id.fill(UNSET_TASK_ID);
            }

        } else {
            
            if (side.is.full.is_ghost) continue ; 
            auto const offset = grace::amr::get_local_quadrants_offset(side.treeid); 
            // not hanging not ghost 
            auto qid = side.is.full.quadid +  offset ; 
            auto & edge = neighbors[qid].edges[side.edge] ; 
            edge.kind = interface_kind_t::PHYS ;
            edge.filled = true ; 
            // grid normal 
            edge.data.phys.dir[0] = dx ;
            edge.data.phys.dir[1] = dy ;
            edge.data.phys.dir[2] = dz ;
            edge.data.phys.in_cbuf = false ;
            edge.data.phys.type = sides.size() == 1 ? amr::element_kind_t::EDGE : amr::element_kind_t::FACE ; 
            neighbors[qid].n_registered_edges++ ; 
            edge.data.phys.task_id.fill(UNSET_TASK_ID);
        }
    }
    
}

inline void fill_full_edge_desc(edge_descriptor_t& desc,
                          int8_t this_edge,
                          int8_t other_edge,
                          size_t qid,
                          bool is_remote,
                          p4est_topidx_t treeid,
                          p4est_quadrant_t const* quad) {
    desc.filled = true; 
    desc.edge = other_edge;
    desc.kind = interface_kind_t::INTERNAL;
    desc.data.full.quad_id = qid;
    desc.data.full.is_remote = is_remote;
    desc.data.full.task_id.fill(UNSET_TASK_ID);
    if (is_remote) {
        desc.data.full.owner_rank =
            p4est_comm_find_owner(grace::amr::forest::get().get(), treeid, quad, 0);
    }
};

inline void fill_hanging_edge_desc(edge_descriptor_t& desc,
                             int8_t this_edge,
                             int8_t other_edge,
                             int8_t islot,
                             size_t qid,
                             bool is_remote,
                             p4est_topidx_t treeid,
                             p4est_quadrant_t const* quad) {
    desc.filled = true; 
    desc.edge = other_edge;
    desc.kind = interface_kind_t::INTERNAL;
    desc.data.hanging.quad_id[islot] = qid;
    desc.data.hanging.is_remote[islot] = is_remote;
    desc.data.hanging.task_id[islot].fill(UNSET_TASK_ID);
    if (is_remote) {
        desc.data.hanging.owner_rank[islot] =
            p4est_comm_find_owner(grace::amr::forest::get().get(), treeid, quad, 0);
    }
};


static void register_edge(p8est_iter_edge_side_t const& s0,
                   p8est_iter_edge_side_t const& s1,
                   std::vector<quad_neighbors_descriptor_t>& neighbors)
{
    
    int8_t e = s0.edge;

    if (s0.is_hanging) {
        // s0 hanging
        for (int is = 0; is < 2; ++is) {
            if (s0.is.hanging.is_ghost[is]) continue; // skip remote on our side
            auto offset = grace::amr::get_local_quadrants_offset(s0.treeid);
            auto qid = s0.is.hanging.quadid[is] + offset;
            auto& desc = neighbors[qid].edges[e];
            neighbors[qid].n_registered_edges++;

            if (s1.is_hanging) {
                // both hanging → SAME level
                auto other_offset = s1.is.hanging.is_ghost[is] ? 0 : grace::amr::get_local_quadrants_offset(s1.treeid);
                desc.level_diff = level_diff_t::SAME ; 
                fill_full_edge_desc(desc, e, s1.edge,
                               s1.is.hanging.quadid[is] + other_offset,
                               s1.is.hanging.is_ghost[is],
                               s1.treeid,
                               s1.is.hanging.quad[is]);
            } else {
                // s0 hanging, s1 full → s1 is COARSER
                auto other_offset = s1.is.full.is_ghost ? 0 : grace::amr::get_local_quadrants_offset(s1.treeid);
                desc.level_diff = level_diff_t::COARSER;
                desc.child_id = is ; 
                fill_full_edge_desc(desc, e, s1.edge,
                               s1.is.full.quadid + other_offset,
                               s1.is.full.is_ghost,
                               s1.treeid,
                               s1.is.full.quad);
            }
        }
    } else {
        // s0 full
        if (s0.is.full.is_ghost) return; // we only register if local
        auto offset = grace::amr::get_local_quadrants_offset(s0.treeid);
        auto qid = s0.is.full.quadid + offset;
        auto& desc = neighbors[qid].edges[e];
        neighbors[qid].n_registered_edges++;

        if (s1.is_hanging) {
            // neighbor is finer
            desc.level_diff = level_diff_t::FINER;
            for (int is = 0; is < 2; ++is) {
                auto other_offset = s1.is.hanging.is_ghost[is] ? 0 : grace::amr::get_local_quadrants_offset(s1.treeid);
                fill_hanging_edge_desc(desc, e, s1.edge,
                                  is,
                                  s1.is.hanging.quadid[is] + other_offset,
                                  s1.is.hanging.is_ghost[is],
                                  s1.treeid,
                                  s1.is.hanging.quad[is]);
            }
        } else {
            // both full → SAME level
            auto other_offset = s1.is.full.is_ghost ? 0 : grace::amr::get_local_quadrants_offset(s1.treeid);
            desc.level_diff = level_diff_t::SAME ; 
            fill_full_edge_desc(desc, e, s1.edge,
                           s1.is.full.quadid + other_offset,
                           s1.is.full.is_ghost,
                           s1.treeid,
                           s1.is.full.quad);
        }
    }
}

inline void fill_hanging_corner_desc(corner_descriptor_t& desc,
                             int8_t this_corner,
                             int8_t other_corner,
                             size_t qid,
                             bool is_remote,
                             p4est_topidx_t treeid,
                             p4est_quadrant_t const* quad) {
    desc.filled = true; 
    desc.corner = other_corner;
    desc.kind = interface_kind_t::INTERNAL;
    desc.data.quad_id = qid;
    desc.data.is_remote = is_remote;
    desc.data.task_id.fill(UNSET_TASK_ID);
    if (is_remote) {
        desc.data.owner_rank =
            p4est_comm_find_owner(grace::amr::forest::get().get(), treeid, quad, 0);
    }
};

static void register_hanging_corner(
    p8est_iter_edge_side_t const& s0,
    p8est_iter_edge_side_t const& s1,
    std::vector<quad_neighbors_descriptor_t>& neighbors
)
{
    int8_t e = s0.edge;
    int8_t c[] = { 
        static_cast<int8_t>(grace::amr::detail::e2c[e][1]), 
        static_cast<int8_t>(grace::amr::detail::e2c[e][0])
    };
    //nb this assumes no relative orientation
    auto const get_opposite_corner = [=] (int8_t icorner ) -> int8_t {
        int8_t ix = (icorner>>0) & 1 ? 0 : 1 ; 
        int8_t iy = (icorner>>1) & 1 ? 0 : 1 ; 
        int8_t iz = (icorner>>2) & 1 ? 0 : 1 ; 
        return ix + (iy<<1) + (iz<<2) ; 
    } ; 

    for( int ic=0; ic<2; ++ic ) {
        if (s0.is.hanging.is_ghost[ic]) continue; // skip remote on our side
        int other_ic = ic ? 0 : 1 ; 
        auto offset = grace::amr::get_local_quadrants_offset(s0.treeid);
        auto qid = s0.is.hanging.quadid[ic] + offset;
        auto& desc = neighbors[qid].corners[c[ic]];
        // both hanging → SAME level
        auto other_offset = s1.is.hanging.is_ghost[other_ic] ? 0 : grace::amr::get_local_quadrants_offset(s1.treeid);

        fill_hanging_corner_desc(
            desc, c[ic], get_opposite_corner(c[ic]),
            s1.is.hanging.quadid[other_ic] + other_offset,
            s1.is.hanging.is_ghost[other_ic],
            s1.treeid,
            s1.is.hanging.quad[other_ic] 
        ) ; 
    }
}

void grace_iterate_edges(p8est_iter_edge_info_t* info, void* user_data)
{
    auto iter_data = reinterpret_cast<p4est_iter_data_t*>(user_data) ;
    auto ghosts = iter_data->ghost_layer ;
    sc_array_view_t<p8est_iter_edge_side_t> sides{&(info->sides)};
    int const nsides = static_cast<int>(sides.size()) ;

    if (nsides == 1) {
        // Domain corner edge: only 1 quadrant, nothing to reflux
        register_physical_boundary_edge(sides, *ghosts);
        return;
    }

    if (nsides < 4) {
        // Boundary edge with 2 sides: register physical BC,
        // then fall through to build reflux descriptor
        register_physical_boundary_edge(sides, *ghosts);
    }

    if (nsides == 4) {
        // Interior edge: pair and register ghost-zone neighbors
        static constexpr int opposite_edge[12] = {
            3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8
        } ;

        auto is_edge_neighbor = [&](int i, int j) {
            return opposite_edge[sides[i].edge] == sides[j].edge;
        } ;

        std::vector<std::array<int,2>> pairs;
        std::vector<bool> found(sides.size(), false);

        for (int in = 0; in < nsides; ++in) {
            if (found[in]) continue;
            for (int jn = in + 1; jn < nsides; ++jn) {
                if (found[jn]) continue;
                if (is_edge_neighbor(in, jn)) {
                    pairs.push_back({in, jn});
                    found[in] = found[jn] = true;
                    break;
                }
            }
        }

        for (auto const& [i0, i1] : pairs) {
            register_edge(sides[i0], sides[i1], *ghosts);
            register_edge(sides[i1], sides[i0], *ghosts);
            if ( sides[i0].is_hanging and sides[i1].is_hanging ) {
                register_hanging_corner(sides[i0],sides[i1], *ghosts) ;
                register_hanging_corner(sides[i1],sides[i0], *ghosts) ;
            }
        }
    }

    // Build reflux descriptor
    int n_fine = 0 ;
    for( int is=0; is<nsides; ++is ) {
        n_fine += sides[is].is_hanging ;
    }

    hanging_edge_reflux_desc_t desc {} ;
    desc.n_sides = nsides ;
    desc.n_fine = n_fine ;
    desc.n_coarse = nsides - n_fine ;
    int ifine{0}, icoarse{0};
    for( int is=0; is<nsides; ++is) {
        auto& side = sides[is] ;
        auto& sdsc = desc.sides[is] ;
        sdsc.edge_id = side.edge ;
        sdsc.off_i = sdsc.off_j = 0 ;
        if ( side.is_hanging ) {
            desc.fine_sides[ifine++] = is ;
            sdsc.is_fine = true ;
            for ( int ic=0; ic<2; ++ic ) {
                sdsc.octants.fine.is_remote[ic] = side.is.hanging.is_ghost[ic] ;
                sdsc.octants.fine.quad_id[ic] = side.is.hanging.is_ghost[ic]
                                              ? side.is.hanging.quadid[ic]
                                              : side.is.hanging.quadid[ic] + grace::amr::get_local_quadrants_offset(side.treeid) ;
                sdsc.octants.fine.owner_rank[ic] = p4est_comm_find_owner(grace::amr::forest::get().get(), side.treeid, side.is.hanging.quad[ic], 0);
            }
        } else {
            desc.coarse_sides[icoarse++] = is ;
            sdsc.is_fine = false ;
            sdsc.octants.coarse.is_remote = side.is.full.is_ghost ;
            sdsc.octants.coarse.quad_id = side.is.full.is_ghost
                                        ? side.is.full.quadid
                                        : side.is.full.quadid + grace::amr::get_local_quadrants_offset(side.treeid) ;
            sdsc.octants.coarse.owner_rank = p4est_comm_find_owner(grace::amr::forest::get().get(), side.treeid, side.is.full.quad, 0);
        }
    }
    if ( n_fine > 0 ) {
        iter_data->reflux_edges->push_back(desc) ;
    } else {
        iter_data->reflux_coarse_edges->push_back(desc) ;
    }
}


}