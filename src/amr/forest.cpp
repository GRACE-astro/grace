/**
 * @file forest.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-02-29
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#include <grace/amr/forest.hh>

#include <grace/amr/amr_flags.hh>
#include <grace/parallel/mpi_wrappers.hh> 
#include <grace/amr/connectivity.hh>
#include <grace/config/config_parser.hh>
#include <grace/system/print.hh>

#include <vector>
#include <iostream>

namespace grace { namespace amr {

struct fmr_box_t {
    fmr_box_t(node_t node) {
        try {
            target_level_delta = node["target_level_delta"].as<int>() ; 
        } catch (const YAML::BadConversion&) {
            ERROR("Cannot retrieve target_level_delta from fmr box") ; 
        }

        try {
            xmin = node["x_min"].as<double>() ; 
        } catch (const YAML::BadConversion&) {
            ERROR("Cannot retrieve x_min from fmr box") ; 
        }

        try {
            xmax = node["x_max"].as<double>() ; 
        } catch (const YAML::BadConversion&) {
            ERROR("Cannot retrieve x_max from fmr box") ; 
        }

        try {
            ymin = node["y_min"].as<double>() ; 
        } catch (const YAML::BadConversion&) {
            ERROR("Cannot retrieve y_min from fmr box") ; 
        }

        try {
            ymax = node["y_max"].as<double>() ; 
        } catch (const YAML::BadConversion&) {
            ERROR("Cannot retrieve y_max from fmr box") ; 
        }

        try {
            zmin = node["z_min"].as<double>() ; 
        } catch (const YAML::BadConversion&) {
            ERROR("Cannot retrieve z_min from fmr box") ; 
        }

        try {
            zmax = node["z_max"].as<double>() ; 
        } catch (const YAML::BadConversion&) {
            ERROR("Cannot retrieve z_max from fmr box") ; 
        }
        
    }
    int target_level_delta ;
    double xmin,xmax,ymin,ymax,zmin,zmax ; 
} ; 
 
struct fmr_context {
    std::vector<fmr_box_t> boxes;
    unsigned base_level;
};

forest_impl_t::forest_impl_t(
    p4est_t* _forest_ptr
) : _p4est(_forest_ptr)
{ }

static int fmr_refine_cback(
    p4est_t* p4est,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quad 
)
{
    auto context = static_cast<fmr_context*>(p4est->user_pointer) ;
    size_t n_boxes = context->boxes.size() ; 
    size_t base_level = context->base_level ; 
    // now compute coordinates of this quad 
    auto pconn = p4est->connectivity ; 
    p4est_qcoord_t qx,qy, qz; 
    qx = quad->x ; qy = quad->y ; qz = quad->z ; 
    double xyz[3] ; 
    p4est_qcoord_to_vertex(pconn,which_tree,qx,qy,qz,xyz) ; 
    double dx_quad = 1./(1<<static_cast<int>(quad->level)) ; 
    // tree spacing (we assume same in all dimensions)
    double dx_tree ; 
    
    auto nv1 = pconn->tree_to_vertex[which_tree*P4EST_CHILDREN] ; 
    auto nv2 = pconn->tree_to_vertex[which_tree*P4EST_CHILDREN+1] ; 
    auto xv1 = pconn->vertices[3UL*nv1] ; auto xv2 = pconn->vertices[3UL*nv2] ; 
    dx_tree = xv2-xv1 ; 
    ASSERT(dx_tree>0, "something wrong") ; 

    dx_quad *= dx_tree ; 

    double x0 = xyz[0];
    double y0 = xyz[1];
    double z0 = xyz[2];
    double x1 = x0 + dx_quad;
    double y1 = y0 + dx_quad;
    double z1 = z0 + dx_quad;
    double xm = (x0+x1)*0.5 ;
    double ym = (y0+y1)*0.5 ; 
    double zm = (z0+z1)*0.5 ; 

    // determine if the quadrant is inside the box ;
    bool need_refine = false ;
    for( int ibox=0; ibox<n_boxes; ++ibox) {
        auto const& box = context->boxes[ibox] ;
        if ( static_cast<int>(quad->level) >= base_level + box.target_level_delta ) continue ; 
        bool inside =
            (box.xmin <= xm && xm <= box.xmax) &&
            (box.ymin <= ym && ym <= box.ymax) &&
            (box.zmin <= zm && zm <= box.zmax);

        need_refine |= inside ; 
    }
    
    return need_refine ? 1 : 0 ; 
}

forest_impl_t::forest_impl_t() 
{ 
    GRACE_INFO("Initializing forest of oct-trees...")  ;
    auto & params       = grace::config_parser::get()   ; 
    auto & connectivity = grace::amr::connectivity::get() ; 
    int min_level( params["amr"]["initial_refinement_level"].as<int>() ) ; 
    _p4est =  p4est_new_ext(  parallel::get_comm_world()          // MPI comm
                            , connectivity.get()                  // Connectivity
                            , 0                                   // min_quadrants 
                            , min_level                           // min_level 
                            , 1                                   // fill_uniform 
                            , sizeof(grace_quadrant_user_data_t)  // data_size
                            , initialize_quadrant                      // init_fn 
                            , nullptr ) ;                         // user_ptr 
    // set up fmr grid if needed
    // first: get the fmr boxes 
    auto n_boxes = grace::get_param<unsigned>("amr","n_fmr_boxes") ; 
    auto base_level = grace::get_param<unsigned>("amr", "initial_refinement_level") ; 
    auto fmr_boxes_params = grace::config_parser::get().get()["amr"]["fmr_boxes"] ; 

    std::vector<fmr_box_t> boxes ; 
    for( int ib=0; ib<n_boxes; ++ib) {
        fmr_box_t box(fmr_boxes_params[ib]) ; 
        boxes.push_back(box) ; 
    }

    if (n_boxes>0) {
        fmr_context context ; 
        context.boxes = boxes ; 
        context.base_level = base_level ; 
        _p4est->user_pointer = &context ; 
        // call refine
        p4est_refine(_p4est, 1, fmr_refine_cback, initialize_quadrant) ;
        // call balance 
        p4est_balance(_p4est, P4EST_CONNECT_FULL, initialize_quadrant) ; 
        // call partition! 
        p4est_partition(_p4est, 1, nullptr) ; 
    }

    GRACE_INFO("Forest initialized with {} ({}) total (local) quadrants."
                 , _p4est->global_num_quadrants, _p4est->local_num_quadrants ) ; 
}

forest_impl_t::~forest_impl_t() 
{ 
    p4est_destroy(_p4est) ; 
}

} } /* namespace grace::amr */