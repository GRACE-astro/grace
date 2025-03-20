/**
 * @file detectors.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-03-12
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#include <grace/utils/grace_utils.hh>
#include <grace/utils/format_utils.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/system/grace_system.hh>
#include <grace/config/config_parser.hh>
#include <grace/amr/p4est_headers.hh> 
#include <grace/amr/grace_amr.hh>
// #include <grace/amr/octree_search.hh>

#include <grace/healpix/detectors.hh> 



namespace grace {


/** @brief
 * forward declarations of two functions in <grace/amr/octree_search.h>
 * since the implementation there is not decoupled from declaration 
*/
int reset_user_data(p4est_t* p4est,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant,
    p4est_locidx_t local_num,
    void* point); 

int 
my_points_function(p4est_t* p4est,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant,
    p4est_locidx_t local_num,
    void* point);
    
// TO DO: once octree_search decouples .cpp from .hh, remove the above!


namespace healpix {


void GRACE_HOST 
get_coord_from_healpix_index(
        const size_t nside, const double radius, const int idx,
        double& x1, double& x2, double& x3){
        // get the coordinate from the healpix ring index form, see arXiv:astro-ph/0409513
        // idx should run from 0 to 6N**2+2N-1 (inclusive) for the northern hemisphere+equitor if zsymm
        // idx should run from 0 to 12N**2-1 (inclusive) when z symmetry is not present 
        // local variables
        size_t  i, j, ss;
        double ph, z, phi, theta, aux, dummy;

        // calculate the northern polar cap
        if (idx<2*(nside-1)*nside){
        ph = (idx+1.)/2;
        aux = int(ph);
        aux = sqrt(ph-sqrt(aux));
        i = int(aux)+1;
        j = idx+1-2*i*(i-1);
        z = 1-i*i/(3.0*nside*nside);
        ss = 1;
        phi = M_PI*(j-ss/2.0)/(2*i);
        } // calculate the northern and southern belt
        else if (idx<10*nside*nside+2*nside){
        ph = idx-2*nside*(nside-1);
        i = int(ph/(4.*nside))+nside;
        j =  std::fmod(ph, 4.*nside)+1;
        z = 4.0/3.0-2*i/(3.0*nside);
        ss = (i-nside+1) % 2;
        phi = M_PI*(j+ss/2.0-1.0)/(2*nside);
        } // calculate the southern polar cap
        else if (idx<12*nside*nside){
        ph = ((12*nside*nside-idx-1)+1.)/2;
        aux = int(ph);
        aux = sqrt(ph-sqrt(aux));
        i = int(aux)+1;
        j = idx+1-2*i*(i-1);
        z = 1-i*i/(3.0*nside*nside);
        ss = 1;
        phi = 2*M_PI-M_PI*(j-ss/2.0)/(2*i);
        }
        else {
        ERROR("An index larger than 12*nside**2 is not allowed in healpix outflow!");
        }        
        // output coordinate according to coordinate type
        
        theta = std::acos(z);
        x1 = std::sin(theta) * std::cos(phi) * radius;
        x2 = std::sin(theta) * std::sin(phi) * radius;
        x3 = radius*z;
}


void GRACE_HOST
healpix_detector::update_detector_info(){
        // Need a p4est
        auto& forest = grace::amr::forest::get() ; 

        size_t first = forest.first_local_tree();
        size_t last = forest.last_local_tree();
        
        // Step 1.:
        // search for the coordinates of the detector that belong to this rank
        // i.e. the healpix indices of the detector that are within the quadrants of this mpi rank

        // Reset the user data to 0
        p4est_search_local_t reset_func = reset_user_data;
        p4est_search_local(forest.get(), false, reset_func, nullptr, nullptr);

        // TO DO 
        // rewrite the detector class so that we use sc_array_t for things from the start 
        auto convert_std_vector_to_sc_array_t 
                = [](const std::vector<double>& coords) ->sc_array_t* {
                        using value_type = std::array<double, 3>;
                
                        sc_array_t* sc_array = new sc_array_t;
                        sc_array->elem_count = coords.size()/3;
                        sc_array->elem_size = sizeof(value_type);
                        sc_array->array = static_cast<char*>(malloc(sc_array->elem_count * sc_array->elem_size));
                        value_type* data = reinterpret_cast<value_type*>(sc_array->array);
                        for (int i = 0; i < sc_array->elem_count; ++i) {
                                data[i] = {coords[3*i + 0], coords[3*i + 1], coords[3*i + 2]};
                        }
                     return sc_array;
                };

        p4est_search_local_t point_search_func = my_points_function;
        p4est_search_local(forest.get(), true , nullptr, point_search_func, convert_std_vector_to_sc_array_t(all_coords_det_));

        // all_coords_det contains coordinates of all pixels
        // how do we mark the pixel --> quadrant mapping in what follows?
        // how do we update the coords_det from it?
        // I think information about which function 
        // how about changing the my_points_function
        // so that in the quadrant, we save the info about th.... 
        //     quadrant->p.user_int += 1;
        // 


        // all_coords_det
        // coords_det (local, on this mpi rank)
        // 
        // indices ? not needed if we just carry out the 
        // at this point we at least know how many finest-level quadrants 
        // correspond to our set of healpix pixels - i.e. we know the size 
        // of indices, which_quadrants, outflows_ !!!
        // now we want to obtain:
            // indices_; 
            // which_quadrants_;
            // outflows_; 
        // how about we just invoke the interpolation at the relevant quadrants?
        // let's say:
        // creatae MDPolicyRange for matching quadrants
        // call the kernel to fill out outflows_
        // if coords_det(i,j,k) is within the kernel (check via the quadrant.p.user_int)
        // , fill out the indices location for bookkeeping (indices.push_back(index))
        //
        // , which_quadrants: (which_quadrants[counter]=quadrant_id and outflows 
        // note: how to make sure the parallel work on which_quadrants will not be overwritten?
        // what kind of counter to use? 
        
            
        size_t it = 0;
        for (size_t i = first; i <= last; i++) // Loop from first to last local tree
        {
            auto tree = forest.tree(i);  // Assuming there is a function to access a tree by index

            // We end up not using this number:
            size_t quadrant_offset = tree.quadrants_offset();  // Number of quadrants in the tree

            it=0;
            for (auto tree_quadrant : tree.quadrants())
            {
                auto quadrant = tree_quadrant;  // Get the j-th quadrant in the tree (adjust if needed)

                if (quadrant.p.user_int != 0)
                {
                    
                    //printf("Quadrant %zu in tree %zu intersects the point %d times.\n", i*quadrant_offset + it, i, quadrant.p.user_int);
                    //printf("Quadrant %zu in tree %zu intersects the point %d times.\n", i*quadrant_offset + it, i, quadrant.p.user_int);

                }
                it++;
                }
            }

}
        
void GRACE_HOST_DEVICE
healpix_detector::update_detector_fluxes(){
        
}

// definition of the compute fluxes function...  
void GRACE_HOST_DEVICE
compute_surface_fluxes(){

// auto& params = grace::config_parser::get() 

}

}
}