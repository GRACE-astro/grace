/**
 * @file tetrads.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2025-01-29
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

#ifndef GRACE_PHYSICS_TETRADS_HH
#define GRACE_PHYSICS_TETRADS_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>



namespace grace {

/**
 * @brief Constructs the consecutive vectors of a null tetrad
 *        used to extract the gravitational signal 
 */
struct quasi_kinnersley_tetrad {

    enum  {
        RE=0,
        IM,
        NCOMPLEX
    } ; 

    constexpr const size_t NCOMPS = 3; // only spatial parts will be saved 

    using real_vector = std::array<double, NCOMPS>
    using complex_vector = std::array<std::array<double, NCOMPLEX>, NCOMPS>
    using tetrad_vectors = std::array<complex_vectors, 4>;
         
    /**
     * @brief return the tetrad vectors at a spatial location
     *        (we follow arxiv:0104063)
     * @param metric 
     * @param pcoords 
     * @return std::array<std::array<std::array<double, 2>, 3>, 4>  
     * @warning consistent with the formulae in WeylScal4 in the EinteinToolkit, we do not fill out the unnecessary (timelike) components of tetrad vectors 
     */
    GRACE_HOST_DEVICE 
    static tetrad_vectors evaluate(grace::metric_array_t const& metric ,
                            grace::coord_array_t<GRACE_NSPACEDIM>  const& pcoords){


        x=pcoords[X];
        y=pcoords[Y];
        z=pcoords[Z];
        // we start by creating an orthonormal set of spatial vectors 
        // as the starting point, we define the spherical coordinate vectors
        // note the polar vector is obtained through spatially covariant curl

        real_vector v_azimuthal{-y,x,0.0}; // \partial_phi
        real_vector v_radial{x,y,z};    // \partial_r
        real vector v_polar = metric.compute_covariant_curl(v_azimuthal, v_radial); // \Theta^i 
        
        // orthonormalization procedure in the order (phi, r, theta)

        // phi
        auto w_azimuthal = v_azimuthal;
        const double omega11 = metric.square_vec(w_azimuthal);
        real_vector e_azimuthal{w_azimuthal[X]/Kokkos::sqrt(omega11), 
                                w_azimuthal[Y]/Kokkos::sqrt(omega11), 
                                w_azimuthal[Z]/Kokkos::sqrt(omega11)};

        // r
        const double omega12 = metric.scalar_product(v_radial, e_azimuthal);
        std::array<double,3> w_radial{v_radial[X] - omega12*e_azimuthal[X],
                                      v_radial[Y] - omega12*e_azimuthal[Y],
                                      v_radial[Z] - omega12*e_azimuthal[Z]};
        const double omega22 = metric.scalar_product(w_radial, w_radial);
        real_vector e_radial{w_radial[X]/Kokkos::sqrt(omega22),
                             w_radial[Y]/Kokkos::sqrt(omega22),
                             w_radial[Z]/Kokkos::sqrt(omega22)};
        // theta             
        const double omega13 = metric.scalar_product(e_azimuthal, v_polar);
        const double omega23 = metric.scalar_product(e_radial, v_polar);
        std::array<double,3> w_polar{v_polar[X] - omega13*e_azimuthal[X] - omega23*e_radial[X],
                                     v_polar[Y] - omega13*e_azimuthal[Y] - omega23*e_radial[Y],
                                     v_polar[Z] - omega13*e_azimuthal[Z] - omega23*e_radial[Z]};                         
        const double omega33 = metric.scalar_product(w_polar, w_polar);
        std::array<double,3> e_polar{w_polar[X]/Kokkos::sqrt(omega33),
                                     w_polar[Y]/Kokkos::sqrt(omega33),
                                     w_polar[Z]/Kokkos::sqrt(omega33)};

        // finally, we have (e_r, e_th, e_phi) that are orthonormal in the g_ij metric
        double const sqrt2 = Kokkos::sqrt(2.0);
        complex_vector ltet{}, ntet{}, mtet{}, mbartet{};
        // ltet[X][IM]=0.0;
        // ltet[Y][IM]=0.0;
        // ltet[Z][IM]=0.0;
        ltet[X][RE]=sqrt2 * e_radial[X];
        ltet[Y][RE]=sqrt2 * e_radial[Y];
        ltet[Z][RE]=sqrt2 * e_radial[Z];
        
        ntet[X][RE]=-sqrt2 * e_radial[X];
        ntet[Y][RE]=-sqrt2 * e_radial[Y];
        ntet[Z][RE]=-sqrt2 * e_radial[Z];

        mtet[X][RE]=sqrt2 * e_polar[X];
        mtet[Y][RE]=sqrt2 * e_polar[Y];
        mtet[Z][RE]=sqrt2 * e_polar[Z];
        mtet[X][IM]=sqrt2 * e_azimuthal[X];
        mtet[Y][IM]=sqrt2 * e_azimuthal[Y];
        mtet[Z][IM]=sqrt2 * e_azimuthal[Z];


        mbartet[X][RE]=sqrt2 * e_polar[X];
        mbartet[Y][RE]=sqrt2 * e_polar[Y];
        mbartet[Z][RE]=sqrt2 * e_polar[Z];
        mbartet[X][IM]=-sqrt2 * e_azimuthal[X];
        mbartet[Y][IM]=-sqrt2 * e_azimuthal[Y];
        mbartet[Z][IM]=-sqrt2 * e_azimuthal[Z];

        return {ltet, ntet, mtet, mbartet};

    }

private: 

    static constexpr int X = 0 ; 
    static constexpr int Y = 1 ; 
    static constexpr int Z = 2 ; 

};


}


#endif /* GRACE_PHYSICS_TETRADS_HH */