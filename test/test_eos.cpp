/**
 * @file test_pwpoly_eos.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
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

#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/scalar_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/system/grace_system.hh>
#include <iostream>
#include <fstream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <hdf5.h>


static void read_dset(const std::string &fileName, const std::string &groupName, const std::string &datasetName, std::vector<double> &data) ;

static std::vector<double> inline 
get_linspace(double const& xmin, double const& xmax, size_t N) {
    double const h{ (xmax-xmin)/N } ; 
    std::vector<double> res(N) ; 
    for( int i=0; i<N; ++i) {
        res[i] = xmin + static_cast<double>(i) * h ; 
    } 
    return std::move(res) ; 
}

template< typename eos_t >
static void test_eos_implementation(double _temp, std::string const& group, std::string const& test_filename)
{
    auto eos = grace::eos::get().get_eos<eos_t>() ; 
    std::vector<double> rho,press,eps,h,entropy,csnd ;

    read_dset(test_filename,group,"rho",rho) ; 
    read_dset(test_filename,group,"press",press) ; 
    read_dset(test_filename,group,"eps",eps) ; 
    read_dset(test_filename,group,"h",h) ; 
    read_dset(test_filename,group,"entropy",entropy) ; 
    read_dset(test_filename,group,"csnd2",csnd) ; 

    size_t const N = rho.size() ; 
    
    Kokkos::View<double *> d_rho("rho", N) ;
    Kokkos::View<double *> d_press("press", N) ;
    Kokkos::View<double *> d_eps("eps", N) ;
    Kokkos::View<double *> d_h("h", N) ;
    Kokkos::View<double *> d_csnd("csnd", N) ;
    Kokkos::View<double *> d_ent("entropy", N) ;

    Kokkos::View<double **> d_err("error", N,35) ; 

    auto h_press = Kokkos::create_mirror_view(d_press);
    auto h_eps   = Kokkos::create_mirror_view(d_eps)  ;
    auto h_h   = Kokkos::create_mirror_view(d_h)  ;
    auto h_csnd   = Kokkos::create_mirror_view(d_csnd)  ;
    auto h_ent   = Kokkos::create_mirror_view(d_ent)  ;

    auto h_err   = Kokkos::create_mirror_view(d_err)  ;
    

    #define DEEP_COPY_VEC_TO_VIEW(vec,view) \
            do { \
                auto host_view = Kokkos::create_mirror_view(view) ; \
                for( int i=0; i < vec.size(); ++i){                 \
                    host_view(i) = vec[i] ;                         \
                }                                                   \
                Kokkos::deep_copy(view,host_view) ;                 \
            } while(0)
    
    DEEP_COPY_VEC_TO_VIEW(rho,d_rho) ; 

    Kokkos::parallel_for("pwp_test_fill",N,
    KOKKOS_LAMBDA (int i){
        double eps,csnd2; 
        double rho{d_rho(i)}, temp{_temp}, ye{0} ; 
        int ww{0} ;
        unsigned int err ; 

        double press ; 
        press = eos.press_eps_csnd2__temp_rho_ye(eps,csnd2,temp,d_rho(i),ye,err) ;
        d_press(i) = press; 
        d_eps(i)  = eps;

        double ploc, tloc, epsloc ;
        ploc = eos.press__eps_rho_ye(eps,d_rho(i),ye,err) ; 
        d_err(i,ww) = math::abs(d_press(i) - ploc); ww++; //0
        
        ploc = eos.press_temp__eps_rho_ye(tloc,eps,d_rho(i),ye,err) ;
        d_err(i,ww) = math::abs(d_press(i) - ploc); ww++; //1
        d_err(i,ww) = math::abs(temp-tloc) ; ww++;        //2

        ploc = eos.press__temp_rho_ye(temp,d_rho(i),ye,err) ;
        d_err(i,ww) = math::abs(d_press(i) - ploc); ww++; //3

        epsloc = eos.eps__temp_rho_ye(temp,rho,ye,err);
        d_err(i,ww) = math::abs(eps-epsloc) ; ww++;       //4
        ploc = d_press(i) ; 

        epsloc = eos.eps__press_temp_rho_ye(ploc,temp,rho,ye,err);
        d_err(i,ww) = math::abs(eps-epsloc) ; ww++;       //5

        double h, hloc; 
        double csnd, csndloc ; 
        ploc = eos.press_h_csnd2__eps_rho_ye(
            h,csnd,eps,rho,ye,err
        ) ; 
        d_err(i,ww) = math::abs(d_press(i) - ploc); ww++; //6

        ploc = eos.press_h_csnd2__temp_rho_ye(
            hloc,csndloc,temp,rho,ye,err
        ) ;
        d_err(i,ww) = math::abs(d_press(i) - ploc); ww++; //7
        d_err(i,ww) = math::abs(h-hloc); ww++;            //8
        d_err(i,ww) = math::abs(csnd-csndloc); ww++;      //9
 
        epsloc = eos.eps_h_csnd2__press_rho_ye(
            hloc,csndloc,press,rho,ye,err
        ) ; 
        d_err(i,ww) = math::abs(h-hloc); ww++;            //10
        d_err(i,ww) = math::abs(csnd-csndloc); ww++;      //11
        d_err(i,ww) = math::abs(eps-epsloc) ; ww++;       //12

        ploc = eos.press_eps_csnd2__temp_rho_ye(
            epsloc,csndloc,temp,rho,ye,err
        );
        d_err(i,ww) = math::abs(csnd-csndloc); ww++;      //13
        d_err(i,ww) = math::abs(eps-epsloc) ; ww++;       //14
        d_err(i,ww) = math::abs(d_press(i) - ploc); ww++; //15

        double entropy, entropyloc;
        ploc = eos.press_h_csnd2_temp_entropy__eps_rho_ye(
            hloc,csndloc,tloc,entropy,eps,rho,ye,err
        );
        d_err(i,ww) = math::abs(d_press(i) - ploc); ww++; //16
        d_err(i,ww) = math::abs(csnd-csndloc); ww++;      //17
        d_err(i,ww) = math::abs(h-hloc); ww++;            //18
        d_err(i,ww) = math::abs(temp-tloc) ; ww++;        //19

        epsloc = eos.eps_csnd2_entropy__temp_rho_ye(
            csndloc,entropyloc,temp,rho,ye,err
        ) ;
        d_err(i,ww) = math::abs(csnd-csndloc); ww++;        //20
        d_err(i,ww) = math::abs(eps-epsloc) ; ww++;         //21
        d_err(i,ww) = math::abs(entropy-entropyloc) ; ww++; //22

        ploc = eos.press_h_csnd2_temp_eps__entropy_rho_ye(
            hloc,csndloc,tloc,epsloc,entropy,rho,ye,err
        );
        d_err(i,ww) = math::abs(csnd-csndloc); ww++;         //23
        d_err(i,ww) = math::abs(eps-epsloc) ; ww++;          //24
        d_err(i,ww) = math::abs(d_press(i) - ploc); ww++;    //25
        d_err(i,ww) = math::abs(h-hloc); ww++;               //26
        d_err(i,ww) = math::abs(temp-tloc) ; ww++;           //27

        epsloc = eos.eps_h_csnd2_temp_entropy__press_rho_ye(
            hloc,csndloc,tloc,entropyloc,d_press(i),rho,ye,err
        ) ;
        d_err(i,ww) = math::abs(csnd-csndloc); ww++;         //28
        d_err(i,ww) = math::abs(eps-epsloc) ; ww++;          //29
        d_err(i,ww) = math::abs(h-hloc); ww++;               //30
        d_err(i,ww) = math::abs(temp-tloc); ww++;            //31
        d_err(i,ww) = math::abs(entropy-entropyloc) ; ww++;  //32

         
        d_h(i)    = h ;
        d_csnd(i) = csnd;
        d_ent(i)  = entropy; 
    }) ; 
    Kokkos::deep_copy(h_press,d_press);
    Kokkos::deep_copy(h_eps, d_eps)   ;
    Kokkos::deep_copy(h_h, d_h)       ; 
    Kokkos::deep_copy(h_csnd, d_csnd) ;
    Kokkos::deep_copy(h_ent, d_ent) ;
    Kokkos::deep_copy(h_err, d_err) ;

    std::ofstream errfile{"err.txt"} ;

    std::ofstream resfile("results.dat"); 
    for( int i=0; i<N; ++i){
        for( int j=0; j<33; ++j) {
            // This is needed to prevent a failure because for T=0 eps cannot be recovered from the entropy
            if( _temp == 0 and j >= 23 and j <= 27 ) 
                    continue; 
            if( h_err(i,j) > 1e-10 ) {
                errfile << "rho " << rho[i] << ", " << j 
                          << ", err " << h_err(i,j) << std::endl ; 
            }
            CHECK_THAT(
            h_err(i,j),
            Catch::Matchers::WithinAbs(0., 1e-10)
        ) ; 
        }
        CHECK_THAT(
            fabs(h_press(i)-press[i]),
            Catch::Matchers::WithinAbs(0., 1e-8)
        ) ; 
        CHECK_THAT(
            fabs(h_eps(i)-eps[i]),
            Catch::Matchers::WithinAbs(0., 1e-8)
        ) ; 
        CHECK_THAT(
            fabs(h_h(i)-h[i]),
            Catch::Matchers::WithinAbs(0., 1e-8)
        ) ; 
        
        CHECK_THAT(
            fabs(h_ent(i)-entropy[i]),
            Catch::Matchers::WithinAbs(0., 1e-8)
        ) ; 
        
        CHECK_THAT(
            fabs(h_csnd(i)-csnd[i]),
            Catch::Matchers::WithinAbs(0., 1e-8)
        ) ;
        #if 0
        resfile << std::setprecision(15) 
                << std::left << std::setw(40) << rho[i]
                << std::left << std::setw(40) << h_press(i)
                << std::left << std::setw(40) << h_eps(i) 
                << std::left << std::setw(40) << h_h(i) 
                << std::left << std::setw(40) << h_ent(i) 
                << std::left << std::setw(40) << h_csnd(i)<< '\n' ; 
        #endif 
    }
    errfile.close() ; 
    resfile.close() ; 
}

TEST_CASE("EOS", "[pwpolytrope]") {
    
    std::vector<double> const temperatures { 0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100. }; 

    auto const eos_type = grace::get_param<std::string>("eos", "eos_type") ;
    if( eos_type == "hybrid" ) {
        auto const cold_eos_type = 
            grace::get_param<std::string>("eos", "cold_eos_type") ;
        if( cold_eos_type == "piecewise_polytrope" ) {
            int i{0} ; 
            for( auto temp: temperatures ) {
                test_eos_implementation<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>(temp,std::to_string(i),"sly.h5") ; 
                ++i;
            }
        } else if ( cold_eos_type == "tabulated" ) {
            ERROR("Not implemented yet.") ;
        }
    } else if ( eos_type == "tabulated" ) {
        ERROR("Not implemented yet.") ; 
    }
}

static void read_dset(const std::string &fileName, const std::string &groupName, const std::string &datasetName, std::vector<double> &data) {
    // Open the HDF5 file
    hid_t file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "Error: unable to open file " << fileName << std::endl;
        return;
    }

    // Open the group
    std::string groupPath = "/" + groupName;
    hid_t group = H5Gopen(file, groupPath.c_str(), H5P_DEFAULT);
    if (group < 0) {
        std::cerr << "Error: unable to open group " << groupPath << std::endl;
        H5Fclose(file);
        return;
    }

    // Open the dataset
    std::string datasetPath = groupPath + "/" + datasetName;
    hid_t dataset = H5Dopen(group, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        std::cerr << "Error: unable to open dataset " << datasetPath << std::endl;
        H5Gclose(group);
        H5Fclose(file);
        return;
    }

    // Get the dataspace
    hid_t dataspace = H5Dget_space(dataset);
    if (dataspace < 0) {
        std::cerr << "Error: unable to get dataspace for " << datasetPath << std::endl;
        H5Dclose(dataset);
        H5Gclose(group);
        H5Fclose(file);
        return;
    }

    // Get the number of elements in the dataset
    hsize_t numElements;
    H5Sget_simple_extent_dims(dataspace, &numElements, NULL);

    // Resize the vector to hold the data
    data.resize(numElements);

    // Read the data into the vector
    H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

    // Close resources
    H5Sclose(dataspace);
    H5Dclose(dataset);
    H5Gclose(group);
    H5Fclose(file);
}