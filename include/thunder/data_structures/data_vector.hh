/**
 * @file data_vector.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-06-13
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference 
 * methods to simulate relativistic astrophysical systems and plasma
 * dynamics.
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
#ifndef DE1341A8_9A07_4040_A152_1C0C7418420F
#define DE1341A8_9A07_4040_A152_1C0C7418420F

#include <Kokkos_Core.hpp>
#include <memory_defaults.hh>

namespace thunder { 

template< size_t ndim=THUNDER_NSPACEDIM > 
class data_vector 
{ 
 private: 
    using data_type = Kokkos::View<double ****, DefaultSpace> ; 
    //! Staggering of the grid function (determines number of points in each 
    //! dimension )    
    bool _sx, _sy, _sz; 
    unsigned int _ngz    ;
    unsigned int _ntl    ; 
 
    data_type _data     ;  //!< The array holding the grid data
    data_type _data_p   ;  //!< Past time level of grid data 
    data_type _data_p_p ;  //!< Second past time level of grid data

 public: 
    /**
     * @brief Construct a new data vector object
     * 
     * @param varname The name of the variable
     * 
     * The data is initialized to be of shape 
     * (num_local_quadrants,num_points_x,num_points_y,num_points_z).
     * Number of points in each dimension includes ghost zones.
     * This routine allocates memory on device.
     */
    data_vector(  std::string const& varname
                , size_t nq, size_t nx, size_t ny, size_t nz, size_t ngz=0 
                , bool sx=false, bool sy=false, bool sz=false
                , int ntl=1) 
        : _sx(sx), _sy(sy), _sz(sz), _ngz(ngz), _ntl(ntl)
        , _data( varname
               , nz + 2*_ngz + static_cast<int>(_sz)
               , ny + 2*_ngz + static_cast<int>(_sy)
               , nx + 2*_ngz + static_cast<int>(_sx), nq)
    {
        if( _ntl > 1 ) { 
            _data_p = Kokkos::View<double ****, DefaultSpace>( varname
                                                             , nz + static_cast<int>(_sz)
                                                             , ny + static_cast<int>(_sy)
                                                             , nx + static_cast<int>(_sx)
                                                             , nq ) ; 
        } 
        if ( _ntl > 2 ) { 
            _data_p_p = Kokkos::View<double ****, DefaultSpace>( varname
                                                               , nz + static_cast<int>(_sz)
                                                               , ny + static_cast<int>(_sy)
                                                               , nx + static_cast<int>(_sx)
                                                               , nq ) ; 
        }
    };  
    /**
     * @brief Destroy the data vector object. Releases memory 
     *        on device. 
     */
    ~data_vector() ; 
    /**
     * @brief Reallocate the data array after 
     *        resizing of the grid. This could be
     *        due to refinement, coarsening, or 
     *        parallel partitioning of the grid. 
     * NB: Re-allocation does \b not preserve the contents
     * of the data vector. 
     */
    void realloc() ; 
    /**
     * @brief Create a mirror view on host.
     * 
     * @return decltype(auto) The mirror view.
     * 
     * A mirror has the same memory layout as the 
     * original view but it resides on host memory.
     * It can be used to transfer data between host 
     * and device. Note that this routine does not 
     * perform any memory transfers. 
     */
    THUNDER_ALWAYS_INLINE HOST decltype(auto) 
    get_host_mirror() 
    {
        return Kokkos::get_mirror_view( (*this).get_physical_data() ) ; 
    }; 
    /**
     * @brief Get the physical data object
     * 
     * @return THUNDER_ALWAYS_INLINE Subview containing physical data at current time.
     * 
     * This method returns a subview of the data where the ghostzones have been removed.
     */
    THUNDER_ALWAYS_INLINE HOST decltype(auto) 
    get_physical_data() 
    { 
        return Kokkos::subview( _data
                              , Kokkos::Pair<int,int>( _ngz, _data.extent(0) - _ngz )
                              , Kokkos::Pair<int,int>( _ngz, _data.extent(1) - _ngz )
                              , Kokkos::Pair<int,int>( _ngz, _data.extent(2) - _ngz )
                              , Kokkos::ALL() )
    }
    /**
     * @brief Get underlying view.
     * 
     * @return data_type The view contained by 
     *         <code>data_vector</code> object. 
     * 
     * This routine is useful to get direct access 
     * to the view e.g. so that it can be copied by 
     * value into a device kernel. Note that copying 
     * the <code>View</code> is cheap but not free,
     * since no memory allocations happen but the 
     * refcount of the object needs to be updated. 
     */
    template< size_t tl = 0 > 
    data_type get() ;    
} ; 

template<ndim> 
template<> 
data_vector<ndim>::data_type data_vector<ndim>::get<0>()
{
    return _data ; 
}

template<ndim> 
template<1> 
data_vector<ndim>::data_type data_vector<ndim>::get<1>()
{
    ASSERT_DBG( num_timelevels > 1, 
                MakeString{} << "Trying to access timelevel 1"  
                << " of gf " << _data.label() << " which only has one"
                << " active timelevel.") ; 
    return _data_p ; 
}

template<ndim> 
template<2> 
data_vector<ndim>::data_type data_vector<ndim>::get<2>()
{
    ASSERT_DBG( num_timelevels > 2, 
                MakeString{} << "Trying to access timelevel 2"  
                << " of gf " << _data.label() << " which does not exist.") ; 
    return _data_p_p ; 
}

template< bool is_vector = false 
        , bool stagger_x = false 
        , bool stagger_y = false 
        , bool stagger_z = false >
class data_vector<is_vector,stagger_x,stagger_y,stagger_z,2>
{
    private:
        using data_type = Kokkos::View<double ***> ; 
} ; 

}

#endif /* DE1341A8_9A07_4040_A152_1C0C7418420F */
