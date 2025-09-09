/**
 * @file grace/utils/lifetime_tracker.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief This header contains the lifetime_tracker class used by singleton_holder
 * @version 0.1
 * @date 2023-03-10
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

#ifndef GRACE_UTILS_LIFETIME_TRACKER_HH
#define GRACE_UTILS_LIFETIME_TRACKER_HH

#include <cstdlib>
#include <algorithm>

#include <grace/errors/assert.hh>

/**
 * @brief Lifetimes of grace singleton objects
 * \ingroup system
 * 
 * Objects get deleted from top to bottom in this 
 * list at finalization.
 */
//*****************************************************************************************************
enum unique_objects_lifetimes {
    SYSTEM_UTILITIES=0,
    MPI_RUNTIME,
    GRACE_PROFILING_RUNTIME,
    KOKKOS_RUNTIME,
    DEVICE_RESOURCES,
    P4EST_RUNTIME,
    GRACE_RUNTIME, 
    AMR_CONNECTIVITY,
    AMR_FOREST,
    AMR_GHOSTS,
    GRACE_COORDINATE_SYSTEM,
    GRACE_VARIABLES,
    GRACE_EOS_STORAGE,
    grace_NUM_GLOBAL_OBJECTS 
} ;

namespace utils {
//**************************************************************************************************
namespace pvt {
//! \cond grace_detail
//**************************************************************************************************
/**
 * @brief lifetime_tracker implementation.
 * \ingroup utils
 * 
 * This class allows the user to register an object's lifetime.
 * Objects will be destroyed in order of increasing longevity 
 * at exit.
 */
//**************************************************************************************************
class lifetime_tracker
{
    public:
    //**************************************************************************************************
    /**
     * @brief Construct a new lifetime tracker object
     * 
     * @param x longevity of tracked object
     */
    //**************************************************************************************************
    lifetime_tracker(unsigned int x): longevity_(x) {}
    //**************************************************************************************************
    /**
     * @brief Destroy the lifetime tracker object (virtual)
     */
    //**************************************************************************************************
    virtual ~lifetime_tracker() = 0; 
    //**************************************************************************************************
    /**
     * @brief compare two longevities
     * 
     * @param longevity longevity of object to be compared
     * @param p pointer to lifetime_tracker 
     * @return true if lifetime_tracker object has larger longevity
     * @return false if lifetime_tracker object has smaller longevity
     */
    //**************************************************************************************************
    friend inline bool compare(unsigned int longevity, 
                               lifetime_tracker* p ) ;
    //**************************************************************************************************
    private:
    //**************************************************************************************************
    unsigned int longevity_; //!< longevity of tracked object 
    //**************************************************************************************************
} ;
//**************************************************************************************************
inline lifetime_tracker::~lifetime_tracker() {} ;
//**************************************************************************************************

//**************************************************************************************************
// Extern data structures. Handled at a very low level, do not mess with them.
using tracker_array = lifetime_tracker ** ;
extern tracker_array ptracker_array ; 
extern unsigned int elements ; 
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief concrete implementation of lifetime_tracker for object of type T.
 * \ingroup utils
 * @tparam destroyer struct which provides the correct deleter for the object.
 */
//**************************************************************************************************
template < class destroyer > 
class concrete_lifetime_tracker : public lifetime_tracker
{
public:
    //**************************************************************************************************
    /**
     * @brief Construct a new concrete lifetime tracker object
     * 
     * @param longevity longevity of object
     */
    //**************************************************************************************************
    concrete_lifetime_tracker(unsigned int longevity )
        : lifetime_tracker(longevity){} 
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Destroy the concrete lifetime tracker object
     * 
     * This deletes the object that was tracked. The call to this 
     * destructor is scheduled at an opportune moment by <code>set_longevity</code>,
     * see related documentation.
     */
    //**************************************************************************************************
    ~concrete_lifetime_tracker() { destroyer::destroy() ; }
    //**************************************************************************************************
}; 

//**************************************************************************************************
/**
 * @brief Function scheduled by \code std::atexit() \endcode.
 * \ingroup utils
 * This function pops objects from the top of the ptracker_array stack in the order 
 * of their longevity, triggering the destruction of the associated concrete_lifetime_tracker
 * which itself destroys the object.
 */
void at_exit_func() ; //!< forward declaration
//**************************************************************************************************

//**************************************************************************************************
inline bool compare(unsigned int longevity, 
                    lifetime_tracker* p ) { return p->longevity_ > longevity; }
//**************************************************************************************************
//! \endcond
} // namespace pvt 

//**************************************************************************************************
/**
 * @brief Set the longevity of dynamically allocated object of type T.
 * \ingroup utils
 * @tparam destroyer struct which provides the correct deleter for the object.
 * @param longevity longevity of the object.
 * 
 * This function schedules the execution of the destructor from <code>concrete_lifetime_tracker</code>
 * by means of <code>std::atexit()</code>.
 */
//**************************************************************************************************
template<class destroyer>
void set_longevity(unsigned int longevity)
{
    pvt::tracker_array ptracker_new = static_cast<pvt::tracker_array> (
        std::realloc(pvt::ptracker_array, sizeof(*pvt::ptracker_array) * (pvt::elements + 1 ) ) 
    ) ;
    ASSERT(ptracker_new != nullptr,
           "Allocation failed." ) ;
    pvt::ptracker_array = ptracker_new ; 
    pvt::lifetime_tracker * p = 
        new pvt::concrete_lifetime_tracker<destroyer>(longevity) ;
    pvt::tracker_array pos = std::upper_bound( pvt::ptracker_array, 
        pvt::ptracker_array+ pvt::elements, 
        longevity, pvt::compare) ;

    std::copy_backward(pos, pvt::ptracker_array + pvt::elements, 
        pvt::ptracker_array + pvt::elements + 1 ) ; 
    *pos = p ; 
    ++(pvt::elements) ;
    std::atexit(pvt::at_exit_func) ; 
}
//**************************************************************************************************


} // namespace utils

 #endif 