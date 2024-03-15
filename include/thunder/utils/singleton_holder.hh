/**
 * @file thunder/utils/singleton_holder.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Contains definition of singleton holder class and related utilities.
 * @version 0.1
 * @date 2023-03-10
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
#ifndef THUNDER_UTILS_SINGLETON_HOLDER_HH
#define THUNDER_UTILS_SINGLETON_HOLDER_HH

#include <type_traits>

#include <thunder/utils/type_name.hh>
#include <thunder/utils/lifetime_tracker.hh>
#include <thunder/utils/creation_policies.hh>
#include <thunder/errors/assert.hh>
#include <thunder/errors/error.hh>

namespace utils {
//******************************************************************************************************
/**
 * \defgroup utils Utilities for Thunder.
 */
//******************************************************************************************************
/**
 * @brief singleton_holder class.
 * \ingroup utils
 * @tparam T type of the object being managed
 * 
 * A singleton_holder manages the lifetime of a unique object of type <code>T</code>.
 * It ensures that at most one instance of <code>T</code> exists at any given time and that 
 * the singleton object lives as long as required by the application. It also 
 * provides a static method to access the reference to the singleton itself.
 */
//******************************************************************************************************
template < typename T,                                 // the type held by the singleton
           template <typename> class creation_policy   // the creation/destruction    
           = memory::default_create>                   // policy for the object
class singleton_holder {
    //**************************************************************************************************
public:
    //**************************************************************************************************
    using Type = singleton_holder<T,creation_policy> ; 
    //**************************************************************************************************
    using object_type  = T ; //!< Type of object held by singleton_holder.
    using pointer_type = T*; //!< Type of pointer to object held by singleton_holder.
    using ref_type     = T&; //!< Type of reference to object held by singleton_holder.
    //**************************************************************************************************

    //**************************************************************************************************
    /**
        * @brief get reference to protected instance.
        * 
        * @return ref_type reference to unique instance of object_type.
        * 
        * Usage: Calling  <code>singleton_holder<T>::get()</code> returns a reference 
        * to the unique instance of type <code>T</code>. To enforce uniqueness of the 
        * singleton, constructors and destructors of <code> T </code> are inaccessible to
        * user code. For this reason it is important to enforce that the result of this function
        * is assigned to a reference and that no copy is performed.
        * \code
        * auto & x = singleton_holder<my_type>::get() //OK 
        * auto x = singleton_holder<my_type>::get() // COMPILATION ERROR : ~my_type() noexcept is inaccessible 
        * \endcode
        */
    [[nodiscard]] static ref_type get() ;
    //**************************************************************************************************

    //**************************************************************************************************
    /**
    * @brief Shorthand to initialize the instance 
    * 
    * @tparam ArgT The parameter pack is forwarded to the object constructor.
    * @param args Forwarded to constructor.
    */
    //**************************************************************************************************
    template < typename ... ArgT > 
    static void initialize( ArgT ... args ) ;
    //**************************************************************************************************

    //**************************************************************************************************
protected:
    //**************************************************************************************************
    friend class pvt::concrete_lifetime_tracker<Type> ; 
    //**************************************************************************************************
    /**
    * @brief Destroy the singleton.
    */
    static void destroy() noexcept;
    //**************************************************************************************************
    
    //**************************************************************************************************
    /**
    * @brief (Never) construct a new singleton holder object
    * 
    * Private because the singleton lifetime is protected from user interference
    */
    singleton_holder();
    //**************************************************************************************************

    //**************************************************************************************************
    /**
    * @brief (Never) destroy the singleton holder object
    * 
    * Private because the singleton lifetime is protected from user interference
    */
    ~singleton_holder();
    //**************************************************************************************************

    //**Static members**********************************************************************************
    static pointer_type instance_ ;    //!< Pointer to protected instance.
    static bool deleted_ ;             //!< Is the pointer valid.     
    //**************************************************************************************************
};
//******************************************************************************************************


//======================================================================================================
//
//  IMPLEMENTATION
//
//======================================================================================================


//******************************************************************************************************
template < typename T,                                 // the type held by the singleton
           template <typename> class creation_policy > // the creation/destruction policy for the object
[[nodiscard]] T& singleton_holder<T,creation_policy>::get() 
{
    if ( ! instance_ ) {
        if constexpr ( std::is_default_constructible<T>::value )
        {
            ASSERT(!deleted_,
                "Access to singleton requested after"
                " the end of its lifetime." ) ; 
            instance_ = creation_policy<T>::create() ;
            set_longevity<Type>(T::longevity) ;
        } else {
            ERROR("Type " << type_name<T>() << " is not default constructible,"
                  " please call singleton_holder<"<<type_name<T>()<<">::initialize(args...) before"
                  " attempting to get().") ;
        }
    }
    return *instance_ ;
}
//*****************************************************************************************************

//*****************************************************************************************************
template < typename T,                                 // the type held by the singleton
           template <typename> class creation_policy > // the creation/destruction policy for the object
template<typename ... ArgT>
void singleton_holder<T,creation_policy>::initialize(ArgT ... args) 
{
    if ( instance_ ) return; 
    ASSERT(!deleted_,
            "Initialization of singleton of type " 
            << utils::type_name<T>() << 
            " requested after"
            " the end of its lifetime." ) ; 
    instance_ = creation_policy<T>::create(std::forward<ArgT>(args)...) ;
    set_longevity<Type>(T::longevity) ;
}
//*****************************************************************************************************

//*****************************************************************************************************
template < typename T,                                 // the type held by the singleton
           template <typename> class creation_policy > // the creation/destruction policy for the object
void singleton_holder<T,creation_policy>::destroy() noexcept
{
    deleted_ = true ;
    creation_policy<T>::destroy(instance_) ;
}
//**************************************************************************************************

// instantiation of static member variables 
//*****************************************************************************************************
template < typename T,                                 // the type held by the singleton
           template <typename> class creation_policy > // the creation/destruction policy for the object
bool singleton_holder<T,creation_policy>::deleted_ = false ;
//*****************************************************************************************************
template < typename T,                                 // the type held by the singleton
           template <typename> class creation_policy > // the creation/destruction policy for the object
T* singleton_holder<T,creation_policy>::instance_ = nullptr ;
//*****************************************************************************************************

//*****************************************************************************************************
} // namespace utils 

#endif 