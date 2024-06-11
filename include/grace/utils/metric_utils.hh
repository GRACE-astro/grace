/**
 * @file metric_utils.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-04
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

#ifndef GRACE_UTILS_METRIC_HH
#define GRACE_UTILS_METRIC_HH

#include <grace_config.h>

#include <grace/utils/math.hh>
#include <grace/utils/constexpr.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <array> 

namespace grace {

/**
 * @brief Helper class for metric tensor manipulations.
 * \ingroup numerics
 */
struct metric_array_t {
/**
 * @brief Constructor. This is marked 
 *        <code>__host__ __device__</code>
 *        to allow for construction in parallel
 *        regions.
 * 
 * @param g_ Array containing the nontrivial components of the spatial 
 *           metric.
 * @param beta_ Array containing components of contravariant shift.
 * @param alp_ Lapse function.
 * NB: The order of metric components should be: (XX,XY,XZ,YY,YZ,ZZ).
 */
GRACE_HOST_DEVICE
metric_array_t( std::array<double,6>const& g_
              , std::array<double,3>const& beta_ 
              , double const& alp_ )
    : _g(g_), _ginv(), _beta(beta_), _alp(alp_), _sqrtg(1.)
{
    _sqrtg = -(math::int_pow<2>(_g[2])*_g[3]) 
             + 2*_g[1]*_g[2]*_g[4] 
             - _g[0]*math::int_pow<2>(_g[4]) 
             - math::int_pow<2>(_g[1])*_g[5] 
             + _g[0]*_g[3]*_g[5]    ;
    _ginv[0] = (_g[3] - math::int_pow<2>(_g[4]))/_sqrtg;
    _ginv[1] = (-_g[1] + _g[2]*_g[4])/_sqrtg;
    _ginv[2] = (-(_g[2]*_g[3]) + _g[1]*_g[4])/_sqrtg;
    _ginv[3] = (2*_g[0] - math::int_pow<2>(_g[2]))/_sqrtg;
    _ginv[4] = (_g[1]*_g[2] - _g[0]*_g[4])/_sqrtg ; 
    _ginv[5] = (-math::int_pow<2>(_g[1]) + _g[0]*_g[3]) / _sqrtg;
}
/**
 * @brief Get a component of the covariant metric.
 * 
 * @param i Component index.
 * @return double The component of the metric.
 * NB: The index i translates to tensor indices as in the 
 * constructor: (XX,XY,XZ,YY,YZ,ZZ).
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
gamma(int i) const {
    return _g[i] ; 
}
/**
 * @brief Get a component of the contravariant metric.
 * 
 * @param i Component index.
 * @return double The component of the metric.
 * NB: The index i translates to tensor indices as in the 
 * constructor: (XX,XY,XZ,YY,YZ,ZZ).
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
invgamma(int i) const {
    return _ginv[i] ;
}
/**
 * @brief Get a component of the contravariant shift.
 * 
 * @param i Component index.
 * @return double The component of the shift vector.
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
beta(int i) const {
    return _beta[i] ; 
}
/**
 * @brief Get the lapse function.
 * 
 * @return double The lapse function.
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
alp() const { return _alp ; }
/**
 * @brief Get the square root of the 
 *        determinant of the spatial metric.
 * 
 * @return double \f$\sqrt{\gamma}\f$.
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
sqrtg() const { return _sqrtg ; }
/**
 * @brief Raise the index of a 3-covector.
 * 
 * @param v Components of the 3-covector.
 * @return std::array<double,3> The 3-vector 
 *         obtained as \f$\gamma^{i j} v_j\f$.
 */
std::array<double,3> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
raise(std::array<double,3> const& v ) const {
    return std::array<double,3> {
          _ginv[XX] * v[X] + _ginv[XY] * v[Y] + _ginv[XZ] * v[Z]
        , _ginv[XY] * v[X] + _ginv[YY] * v[Y] + _ginv[YZ] * v[Z]
        , _ginv[XZ] * v[X] + _ginv[YZ] * v[Y] + _ginv[ZZ] * v[Z]
    } ; 
}
/**
 * @brief Lower the index of a 3-vector.
 * 
 * @param v Components of the 3-vector.
 * @return std::array<double,3> The 3-covector 
 *         obtained as \f$\gamma_{i j} v^j\f$.
 */
std::array<double,3> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
lower(std::array<double,3> const& v ) const {
    return std::array<double,3> {
          _g[XX] * v[X] + _g[XY] * v[Y] + _g[XZ] * v[Z]
        , _g[XY] * v[X] + _g[YY] * v[Y] + _g[YZ] * v[Z]
        , _g[XZ] * v[X] + _g[YZ] * v[Y] + _g[ZZ] * v[Z]
    } ; 
}
/**
 * @brief Compute the square norm of a 3-vector.
 * 
 * @param v Components of the 3-vector.
 * @return double \f$v^i v_i\f$.
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
square_vec(std::array<double,3> const& v ) const {
    return  _g[XX] * v[X] * v[X] 
          + _g[YY] * v[Y] * v[Y] 
          + _g[ZZ] * v[Z] * v[Z] 
          + 2. * ( _g[XY] * v[X] * v[Y]  
                 + _g[XZ] * v[X] * v[Z] 
                 + _g[YZ] * v[Y] * v[Z] ) ; 
}
/**
 * @brief Compute the square norm of a 3-covector.
 * 
 * @param v Components of the 3-covector.
 * @return double \f$v^i v_i\f$.
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
square_covec(std::array<double,3> const& v ) const {
    return  _ginv[XX] * v[X] * v[X] 
          + _ginv[YY] * v[Y] * v[Y] 
          + _ginv[ZZ] * v[Z] * v[Z] 
          + 2. * ( _ginv[XY] * v[X] * v[Y]  
                 + _ginv[XZ] * v[X] * v[Z] 
                 + _ginv[YZ] * v[Y] * v[Z] ) ; 
}
/**
 * @brief Compute the full contraction of two
 *        symmetric rank 2 spatial tensors given
 *        their contravariant components.
 * 
 * @param A Contravariant components of the first 2-tensor.
 * @param B Contravariant components of the other 2-tensor.
 * @return double \f$A^{i j} B_{i j}\f$.
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
contract_symm_2tensors_up( std::array<double,6> const& A
                         , std::array<double,6> const& B ) const 
{
    return 2*A[XZ]*B[XX]*_g[XX]*_g[XZ] + 2*A[YZ]*B[XX]*_g[XY]*_g[XZ] + 2*A[XZ]*B[XY]*_g[XY]*_g[XZ] + 2*A[YY]*B[XY]*_g[XY]*_g[YY] + 
           2*A[YZ]*B[XY]*_g[XZ]*_g[YY] + 2*A[XZ]*B[XY]*_g[XX]*_g[YZ] + 2*A[YZ]*B[XY]*_g[XY]*_g[YZ] + 2*A[YY]*B[XZ]*_g[XY]*_g[YZ] + 
           2*A[XZ]*B[YY]*_g[XY]*_g[YZ] + 2*A[ZZ]*B[XY]*_g[XZ]*_g[YZ] + 2*A[YZ]*B[XZ]*_g[XZ]*_g[YZ] + 2*A[XZ]*B[YZ]*_g[XZ]*_g[YZ] + 
           2*A[YZ]*B[YY]*_g[YY]*_g[YZ] + 2*A[YY]*B[YZ]*_g[YY]*_g[YZ] + 
           2*(A[XZ]*B[XZ]*_g[XX] + A[YZ]*B[XZ]*_g[XY] + A[XZ]*B[YZ]*_g[XY] + A[ZZ]*B[XZ]*_g[XZ] + A[XZ]*B[ZZ]*_g[XZ] + 
           A[YZ]*B[YZ]*_g[YY] + A[ZZ]*B[YZ]*_g[YZ] + A[YZ]*B[ZZ]*_g[YZ])*_g[ZZ] + A[YY]*B[XX]*math::int_pow<2>(_g[XY]) + 
           2*A[XY]*(B[XX]*_g[XX]*_g[XY] + B[XZ]*_g[XY]*_g[XZ] + B[XY]*_g[XX]*_g[YY] + B[YY]*_g[XY]*_g[YY] + B[YZ]*_g[XZ]*_g[YY] + 
           B[XZ]*_g[XX]*_g[YZ] + B[YZ]*_g[XY]*_g[YZ] + B[ZZ]*_g[XZ]*_g[YZ] + B[XY]*math::int_pow<2>(_g[XY])) + A[ZZ]*B[XX]*math::int_pow<2>(_g[XZ]) + 
           2*A[XZ]*B[XZ]*math::int_pow<2>(_g[XZ]) + A[XX]*(2*B[XY]*_g[XX]*_g[XY] + 2*B[XZ]*_g[XX]*_g[XZ] + 2*B[YZ]*_g[XY]*_g[XZ] + 
           B[XX]*math::int_pow<2>(_g[XX]) + B[YY]*math::int_pow<2>(_g[XY]) + B[ZZ]*math::int_pow<2>(_g[XZ])) + A[YY]*B[YY]*math::int_pow<2>(_g[YY]) + 
           A[ZZ]*B[YY]*math::int_pow<2>(_g[YZ]) + 2*A[YZ]*B[YZ]*math::int_pow<2>(_g[YZ]) + A[YY]*B[ZZ]*math::int_pow<2>(_g[YZ]) + A[ZZ]*B[ZZ]*math::int_pow<2>(_g[ZZ]) ; 
}
/**
 * @brief Compute the full contraction of two
 *        symmetric rank 2 spatial tensors given
 *        their covariant components.
 * 
 * @param A Covariant components of the first 2-tensor.
 * @param B Covariant components of the other 2-tensor.
 * @return double \f$A^{i j} B_{i j}\f$.
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
contract_symm_2tensors_low( std::array<double,6> const& A
                          , std::array<double,6> const& B ) const 
{
    return 2*A[XZ]*B[XX]*_ginv[XX]*_ginv[XZ] + 2*A[YZ]*B[XX]*_ginv[XY]*_ginv[XZ] + 2*A[XZ]*B[XY]*_ginv[XY]*_ginv[XZ] + 2*A[YY]*B[XY]*_ginv[XY]*_ginv[YY] + 
           2*A[YZ]*B[XY]*_ginv[XZ]*_ginv[YY] + 2*A[XZ]*B[XY]*_ginv[XX]*_ginv[YZ] + 2*A[YZ]*B[XY]*_ginv[XY]*_ginv[YZ] + 2*A[YY]*B[XZ]*_ginv[XY]*_ginv[YZ] + 
           2*A[XZ]*B[YY]*_ginv[XY]*_ginv[YZ] + 2*A[ZZ]*B[XY]*_ginv[XZ]*_ginv[YZ] + 2*A[YZ]*B[XZ]*_ginv[XZ]*_ginv[YZ] + 2*A[XZ]*B[YZ]*_ginv[XZ]*_ginv[YZ] + 
           2*A[YZ]*B[YY]*_ginv[YY]*_ginv[YZ] + 2*A[YY]*B[YZ]*_ginv[YY]*_ginv[YZ] + 
           2*(A[XZ]*B[XZ]*_ginv[XX] + A[YZ]*B[XZ]*_ginv[XY] + A[XZ]*B[YZ]*_ginv[XY] + A[ZZ]*B[XZ]*_ginv[XZ] + A[XZ]*B[ZZ]*_ginv[XZ] + 
           A[YZ]*B[YZ]*_ginv[YY] + A[ZZ]*B[YZ]*_ginv[YZ] + A[YZ]*B[ZZ]*_ginv[YZ])*_ginv[ZZ] + A[YY]*B[XX]*math::int_pow<2>(_ginv[XY]) + 
           2*A[XY]*(B[XX]*_ginv[XX]*_ginv[XY] + B[XZ]*_ginv[XY]*_ginv[XZ] + B[XY]*_ginv[XX]*_ginv[YY] + B[YY]*_ginv[XY]*_ginv[YY] + B[YZ]*_ginv[XZ]*_ginv[YY] + 
           B[XZ]*_ginv[XX]*_ginv[YZ] + B[YZ]*_ginv[XY]*_ginv[YZ] + B[ZZ]*_ginv[XZ]*_ginv[YZ] + B[XY]*math::int_pow<2>(_ginv[XY])) + A[ZZ]*B[XX]*math::int_pow<2>(_ginv[XZ]) + 
           2*A[XZ]*B[XZ]*math::int_pow<2>(_ginv[XZ]) + A[XX]*(2*B[XY]*_ginv[XX]*_ginv[XY] + 2*B[XZ]*_ginv[XX]*_ginv[XZ] + 2*B[YZ]*_ginv[XY]*_ginv[XZ] + 
           B[XX]*math::int_pow<2>(_ginv[XX]) + B[YY]*math::int_pow<2>(_ginv[XY]) + B[ZZ]*math::int_pow<2>(_ginv[XZ])) + A[YY]*B[YY]*math::int_pow<2>(_ginv[YY]) + 
           A[ZZ]*B[YY]*math::int_pow<2>(_ginv[YZ]) + 2*A[YZ]*B[YZ]*math::int_pow<2>(_ginv[YZ]) + A[YY]*B[ZZ]*math::int_pow<2>(_ginv[YZ]) + A[ZZ]*B[ZZ]*math::int_pow<2>(_ginv[ZZ]) ; 
}
/**
 * @brief Compute the full contraction of two
 *        symmetric rank 2 spatial tensors.
 * 
 * @param A Covariant components of the first 2-tensor.
 * @param B Contravariant components of the other 2-tensor.
 * @return double \f$A_{i j} B^{i j}\f$.
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
contract_symm_2tensors( std::array<double,6> const& A
                      , std::array<double,6> const& B ) const 
{
    return A[XX]*B[XX] + 2*A[XY]*B[XY] + 2*A[XZ]*B[XZ] + A[YY]*B[YY] + 2*A[YZ]*B[YZ] + A[ZZ]*B[ZZ];
}

std::array<double,6> _g, _ginv ; //!< Spatial metric and inverse components.
std::array<double,3> _beta     ; //!< Contravariant shift vector. 
double _alp               ; //!< Lapse function.
double _sqrtg             ; //!< Square root of spatial metric determinant.

 private: 
    //! Helpers for tensor and vector indices.
    static constexpr int XX = 0 ; 
    static constexpr int XY = 1 ; 
    static constexpr int XZ = 2 ; 
    static constexpr int YY = 3 ; 
    static constexpr int YZ = 4 ; 
    static constexpr int ZZ = 5 ; 
    static constexpr int X = 0 ; 
    static constexpr int Y = 1 ; 
    static constexpr int Z = 2 ; 

} ;

}

#endif /* GRACE_UTILS_METRIC_HH */