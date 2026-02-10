/**
 * @file tabulated.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2026-02-06
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
#ifndef GRACE_PHYSICS_TABEOS_HH
#define GRACE_PHYSICS_TABEOS_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/physics/eos/eos_base.hh>

#include <Kokkos_Core.hpp>

namespace grace {

// interpolator for tabeos 
// spacing assumed constant 
struct tabeos_linterp_t {
    tabeos_linterp_t(
        Kokkos::View<double ****> tabs,
        Kokkos::View<double *> ar, 
        Kokkos::View<double *> at, 
        Kokkos::View<double *> ay
    ) : _tables(tabs), _logrho(ar), _logT(at), _ye(ay) 
    {
        idr =  1./(_logrho(1)-_logrho(0)) ; 
        idt = 1./(_logT(1)-_logT(0))      ; 
        idy = 1./(_ye(1)-_ye(0))          ; 
    }

    double KOKKOS_INLINE_FUNCTION lrho(int idx) const { return _logrho(idx) ; } 
    double KOKKOS_INLINE_FUNCTION ltemp(int idx) const { return _logT(idx) ; } 
    double KOKKOS_INLINE_FUNCTION ye(int idx) const { return _ye(idx) ; } 

    double KOKKOS_ALWAYS_INLINE operator() (int irho, int itemp, int iye, int varidx) const {
        return _tables(irho,itemp,iye,varidx) ; 
    }

    double KOKKOS_INLINE_FUNCTION
    interp(double lrho, double ltemp, double ye, int const& idx) const 
    {
        std::array<double,1> res ;
        interp(lrho,ltemp,ye,std::array<int,1>(idx),res) ; 
        return res[0] ;
    }

    template< int N >
    void KOKKOS_INLINE_FUNCTION
    interp(double lrho, double ltemp, double ye, std::array<int,N> const& idx,  std::array<double,N>& res) const 
    {
        for( int iv=0; iv<N; ++iv) res[iv] = 0 ; 
        // indices
        int i,j,k;
        getidx(lrho,ltemp,ye,i,j,k) ; 
        // weights 
        double wr[2],wt[2],wy[2]; 
        getw(lrho,i,_logrho,idr,wr) ;
        getw(ltemp,j,_logtemp,idt,wt) ;
        getw(ye,k,_ye,idy,wy) ; 
        // interpolate 
        for( int ii=0; ii<1; ++ii) {
            for( int jj=0; jj<1; ++jj) {
                for( int kk=0; kk<1; ++kk) {
                    for( int iv=0; iv<N; ++iv) {
                        res[iv] += wr[ii] * wt[jj] * wy[kk] * _tables(
                            i+ii, j+jj, k+kk, idx[iv]
                        ) ; 
                    }
                }
            }
        }
    } 

    void KOKKOS_INLINE_FUNCTION
    getidx(double x, double y, double z, int& i, int& j, int& k) const {
        i = idr * x ;
        j = idt * y ; 
        k = idy * z ; 
    }

    void KOKKOS_INLINE_FUNCTION
    getw(double x, int i, readonly_view_t<double> ax, double ih, double* w) const {
        double lam = (x - ax(i)) * ih ; 
        w[0] = 1.-lam ;
        w[1] = lam    ; 
    }

    readonly_view_t<double> _logrho, _logT, _ye ; 
    Kokkos::View<double ****> _tables ;

    double idr,idt,idy;
} ; 

/**
 * @brief Concrete EOS type corresponding to 
 *        a tabulated EOS.
 * \ingroup eos
 * @tparam cold_eos_t Type of cold EOS. 
 * The methods of this class are explicit implementations
 * of public methods from <code>eos_base_t</code>.
 */
template < typename cold_eos_t >
class tabulated_eos_t
    : public eos_base_t<tabulated_eos_t> 
{
    /**************************************************************************************/
    using err_t  = unsigned int ; 
    using base_t = eos_base_t<tabulated_eos_t> ;
    /**************************************************************************************/
    enum TEOS_VIDX : int {
        TABPRESS,
        TABEN,
        TABCSND2,
        TABENTROPY,
        TABMUE,
        TABMUP,
        
        N_TAB_VARS 
    } ; 
    /**************************************************************************************/
    tabulated_eos_t() = default ; 
    /**************************************************************************************/
    tabulated_eos_t(
        Kokkos::View<double *, grace::default_space> _tabeos, 
        Kokkos::View<double *,  grace::default_space> _logrho  , 
        Kokkos::View<double *,  grace::default_space> _logT    , 
        Kokkos::View<double *,  grace::default_space> _ye      ,
        double _energy_shift,
        double _rhomax, double _rhomin,
        double _tempmax, double _tempmin,
        double _yemax, double _yemin,
        double _baryon_mass, 
        double _c2p_epsmin, double _c2p_epsmax,
        double _c2p_hmin, double _c2p_hmax,
        bool _atmo_is_beta_eq
    ) : tables(_tabeos,_logrho,_logT,_ye)
      , nrho(_logrho.size())
      , nT(_logT.size())
      , nye(_ye.size())
      , base_t(
        _energy_shift, 
        _rhomax, _rhomin,
        _tempmax, _tempmin,
        _yemax, _yemin,
        _baryon_mass,
        _c2p_epsmin, _c2p_epsmax,
        _c2p_hmin, _c2p_hmax,
        _atmo_is_beta_eq,
        false )
    { }
    /**************************************************************************************/
    /**
     * @brief Get pressure given eps rho and ye.
     * 
     * @param eps Specific internal energy.
     * @param rho Rest-mass density.
     * @param ye  Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press__eps_rho_ye_impl(double& eps, double& rho, double& ye, err_t& err) const 
    {
        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ; 
        auto lrho = Kokkos::log(rho) ; 
        auto ltemp = ltemp__eps_lrho_ye(eps-this->energy_shift,lrho,ye,err) ; 
        return tables.interp(lrho,ltemp,ye,TABPRESS) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Get pressure and temperature given eps rho and ye.
     * 
     * @param temp Temperature.
     * @param eps Specific internal energy.
     * @param rho Rest-mass density.
     * @param ye  Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_temp__eps_rho_ye_impl(double& temp, double& eps, double& rho, double& ye, err_t& err) const 
    {
        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ; 
        auto lrho = Kokkos::log(rho) ; 
        auto ltemp = ltemp__eps_lrho_ye(eps-this->energy_shift,lrho,ye,err) ;
        temp = Kokkos::exp(ltemp) ; 
        return tables.interp(lrho,ltemp,ye,TABPRESS) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Get pressure given temperature rho and ye.
     * 
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye  Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE 
    press__temp_rho_ye_impl(double& temp, double& rho, double& ye, err_t& err) const 
    {
        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ; 
        double lrho  = Kokkos::log(rho)  ;
        double ltemp = Kokkos::log(temp) ; 
        return tables.interp(lrho,ltemp,ye,TABPRESS) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Get eps given temperature rho and ye.
     * 
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The specific internal energy.
     */
    double GRACE_HOST_DEVICE 
    eps__temp_rho_ye_impl(double& temp, double& rho, double& ye, err_t& err) const 
    {
        limit_rho(rho,err)   ; 
        limit_ye(ye,err)     ; 
        limit_temp(temp,err) ; 
        return tables.interp(Kokkos::log(rho),Kokkos::log(temp),ye,TABEPS) + this->energy_shift; 
    }
    /**************************************************************************************/
    /**
     * @brief Pressure, specific enthalpy and 
     *        square sound speed given eps, 
     *        rho and ye.
     * 
     * @param h Specific enthalpy.
     * @param csnd2 Sound speed squared.
     * @param eps Specific internal energy.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_h_csnd2__eps_rho_ye_impl( double &h, double &csnd2, double &eps
                             , double &rho, double &ye
                             , error_type &err) const 
    {

        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ; 
        auto lrho = Kokkos::log(rho) ; 
        auto ltemp = ltemp__eps_lrho_ye(eps-this->_energy_shift,lrho,ye,err) ;
        auto press = tables.interp(lrho,ltemp,ye,TABPRESS) ; 
        csnd2 = tables.interp(lrho,ltemp,ye,TABCSND2) ; 
        h = 1 + eps + press/rho ; 
        return press ; 
    }
    /**************************************************************************************/
    /**
     * @brief Pressure, specific enthalpy and 
     *        square sound speed given temp, 
     *        rho and ye.
     * 
     * @param h Specific enthalpy.
     * @param csnd2 Sound speed squared.
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_h_csnd2__temp_rho_ye_impl( double &h, double &csnd2, double &temp
                              , double &rho, double &ye
                              , error_type &err) const 
    {
        limit_rho(rho,err)   ; 
        limit_ye(ye,err)     ; 
        limit_temp(temp,err) ; 
        auto lrho = Kokkos::log(rho) ; 
        auto ltemp = Kokkos::log(temp) ; 
        eps = tables.interp(lrho,ltemp,ye,TABEPS) + this->energy_shift; 
        auto press = tables.interp(lrho,ltemp,ye,TABPRESS) ; 
        csnd2 = tables.interp(lrho,ltemp,ye,TABCSND2) ; 
        h = 1 + eps + press/rho ; 
        return press ; 
    }
    /**************************************************************************************/
    /**
     * @brief Pressure, specific internal energy
     *        and square sound speed given temperature,
     *        rho and ye.
     * 
     * @param eps Specific internal energy.
     * @param csnd2 Square sound speed.
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_eps_csnd2__temp_rho_ye( double &eps, double &csnd2, double &temp
                                , double &rho, double &ye
                                , error_type& err) const 
    {
        limit_rho(rho,err)   ; 
        limit_ye(ye,err)     ; 
        limit_temp(temp,err) ; 
        auto lrho = Kokkos::log(rho) ; 
        auto ltemp = Kokkos::log(temp) ;   
        eps = tables.interp(lrho,ltemp,ye,TABEPS) + this->energy_shift; 
        auto press = tables.interp(lrho,ltemp,ye,TABPRESS) ; 
        csnd2 = tables.interp(lrho,ltemp,ye,TABCSND2) ; 
        return press ; 
    }
    /**************************************************************************************/
    /**
     * @brief Pressure, specific enthalpy,
     *        square sound speed, temperature
     *        and entropy given epsilon,
     *        rho and ye.
     * 
     * @param h Specific enthalpy.
     * @param csnd2 Square sound speed.
     * @param temp Temperature.
     * @param entropy Entropy per baryon.
     * @param eps Specific internal energy.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_entropy__eps_rho_ye( double& h, double& csnd2, double& temp 
                                          , double& entropy, double& eps 
                                          , double& rho, double& ye 
                                          , error_type& err ) const 
    {
        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ; 
        auto lrho = Kokkos::log(rho) ; 
        auto ltemp = ltemp__eps_lrho_ye(eps-this->_energy_shift,lrho,ye,err) ;
        temp = Kokkos::exp(ltemp) ; 
        auto press = tables.interp(lrho,ltemp,ye,TABPRESS) ; 
        csnd2 = tables.interp(lrho,ltemp,ye,TABCSND2) ; 
        h = 1 + eps + press/rho ; 
        entropy = tables.interp(lrho,ltemp,ye,TABENTROPY) ; 
        return press ; 
    }
    /**************************************************************************************/
    /**
     * @brief Epsilon, square sound speed and entropy
     *        given temperature, rho and ye.
     * 
     * @param csnd2 Square sound speed.
     * @param entropy Entropy per baryon.
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The specific internal energy.
     */
    double GRACE_HOST_DEVICE
    eps_csnd2_entropy__temp_rho_ye( double& csnd2, double& entropy, double& temp 
                                  , double& rho, double& ye 
                                  , error_type& err ) const 
    {
        
    }
    /**************************************************************************************/
    /**************************************************************************************/
    /**************************************************************************************/
    /**************************************************************************************/
    /**************************************************************************************/
    /**************************************************************************************/
    /*                                      COLD EOS UTILS                                */
    /**************************************************************************************/
    /**
     * @brief Get cold pressure given rho and ye.
     * 
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure at \f$T=0\f$
     */
    double GRACE_HOST_DEVICE
    press_cold__rho_impl(double& rho, err_t& err) const 
    {
        limit_rho(rho,err)   ; 
        double lrho = Kokkos::log(rho) ; 
        return cold_table.interp(lrho,CTABPRESS) ;
    }
    /**************************************************************************************/
    /**
     * @brief Get rest mass density given P at T=0.
     * 
     * @param press Pressure.
     * @param err Error code.
     * @return double The pressure at \f$T=0\f$
     */
    double GRACE_HOST_DEVICE
    rho__press_cold_impl(double& press_cold, err_t& err) const 
    {
        // rootfind 
        auto rootfun = [this, press_cold] (double lrho) {
            return cold_table.interp(lrho,CTABPRESS) - press_cold ; 
        } ; 
        double lrho = utils::brent(rootfun,lrhomin,lrhomax,1e-10/*fixme tol??*/) ; 
        return Kokkos::exp(lrho) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Get cold energy density given press and ye.
     * 
     * @param press Pressure.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure at \f$T=0\f$
     */
    double GRACE_HOST_DEVICE
    rho__energy_cold_impl(double& e_cold, err_t& err) const 
    {
        // e is rho(1+eps)
        // fixme what is the sign of energy shift??
        // fixme fixme do we ecen need to shift for the cold table??  
        auto rootfun = [this, e_cold] (double lrho) {
            double eps = cold_table.interp(lrho,CTABEPS) + this->energy_shift; 
            double rho = Kokkos::exp(lrho) ; 
            return rho * ( 1. + eps ) - e_cold ; 
        } ; 
        return utils::brent(rootfun,lrhomin,lrhomax,1e-10/*fixme tol??*/) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Get cold energy density given press and ye.
     * 
     * @param press Pressure.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure at \f$T=0\f$
     */
    double GRACE_HOST_DEVICE
    energy_cold__press_cold_impl(double& press_cold, err_t& err) const 
    {
        auto rootfun = [this, press_cold] (double lrho) {
            return cold_table.interp(lrho,CTABPRESS) - press_cold ; 
        } ; 
        double lrho = utils::brent(rootfun,lrhomin,lrhomax,1e-10/*fixme tol??*/) ;
        double eps = cold_table.interp(lrho,CTABEPS) + this->energy_shift ; 
        return Kokkos::exp(lrho) * ( 1. + eps ) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Cold specific internal energy given rho and ye.
     * 
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The specific internal energy at \f$T=0\f$
     */
    double GRACE_HOST_DEVICE
    eps_cold__rho(double& rho,  error_type& err) const 
    {
        // fixme what is this even needed for?? 
        auto rootfun = [this, press_cold] (double lrho) {
            return cold_table.interp(lrho,CTABPRESS) - press_cold ; 
        } ; 
        double lrho = utils::brent(rootfun,lrhomin,lrhomax,1e-10/*fixme tol??*/) ;
        return cold_table.interp(lrho,CTABEPS) + this->energy_shift ; 
    }
    /**************************************************************************************/
    /**
     * @brief Electron fraction on cold table given rest mass dens
     * 
     * @param rho Rest-mass density.
     * @param err Error code.
     * @return double The electron fraction
     */
    double GRACE_HOST_DEVICE
    ye_cold__rho_impl(double& rho,  error_type& err) const 
    {
        double lrho = Kokkos::log(rho) ; 
        return cold_table.interp(lrho,CTABYE) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Electron fraction on cold table given pressure
     * 
     * @param press Cold pressure.
     * @param err Error code.
     * @return double The electron fraction
     */
    double GRACE_HOST_DEVICE
    ye_cold__press_impl(double& press,  error_type& err) const 
    {
        double rho = rho__press_cold_impl(press,err) ; 
        double lrho = Kokkos::log(rho) ; 
        return cold_table.interp(lrho,CTABYE) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Temperature of cold slice given rest mass dens
     * 
     * @param rho Rest-mass density.
     * @param err Error code.
     * @return double The temperature 
     */
    double GRACE_HOST_DEVICE
    temp_cold__rho_impl(double& rho,  error_type& err) const 
    {
        double lrho = Kokkos::log(rho) ; 
        return cold_table.interp(lrho,CTABTEMP) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Temperature of cold slice given rest mass dens
     * 
     * @param rho Rest-mass density.
     * @param err Error code.
     * @return double The temperature 
     */
    double GRACE_HOST_DEVICE
    entropy_cold__rho_impl(double& rho,  error_type& err) const 
    {
        double lrho = Kokkos::log(rho) ; 
        return cold_table.interp(lrho,CTABENTROPY) ; 
    }
    /**************************************************************************************/
    /**
     * @brief get temperature at rho, ye
     */
    double temp__eps_rho_ye(
        double& eps, double& rho, double& ye, err_t& err
    )
    {
        /*****************************************/
        limit_eps_rho_ye(eps,rho,ye,err) ; 
        /*****************************************/
        double lrho = Kokkos::log(rho) ; 
        /*****************************************/
        /*****************************************/
        return Kokkos::exp(ltemp__eps_lrho_ye(eps,lrho,ye,err)) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Get eps range at rho, ye
     */
    void KOKKOS_FUNCTION 
    eps_range__rho_ye(double& eps_min, double& eps_max, double& rho, double& ye, err_t& err) {
        limit_rho(rho,err) ; 
        limit_ye(ye,err) ; 
        eps_min = tables.interp(lrho,ltempmin,ye,TABEPS) ;
        eps_max = tables.interp(lrho,ltempmax,ye,TABEPS) ;
    }
    /**************************************************************************************/
    /**
     * @brief Get eps range at rho, ye
     */
    void KOKKOS_FUNCTION 
    entropy_range__rho_ye(double& s_min, double& s_max, double& rho, double& ye, err_t& err) {
        limit_rho(rho,err) ; 
        limit_ye(ye,err) ; 
        s_min = tables.interp(lrho,ltempmin,ye,TABENTROPY) ;
        s_max = tables.interp(lrho,ltempmax,ye,TABENTROPY) ;
    }
    /**************************************************************************************/
    private:
    /**************************************************************************************/
    // following functions are intended for internal use
    // they are unsafe in that they do not check arguments,
    // and generally work in logrho for efficiency 
    /**************************************************************************************/
    void KOKKOS_INLINE_FUNCTION 
    limit_rho(double& rho, err_t& err) {
        err = SUCCESS ; 
        if( rho < this->eos_rhomin ) {
            rho = (1+1e-5) * this->eos_rhomin;
            err = EOS_RHO_TOO_LOW;
        } 
        if( rho > this->eos_rhomax ) {
            rho = (1-1e-5) * this->eos_rhomax;
            err = EOS_RHO_TOO_HIGH;
        }
    }
    /**************************************************************************************/
    void KOKKOS_INLINE_FUNCTION 
    limit_ye(double& ye, err_t& err) {
        err = SUCCESS ; 
        if( ye < this->eos_yemin ) {
            ye = (1+1e-2) * this->eos_yemin;
            err = EOS_YE_TOO_LOW;
        } 
        if( ye > this->eos_yemax ) {
            ye = (1-1e-2) * this->eos_rhomax;
            err = EOS_YE_TOO_HIGH;
        }
    }
    /**************************************************************************************/
    void KOKKOS_INLINE_FUNCTION 
    limit_temp(double& temp, err_t& err) {
        err = SUCCESS ; 
        if( temp < this->eos_tempmin ) {
            temp = (1+1e-2) * this->eos_tempmin;
            err = EOS_TEMPERATURE_TOO_LOW;
        } 
        if( temp > this->eos_tempmax ) {
            temp = (1-1e-2) * this->eos_tempmax;
            err = EOS_TEMPERATURE_TOO_HIGH;
        }
    }
    /**************************************************************************************/
    void KOKKOS_INLINE_FUNCTION
    limit_eps_rho_ye(double& eps, double& rho, double& ye, err_t& err)
    {
        // this call limits rho and ye 
        double epsmin, epsmax;
        eps_range__rho_ye(epsmin,epsmax,rho,ye,err) ; 
        if ( eps<epsmin) {
            eps = (1+1e-5)*epsmin ; 
            err = ERR_EPS_TOO_LOW ; 
        }
        if ( eps>epsmax ) {
            eps = (1-1e-5)*epsmax;
            err = ERR_EPS_TOO_HIGH ; 
        }
    }  
    /**************************************************************************************/
    void KOKKOS_INLINE_FUNCTION
    limit_eps_lrho_ye(double& eps, double& lrho, double& ye, err_t& err)
    {
        // this call limits rho and ye 
        double epsmin, epsmax;
        eps_min = tables.interp(lrho,ltempmin,ye,TABEPS) ;
        eps_max = tables.interp(lrho,ltempmax,ye,TABEPS) ;
        if ( eps<epsmin) {
            eps = (1+1e-5)*epsmin ; 
            err = ERR_EPS_TOO_LOW ; 
        }
        if ( eps>epsmax ) {
            eps = (1-1e-5)*epsmax;
            err = ERR_EPS_TOO_HIGH ; 
        }
    }  
    /**************************************************************************************/
    // no checks, takes and returns log! 
    double KOKKOS_INLINE_FUNCTION
    ltemp__eps_lrho_ye(double& eps, double& lrho, double& ye, err_t& err)
    {
        auto rootfun = [this,lrho,ye,eps] (double lt) {
            return tables.interp(lrho,lt,ye,TABEPS) - eps ;  
        } ; 
        /*****************************************/
        return utils::brent(rootfun,ltempmin,ltempmax, 1e-10/*FIXME tolerance?*/) ; 
    } 
    /**************************************************************************************/
    tabeos_linterp_t tables; 
    /**************************************************************************************/
    coldtab_linterp_t cold_table ;
    /**************************************************************************************/
    double lrhomin, lrhomax ;
    /**************************************************************************************/
    double ltempmin, ltempmax ; 
    /**************************************************************************************/
    int nrho,nT,nye ; 
    /**************************************************************************************/
} ; 

} /* namespace grace */
#endif /*GRACE_PHYSICS_TABEOS_HH*/