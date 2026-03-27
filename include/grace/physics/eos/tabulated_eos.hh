/**
 * @file tabulated.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de) & Khalil Pierre (pierre@itp.uni-frankfurt.de)
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
#include <grace/data_structures/memory_defaults.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/utils/rootfinding.hh>

#include <Kokkos_Core.hpp>

namespace grace {

// interpolator for tabeos 
// spacing assumed constant 
struct tabeos_linterp_t {

    tabeos_linterp_t() = default ; 

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

    double KOKKOS_INLINE_FUNCTION operator() (int irho, int itemp, int iye, int varidx) const {
        return _tables(irho,itemp,iye,varidx) ; 
    }

    double KOKKOS_INLINE_FUNCTION
    interp(double lrho, double ltemp, double ye, int const& idx) const 
    {
        std::array<double,1> res ;
        std::array<int,1> _idx{idx} ; 
        interp<1>(lrho,ltemp,ye,_idx,res) ; 
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
        getw(ltemp,j,_logT,idt,wt) ;
        getw(ye,k,_ye,idy,wy) ; 
        // interpolate 
        for( int ii=0; ii<2; ++ii) {
            for( int jj=0; jj<2; ++jj) {
                for( int kk=0; kk<2; ++kk) {
                    double weight =  wr[ii] * wt[jj] * wy[kk];
                    for( int iv=0; iv<N; ++iv) {
                        res[iv] += weight * _tables(
                            i+ii, j+jj, k+kk, idx[iv]
                        ) ; 
                    }
                }
            }
        }
    } 

    void KOKKOS_INLINE_FUNCTION
    getidx(double x, double y, double z, int& i, int& j, int& k) const {
        i = Kokkos::max(
            0UL,
            Kokkos::min(
                static_cast<size_t>((x - _logrho(0)) * idr), 
                _logrho.extent(0)-2
            )
        ) ;
        j = Kokkos::max(
            0UL,
            Kokkos::min(
                static_cast<size_t>((y - _logT(0))   * idt), 
                _logT.extent(0)-2
            )
        ) ;
        
        k = Kokkos::max(
            0UL,
            Kokkos::min(
                static_cast<size_t>((z - _ye(0))     * idy), 
                _ye.extent(0)-2
            )
        );
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

// interpolator for tabeos 
// spacing assumed constant 
struct cold_eos_linterp_t {

    cold_eos_linterp_t() = default ;

    cold_eos_linterp_t(
        Kokkos::View<double **> tabs,
        Kokkos::View<double *> ar
    ) : _tables(tabs), _logrho(ar)
    {
        idr =  1./(_logrho(1)-_logrho(0)) ; 
    }

    double KOKKOS_INLINE_FUNCTION lrho(int idx) const { return _logrho(idx) ; } 

    double KOKKOS_INLINE_FUNCTION operator() (int irho, int varidx) const {
        return _tables(irho,varidx) ; 
    }

    double KOKKOS_INLINE_FUNCTION
    interp(double lrho, int const& idx) const 
    {
        std::array<double,1> res ;
        std::array<int,1> _idx{idx} ; 
        interp<1>(lrho,_idx,res) ; 
        return res[0] ;
    }

    template< int N >
    void KOKKOS_INLINE_FUNCTION
    interp(double lrho, std::array<int,N> const& idx,  std::array<double,N>& res) const 
    {
        for( int iv=0; iv<N; ++iv) res[iv] = 0 ; 
        // indices
        int i;
        getidx(lrho,i) ; 
        // weights 
        double wr[2] ; 
        getw(lrho,i,_logrho,idr,wr) ;
        // interpolate 
        for( int iv=0; iv<N; ++iv) {
            res[iv] = wr[0] * _tables(i,idx[iv]) + wr[1] * _tables(i+1,idx[iv]) ; 
        }
    } 

    void KOKKOS_INLINE_FUNCTION
    getidx(double x, int& i) const {
        i = Kokkos::max(
            0UL,
            Kokkos::min(
                static_cast<size_t>((x - _logrho(0)) * idr),
                _logrho.extent(0)-2
            )
        ) ;
    }

    void KOKKOS_INLINE_FUNCTION
    getw(double x, int i, readonly_view_t<double> ax, double ih, double* w) const {
        double lam = (x - ax(i)) * ih ; 
        w[0] = 1.-lam ;
        w[1] = lam    ; 
    }

    readonly_view_t<double> _logrho ; 
    Kokkos::View<double **> _tables  ;

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
class tabulated_eos_t
    : public eos_base_t<tabulated_eos_t> 
{
    /**************************************************************************************/
    using err_t  = eos_err_t ; 
    using base_t = eos_base_t<tabulated_eos_t> ;
    public:
    /**************************************************************************************/
    enum TEOS_VIDX : int {
        TABPRESS=0,
        TABEPS,
        TABCSND2,
        TABENTROPY,
        TABMUE,
        TABMUP,
        TABMUN,
        TABXA,
        TABXH,
        TABXN,
        TABXP,
        TABABAR,
        TABZBAR,
        N_TAB_VARS 
    } ; 
    /**************************************************************************************/
    enum COLD_TEOS_VIDX : int {
        CTABTEMP=0,
        CTABYE,
        CTABPRESS,
        CTABEPS,
        CTABCSND2,
        CTABENTROPY,
        N_CTAB_VARS 
    } ;
    /**************************************************************************************/
    /**************************************************************************************/
    tabulated_eos_t() = default ; 
    /**************************************************************************************/
    tabulated_eos_t(
        Kokkos::View<double ****, grace::default_space> _tabeos, 
        Kokkos::View<double *,  grace::default_space> _logrho  , 
        Kokkos::View<double *,  grace::default_space> _logT    , 
        Kokkos::View<double *,  grace::default_space> _ye      ,
        Kokkos::View<double **, grace::default_space> _cold_tabeos      ,
        Kokkos::View<double *,  grace::default_space> _cold_tabeos_logrho  ,
        double _rhomax, double _rhomin,
        double _tempmax, double _tempmin,
        double _yemax, double _yemin,
        double _baryon_mass, double _energy_shift,
        double _c2p_epsmin, double _c2p_epsmax,
        double _c2p_hmin, double _c2p_hmax,
        double _c2p_temp_atm,
        double _c2p_ye_atm,
        bool _atmo_is_beta_eq
    ) : tables(_tabeos,_logrho,_logT,_ye)
      , cold_table(_cold_tabeos, _cold_tabeos_logrho)
      , nrho(_logrho.size())
      , nT(_logT.size())
      , nye(_ye.size())
      , energy_shift(_energy_shift)
      , base_t(
        _rhomax, _rhomin,
        _tempmax, _tempmin,
        _yemax, _yemin,
        _baryon_mass,
        _c2p_epsmin, _c2p_epsmax,
        _c2p_hmin, _c2p_hmax,
        _c2p_temp_atm,
        _c2p_ye_atm,
        _atmo_is_beta_eq,
        false )
    { 
        lrhomin = _logrho[0] ; lrhomax = _logrho[_logrho.size()-1] ; 
        ltempmin = _logT[0] ; ltempmax = _logT[_logT.size()-1] ; 
    }
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
        // this call imposes limits on eps ! 
        auto ltemp = ltemp__eps_lrho_ye(eps,lrho,ye,err) ; 
        return press__lrho_ltemp_ye(lrho,ltemp,ye) ; 
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
        auto lrho  = Kokkos::log(rho) ; 
        // this call limits epsilon! 
        auto ltemp = ltemp__eps_lrho_ye(eps,lrho,ye,err) ;
        temp = Kokkos::exp(ltemp) ; 
        return press__lrho_ltemp_ye(lrho,ltemp,ye) ; 
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
        limit_rho(rho,err)   ; 
        limit_ye(ye,err)     ; 
        limit_temp(temp,err) ; 
        double lrho  = Kokkos::log(rho)  ;
        double ltemp = Kokkos::log(temp) ; 
        return press__lrho_ltemp_ye(lrho,ltemp,ye) ; 
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
        double lrho = Kokkos::log(rho) ; 
        double ltemp = Kokkos::log(temp) ; 
        auto leps = tables.interp(lrho,ltemp,ye,TABEPS) ;
        return  Kokkos::exp(leps) - energy_shift ; 
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
                             , err_t &err) const 
    {

        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ; 
        auto lrho = Kokkos::log(rho) ; 
        auto ltemp = ltemp__eps_lrho_ye(eps,lrho,ye,err) ;
        auto press = press__lrho_ltemp_ye(lrho,ltemp,ye) ; 
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
                              , err_t &err) const 
    {
        limit_rho(rho,err)   ; 
        limit_ye(ye,err)     ; 
        limit_temp(temp,err) ; 
        auto lrho = Kokkos::log(rho) ; 
        auto ltemp = Kokkos::log(temp) ; 
        double eps = Kokkos::exp(tables.interp(lrho,ltemp,ye,TABEPS)) - energy_shift ; 
        auto press = press__lrho_ltemp_ye(lrho,ltemp,ye) ; 
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
    press_eps_csnd2__temp_rho_ye_impl( double &eps, double &csnd2, double &temp
                                , double &rho, double &ye
                                , err_t& err) const 
    {
        limit_rho(rho,err)   ; 
        limit_ye(ye,err)     ; 
        limit_temp(temp,err) ; 
        auto lrho = Kokkos::log(rho) ; 
        auto ltemp = Kokkos::log(temp) ;   
        eps = Kokkos::exp(tables.interp(lrho,ltemp,ye,TABEPS)) - energy_shift ;
        auto press = press__lrho_ltemp_ye(lrho,ltemp,ye) ; 
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
    press_h_csnd2_temp_entropy__eps_rho_ye_impl( double& h, double& csnd2, double& temp 
                                          , double& entropy, double& eps 
                                          , double& rho, double& ye 
                                          , err_t& err ) const 
    {
        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ; 
        auto lrho = Kokkos::log(rho) ;
        // this call limits eps 
        auto ltemp = ltemp__eps_lrho_ye(eps,lrho,ye,err) ;
        temp = Kokkos::exp(ltemp) ; 
        auto press = press__lrho_ltemp_ye(lrho,ltemp,ye) ; 
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
    eps_csnd2_entropy__temp_rho_ye_impl( double& csnd2, double& entropy, double& temp 
                                  , double& rho, double& ye 
                                  , err_t& err ) const 
    {
        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ; 
        limit_temp(temp,err) ; 
        auto lrho  = Kokkos::log(rho) ;
        auto ltemp = Kokkos::log(temp) ; 
        csnd2 = tables.interp(lrho,ltemp,ye,TABCSND2) ; 
        entropy = tables.interp(lrho,ltemp,ye,TABENTROPY) ; 
        return Kokkos::exp(tables.interp(lrho,ltemp,ye,TABEPS)) - energy_shift ;
    }
    /**************************************************************************************/
    double GRACE_HOST_DEVICE
    press_eps_csnd2_entropy__temp_rho_ye_impl( double& eps, double& csnd2, double& entropy, double& temp 
                                  , double& rho, double& ye 
                                  , err_t& err ) const
    {
        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ; 
        limit_temp(temp,err) ; 
        auto lrho  = Kokkos::log(rho) ;
        auto ltemp = Kokkos::log(temp) ; 
        eps = Kokkos::exp(tables.interp(lrho,ltemp,ye,TABEPS)) - energy_shift ;
        csnd2 = tables.interp(lrho,ltemp,ye,TABCSND2) ; 
        entropy = tables.interp(lrho,ltemp,ye,TABENTROPY) ; 
        return press__lrho_ltemp_ye(lrho,ltemp,ye) ; 
    }
    
    /**************************************************************************************/
    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_eps__entropy_rho_ye_impl( double& h, double& csnd2, double& temp
                                               , double& eps, double& entropy, double& rho 
                                               , double& ye, err_t& err ) const 
    {
        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ;
        limit_entropy_rho_ye(entropy,rho,ye,err) ; 
        double lrho = Kokkos::log(rho) ; 
        auto ltemp = ltemp__entropy_lrho_ye(entropy,lrho,ye) ; 
        temp = Kokkos::exp(ltemp) ; 
        eps = Kokkos::exp(tables.interp(lrho,ltemp,ye,TABEPS)) - energy_shift ;
        double press = press__lrho_ltemp_ye(lrho,ltemp,ye) ; 
        csnd2 = tables.interp(lrho,ltemp,ye,TABCSND2) ; 
        h = 1 + eps + press/rho ;
        return press ; 
    }
    /**************************************************************************************/
    double GRACE_HOST_DEVICE
    mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_impl( 
        double &mup, double &mun, double &Xa, double &Xh, double &Xn, double &Xp
      , double &Abar, double &Zbar, double &temp, double &rho, double &ye
      , err_t &err) const 
    {
        limit_rho(rho,err) ; 
        limit_ye(ye,err)   ; 
        limit_temp(temp,err) ; 
        auto lrho  = Kokkos::log(rho) ;
        auto ltemp = Kokkos::log(temp) ;

        double mue = tables.interp(lrho,ltemp,ye,TABMUE) ; 
        mup = tables.interp(lrho,ltemp,ye,TABMUP) ; 
        mun = tables.interp(lrho,ltemp,ye,TABMUN) ; 
        Xa = tables.interp(lrho,ltemp,ye,TABXA) ; 
        Xh = tables.interp(lrho,ltemp,ye,TABXH) ; 
        Xn = tables.interp(lrho,ltemp,ye,TABXN) ; 
        Xp = tables.interp(lrho,ltemp,ye,TABXP) ; 
        Abar = tables.interp(lrho,ltemp,ye,TABABAR) ; 
        Zbar = tables.interp(lrho,ltemp,ye,TABZBAR) ; 
        return mue ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_h_csnd2_temp_entropy__press_rho_ye_impl( double& h, double& csnd2, double& temp
                                               , double& entropy, double& press, double& rho 
                                               , double& ye, err_t& err) const 
    {
        Kokkos::abort("The function eps_h_csnd2_temp_entropy__press_rho_ye_impl is not well defined for a tabulated eos.") ; 
    }
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
        return Kokkos::exp(cold_table.interp(lrho,CTABPRESS)) ;
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
        auto lp = cold_lpress__press_limited(press_cold,err) ; 
        // rootfind 
        auto rootfun = [this, lp] (double lrho) {
            return cold_table.interp(lrho,CTABPRESS) - lp ; 
        } ; 
        auto lrmin = cold_table._logrho(0);
        auto lrmax = cold_table._logrho(cold_table._logrho.size()-1) ; 
        double lrho = utils::brent(rootfun,lrmin,lrmax,1e-14/*fixme tol??*/) ; 
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
        // find minimum and maximum energy 
        int n = cold_table._logrho.size() ;
        double eps_min = Kokkos::exp(cold_table._tables(0,CTABEPS)) - energy_shift ;  
        double eps_max = Kokkos::exp(cold_table._tables(n-1,CTABEPS)) - energy_shift ;
        double e_min = ( 1 + eps_min ) * Kokkos::exp(cold_table._logrho(0)) ; 
        double e_max = ( 1 + eps_max ) * Kokkos::exp(cold_table._logrho(n-1)) ; 
        if ( e_cold < e_min ) {
            e_cold = e_min ; 
            err.set(EOS_EPS_TOO_LOW) ; 
            return Kokkos::exp(cold_table._logrho(0));
        }
        if ( e_cold > e_max) {
            e_cold = e_max ; 
            err.set(EOS_EPS_TOO_HIGH) ; 
            return Kokkos::exp(cold_table._logrho(n-1));
        }

        auto rootfun = [this, e_cold] (double lrho) {
            double eps = Kokkos::exp(cold_table.interp(lrho,CTABEPS)) - energy_shift  ; 
            double e = ( 1 + eps ) * Kokkos::exp(lrho) ; 
            return e - e_cold ; 
        } ; 
        return Kokkos::exp(utils::brent(rootfun, 
            cold_table._logrho(0), cold_table._logrho(n-1), 1e-14));
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
        auto lp_cold = cold_lpress__press_limited(press_cold,err) ; 
        auto rootfun = [this, lp_cold] (double lrho) {
            return cold_table.interp(lrho,CTABPRESS) - lp_cold ; 
        } ; 
        auto lrmin = cold_table._logrho(0);
        auto lrmax = cold_table._logrho(cold_table._logrho.size()-1) ;
        double lrho = utils::brent(rootfun,lrmin,lrmax,1e-14/*fixme tol??*/) ;
        double eps = Kokkos::exp(cold_table.interp(lrho,CTABEPS)) - energy_shift  ; 
        return Kokkos::exp(lrho) * (1. + eps)  ; 
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
    eps_cold__rho(double& rho,  err_t& err) const 
    {
        limit_rho(rho,err) ; 
        double lrho = Kokkos::log(rho) ; 
        return Kokkos::exp(cold_table.interp(lrho,CTABEPS)) - energy_shift; 
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
    ye_cold__rho_impl(double& rho,  err_t& err) const 
    {
        limit_rho(rho,err) ; 
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
    ye_cold__press_impl(double& press,  err_t& err) const 
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
    temp_cold__rho_impl(double& rho,  err_t& err) const 
    {
        limit_rho(rho,err) ; 
        double lrho = Kokkos::log(rho) ; 
        return Kokkos::exp(cold_table.interp(lrho,CTABTEMP)) ; 
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
    entropy_cold__rho_impl(double& rho,  err_t& err) const 
    {
        limit_rho(rho,err) ; 
        double lrho = Kokkos::log(rho) ; 
        return cold_table.interp(lrho,CTABENTROPY) ; 
    }
    /**************************************************************************************/
    /**
     * @brief Get eps range at rho, ye
     */
    void KOKKOS_FUNCTION 
    eps_range__rho_ye(double& eps_min, double& eps_max, double& rho, double& ye, err_t& err) const {
        limit_rho(rho,err) ; 
        limit_ye(ye,err) ; 
        double lrho = Kokkos::log(rho) ; 
        eps_min = Kokkos::exp(tables.interp(lrho,ltempmin,ye,TABEPS)) - energy_shift ;
        eps_max = Kokkos::exp(tables.interp(lrho,ltempmax,ye,TABEPS)) - energy_shift ;
    }
    /**************************************************************************************/
    /**
     * @brief Get eps range at rho, ye
     */
    void KOKKOS_FUNCTION 
    entropy_range__rho_ye(double& s_min, double& s_max, double& rho, double& ye, err_t& err) const {
        limit_rho(rho,err) ; 
        limit_ye(ye,err) ; 
        double lrho = Kokkos::log(rho) ;
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
    limit_rho(double& rho, err_t& err) const {
        if( rho < this->eos_rhomin ) {
            rho = (1+1e-5) * this->eos_rhomin;
            err.set(EOS_RHO_TOO_LOW);
        } 
        if( rho > this->eos_rhomax ) {
            rho = (1-1e-5) * this->eos_rhomax;
            err.set(EOS_RHO_TOO_HIGH);
        }
    }
    /**************************************************************************************/
    void KOKKOS_INLINE_FUNCTION 
    limit_ye(double& ye, err_t& err) const {
        if( ye < this->eos_yemin ) {
            ye = (1+1e-2) * this->eos_yemin;
            err.set(EOS_YE_TOO_LOW);
        } 
        if( ye > this->eos_yemax ) {
            ye = (1-1e-2) * this->eos_yemax;
            err.set(EOS_YE_TOO_HIGH);
        }
    }
    /**************************************************************************************/
    void KOKKOS_INLINE_FUNCTION 
    limit_temp(double& temp, err_t& err) const {
        if( temp < this->eos_tempmin ) {
            temp = (1+1e-2) * this->eos_tempmin;
            err.set(EOS_TEMPERATURE_TOO_LOW);
        } 
        if( temp > this->eos_tempmax ) {
            temp = (1-1e-2) * this->eos_tempmax;
            err.set(EOS_TEMPERATURE_TOO_HIGH);
        }
    }
    /**************************************************************************************/
    void KOKKOS_INLINE_FUNCTION
    limit_entropy_rho_ye(double& entropy, double& rho, double& ye, err_t& err) const 
    {
        // this call limits rho and ye 
        double smin, smax;
        entropy_range__rho_ye(smin,smax,rho,ye,err) ; 
        if ( entropy<smin) {
            entropy = smin ; 
            err.set(EOS_ENTROPY_TOO_LOW) ; 
        }
        if ( entropy>smax ) {
            entropy = smax;
            err.set(EOS_ENTROPY_TOO_HIGH) ; 
        }
    } 
    /**************************************************************************************/
    // no checks, takes and returns log! 
    double KOKKOS_INLINE_FUNCTION
    ltemp__eps_lrho_ye(double& eps, double& lrho, double& ye, err_t& err) const
    {   
        auto leps = Kokkos::log(eps + energy_shift) ; 
        auto lepsmin = tables.interp(lrho,ltempmin,ye,TABEPS) ;
        auto lepsmax = tables.interp(lrho,ltempmax,ye,TABEPS) ; 
        if ( leps <= lepsmin ) {
            eps = Kokkos::exp(lepsmin) - energy_shift ; 
            err.set(EOS_EPS_TOO_LOW) ; 
            return ltempmin ; 
        }
        if ( leps >= lepsmax ) {
            eps = Kokkos::exp(lepsmax) - energy_shift ; 
            err.set(EOS_EPS_TOO_HIGH) ; 
            return ltempmax ; 
        }
        /************************************************/
        auto rootfun = [this,lrho,ye,leps] (double lt) {
            return tables.interp(lrho,lt,ye,TABEPS) - leps ;  
        } ; 
        /************************************************/
        return utils::brent(rootfun,ltempmin,ltempmax, 1e-14/*FIXME tolerance?*/) ; 
    } 
    /**************************************************************************************/
    // no checks, takes and returns log! 
    double KOKKOS_INLINE_FUNCTION
    ltemp__entropy_lrho_ye(double& entropy, double& lrho, double& ye) const
    {   
        /************************************************/
        auto rootfun = [this,lrho,ye,entropy] (double lt) {
            return tables.interp(lrho,lt,ye,TABENTROPY) - entropy ;  
        } ; 
        /************************************************/
        return utils::brent(rootfun,ltempmin,ltempmax, 1e-14/*FIXME tolerance?*/) ; 
    } 
    /**************************************************************************************/
    /**************************************************************************************/
    double KOKKOS_INLINE_FUNCTION
    press__lrho_ltemp_ye(double const lrho, double const ltemp, double const ye) const 
    {
        return Kokkos::exp(tables.interp(lrho,ltemp,ye,TABPRESS)) ; 
    }
    /**************************************************************************************/
    /**************************************************************************************/
    double KOKKOS_INLINE_FUNCTION
    cold_lpress__press_limited(double& press_cold, err_t& err) const {
        double p_min = Kokkos::exp(cold_table._tables(0,CTABPRESS)) ;
        int n = cold_table._logrho.size() ;  
        double p_max = Kokkos::exp(cold_table._tables(n-1,CTABPRESS)) ; 
        if ( press_cold < p_min ) {
            err.set(EOS_RHO_TOO_LOW) ; // ... 
            press_cold = p_min * (1+1e-10) ; 
        }
        if ( press_cold > p_max ) {
            err.set(EOS_RHO_TOO_HIGH) ;// ... 
            press_cold = p_max * (1-1e-10) ; 
        }
        return Kokkos::log(press_cold) ; 
    } 
    /**************************************************************************************/
    public: 
    /**************************************************************************************/
    tabeos_linterp_t tables; 
    /**************************************************************************************/
    cold_eos_linterp_t cold_table ;
    /**************************************************************************************/
    double lrhomin, lrhomax ;
    /**************************************************************************************/
    double energy_shift ; 
    /**************************************************************************************/
    double ltempmin, ltempmax ; 
    /**************************************************************************************/
    int nrho,nT,nye ; 
    /**************************************************************************************/
} ; 

} /* namespace grace */
#endif /*GRACE_PHYSICS_TABEOS_HH*/