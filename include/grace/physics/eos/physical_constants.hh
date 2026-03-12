/**
 * @file physical_constants.hh
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

namespace grace { namespace physical_constants {

/*
# Units are CGS unless specified 
# speed of light
clight = 29979245800
# Planck constant eV s 
h_eVs = 6.582119569e-16 
# Boltzmann constant 
k_evKm1 = 8.617333262e-5 
# Charge constant (C)
e_charge = 1.602176634e-19 
# statC 
e_cgs = 4.80320425e-10 
# Fine structure constant 
alpha_fine = 1./137.
# Stefan Boltzmann constant 
sigma_cgs = 5.670374419e-5 # erg cmˆ-2 sˆ-1 Kˆ-4 
# Solar mass 
Msun_cgs = 1.988475e33 
# G constant 
G_cgs = 6.67430e-8 
G_pcMsunm1 = 4.3009172706e-3 # in parsec / Msun (km/s)ˆ2 
# Electron mass 
me_MeV = 0.51099895069
me_KeV = 510.99895069
# conversions 

# Length
cm_to_m = 1e2 
m_to_cm = 1e-2 
cm_to_km = 1e5 
km_to_cm = 1e-5
mum_to_cm = 1e-4
cm_to_mum = 1e4 
pc_to_km = 3.0857e13 
AU_to_km = 149597870.7
ly_to_km = 9460730472580.8 
km_to_pc = 1/pc_to_km
angstrom_to_nm = 0.1 
nm_to_angstrom = 10 
# Assuming c = G = Msun = 1  
Msun_to_cm = G_cgs * Msun_cgs / clight**2
cm_to_Msun = 1./Msun_to_cm 
Msun_to_pc = Msun_to_cm * cm_to_km * km_to_pc

# Time 
hour_to_s = 60 * 60 
day_to_s = 24*hour_to_s 
year_to_s = 365 * day_to_s 
s_to_year = 1/year_to_s
Msun_to_s = Msun_to_cm / clight 
s_to_Msun = 1./Msun_to_s
ms_to_Msun = s_to_Msun * 1e-3 
Msun_to_ms = 1e03 * Msun_to_s

# Temperature 
eV_to_K = 1.0/k_evKm1
keV_to_K = eV_to_K * 1e03 
MeV_to_K = keV_to_K * 1e03 
# Energy 
J_to_erg = 1e07 
erg_to_J = 1/J_to_erg
eV_to_erg = e_charge * J_to_erg
erg_to_eV = 1/eV_to_erg
erg_to_keV = erg_to_eV * 1e-3 
erg_to_MeV = erg_to_eV * 1e-6

# Mass 
eV_to_g = eV_to_erg/clight**2
MeV_to_g = eV_to_g * 1e6 

# Boltzmann constant in CGS 
k_cgs = k_evKm1 * eV_to_erg
# Planck constant in CGS 
h_cgs = h_eVs * eV_to_erg
# Electron mass revisited 
me_cgs = me_MeV * MeV_to_g
*/
#define CONSTDEF(x,y) static constexpr double x = y 

CONSTDEF(clight,29979245800);
CONSTDEF(h_eVs,6.582119569e-16);
CONSTDEF(k_evKm1,8.617333262e-5);
CONSTDEF(e_charge,1.602176634e-19);
CONSTDEF(e_cgs,4.80320425e-10);
CONSTDEF(alpha_fine,1./137.);
CONSTDEF(sigma_cgs,5.670374419e-5);
CONSTDEF(Msun_cgs,1.988475e33);
CONSTDEF(G_cgs,6.67430e-8);
CONSTDEF(me_KeV,510.99895069);
CONSTDEF(me_MeV,0.51099895069);
CONSTDEF(mp_MeV,938.27208943 );
CONSTDEF(J_to_erg,1e07 ); 


CONSTDEF(Msun_to_cm,G_cgs * Msun_cgs / (clight*clight));
CONSTDEF(Msun_to_s,Msun_to_cm / clight);
CONSTDEF(Msun_to_erg,Msun_cgs*clight*clight);
CONSTDEF(eV_to_erg, e_charge * J_to_erg);
CONSTDEF(erg_to_eV, 1./eV_to_erg);
CONSTDEF(erg_to_keV,erg_to_eV * 1e-3);
CONSTDEF(eV_to_g,eV_to_erg/SQR(clight));
CONSTDEF(MeV_to_g,eV_to_g * 1e6);

CONSTDEF(k_cgs,k_evKm1 * eV_to_erg);
CONSTDEF(h_cgs,h_eVs * eV_to_erg);
CONSTDEF(me_cgs,me_MeV * MeV_to_g);
CONSTDEF(mp_cgs,mp_MeV * MeV_to_g);
CONSTDEF(mnuc_CGS, mp_cgs) ; 

CONSTDEF(sigma_T,  6.6524587051e-25);
} } /* namespace grace::physical_constants */