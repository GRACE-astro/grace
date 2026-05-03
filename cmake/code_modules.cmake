option(GRACE_ENABLE_BURGERS  "Enable Burgers equation module" OFF) 
option(GRACE_ENABLE_SCALAR_ADV  "Enable scalar advection equation module" OFF) 
option(GRACE_ENABLE_GRMHD   "Enable GRMHD equation module"  ON)

# M1
option(GRACE_ENABLE_M1 "Enable M1 neutrino transport" OFF)
option(M1_NU_THREESPECIES "Enable 3-species neutrino M1 scheme" OFF)
option(M1_NU_FIVESPECIES "Enable 5-species neutrino M1 scheme" OFF)

# --- enforce hierarchy ---
if(M1_NU_THREESPECIES)
    set(GRACE_ENABLE_M1 ON)
    message(STATUS "M1 with 3-Species enabled.")
endif()

if(M1_NU_FIVESPECIES)
    set(M1_NU_THREESPECIES ON)
    set(GRACE_ENABLE_M1 ON)
    message(STATUS "M1 with 5-Species enabled.")
endif()
# M1

option(GRACE_FREEZE_HYDRO "Freeze hydrodynamics evolution" OFF)

if( GRACE_ENABLE_SCALAR_ADV )
    message(STATUS "Scalar advection module enabled.")
    set(GRACE_ENABLE_GRMHD OFF)
endif()
if( GRACE_ENABLE_BURGERS )
    message(STATUS "Burgers module enabled.")
    set(GRACE_ENABLE_GRMHD OFF)
endif()
if( GRACE_ENABLE_GRMHD )
    option(GRACE_GRMHD_USE_GS "Use Gardiner Stone method of EMF evaluation" OFF)
    message(STATUS "GRMHD module enabled.")
    set(GRACE_ENABLE_BURGERS OFF)
endif()

if( GRACE_ENABLE_FUKA )
    message(STATUS "FUKA module enabled.")
endif()
