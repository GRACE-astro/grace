# GRMHD is the only evolved-equations module GRACE currently supports.
# (Burgers and scalar-advection were dropped years ago; their source
# files no longer exist.)  Optional add-ons sit alongside it.
option(GRACE_ENABLE_M1 "Enable M1 radiation transport" OFF)
option(GRACE_FREEZE_HYDRO "Freeze hydrodynamics evolution" OFF)

# First-Order Flux Correction.  Stage-3 flagger + stage-4 donor-cell/LLF
# recompute at faces of cells whose tentative HO update would have required
# c2p flooring.  Default ON — it's a safety net with negligible cost on
# clean states.  Disable for symmetry-preservation diagnostics or when
# bisecting a flux-related bug.
option(GRACE_ENABLE_FOFC "Enable First-Order Flux Correction" ON)

# GRMHD Riemann solver selection (compile-time).
#   HLL — 2-wave HLLE (default).
#   ADV — "advanced": HLLD for MHD, HLLC for pure-hydro states; HLLE fallback.
#   LLF — Local Lax-Friedrichs (Rusanov): symmetric HLL with cmax/cmin
#         clamped to the largest local |fast magnetosonic speed|.
set(GRACE_RIEMANN_SOLVER "HLL" CACHE STRING
    "GRMHD Riemann solver (HLL|ADV|LLF)")
set_property(CACHE GRACE_RIEMANN_SOLVER PROPERTY STRINGS HLL ADV LLF)
if(NOT GRACE_RIEMANN_SOLVER MATCHES "^(HLL|ADV|LLF)$")
    message(FATAL_ERROR
        "GRACE_RIEMANN_SOLVER=${GRACE_RIEMANN_SOLVER} is not one of HLL, ADV, LLF.")
endif()
message(STATUS "GRMHD Riemann solver: ${GRACE_RIEMANN_SOLVER}")

# EMF reconstruction scheme for constrained transport (compile-time).
#   GS  — Gardiner-Stone EMF (default): edge EMFs reconstructed in-kernel from
#         face fluxes during the directional flux sweep.
#   UCT — Upwind constrained transport: edge EMFs computed in a separate pass
#         from face-centered vtilde and the staggered B field.
set(GRACE_EMF_SCHEME "GS" CACHE STRING
    "CT EMF reconstruction scheme (GS|UCT)")
set_property(CACHE GRACE_EMF_SCHEME PROPERTY STRINGS GS UCT)
if(NOT GRACE_EMF_SCHEME MATCHES "^(GS|UCT)$")
    message(FATAL_ERROR
        "GRACE_EMF_SCHEME=${GRACE_EMF_SCHEME} is not one of GS, UCT.")
endif()
message(STATUS "EMF scheme: ${GRACE_EMF_SCHEME}")

if( GRACE_ENABLE_FUKA )
    message(STATUS "FUKA module enabled.")
endif()

