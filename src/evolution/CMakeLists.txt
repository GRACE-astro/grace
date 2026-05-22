register_grace_object(grace_evolution auxiliaries.cpp evolve.cpp find_stable_timestep.cpp refluxing.cpp)
register_grace_object(grace_initial_data initial_data.cpp)

# Opt-in diagnostics for the EMF reflux apply kernel — gates the printf in
# reflux_emf_apply_coarse_face that traces the data flow per descriptor.
# Toggle with cmake -DGRACE_DEBUG_REFLUX=ON ...
option(GRACE_DEBUG_REFLUX "Enable printf diagnostics in reflux apply + comm-pattern build" OFF)
if(GRACE_DEBUG_REFLUX)
    target_compile_definitions(grace_evolution PRIVATE DEBUG_REFLUX_FACE)
endif()

# Optional: print VGPR/scratch/occupancy per device kernel (and optionally dump
# GCN asm) for the Z4c kernels, which live in evolve.cpp.  Toggle with
#   cmake -DGRACE_DUMP_GPU_RESOURCE_USAGE=ON ...
option(GRACE_DUMP_GPU_RESOURCE_USAGE "Emit per-kernel VGPR/scratch diagnostics (HIP/amdclang++)" OFF)
if(GRACE_DUMP_GPU_RESOURCE_USAGE)
    # Target-level so the flags always propagate through Kokkos's launcher
    # (set_source_files_properties has directory-scope quirks with OBJECT libs).
    # Affects all 4 files in grace_evolution, which is fine for diagnostics.
    target_compile_options(grace_evolution PRIVATE
        -Rpass-analysis=kernel-resource-usage
        --save-temps=obj)
endif()