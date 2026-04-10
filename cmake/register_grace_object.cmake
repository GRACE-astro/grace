function(register_grace_object_conditional target_name condition)
    set(sources "${ARGN}")
    # Check if the condition is true
    if(${condition})
        # Add the object library for the target
        add_library(${target_name} OBJECT ${sources})

        # Set the include directories, compile options, and link libraries for the target
        target_include_directories(${target_name} PRIVATE "${HEADER_DIR}")
        target_link_libraries(${target_name} PRIVATE
            yaml_cpp::yaml
            MPI::MPI_CXX
            OpenMP::OpenMP_CXX
            p4est::sc
            p4est::p4est 
            Kokkos::kokkos 
            spdlog::spdlog
            HDF5::HDF5
            ZLIB::ZLIB
            # FUKA handling:
            $<$<BOOL:${GRACE_ENABLE_FUKA}>:Kadath::kadath>
            $<$<BOOL:${GRACE_ENABLE_VTK}>:VTK::VTK>
            $<$<BOOL:${GRACE_ENABLE_PROFILING}>:GRACE_GPUProfiling>
            $<$<BOOL:${GRACE_ENABLE_LORENE}>:LORENE::LORENE>
            $<$<BOOL:${GRACE_ENABLE_TWO_PUNCTURES}>:TwoPunctures::TwoPunctures>)
        # Init VTK modules if support is requested 
        if ( GRACE_ENABLE_VTK )
            vtk_module_autoinit(
                TARGETS grace
                MODULES ${VTK_LIBRARIES}
            )
        endif()
        # Register the object files of the target into the grace_objects interface library
        target_sources(grace_objects INTERFACE $<TARGET_OBJECTS:${target_name}>)
    endif()
endfunction()

function(register_grace_object target_name)
    set(sources "${ARGN}")
    # Add the object library for the target
    add_library(${target_name} OBJECT ${sources})

    # Set the include directories, compile options, and link libraries for the target
    target_include_directories(${target_name} PRIVATE "${HEADER_DIR}")
    target_link_libraries(${target_name} PRIVATE
        yaml_cpp::yaml
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        p4est::sc
        p4est::p4est 
        Kokkos::kokkos 
        spdlog::spdlog
        HDF5::HDF5
        ZLIB::ZLIB
        # FUKA handling:
        $<$<BOOL:${GRACE_ENABLE_FUKA}>:Kadath::kadath>
        $<$<BOOL:${GRACE_ENABLE_VTK}>:VTK::VTK>
        $<$<BOOL:${GRACE_ENABLE_PROFILING}>:GRACE_GPUProfiling>
        $<$<BOOL:${GRACE_ENABLE_LORENE}>:LORENE::LORENE>
        $<$<BOOL:${GRACE_ENABLE_TWO_PUNCTURES}>:TwoPunctures::TwoPunctures>)
    # Register the object files of the target into the grace_objects interface library
    target_sources(grace_objects INTERFACE $<TARGET_OBJECTS:${target_name}>)
endfunction()
