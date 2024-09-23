function(register_grace_object_conditional target_name sources condition)
    # Check if the condition is true
    if(${condition})
        # Add the object library for the target
        add_library(${target_name} OBJECT ${sources})

        # Set the include directories, compile options, and link libraries for the target
        target_include_directories(${target_name} PRIVATE "${HEADER_DIR}")
        target_compile_options(${target_name} PRIVATE -Wno-dangling-field)
        target_link_libraries(${target_name} PRIVATE
            yaml_cpp::yaml
            MPI::MPI_CXX
            OpenMP::OpenMP_CXX
            p4est::sc
            p4est::p4est 
            Kokkos::kokkos 
            spdlog::spdlog
            HDF5::HDF5
            ZLIB::ZLIB)

        # Register the object files of the target into the grace_objects interface library
        target_sources(grace_objects INTERFACE $<TARGET_OBJECTS:${target_name}>)
    endif()
endfunction()

function(register_grace_object target_name sources)
    # Check if the condition is true
    if(${condition})
        # Add the object library for the target
        add_library(${target_name} OBJECT ${sources})

        # Set the include directories, compile options, and link libraries for the target
        target_include_directories(${target_name} PRIVATE "${HEADER_DIR}")
        target_compile_options(${target_name} PRIVATE -Wno-dangling-field)
        target_link_libraries(${target_name} PRIVATE
            yaml_cpp::yaml
            MPI::MPI_CXX
            OpenMP::OpenMP_CXX
            p4est::sc
            p4est::p4est 
            Kokkos::kokkos 
            spdlog::spdlog
            HDF5::HDF5
            ZLIB::ZLIB)

        # Register the object files of the target into the grace_objects interface library
        target_sources(grace_objects INTERFACE $<TARGET_OBJECTS:${target_name}>)
    endif()
endfunction()
