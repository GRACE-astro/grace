find_package(Doxygen REQUIRED dot)

find_program(SPHINXBUILD sphinx-build)
find_package(Python REQUIRED COMPONENTS Interpreter)

if(NOT SPHINXBUILD)
  message(FATAL_ERROR "The 'sphinx-build' command was not found")
endif()

find_package(LATEX COMPONENTS PDFLATEX)
find_program(LATEXMK latexmk)
if(NOT LATEX_PDFLATEX_FOUND OR NOT LATEXMK)
  message(WARNING "LaTeX components not found. PDF build will not be available.")
endif()

set(GRACE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(GRACE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GRACE_SOURCE_DIR ${CMAKE_SOURCE_DIR})

set(DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/grace.doxyfile.in")
set(DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/grace.doxyfile")

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

set(SPHINX_CONF_IN "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in")
set(SPHINX_CONF_OUT "${CMAKE_CURRENT_SOURCE_DIR}/conf.py")

configure_file(${SPHINX_CONF_IN} ${SPHINX_CONF_OUT} @ONLY)

set(TABLE_GEN_IN "${CMAKE_CURRENT_SOURCE_DIR}/generate_rst_file_from_yaml.py.in")
set(TABLE_GEN_OUT "${CMAKE_CURRENT_BINARY_DIR}/generate_rst_file_from_yaml.py")

configure_file(${TABLE_GEN_IN} ${TABLE_GEN_OUT} @ONLY)

add_custom_target(rst_files
        COMMAND ${Python_EXECUTABLE} ${TABLE_GEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating rst files."
        VERBATIM)

add_custom_target(doc_doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
add_custom_target(doc_sphinx
        COMMAND ${SPHINXBUILD} -b html ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating documentation with Sphinx"
        VERBATIM)
add_dependencies(doc_sphinx doc_doxygen)
add_dependencies(doc_sphinx rst_files)
add_custom_target(doc ALL DEPENDS rst_files doc_doxygen doc_sphinx)

